"""Shared client, target resolution, and safe update helpers for Zenodo maintenance."""

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from python_utilities.generate_zenodo_record_lists import RejectRedirects

PRODUCTION_API_ROOT = "https://zenodo.org/api"
SANDBOX_API_ROOT = "https://sandbox.zenodo.org/api"
ALLOWED_API_ROOTS = {PRODUCTION_API_ROOT, SANDBOX_API_ROOT}

USER_AGENT = "tsml-archive"
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT = 120
TRANSIENT_STATUS_CODES = {429, 502, 503, 504}

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
PRODUCTION_TOKEN_PATH = REPOSITORY_ROOT / ".zenodo_token"
SANDBOX_TOKEN_PATH = REPOSITORY_ROOT / ".zenodo_sandbox_token"
CURRENT_VERSIONS_PATH = REPOSITORY_ROOT / "zenodo_records" / "current-versions"


class ZenodoMaintenanceError(RuntimeError):
    """Report an unsafe, misconfigured, or unsuccessful Zenodo maintenance action."""


@dataclass(frozen=True)
class RecordReference:
    """Identify a Zenodo record and, when known locally, its expected title."""

    record_id: str
    title: str | None = None


@dataclass(frozen=True)
class PlannedChange:
    """Describe one verified metadata update ready for preview or publication."""

    record_id: str
    title: str
    old_value: Any
    new_value: Any
    mutate_metadata: Callable[[dict], None]
    extract_value: Callable[[dict, str], Any]


def normalize_api_root(api_root):
    """Validate that ``api_root`` is an allowed HTTPS Zenodo API endpoint."""
    normalized = api_root.rstrip("/")
    if normalized not in ALLOWED_API_ROOTS:
        raise ZenodoMaintenanceError(
            f"Unsupported Zenodo API root {api_root!r}; expected one of "
            f"{sorted(ALLOWED_API_ROOTS)}."
        )
    return normalized


def resolve_token_path(api_root, token_path=None):
    """Select and validate the credential path for ``api_root`` without reading it."""
    normalized_root = normalize_api_root(api_root)
    is_sandbox = normalized_root == SANDBOX_API_ROOT
    if token_path is None:
        return SANDBOX_TOKEN_PATH if is_sandbox else PRODUCTION_TOKEN_PATH

    resolved = Path(token_path).resolve()
    if is_sandbox and resolved == PRODUCTION_TOKEN_PATH.resolve():
        raise ZenodoMaintenanceError(
            "Refusing to use the production .zenodo_token with sandbox.zenodo.org."
        )
    if not is_sandbox and resolved == SANDBOX_TOKEN_PATH.resolve():
        raise ZenodoMaintenanceError(
            "Refusing to use .zenodo_sandbox_token with production zenodo.org."
        )
    return Path(token_path)


def load_required_token(path):
    """Load a non-empty Zenodo token required for an explicit ``--apply`` run."""
    if not path.is_file():
        raise ZenodoMaintenanceError(
            f"Cannot apply changes without a token at {path}."
        )
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ZenodoMaintenanceError(
            f"Cannot apply changes because {path} is empty."
        )
    return token


def confirm_apply(count, api_root, input_fn=None):
    """Require explicit interactive ``APPLY`` confirmation before mutating Zenodo."""
    if input_fn is None:
        input_fn = input

    environment = (
        "sandbox" if normalize_api_root(api_root) == SANDBOX_API_ROOT else "production"
    )
    prompt = (
        f"This will modify and publish {count} record(s) on {environment} Zenodo. "
        "Type APPLY to continue: "
    )
    try:
        response = input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return response == "APPLY"


def _format_http_error(error, method, url):
    """Build a concise diagnostic message from a non-transient Zenodo HTTPError."""
    detail = ""
    if error.fp is not None:
        try:
            raw = error.read().decode("utf-8", errors="replace").strip()
            if raw:
                detail = f" — {raw[:500]}"
        except OSError:
            detail = ""
    return f"Zenodo {method} {url} failed with HTTP {error.code}: {error.reason}{detail}"


class ZenodoClient:
    """Minimal HTTPS Zenodo client with redirect protection for bearer tokens."""

    def __init__(self, token=None, api_root=PRODUCTION_API_ROOT):
        self.api_root = normalize_api_root(api_root)
        self.opener = build_opener(RejectRedirects()) if token else build_opener()
        self.opener.addheaders = [
            ("Accept", "application/json"),
            ("User-Agent", USER_AGENT),
        ]
        if token:
            self.opener.addheaders.append(("Authorization", f"Bearer {token}"))

    def request_json(self, method, path, payload=None):
        """Send an API request, retrying transient errors on idempotent methods."""
        url = f"{self.api_root}/{path.lstrip('/')}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        attempts = MAX_ATTEMPTS if method in {"GET", "PUT"} else 1
        last_error = None
        for attempt in range(1, attempts + 1):
            request = Request(url, data=data, headers=headers, method=method)
            try:
                with self.opener.open(request, timeout=REQUEST_TIMEOUT) as response:
                    body = response.read()
                return json.loads(body) if body else None
            except HTTPError as error:
                if error.code not in TRANSIENT_STATUS_CODES:
                    raise ZenodoMaintenanceError(
                        _format_http_error(error, method, url)
                    ) from error
                last_error = error
                retry_after = error.headers.get("Retry-After")
            except (TimeoutError, URLError) as error:
                last_error = error
                retry_after = None

            if attempt < attempts:
                delay = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 20
                )
                time.sleep(delay)

        raise ZenodoMaintenanceError(
            f"Too many transient errors calling {url}."
        ) from last_error

    def get_public_record_metadata(self, record_id):
        """Fetch a record's title and metadata dict via the public read-only API."""
        record = self.request_json("GET", f"records/{record_id}")
        metadata = record.get("metadata") if isinstance(record, dict) else None
        if not isinstance(metadata, dict):
            raise ZenodoMaintenanceError(
                f"Zenodo record {record_id} does not have usable metadata."
            )
        title = metadata.get("title")
        if not isinstance(title, str) or not title:
            raise ZenodoMaintenanceError(
                f"Zenodo record {record_id} does not have a usable title."
            )
        return title, metadata


def _meaningful_lines(path):
    """Yield ``(line_number, stripped_line)`` for non-empty, non-comment lines."""
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            yield line_number, line


def _parse_titled_reference(line, source_label):
    """Parse one ``title: record_id`` entry."""
    title, separator, record_id = line.rpartition(": ")
    if not separator or not title or not record_id.isdecimal():
        raise ValueError(
            f"{source_label} must be a numeric record ID, a title from the "
            "record index, or a 'title: record-id' entry."
        )
    return RecordReference(record_id, title)


def load_record_index(path):
    """Load unique ``title -> record_id`` mappings from a Zenodo record list."""
    records = {}
    record_ids = set()
    for line_number, line in _meaningful_lines(path):
        reference = _parse_titled_reference(line, f"{path}:{line_number}")
        if reference.title in records:
            raise ValueError(f"{path}:{line_number} repeats {reference.title!r}.")
        if reference.record_id in record_ids:
            raise ValueError(
                f"{path}:{line_number} repeats record ID {reference.record_id}."
            )
        records[reference.title] = reference.record_id
        record_ids.add(reference.record_id)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def resolve_record_references(
    record_file=None,
    explicit_records=None,
    index_path=CURRENT_VERSIONS_PATH,
):
    """Resolve file and/or CLI record specs into unique ``RecordReference`` items."""
    raw_entries = []
    if record_file is not None:
        file_entries = list(_meaningful_lines(record_file))
        if not file_entries:
            raise ValueError(f"{record_file} does not contain any records.")
        for line_number, line in file_entries:
            raw_entries.append((f"{record_file}:{line_number}", line))

    for idx, value in enumerate(explicit_records or (), start=1):
        cleaned = value.strip() if isinstance(value, str) else ""
        if not cleaned:
            raise ValueError(f"--record #{idx} must not be empty.")
        raw_entries.append((f"--record #{idx}", cleaned))

    if not raw_entries:
        raise ValueError(
            "Specify a record_file and/or at least one --record target."
        )

    parsed_rows = []
    unresolved_titles = []
    for label, text in raw_entries:
        if text.isdecimal():
            parsed_rows.append((label, RecordReference(text)))
            continue
        try:
            reference = _parse_titled_reference(text, label)
        except ValueError:
            unresolved_titles.append(text)
            reference = RecordReference("", text)
        parsed_rows.append((label, reference))

    if unresolved_titles:
        index = load_record_index(index_path)
        missing = sorted(set(unresolved_titles).difference(index))
        if missing:
            raise ValueError(
                f"Records missing from {index_path}: " + ", ".join(missing)
            )
        parsed_rows = [
            (
                label,
                RecordReference(index[ref.title], ref.title)
                if not ref.record_id
                else ref,
            )
            for label, ref in parsed_rows
        ]

    references = []
    seen_ids = set()
    seen_titles = set()
    for label, ref in parsed_rows:
        if ref.record_id in seen_ids:
            raise ValueError(f"{label} repeats record ID {ref.record_id}.")
        if ref.title is not None and ref.title in seen_titles:
            raise ValueError(f"{label} repeats {ref.title!r}.")
        seen_ids.add(ref.record_id)
        if ref.title is not None:
            seen_titles.add(ref.title)
        references.append(ref)
    return references


def _deposition_path(record_id):
    return f"deposit/depositions/{record_id}"


def _deposition_metadata(deposition, record_id):
    metadata = deposition.get("metadata") if isinstance(deposition, dict) else None
    if not isinstance(metadata, dict):
        raise ZenodoMaintenanceError(
            f"Zenodo deposition {record_id} does not have usable metadata."
        )
    return metadata


def is_published_locked(deposition):
    """Return True when a published deposition is locked (``state == 'done'``)."""
    return (
        isinstance(deposition, dict)
        and deposition.get("submitted") is True
        and deposition.get("state") == "done"
    )


def is_open_edit_session(deposition):
    """Return True when a published deposition is unlocked (``state == 'inprogress'``)."""
    return (
        isinstance(deposition, dict)
        and deposition.get("submitted") is True
        and deposition.get("state") == "inprogress"
    )


def discard_open_edit(client, record_id):
    """Discard an open metadata edit session if ``record_id`` is ``inprogress``."""
    path = _deposition_path(record_id)
    deposition = client.request_json("GET", path)
    if is_published_locked(deposition):
        return
    if not is_open_edit_session(deposition):
        state = deposition.get("state") if isinstance(deposition, dict) else None
        submitted = (
            deposition.get("submitted") if isinstance(deposition, dict) else None
        )
        raise ZenodoMaintenanceError(
            f"Could not determine edit state for record {record_id} "
            f"(state={state!r}, submitted={submitted!r})."
        )

    discarded = client.request_json("POST", f"{path}/actions/discard")
    verified = (
        discarded if isinstance(discarded, dict) else client.request_json("GET", path)
    )
    if not is_published_locked(verified):
        state = verified.get("state") if isinstance(verified, dict) else None
        submitted = verified.get("submitted") if isinstance(verified, dict) else None
        raise ZenodoMaintenanceError(
            f"Zenodo did not confirm that record {record_id} returned to state='done' "
            f"(state={state!r}, submitted={submitted!r})."
        )


def apply_record_change(client, change, *, discard_existing_edit=False):
    """Unlock, verify, update, and republish one record, discarding on failure."""
    path = _deposition_path(change.record_id)
    before = client.request_json("GET", path)

    if is_open_edit_session(before):
        draft_meta = _deposition_metadata(before, change.record_id)
        draft_title = draft_meta.get("title")
        draft_value = change.extract_value(draft_meta, change.record_id)
        if not discard_existing_edit:
            raise ZenodoMaintenanceError(
                f"Record {change.record_id} already has an open edit session "
                "(state='inprogress'); pass --discard-existing-edit to reset an "
                "unedited draft, or check the record in Zenodo."
            )
        if draft_title != change.title or draft_value != change.old_value:
            raise ZenodoMaintenanceError(
                f"Record {change.record_id} has an open edit session with modified "
                "metadata; refusing to discard it automatically."
            )
        discard_open_edit(client, change.record_id)
        before = client.request_json("GET", path)

    if not is_published_locked(before):
        state = before.get("state") if isinstance(before, dict) else None
        submitted = before.get("submitted") if isinstance(before, dict) else None
        raise ZenodoMaintenanceError(
            f"Record {change.record_id} is not a locked published record "
            f"(state={state!r}, submitted={submitted!r})."
        )

    metadata = _deposition_metadata(before, change.record_id)
    current_title = metadata.get("title")
    current_value = change.extract_value(metadata, change.record_id)
    if current_title != change.title or current_value != change.old_value:
        raise ZenodoMaintenanceError(
            f"Record {change.record_id} changed after the preview; refusing to "
            "overwrite newer metadata."
        )

    edit_attempted = False
    try:
        edit_attempted = True
        client.request_json("POST", f"{path}/actions/edit")

        draft = client.request_json("GET", path)
        if not is_open_edit_session(draft):
            raise ZenodoMaintenanceError(
                f"Zenodo did not confirm that record {change.record_id} entered "
                "state='inprogress'."
            )
        draft_metadata = _deposition_metadata(draft, change.record_id)
        if (
            draft_metadata.get("title") != change.title
            or change.extract_value(draft_metadata, change.record_id)
            != change.old_value
        ):
            raise ZenodoMaintenanceError(
                f"Record {change.record_id} changed while it was being unlocked."
            )

        updated_metadata = deepcopy(draft_metadata)
        change.mutate_metadata(updated_metadata)
        client.request_json("PUT", path, {"metadata": updated_metadata})

        published = client.request_json("POST", f"{path}/actions/publish")
        verified_pub = (
            published
            if isinstance(published, dict)
            else client.request_json("GET", path)
        )
        if not is_published_locked(verified_pub):
            state = (
                verified_pub.get("state") if isinstance(verified_pub, dict) else None
            )
            submitted = (
                verified_pub.get("submitted")
                if isinstance(verified_pub, dict)
                else None
            )
            raise ZenodoMaintenanceError(
                f"Zenodo did not confirm that record {change.record_id} returned "
                f"to state='done' after publishing "
                f"(state={state!r}, submitted={submitted!r})."
            )
    except BaseException as update_error:
        if edit_attempted:
            try:
                discard_open_edit(client, change.record_id)
            except BaseException as cleanup_error:
                raise ZenodoMaintenanceError(
                    f"Updating record {change.record_id} failed and its edit session "
                    "could not be safely discarded; check the record in Zenodo."
                ) from cleanup_error
        raise update_error


def format_json_value(value):
    """Format a tuple or list value compactly as JSON for terminal diffs."""
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, ensure_ascii=False)


def clean_tags(tags):
    """Validate, strip, and deduplicate one or more keyword tags."""
    if not isinstance(tags, (list, tuple)) or not tags:
        raise ValueError("At least one --tag value is required.")

    cleaned = []
    for raw_tag in tags:
        if not isinstance(raw_tag, str) or not raw_tag.strip():
            raise ValueError("Every configured tag must be a non-empty string.")
        tag = raw_tag.strip()
        if tag in cleaned:
            raise ValueError(f"Duplicate configured tag: {tag}")
        cleaned.append(tag)
    return tuple(cleaned)


def extract_record_tags(metadata, record_id):
    """Extract and validate a record's ``keywords`` list as a tuple of strings."""
    tags = metadata.get("keywords", [])
    if not isinstance(tags, list) or any(
        not isinstance(tag, str) or not tag for tag in tags
    ):
        raise ZenodoMaintenanceError(
            f"Zenodo record {record_id} does not have a usable keywords list."
        )
    return tuple(tags)


def add_common_arguments(parser):
    """Add target selection, index, sandbox, token, and apply flags to ``parser``."""
    parser.add_argument(
        "record_file",
        nargs="?",
        type=Path,
        help=(
            "optional file containing record IDs, titles from the record index, "
            "or 'title: record-id' entries"
        ),
    )
    parser.add_argument(
        "--record",
        dest="records",
        action="append",
        default=[],
        metavar="TARGET",
        help="target record ID, title, or 'title: record-id' (repeatable)",
    )
    parser.add_argument(
        "--record-index",
        type=Path,
        default=CURRENT_VERSIONS_PATH,
        help="title-to-ID index used for title-only targets",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="target sandbox.zenodo.org instead of production Zenodo",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help="override token file path used only with --apply",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="edit and publish records after preview and confirmation",
    )
    parser.add_argument(
        "--discard-existing-edit",
        action="store_true",
        help="discard a pre-existing unedited open draft session before applying",
    )
    return parser


def run_metadata_update(
    references,
    build_change,
    *,
    apply=False,
    discard_existing_edit=False,
    api_root=PRODUCTION_API_ROOT,
    token_path=None,
    client_factory=None,
    prefetched_metadata=None,
    input_fn=None,
    noun="record metadata",
):
    """Preview changes via public GET, then optionally confirm and publish them."""
    if client_factory is None:
        client_factory = ZenodoClient
    normalized_root = normalize_api_root(api_root)
    effective_token_path = (
        resolve_token_path(normalized_root, token_path) if apply else None
    )

    read_client = client_factory(None, normalized_root)
    planned = []
    pending = []

    for ref in references:
        if prefetched_metadata and ref.record_id in prefetched_metadata:
            title, metadata = prefetched_metadata[ref.record_id]
        else:
            title, metadata = read_client.get_public_record_metadata(ref.record_id)
        if ref.title is not None and ref.title != title:
            raise ZenodoMaintenanceError(
                f"Record {ref.record_id} title mismatch: expected {ref.title!r}, "
                f"found {title!r}."
            )
        change = build_change(ref.record_id, title, metadata)
        planned.append(change)
        changed = change.old_value != change.new_value
        label = "PLAN" if apply and changed else "CHANGE" if changed else "UNCHANGED"
        print(f"{label}\t{change.record_id}\t{change.title}")
        if changed:
            print(f"  {format_json_value(change.old_value)}")
            print(f"  -> {format_json_value(change.new_value)}")
            pending.append(change)

    skipped = len(planned) - len(pending)
    if not apply:
        print(
            f"Preview only: {len(pending)} {noun} update(s) would change; "
            f"{skipped} already match. No edit sessions were opened."
        )
        print("Re-run with --apply to publish these changes.")
        return 0

    if not pending:
        print(f"Nothing to apply: all {skipped} record(s) already match.")
        return 0

    if not confirm_apply(len(pending), normalized_root, input_fn=input_fn):
        print("Cancelled. No records were changed.")
        return 1

    token = load_required_token(effective_token_path)
    write_client = client_factory(token, normalized_root)
    for change in pending:
        apply_record_change(
            write_client,
            change,
            discard_existing_edit=discard_existing_edit,
        )
        print(f"UPDATED\t{change.record_id}\t{change.title}")

    print(f"Published {len(pending)} {noun} change(s); skipped {skipped} record(s).")
    return 0
