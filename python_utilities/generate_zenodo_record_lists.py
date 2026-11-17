"""Generate current- and previous-version lists for the TSML Zenodo community."""

import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import build_opener

API_URL = "https://zenodo.org/api/records"
USER_AGENT = "tsml-archive"
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT = 120

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
TOKEN_PATH = REPOSITORY_ROOT / ".zenodo_token"
RECORDS_DIR = REPOSITORY_ROOT / "zenodo_records"
CURRENT_VERSIONS_PATH = RECORDS_DIR / "current-versions"
PREVIOUS_VERSIONS_PATH = RECORDS_DIR / "previous-versions"


def confirm_run(input_fn=None):
    """Confirm the production read and local writes before continuing."""
    if input_fn is None:
        input_fn = input

    try:
        response = input_fn(
            "This reads production Zenodo and replaces the local record lists. "
            "Press Enter to continue: "
        )
    except (EOFError, KeyboardInterrupt):
        return False

    return response == ""


def load_token():
    """Load the optional Zenodo token from the repository root."""
    if not TOKEN_PATH.is_file():
        print("Zenodo token not loaded; using unauthenticated requests.")
        return None

    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        print("Zenodo token not loaded; the token file is empty.")
        return None

    print(f"Zenodo token loaded from {TOKEN_PATH.relative_to(REPOSITORY_ROOT)}.")
    return token


def create_opener(token):
    """Create a URL opener, with Zenodo authentication when available."""
    opener = build_opener()
    opener.addheaders = [("User-Agent", USER_AGENT)]
    if token:
        opener.addheaders.append(("Authorization", f"Bearer {token}"))
    return opener


def get_page(opener, page, page_size):
    """Retrieve one page of records, retrying transient server errors."""
    params = {
        "all_versions": "true",
        "communities": "tsml",
        "page": page,
        "size": page_size,
        "sort": "newest",
    }

    url = f"{API_URL}?{urlencode(params)}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with opener.open(url, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in {429, 502, 503, 504}:
                raise
            retry_after = error.headers.get("Retry-After")
        except (TimeoutError, URLError):
            retry_after = None

        if attempt < MAX_ATTEMPTS:
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 20
            time.sleep(delay)

    raise RuntimeError(f"Too many transient errors retrieving page {page}.")


def fetch_records(opener, page_size):
    """Retrieve every version of every record in the Zenodo community."""
    records = []
    expected_total = None
    page = 1

    while True:
        data = get_page(opener, page, page_size)
        hit_data = data.get("hits", {})
        hits = hit_data.get("hits", [])
        if expected_total is None:
            expected_total = hit_data.get("total")

        records.extend(hits)
        if not hits or (
            expected_total is not None and len(records) >= expected_total
        ):
            break
        page += 1

    if expected_total is not None and len(records) != expected_total:
        raise ValueError(
            f"Zenodo reported {expected_total} records but returned {len(records)}."
        )

    return records


def record_sort_key(record):
    """Return a key that places the newest version first."""
    return record.get("created", ""), int(record["id"])


def split_record_versions(records):
    """Split records into the newest and previous versions of each concept."""
    versions_by_concept = defaultdict(list)

    for record in records:
        record_id = record.get("id")
        title = record.get("metadata", {}).get("title")
        if record_id is None or not title:
            raise ValueError("Zenodo returned a record without an ID or title.")

        concept_id = record.get("conceptrecid") or record_id
        versions_by_concept[str(concept_id)].append(record)

    current_versions = []
    previous_versions = []
    for versions in versions_by_concept.values():
        versions.sort(key=record_sort_key, reverse=True)
        current_versions.append(versions[0])
        previous_versions.extend(versions[1:])

    current_versions.sort(key=record_sort_key, reverse=True)
    previous_versions.sort(key=record_sort_key, reverse=True)
    return current_versions, previous_versions


def write_record_list(path, records):
    """Atomically write records in the repository's ``title: ID`` format."""
    contents = "".join(
        f"{record['metadata']['title']}: {record['id']}\n" for record in records
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def validate_unique_titles(records):
    """Raise an error if current-version titles are not unique."""
    seen = set()
    duplicates = set()
    for record in records:
        title = record["metadata"]["title"]
        if title in seen:
            duplicates.add(title)
        seen.add(title)

    if duplicates:
        duplicate_titles = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate current-version titles: {duplicate_titles}")


def main(input_fn=None):
    """Fetch Zenodo records and update the two repository record lists."""
    if not confirm_run(input_fn):
        return 1

    token = load_token()
    page_size = 100 if token else 25
    opener = create_opener(token)

    records = fetch_records(opener, page_size)
    current_versions, previous_versions = split_record_versions(records)
    validate_unique_titles(current_versions)

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    write_record_list(CURRENT_VERSIONS_PATH, current_versions)
    write_record_list(PREVIOUS_VERSIONS_PATH, previous_versions)

    print(
        f"Wrote {len(current_versions)} current versions to "
        f"{CURRENT_VERSIONS_PATH.relative_to(REPOSITORY_ROOT)}."
    )
    print(
        f"Wrote {len(previous_versions)} previous versions to "
        f"{PREVIOUS_VERSIONS_PATH.relative_to(REPOSITORY_ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
