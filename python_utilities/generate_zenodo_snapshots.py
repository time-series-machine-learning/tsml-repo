"""Generate version and file snapshots for the TSML Zenodo community."""

import json
from collections import defaultdict
from datetime import datetime, timezone

from python_utilities import generate_zenodo_record_lists as record_lists

MANIFEST_DIR = record_lists.REPOSITORY_ROOT / "manifest"
SCHEMA_DIR = MANIFEST_DIR / "schema"
VERSIONS_SNAPSHOT_PATH = MANIFEST_DIR / "zenodo_versions.jsonl"
FILES_SNAPSHOT_PATH = MANIFEST_DIR / "zenodo_files.jsonl"
SNAPSHOT_PATH = MANIFEST_DIR / "snapshot.jsonl"
VERSION_SCHEMA_PATH = SCHEMA_DIR / "zenodo_version.schema.json"
FILE_SCHEMA_PATH = SCHEMA_DIR / "zenodo_file.schema.json"
SNAPSHOT_SCHEMA_PATH = SCHEMA_DIR / "snapshot.schema.json"


def confirm_run(input_fn=None):
    """Confirm the production read and local writes before continuing."""
    if input_fn is None:
        input_fn = input

    try:
        response = input_fn(
            "This reads production Zenodo and replaces the local snapshots. "
            "Press Enter to continue: "
        )
    except (EOFError, KeyboardInterrupt):
        return False

    return response == ""


def build_version_rows(records, current_versions):
    """Normalize published record versions for the JSONL snapshot."""
    current_ids = {int(record["id"]) for record in current_versions}
    rows = []

    for record in records:
        record_id = int(record["id"])
        concept_id = int(record.get("conceptrecid") or record_id)
        title = record.get("metadata", {}).get("title")
        record_url = record.get("links", {}).get("self_html")
        if not title or not record_url:
            raise ValueError(
                f"Zenodo record {record_id} has no title or record URL."
            )

        rows.append(
            {
                "concept_id": concept_id,
                "record_id": record_id,
                "title": title,
                "is_current": record_id in current_ids,
                "record_url": record_url,
            }
        )

    return sorted(
        rows,
        key=lambda row: (row["concept_id"], row["record_id"]),
    )


def build_file_rows(records):
    """Normalize every file attached to the published record versions."""
    rows = []

    for record in records:
        record_id = int(record["id"])
        concept_id = int(record.get("conceptrecid") or record_id)
        files = record.get("files")
        if not isinstance(files, list):
            raise ValueError(f"Zenodo record {record_id} has no file list.")

        for file_data in files:
            links = file_data.get("links", {})
            required = {
                "checksum": file_data.get("checksum"),
                "download URL": links.get("self"),
                "key": file_data.get("key"),
                "size": file_data.get("size"),
            }
            missing = [
                name
                for name, value in required.items()
                if value is None or value == ""
            ]
            if missing:
                missing_fields = ", ".join(missing)
                raise ValueError(
                    f"A file in Zenodo record {record_id} is missing: "
                    f"{missing_fields}."
                )

            rows.append(
                {
                    "concept_id": concept_id,
                    "record_id": record_id,
                    "key": required["key"],
                    "size": int(required["size"]),
                    "checksum": required["checksum"],
                    "download_url": required["download URL"],
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["concept_id"],
            row["record_id"],
            row["key"].casefold(),
            row["key"],
        ),
    )


def jsonl_contents(rows):
    """Serialize rows as deterministic newline-delimited JSON."""
    return "".join(
        f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows
    )


def utc_now():
    """Return the current UTC time in a compact ISO 8601 format."""
    value = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return value.replace("+00:00", "Z")


def build_snapshot(version_rows, file_rows, retrieved_at):
    """Build one timestamped entry for the snapshot history."""
    current_count = sum(row["is_current"] for row in version_rows)
    return {
        "community": "tsml",
        "retrieved_at": retrieved_at,
        "version_count": len(version_rows),
        "current_record_count": current_count,
        "previous_version_count": len(version_rows) - current_count,
        "file_count": len(file_rows),
    }


def validate_snapshot(version_rows, file_rows, snapshot):
    """Validate relationships and counts across the generated snapshots."""
    if not version_rows:
        raise ValueError("The Zenodo snapshot contains no record versions.")
    records = {}
    current_by_concept = defaultdict(int)
    for row in version_rows:
        record_id = row["record_id"]
        if record_id in records:
            raise ValueError(f"Duplicate version record ID: {record_id}.")
        records[record_id] = row["concept_id"]
        current_by_concept[row["concept_id"]] += row["is_current"]

    invalid_concepts = [
        concept_id
        for concept_id, current_count in current_by_concept.items()
        if current_count != 1
    ]
    if invalid_concepts:
        concepts = ", ".join(str(value) for value in sorted(invalid_concepts))
        raise ValueError(f"Concepts without exactly one current version: {concepts}.")

    seen_files = set()
    for row in file_rows:
        record_id = row["record_id"]
        if record_id not in records:
            raise ValueError(f"File references unknown record ID: {record_id}.")
        if row["concept_id"] != records[record_id]:
            raise ValueError(f"File concept does not match record ID {record_id}.")

        identity = record_id, row["key"]
        if identity in seen_files:
            raise ValueError(
                f"Duplicate file for record {record_id}: {row['key']}."
            )
        seen_files.add(identity)

    retrieved_at = snapshot.get("retrieved_at")
    if not retrieved_at:
        raise ValueError("Snapshot history entry has no retrieval time.")
    if snapshot != build_snapshot(version_rows, file_rows, retrieved_at):
        raise ValueError("Snapshot summary does not match the generated rows.")


def write_jsonl(path, rows):
    """Atomically write deterministic newline-delimited JSON."""
    contents = jsonl_contents(rows)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def snapshot_counts_have_changed(path, snapshot):
    """Compare generated counts with the latest snapshot history entry."""
    if not path.is_file():
        return True

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return True

    latest = json.loads(lines[-1])
    count_fields = (
        "version_count",
        "current_record_count",
        "previous_version_count",
        "file_count",
    )
    return any(latest.get(field) != snapshot[field] for field in count_fields)


def append_jsonl(path, row):
    """Atomically append one row to a JSONL history file."""
    contents = path.read_text(encoding="utf-8") if path.is_file() else ""
    if contents and not contents.endswith("\n"):
        contents += "\n"
    contents += jsonl_contents([row])

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def main(input_fn=None):
    """Fetch public Zenodo metadata and update the local snapshots."""
    if not confirm_run(input_fn):
        return 1

    print("Using unauthenticated Zenodo requests.")
    opener = record_lists.create_opener(None)
    records = record_lists.fetch_records(opener, page_size=25)
    current_versions, _ = record_lists.split_record_versions(records)
    version_rows = build_version_rows(records, current_versions)
    file_rows = build_file_rows(records)
    snapshot = build_snapshot(version_rows, file_rows, utc_now())
    validate_snapshot(version_rows, file_rows, snapshot)
    add_history = snapshot_counts_have_changed(SNAPSHOT_PATH, snapshot)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(VERSIONS_SNAPSHOT_PATH, version_rows)
    write_jsonl(FILES_SNAPSHOT_PATH, file_rows)
    if add_history:
        append_jsonl(SNAPSHOT_PATH, snapshot)
        print("Added an entry to the snapshot history.")
    else:
        print("Snapshot counts unchanged; history not updated.")

    print(
        f"Wrote {len(version_rows)} versions and {len(file_rows)} files to "
        f"{MANIFEST_DIR.relative_to(record_lists.REPOSITORY_ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
