"""Generate the TSML record catalogue from local snapshots and lists."""

import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
MANIFEST_DIR = REPOSITORY_ROOT / "manifest"
VERSIONS_SNAPSHOT_PATH = MANIFEST_DIR / "zenodo_versions.jsonl"
RECORDS_CATALOG_PATH = MANIFEST_DIR / "records.jsonl"
RECORD_LISTS_DIR = REPOSITORY_ROOT / "record_lists"
RECORD_KINDS = {"aggregate", "dataset"}


def read_jsonl(path):
    """Read non-empty JSONL rows, or return an empty list if absent."""
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path, rows):
    """Atomically write deterministic newline-delimited JSON."""
    contents = "".join(
        f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def load_record_memberships(directory):
    """Map record names to the record-list files containing them."""
    memberships = defaultdict(set)
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name.casefold() == "readme.md":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                memberships[name].add(path.name)

    return {
        name: sorted(list_names)
        for name, list_names in memberships.items()
    }


def load_previous_kinds(rows):
    """Index manually assigned record kinds by Zenodo concept ID."""
    previous_kinds = {}
    for row in rows:
        concept_id = int(row["concept_id"])
        if concept_id in previous_kinds:
            raise ValueError(f"Duplicate catalogue concept ID: {concept_id}.")

        kind = row.get("kind", "")
        if kind and kind not in RECORD_KINDS:
            raise ValueError(f"Invalid kind for concept {concept_id}: {kind}.")
        previous_kinds[concept_id] = kind

    return previous_kinds


def build_record_rows(version_rows, memberships, previous_rows):
    """Build current records while preserving previous manual kinds."""
    previous_kinds = load_previous_kinds(previous_rows)
    current_versions = [row for row in version_rows if row["is_current"]]
    current_concepts = {row["concept_id"] for row in current_versions}
    stale_concepts = set(previous_kinds) - current_concepts
    if stale_concepts:
        concepts = ", ".join(str(value) for value in sorted(stale_concepts))
        raise ValueError(f"Catalogue concepts not found on Zenodo: {concepts}.")

    rows = []
    seen_names = set()
    for version in current_versions:
        name = version["title"]
        if name in seen_names:
            raise ValueError(f"Duplicate current Zenodo title: {name}.")
        seen_names.add(name)
        if name not in memberships:
            raise ValueError(f"Current Zenodo record is not in record_lists: {name}.")

        concept_id = version["concept_id"]
        row = {
            "concept_id": concept_id,
            "current_record_id": version["record_id"],
            "name": name,
            "record_lists": memberships[name],
            "record_url": version["record_url"],
        }
        row["kind"] = previous_kinds.get(concept_id, "")
        rows.append(row)

    unused_memberships = set(memberships) - seen_names
    if unused_memberships:
        names = ", ".join(sorted(unused_memberships))
        raise ValueError(f"record_lists entries not found on Zenodo: {names}.")

    return sorted(
        rows,
        key=lambda row: (row["name"].casefold(), row["name"], row["concept_id"]),
    )


def main():
    """Regenerate records.jsonl while retaining manually assigned kinds."""
    version_rows = read_jsonl(VERSIONS_SNAPSHOT_PATH)
    previous_rows = read_jsonl(RECORDS_CATALOG_PATH)
    memberships = load_record_memberships(RECORD_LISTS_DIR)
    rows = build_record_rows(version_rows, memberships, previous_rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(RECORDS_CATALOG_PATH, rows)

    missing_kind = sum(not row["kind"] for row in rows)
    print(f"Wrote {len(rows)} records to manifest/records.jsonl.")
    if missing_kind:
        print(f"{missing_kind} records have an empty kind and require review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
