"""Single-pass anonymous refresh of TSML Zenodo record lists, snapshots, and catalogue.

Fetches public Zenodo community metadata once without loading credentials,
updates ``zenodo_records/*`` and ``manifest/*``, and prints any follow-up actions
needed for ``record_lists/*``, ``kind`` assignments, or ``datasets.jsonl``::

    python -m python_utilities.refresh_zenodo_inventories
    python -m python_utilities.refresh_zenodo_inventories --dry-run --yes
"""

import argparse
import sys

from python_utilities import (
    generate_catalog,
    generate_dataset_catalog,
    generate_zenodo_record_lists as record_lists,
    generate_zenodo_snapshots as snapshots,
)


def confirm_refresh(dry_run=False, input_fn=None):
    """Confirm the public Zenodo read (and local writes unless ``dry_run``)."""
    if input_fn is None:
        input_fn = input

    prompt = (
        "This reads public production Zenodo metadata without writing local files. "
        "Press Enter to continue: "
        if dry_run
        else "This reads public production Zenodo metadata and refreshes local "
        "inventories. Press Enter to continue: "
    )
    try:
        response = input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return response == ""


def summarize_version_changes(previous_versions, new_versions):
    """Identify newly published concepts and version-bumped concepts."""
    old_current = {
        row["concept_id"]: row for row in previous_versions if row["is_current"]
    }
    new_current = {
        row["concept_id"]: row for row in new_versions if row["is_current"]
    }

    added_concepts = []
    bumped_concepts = []
    renamed_concepts = []

    for concept_id, row in sorted(
        new_current.items(), key=lambda item: (item[1]["title"].casefold(), item[0])
    ):
        prev = old_current.get(concept_id)
        if prev is None:
            added_concepts.append(row)
        else:
            if prev["record_id"] != row["record_id"]:
                bumped_concepts.append((prev, row))
            if prev["title"] != row["title"]:
                renamed_concepts.append((prev, row))

    removed_concepts = [
        row
        for concept_id, row in sorted(old_current.items())
        if concept_id not in new_current
    ]
    return {
        "added": added_concepts,
        "bumped": bumped_concepts,
        "renamed": renamed_concepts,
        "removed": removed_concepts,
    }


def compute_dataset_refresh_targets(record_rows, dataset_rows, bumped_concept_ids):
    """Return dataset record names needing ``generate_dataset_catalog.py``."""
    base_datasets = {row["dataset_name"] for row in dataset_rows}
    targets = []
    for row in record_rows:
        if row.get("kind") != "dataset":
            continue
        name = row["name"]
        if name not in base_datasets or row["concept_id"] in bumped_concept_ids:
            targets.append(name)
    return targets


def create_argument_parser():
    """Create the command-line parser for single-pass inventory refresh."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch public TSML Zenodo records once and refresh local inventories."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and report changes without modifying local files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive Enter confirmation prompt",
    )
    return parser


def run_refresh(
    *,
    dry_run=False,
    yes=False,
    input_fn=None,
    records_dir=None,
    manifest_dir=None,
    record_lists_dir=None,
):
    """Execute the single-pass refresh and print a maintainer follow-up summary."""
    if not yes and not confirm_refresh(dry_run=dry_run, input_fn=input_fn):
        return 1

    records_dir = records_dir or record_lists.RECORDS_DIR
    manifest_dir = manifest_dir or snapshots.MANIFEST_DIR
    record_lists_dir = record_lists_dir or generate_catalog.RECORD_LISTS_DIR

    current_versions_path = records_dir / "current-versions"
    previous_versions_path = records_dir / "previous-versions"
    versions_snapshot_path = manifest_dir / "zenodo_versions.jsonl"
    files_snapshot_path = manifest_dir / "zenodo_files.jsonl"
    snapshot_history_path = manifest_dir / "snapshot.jsonl"
    records_catalog_path = manifest_dir / "records.jsonl"
    dataset_catalog_path = manifest_dir / "datasets.jsonl"

    print("Fetching public TSML Zenodo records (unauthenticated)...")
    opener = record_lists.create_opener(None)
    raw_records = record_lists.fetch_records(opener, page_size=25)
    current_versions, previous_versions = record_lists.split_record_versions(
        raw_records
    )
    record_lists.validate_unique_titles(current_versions)

    version_rows = snapshots.build_version_rows(raw_records, current_versions)
    file_rows = snapshots.build_file_rows(raw_records)
    snapshot = snapshots.build_snapshot(
        version_rows, file_rows, snapshots.utc_now()
    )
    snapshots.validate_snapshot(version_rows, file_rows, snapshot)

    old_version_rows = generate_catalog.read_jsonl(versions_snapshot_path)
    previous_record_rows = generate_catalog.read_jsonl(records_catalog_path)
    dataset_rows = generate_dataset_catalog.read_jsonl(dataset_catalog_path)
    changes = summarize_version_changes(old_version_rows, version_rows)
    add_history = snapshots.snapshot_counts_have_changed(
        snapshot_history_path, snapshot
    )

    memberships = generate_catalog.load_record_memberships(record_lists_dir)
    current_titles = {row["title"] for row in version_rows if row["is_current"]}
    missing_from_lists = sorted(current_titles - set(memberships))
    stale_in_lists = sorted(set(memberships) - current_titles)

    rebuilt_records = None
    catalog_error = None
    if not missing_from_lists and not stale_in_lists:
        try:
            rebuilt_records = generate_catalog.build_record_rows(
                version_rows, memberships, previous_record_rows
            )
        except ValueError as error:
            catalog_error = str(error)

    if not dry_run:
        records_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        record_lists.write_record_list(current_versions_path, current_versions)
        record_lists.write_record_list(previous_versions_path, previous_versions)
        snapshots.write_jsonl(versions_snapshot_path, version_rows)
        snapshots.write_jsonl(files_snapshot_path, file_rows)
        if add_history:
            snapshots.append_jsonl(snapshot_history_path, snapshot)
        if rebuilt_records is not None:
            generate_catalog.write_jsonl(records_catalog_path, rebuilt_records)

    mode = "Dry run" if dry_run else "Refreshed"
    print(
        f"{mode}: {len(current_versions)} current records, "
        f"{len(previous_versions)} previous versions, {len(file_rows)} files."
    )
    if add_history:
        print(
            "Snapshot counts changed "
            + ("(would append to snapshot.jsonl)." if dry_run else "(appended to snapshot.jsonl).")
        )
    else:
        print("Snapshot counts unchanged.")

    for row in changes["added"]:
        print(
            f"NEW RECORD\t{row['title']} (record {row['record_id']}, "
            f"concept {row['concept_id']})"
        )
    for prev, curr in changes["bumped"]:
        print(
            f"NEW VERSION\t{curr['title']} (concept {curr['concept_id']}: "
            f"{prev['record_id']} -> {curr['record_id']})"
        )
    for prev, curr in changes["renamed"]:
        print(
            f"RENAMED\tconcept {curr['concept_id']}: {prev['title']!r} -> {curr['title']!r}"
        )

    if missing_from_lists:
        print(
            "ACTION REQUIRED: Add new record title(s) to record_lists/* and "
            "re-run generate_catalog.py: " + ", ".join(missing_from_lists),
            file=sys.stderr,
        )
        return 1
    if stale_in_lists:
        print(
            "ACTION REQUIRED: Remove or update stale record_lists/* title(s): "
            + ", ".join(stale_in_lists),
            file=sys.stderr,
        )
        return 1
    if catalog_error is not None:
        print(f"ACTION REQUIRED: {catalog_error}", file=sys.stderr)
        return 1

    missing_kinds = [row["name"] for row in rebuilt_records if not row["kind"]]
    if missing_kinds:
        print(
            "ACTION REQUIRED: Assign 'kind' ('dataset' or 'aggregate') in "
            "manifest/records.jsonl for: " + ", ".join(missing_kinds)
        )

    bumped_ids = {curr["concept_id"] for _, curr in changes["bumped"]}
    dataset_targets = compute_dataset_refresh_targets(
        rebuilt_records, dataset_rows, bumped_ids
    )
    if dataset_targets:
        print(
            "ACTION REQUIRED: Generate or refresh manifest/datasets.jsonl rows "
            "(with --replace-existing) for: " + ", ".join(dataset_targets)
        )
    return 0


def main(argv=None, input_fn=None):
    """Run the command-line single-pass refresh utility."""
    args = create_argument_parser().parse_args(argv)
    try:
        return run_refresh(
            dry_run=args.dry_run,
            yes=args.yes,
            input_fn=input_fn,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
