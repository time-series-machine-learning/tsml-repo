"""Synchronize dataset record keywords on Zenodo from repository manifests.

Derives expected tags per dataset record from ``manifest/records.jsonl`` and
``manifest/datasets.jsonl`` while preserving unmanaged tags (such as
``ucr archive`` and ``uea archive``). Preview is the default::

    python -m python_utilities.sync_zenodo_tags
    python -m python_utilities.sync_zenodo_tags --include-length-tags --include-missing-tags
    python -m python_utilities.sync_zenodo_tags --record Sleep --apply
"""

import argparse
import sys

from python_utilities import (
    audit_zenodo_records as audit,
    generate_catalog,
    generate_dataset_catalog,
    generate_zenodo_record_lists as record_lists,
    zenodo_maintenance as zm,
)

BULK_PREFETCH_THRESHOLD = 10


def compute_synced_tags(old_tags, required_tags, forbidden_tags):
    """Add missing required tags and drop forbidden tags, keeping unmanaged tags."""
    desired_set = (set(old_tags) - set(forbidden_tags)) | set(required_tags)
    if set(old_tags) == desired_set and len(old_tags) == len(desired_set):
        return tuple(old_tags)
    return tuple(sorted(desired_set))


def load_dataset_rule_inputs(
    records_path=generate_catalog.RECORDS_CATALOG_PATH,
    datasets_path=generate_dataset_catalog.DATASET_CATALOG_PATH,
):
    """Load dataset records indexed by ``str(current_record_id)`` and base stats."""
    record_rows = generate_catalog.read_jsonl(records_path)
    dataset_rows = generate_dataset_catalog.read_jsonl(datasets_path)
    datasets_by_name = {row["dataset_name"]: row for row in dataset_rows}

    by_record_id = {}
    ordered_refs = []
    for row in record_rows:
        if row.get("kind") != "dataset":
            continue
        name = row["name"]
        base_ds = datasets_by_name.get(name)
        if base_ds is None:
            raise ValueError(
                f"Dataset record {name!r} has no base row in {datasets_path}."
            )
        rec_id = str(row["current_record_id"])
        by_record_id[rec_id] = (row, base_ds)
        ordered_refs.append(zm.RecordReference(rec_id, name))
    return ordered_refs, by_record_id


def make_tag_synchronizer(
    rules_by_record_id,
    *,
    include_length_tags=False,
    include_missing_tags=False,
):
    """Build a ``PlannedChange`` factory synchronizing each dataset's keywords."""

    def build_change(record_id, title, metadata):
        if record_id not in rules_by_record_id:
            raise zm.ZenodoMaintenanceError(
                f"Record {record_id} ({title}) is not a current 'dataset' record "
                "in manifest/records.jsonl."
            )
        record_row, base_ds = rules_by_record_id[record_id]
        required, forbidden = audit.expected_dataset_tags(
            record_row,
            base_ds,
            include_length_tags=include_length_tags,
            include_missing_tags=include_missing_tags,
        )
        old_tags = zm.extract_record_tags(metadata, record_id)
        new_tags = compute_synced_tags(old_tags, required, forbidden)

        def mutate(draft_metadata):
            draft_metadata["keywords"] = list(new_tags)

        return zm.PlannedChange(
            record_id=record_id,
            title=title,
            old_value=old_tags,
            new_value=new_tags,
            mutate_metadata=mutate,
            extract_value=zm.extract_record_tags,
        )

    return build_change


def prefetch_community_metadata():
    """Fetch all public TSML community records in 15 paginated GET requests."""
    opener = record_lists.create_opener(None)
    raw_records = record_lists.fetch_records(opener, page_size=25)
    prefetched = {}
    for rec in raw_records:
        rec_id = str(rec.get("id", ""))
        metadata = rec.get("metadata")
        title = metadata.get("title") if isinstance(metadata, dict) else None
        if rec_id and isinstance(title, str) and title:
            prefetched[rec_id] = (title, metadata)
    return prefetched


def create_argument_parser():
    """Create the command-line parser for catalogue-driven tag synchronization."""
    parser = argparse.ArgumentParser(
        description=(
            "Preview or synchronize Zenodo dataset keywords from repository manifests. "
            "Defaults to all current dataset records when no target is specified."
        )
    )
    zm.add_common_arguments(parser)
    parser.add_argument(
        "--include-length-tags",
        action="store_true",
        help="synchronize 'equal length' / 'unequal length' from base dataset stats",
    )
    parser.add_argument(
        "--include-missing-tags",
        action="store_true",
        help="synchronize 'missing values' / 'no missing values' from base dataset stats",
    )
    parser.add_argument(
        "--records-manifest",
        type=generate_catalog.Path,
        default=generate_catalog.RECORDS_CATALOG_PATH,
        help="path to manifest/records.jsonl",
    )
    parser.add_argument(
        "--datasets-manifest",
        type=generate_catalog.Path,
        default=generate_dataset_catalog.DATASET_CATALOG_PATH,
        help="path to manifest/datasets.jsonl",
    )
    return parser


def main(argv=None, input_fn=None):
    """Run the catalogue-driven tag synchronization CLI."""
    args = create_argument_parser().parse_args(argv)
    api_root = zm.SANDBOX_API_ROOT if args.sandbox else zm.PRODUCTION_API_ROOT
    try:
        all_dataset_refs, rules_by_record_id = load_dataset_rule_inputs(
            records_path=args.records_manifest,
            datasets_path=args.datasets_manifest,
        )
        if args.record_file is None and not args.records:
            references = all_dataset_refs
        else:
            references = zm.resolve_record_references(
                record_file=args.record_file,
                explicit_records=args.records,
                index_path=args.record_index,
            )

        prefetched = None
        if (
            api_root == zm.PRODUCTION_API_ROOT
            and len(references) > BULK_PREFETCH_THRESHOLD
        ):
            prefetched = prefetch_community_metadata()

        build_change = make_tag_synchronizer(
            rules_by_record_id,
            include_length_tags=args.include_length_tags,
            include_missing_tags=args.include_missing_tags,
        )
        return zm.run_metadata_update(
            references,
            build_change,
            apply=args.apply,
            discard_existing_edit=args.discard_existing_edit,
            api_root=api_root,
            token_path=args.token_file,
            prefetched_metadata=prefetched,
            input_fn=input_fn,
            noun="tag sync",
        )
    except (OSError, ValueError, zm.ZenodoMaintenanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
