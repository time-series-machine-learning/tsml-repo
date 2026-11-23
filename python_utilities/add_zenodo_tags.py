"""Preview or add keyword tags to selected Zenodo records.

Preview is the default and uses only public read-only API requests::

    python -m python_utilities.add_zenodo_tags --record Sleep --tag univariate
    python -m python_utilities.add_zenodo_tags record_lists/rehab --tag rehab

Pass ``--apply`` only after reviewing the preview.
"""

import argparse
import sys

from python_utilities import zenodo_maintenance as zm


def compute_added_tags(old_tags, tags_to_add):
    """Return existing tags if all requested tags exist, or sorted union otherwise."""
    if set(tags_to_add).issubset(old_tags):
        return tuple(old_tags)
    return tuple(sorted(set(old_tags).union(tags_to_add)))


def make_tag_adder(tags):
    """Create a ``build_change`` callback that adds ``tags`` to record keywords."""
    clean = zm.clean_tags(tags)

    def build_change(record_id, title, metadata):
        old_tags = zm.extract_record_tags(metadata, record_id)
        new_tags = compute_added_tags(old_tags, clean)

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


def create_argument_parser():
    """Create the command-line parser for tag additions."""
    parser = argparse.ArgumentParser(
        description="Preview or add keyword tags to selected Zenodo records."
    )
    zm.add_common_arguments(parser)
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        required=True,
        metavar="TAG",
        help="keyword tag to add (repeatable)",
    )
    return parser


def main(argv=None, input_fn=None):
    """Run the tag-addition CLI."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    api_root = zm.SANDBOX_API_ROOT if args.sandbox else zm.PRODUCTION_API_ROOT
    try:
        build_change = make_tag_adder(args.tags)
        references = zm.resolve_record_references(
            record_file=args.record_file,
            explicit_records=args.records,
            index_path=args.record_index,
        )
        return zm.run_metadata_update(
            references,
            build_change,
            apply=args.apply,
            discard_existing_edit=args.discard_existing_edit,
            api_root=api_root,
            token_path=args.token_file,
            input_fn=input_fn,
            noun="tag",
        )
    except (OSError, ValueError, zm.ZenodoMaintenanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
