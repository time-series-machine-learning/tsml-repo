"""Preview or remove keyword tags from selected Zenodo records.

Preview is the default and uses only public read-only API requests::

    python -m python_utilities.remove_zenodo_tags \
        --record BenzeneConcentration_disc --tag multiverse-core

Pass ``--apply`` only after reviewing the preview.
"""

import argparse
import sys

from python_utilities import zenodo_maintenance as zm


def compute_removed_tags(old_tags, tags_to_remove):
    """Return remaining tags in existing order after removing ``tags_to_remove``."""
    removals = set(tags_to_remove)
    return tuple(tag for tag in old_tags if tag not in removals)


def make_tag_remover(tags):
    """Create a ``build_change`` callback that removes ``tags`` from keywords."""
    clean = zm.clean_tags(tags)

    def build_change(record_id, title, metadata):
        old_tags = zm.extract_record_tags(metadata, record_id)
        new_tags = compute_removed_tags(old_tags, clean)

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
    """Create the command-line parser for tag removals."""
    parser = argparse.ArgumentParser(
        description="Preview or remove keyword tags from selected Zenodo records."
    )
    zm.add_common_arguments(parser)
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        required=True,
        metavar="TAG",
        help="keyword tag to remove (repeatable)",
    )
    return parser


def main(argv=None, input_fn=None):
    """Run the tag-removal CLI."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    api_root = zm.SANDBOX_API_ROOT if args.sandbox else zm.PRODUCTION_API_ROOT
    try:
        build_change = make_tag_remover(args.tags)
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
