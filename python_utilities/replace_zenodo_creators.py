"""Preview or replace the creator list of selected Zenodo records.

Preview is the default and uses only public read-only API requests::

    python -m python_utilities.replace_zenodo_creators \
        record_lists/rehab --creator "TSML Archive Maintainers"

Pass ``--apply`` only after reviewing the preview.
"""

import argparse
import sys

from python_utilities import zenodo_maintenance as zm


def clean_creator_names(names):
    """Validate, strip, and deduplicate replacement creator display names."""
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("At least one --creator value is required.")

    clean_names = []
    for raw_name in names:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Creator names cannot be empty.")
        name = raw_name.strip()
        if name in clean_names:
            raise ValueError(f"Duplicate creator name: {name}")
        clean_names.append(name)
    return tuple(clean_names)


def extract_creator_names(metadata, record_id):
    """Extract display names from legacy or InvenioRDM Zenodo creator structures."""
    creators = metadata.get("creators")
    if not isinstance(creators, list) or not creators:
        raise zm.ZenodoMaintenanceError(
            f"Zenodo record {record_id} does not have a usable creators list."
        )

    names = []
    for creator in creators:
        name = creator.get("name") if isinstance(creator, dict) else None
        if not name and isinstance(creator, dict):
            person_or_org = creator.get("person_or_org")
            if isinstance(person_or_org, dict):
                name = person_or_org.get("name")
        if not isinstance(name, str) or not name:
            raise zm.ZenodoMaintenanceError(
                f"Zenodo record {record_id} has a creator without a usable name."
            )
        names.append(name)
    return tuple(names)


def make_creator_replacer(creator_names):
    """Create a ``build_change`` callback replacing ``creators`` with ``creator_names``."""
    new_names = clean_creator_names(creator_names)

    def build_change(record_id, title, metadata):
        old_names = extract_creator_names(metadata, record_id)

        def mutate(draft_metadata):
            draft_metadata["creators"] = [{"name": name} for name in new_names]

        return zm.PlannedChange(
            record_id=record_id,
            title=title,
            old_value=old_names,
            new_value=new_names,
            mutate_metadata=mutate,
            extract_value=extract_creator_names,
        )

    return build_change


def create_argument_parser():
    """Create the command-line parser for creator replacements."""
    parser = argparse.ArgumentParser(
        description="Preview or replace the complete creator list of Zenodo records."
    )
    zm.add_common_arguments(parser)
    parser.add_argument(
        "--creator",
        dest="creators",
        action="append",
        required=True,
        metavar="NAME",
        help="replacement creator display name (repeatable)",
    )
    return parser


def main(argv=None, input_fn=None):
    """Run the creator-replacement CLI."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    api_root = zm.SANDBOX_API_ROOT if args.sandbox else zm.PRODUCTION_API_ROOT
    try:
        build_change = make_creator_replacer(args.creators)
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
            noun="creator",
        )
    except (OSError, ValueError, zm.ZenodoMaintenanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
