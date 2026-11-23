"""Preview or update descriptions and related identifiers on Zenodo records.

Preview is the default and uses only public read-only API requests::

    python -m python_utilities.update_zenodo_descriptions \
        --record 18300287 \
        --replace "processed to equal length" "truncated to unequal lengths"

    python -m python_utilities.update_zenodo_descriptions \
        record_lists/rehab \
        --add-related 10.48550/arXiv.2507.21018 isSupplementTo

Pass ``--apply`` only after reviewing the preview.
"""

import argparse
import sys
from pathlib import Path

from python_utilities import zenodo_maintenance as zm

ALLOWED_RELATIONS = {
    rel.lower(): rel
    for rel in (
        "isCitedBy",
        "cites",
        "isSupplementTo",
        "isSupplementedBy",
        "isContinuedBy",
        "continues",
        "isDescribedBy",
        "describes",
        "hasMetadata",
        "isMetadataFor",
        "isPartOf",
        "hasPart",
        "isPublishedIn",
        "isReferencedBy",
        "references",
        "isDocumentedBy",
        "documents",
        "isCompiledBy",
        "compiles",
        "isVariantFormOf",
        "isOriginalFormOf",
        "isIdenticalTo",
        "isReviewedBy",
        "reviews",
        "isDerivedFrom",
        "isSourceOf",
        "isRequiredBy",
        "requires",
        "isObsoletedBy",
        "obsoletes",
        "isAlternateIdentifier",
    )
}

SYNTHETIC_VERSION_RELATIONS = {
    "hasversion",
    "isversionof",
    "isnewversionof",
    "ispreviousversionof",
}


def normalize_relation(relation):
    """Validate and canonicalize a Zenodo ``related_identifiers`` relation type."""
    key = relation.strip().lower() if isinstance(relation, str) else ""
    if key in SYNTHETIC_VERSION_RELATIONS:
        raise ValueError(
            f"Relation {relation!r} is managed automatically by Zenodo versioning."
        )
    canonical = ALLOWED_RELATIONS.get(key)
    if canonical is None:
        raise ValueError(
            f"Unsupported relation {relation!r}; expected one of "
            f"{sorted(ALLOWED_RELATIONS.values())}."
        )
    return canonical


def extract_description(metadata, record_id):
    """Extract a record's ``description`` string (or ``''`` when absent)."""
    description = metadata.get("description")
    if description is None:
        return ""
    if not isinstance(description, str):
        raise zm.ZenodoMaintenanceError(
            f"Zenodo record {record_id} has a non-string description."
        )
    return description


def extract_user_related_identifiers(metadata, record_id):
    """Extract user-managed ``(identifier, relation)`` pairs, ignoring version links."""
    raw_items = metadata.get("related_identifiers", [])
    if raw_items is None:
        return ()
    if not isinstance(raw_items, list):
        raise zm.ZenodoMaintenanceError(
            f"Zenodo record {record_id} has an invalid related_identifiers field."
        )

    pairs = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise zm.ZenodoMaintenanceError(
                f"Zenodo record {record_id} has a non-object related identifier."
            )
        identifier = item.get("identifier")
        relation = item.get("relation")
        if not isinstance(identifier, str) or not identifier:
            raise zm.ZenodoMaintenanceError(
                f"Zenodo record {record_id} has a related identifier without 'identifier'."
            )
        if not isinstance(relation, str) or not relation:
            continue
        if relation.lower() in SYNTHETIC_VERSION_RELATIONS:
            continue
        pairs.append((identifier, ALLOWED_RELATIONS.get(relation.lower(), relation)))
    return tuple(pairs)


def compute_updated_description(
    old_description,
    *,
    replacement_text=None,
    replacements=(),
    append_blocks=(),
):
    """Compute the updated description string and ensure it stays non-empty."""
    if replacement_text is not None:
        updated = replacement_text
    else:
        updated = old_description
        for old_sub, new_sub in replacements:
            updated = updated.replace(old_sub, new_sub)
        for block in append_blocks:
            stripped = block.strip()
            if stripped and stripped not in updated:
                if not updated:
                    updated = stripped
                else:
                    sep = "" if updated.endswith("\n") else "\n"
                    updated = f"{updated}{sep}{stripped}"

    if not updated.strip():
        raise ValueError("Updated description must not be empty.")
    return updated


def compute_updated_related_list(existing_dicts, add_specs=(), remove_identifiers=()):
    """Apply additions and removals to raw ``related_identifiers`` dicts."""
    removals = set(remove_identifiers)
    kept = []
    seen = set()

    for item in existing_dicts:
        identifier = item.get("identifier", "")
        relation = item.get("relation", "")
        if relation.lower() in SYNTHETIC_VERSION_RELATIONS:
            continue
        if identifier in removals:
            continue
        norm_rel = ALLOWED_RELATIONS.get(relation.lower(), relation)
        kept.append({**item, "relation": norm_rel})
        seen.add((identifier, norm_rel))

    for identifier, relation in add_specs:
        if (identifier, relation) not in seen:
            kept.append({"identifier": identifier, "relation": relation})
            seen.add((identifier, relation))

    return kept


def validate_operations(
    *,
    description_file=None,
    replacements=(),
    append_blocks=(),
    add_related=(),
    remove_related=(),
):
    """Validate CLI mutation arguments and return normalized configuration."""
    if description_file is not None and (replacements or append_blocks):
        raise ValueError(
            "--description-file cannot be combined with --replace or --append-html."
        )

    replacement_text = None
    if description_file is not None:
        replacement_text = Path(description_file).read_text(encoding="utf-8").strip()
        if not replacement_text:
            raise ValueError(f"Description file {description_file} is empty.")

    clean_replacements = []
    for old_sub, new_sub in replacements:
        if not old_sub:
            raise ValueError("--replace OLD string must not be empty.")
        if old_sub == new_sub:
            raise ValueError(f"--replace OLD and NEW are identical: {old_sub!r}.")
        clean_replacements.append((old_sub, new_sub))

    clean_appends = []
    for block in append_blocks:
        if not isinstance(block, str) or not block.strip():
            raise ValueError("--append-html value must not be empty.")
        clean_appends.append(block.strip())

    clean_add_related = []
    for identifier, relation in add_related:
        ident = identifier.strip()
        if not ident:
            raise ValueError("--add-related IDENTIFIER must not be empty.")
        norm_rel = normalize_relation(relation)
        if (ident, norm_rel) not in clean_add_related:
            clean_add_related.append((ident, norm_rel))

    clean_remove_related = []
    for identifier in remove_related:
        ident = identifier.strip()
        if not ident:
            raise ValueError("--remove-related IDENTIFIER must not be empty.")
        if ident not in clean_remove_related:
            clean_remove_related.append(ident)

    if (
        replacement_text is None
        and not clean_replacements
        and not clean_appends
        and not clean_add_related
        and not clean_remove_related
    ):
        raise ValueError(
            "Specify at least one operation (--description-file, --replace, "
            "--append-html, --add-related, or --remove-related)."
        )

    return {
        "replacement_text": replacement_text,
        "replacements": tuple(clean_replacements),
        "append_blocks": tuple(clean_appends),
        "add_related": tuple(clean_add_related),
        "remove_related": tuple(clean_remove_related),
    }


def make_description_and_related_updater(**ops):
    """Create a ``PlannedChange`` factory for description and/or related links."""
    config = validate_operations(**ops)
    edits_description = bool(
        config["replacement_text"] is not None
        or config["replacements"]
        or config["append_blocks"]
    )
    edits_related = bool(config["add_related"] or config["remove_related"])

    def extract_state(metadata, record_id):
        state = {}
        if edits_description:
            state["description"] = extract_description(metadata, record_id)
        if edits_related:
            state["related_identifiers"] = list(
                extract_user_related_identifiers(metadata, record_id)
            )
        return state

    def build_change(record_id, title, metadata):
        old_state = extract_state(metadata, record_id)
        new_state = dict(old_state)

        new_description = None
        if edits_description:
            new_description = compute_updated_description(
                old_state["description"],
                replacement_text=config["replacement_text"],
                replacements=config["replacements"],
                append_blocks=config["append_blocks"],
            )
            new_state["description"] = new_description

        if edits_related:
            updated_dicts = compute_updated_related_list(
                metadata.get("related_identifiers") or [],
                add_specs=config["add_related"],
                remove_identifiers=config["remove_related"],
            )
            new_state["related_identifiers"] = [
                (item["identifier"], item["relation"]) for item in updated_dicts
            ]

        def mutate(draft_metadata):
            if edits_description:
                draft_metadata["description"] = new_description
            if edits_related:
                updated_draft_related = compute_updated_related_list(
                    draft_metadata.get("related_identifiers") or [],
                    add_specs=config["add_related"],
                    remove_identifiers=config["remove_related"],
                )
                if updated_draft_related:
                    draft_metadata["related_identifiers"] = updated_draft_related
                else:
                    draft_metadata.pop("related_identifiers", None)

        return zm.PlannedChange(
            record_id=record_id,
            title=title,
            old_value=old_state,
            new_value=new_state,
            mutate_metadata=mutate,
            extract_value=extract_state,
        )

    return build_change


def create_argument_parser():
    """Create the command-line parser for description and related-ID updates."""
    parser = argparse.ArgumentParser(
        description=(
            "Preview or update descriptions and related identifiers on Zenodo records."
        )
    )
    zm.add_common_arguments(parser)
    parser.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help="replace record description with the UTF-8 contents of this file",
    )
    parser.add_argument(
        "--replace",
        dest="replacements",
        action="append",
        nargs=2,
        default=[],
        metavar=("OLD", "NEW"),
        help="replace exact substring OLD with NEW in description (repeatable)",
    )
    parser.add_argument(
        "--append-html",
        dest="append_blocks",
        action="append",
        default=[],
        metavar="HTML",
        help="append an HTML snippet if not already in description (repeatable)",
    )
    parser.add_argument(
        "--add-related",
        dest="add_related",
        action="append",
        nargs=2,
        default=[],
        metavar=("IDENTIFIER", "RELATION"),
        help="add a related identifier (e.g. DOI/URL + isSupplementTo/cites)",
    )
    parser.add_argument(
        "--remove-related",
        dest="remove_related",
        action="append",
        default=[],
        metavar="IDENTIFIER",
        help="remove user related identifier(s) matching IDENTIFIER (repeatable)",
    )
    return parser


def main(argv=None, input_fn=None):
    """Run the description / related-identifier update CLI."""
    args = create_argument_parser().parse_args(argv)
    api_root = zm.SANDBOX_API_ROOT if args.sandbox else zm.PRODUCTION_API_ROOT
    try:
        build_change = make_description_and_related_updater(
            description_file=args.description_file,
            replacements=args.replacements,
            append_blocks=args.append_blocks,
            add_related=args.add_related,
            remove_related=args.remove_related,
        )
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
            noun="description/related",
        )
    except (OSError, ValueError, zm.ZenodoMaintenanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
