"""Read-only live Zenodo auditor checking inventory parity and record metadata.

Paginates public TSML community records using unauthenticated GET requests and
verifies that local inventories, manifests, and metadata conventions match::

    python -m python_utilities.audit_zenodo_records
"""

import argparse
import sys

from python_utilities import (
    generate_catalog,
    generate_dataset_catalog,
    generate_zenodo_record_lists as record_lists,
    generate_zenodo_snapshots as snapshots,
)
from python_utilities.replace_zenodo_creators import extract_creator_names
from python_utilities.zenodo_maintenance import extract_record_tags

EXPECTED_CREATORS = ("TSML Archive Maintainers",)
EXPECTED_LICENSE = "cc-by-4.0"
EXPECTED_ACCESS_RIGHT = "open"
EXPECTED_RESOURCE_TYPE = "dataset"
REQUIRED_COMMON_TAGS = {"time series", "tsml archive"}

COLLECTION_LIST_TO_TAG = {
    "rehab": "rehab",
    "classification-mv": "multiverse",
    "classification-mv-core65": "multiverse-core",
    "eeg": "eeg",
    "monster": "monster",
    "classification-uv-redux30": "bake off redux",
}


def expected_dataset_tags(
    record_row,
    base_dataset_row,
    *,
    include_length_tags=False,
    include_missing_tags=False,
):
    """Return ``(required_tags, forbidden_tags)`` for one dataset record."""
    required = set(REQUIRED_COMMON_TAGS)
    forbidden = set()

    if base_dataset_row["task"] == "classification":
        required.add("classification")
        forbidden.update({"extrinsic regression", "tser archive"})
    else:
        required.update({"extrinsic regression", "tser archive"})
        forbidden.add("classification")

    if base_dataset_row["statistics"]["n_channels"] == 1:
        required.add("univariate")
        forbidden.add("multivariate")
    else:
        required.add("multivariate")
        forbidden.add("univariate")

    memberships = set(record_row["record_lists"])
    for list_name, tag in COLLECTION_LIST_TO_TAG.items():
        if list_name in memberships:
            required.add(tag)
        else:
            forbidden.add(tag)

    if include_length_tags:
        lengths = base_dataset_row["statistics"]["series_length"]
        if lengths["min"] == lengths["max"]:
            required.add("equal length")
            forbidden.add("unequal length")
        else:
            required.add("unequal length")
            forbidden.add("equal length")

    if include_missing_tags:
        if base_dataset_row["statistics"]["missing_value_count"] == 0:
            required.add("no missing values")
            forbidden.add("missing values")
        else:
            required.add("missing values")
            forbidden.add("no missing values")

    return required, forbidden


def _extract_community_ids(metadata):
    communities = metadata.get("communities", [])
    ids = set()
    if isinstance(communities, list):
        for item in communities:
            if isinstance(item, dict):
                cid = item.get("id") or item.get("identifier")
                if isinstance(cid, str) and cid:
                    ids.add(cid)
    return ids


def _extract_license_id(metadata):
    lic = metadata.get("license")
    if isinstance(lic, dict):
        value = lic.get("id")
        return value.lower() if isinstance(value, str) else None
    if isinstance(lic, str):
        return lic.lower()
    return None


def audit_records_against_repository(
    raw_records,
    *,
    records_dir=None,
    manifest_dir=None,
    check_descriptions=False,
    check_length_tags=False,
    check_missing_tags=False,
):
    """Compare fetched public Zenodo records against local files and tag rules."""
    records_dir = records_dir or record_lists.RECORDS_DIR
    manifest_dir = manifest_dir or snapshots.MANIFEST_DIR

    issues = []
    current_versions, previous_versions = record_lists.split_record_versions(
        raw_records
    )
    try:
        record_lists.validate_unique_titles(current_versions)
    except ValueError as error:
        issues.append(str(error))

    version_rows = snapshots.build_version_rows(raw_records, current_versions)
    file_rows = snapshots.build_file_rows(raw_records)
    live_snapshot = snapshots.build_snapshot(
        version_rows, file_rows, "1970-01-01T00:00:00Z"
    )
    snapshots.validate_snapshot(version_rows, file_rows, live_snapshot)

    expected_current_txt = "".join(
        f"{rec['metadata']['title']}: {rec['id']}\n" for rec in current_versions
    )
    expected_previous_txt = "".join(
        f"{rec['metadata']['title']}: {rec['id']}\n" for rec in previous_versions
    )

    actual_current_txt = (records_dir / "current-versions").read_text(encoding="utf-8")
    actual_previous_txt = (records_dir / "previous-versions").read_text(
        encoding="utf-8"
    )
    if actual_current_txt != expected_current_txt:
        issues.append(
            "zenodo_records/current-versions is out of sync with live Zenodo."
        )
    if actual_previous_txt != expected_previous_txt:
        issues.append(
            "zenodo_records/previous-versions is out of sync with live Zenodo."
        )

    actual_versions_txt = (manifest_dir / "zenodo_versions.jsonl").read_text(
        encoding="utf-8"
    )
    if actual_versions_txt != snapshots.jsonl_contents(version_rows):
        issues.append(
            "manifest/zenodo_versions.jsonl is out of sync with live Zenodo."
        )

    actual_files_txt = (manifest_dir / "zenodo_files.jsonl").read_text(
        encoding="utf-8"
    )
    if actual_files_txt != snapshots.jsonl_contents(file_rows):
        issues.append("manifest/zenodo_files.jsonl is out of sync with live Zenodo.")

    snapshot_rows = generate_catalog.read_jsonl(manifest_dir / "snapshot.jsonl")
    if not snapshot_rows:
        issues.append("manifest/snapshot.jsonl is empty.")
    else:
        latest = snapshot_rows[-1]
        for field in (
            "version_count",
            "current_record_count",
            "previous_version_count",
            "file_count",
        ):
            if latest.get(field) != live_snapshot[field]:
                issues.append(
                    f"manifest/snapshot.jsonl {field}={latest.get(field)} does not "
                    f"match live Zenodo ({live_snapshot[field]})."
                )

    catalog_rows = generate_catalog.read_jsonl(manifest_dir / "records.jsonl")
    catalog_by_concept = {row["concept_id"]: row for row in catalog_rows}
    datasets_by_name = {
        row["dataset_name"]: row
        for row in generate_dataset_catalog.read_jsonl(manifest_dir / "datasets.jsonl")
    }

    for rec in sorted(
        current_versions, key=lambda r: (r["metadata"]["title"].casefold(), int(r["id"]))
    ):
        rec_id = int(rec["id"])
        concept_id = int(rec.get("conceptrecid") or rec_id)
        metadata = rec.get("metadata", {})
        title = metadata.get("title", f"<record {rec_id}>")
        label = f"{title} (https://zenodo.org/records/{rec_id})"

        if metadata.get("access_right") != EXPECTED_ACCESS_RIGHT:
            issues.append(
                f"{label}: access_right={metadata.get('access_right')!r}, "
                f"expected {EXPECTED_ACCESS_RIGHT!r}."
            )
        resource_type = (metadata.get("resource_type") or {}).get("type")
        if resource_type != EXPECTED_RESOURCE_TYPE:
            issues.append(
                f"{label}: resource_type={resource_type!r}, "
                f"expected {EXPECTED_RESOURCE_TYPE!r}."
            )
        license_id = _extract_license_id(metadata)
        if license_id != EXPECTED_LICENSE:
            issues.append(
                f"{label}: license={license_id!r}, expected {EXPECTED_LICENSE!r}."
            )
        if "tsml" not in _extract_community_ids(metadata):
            issues.append(f"{label}: missing 'tsml' community membership.")
        if check_descriptions:
            description = metadata.get("description")
            if not isinstance(description, str) or not description.strip():
                issues.append(f"{label}: missing description.")

        try:
            creators = extract_creator_names(metadata, str(rec_id))
            if creators != EXPECTED_CREATORS:
                issues.append(
                    f"{label}: creators={creators!r}, expected {EXPECTED_CREATORS!r}."
                )
        except RuntimeError as error:
            issues.append(f"{label}: {error}")

        try:
            tags = set(extract_record_tags(metadata, str(rec_id)))
        except RuntimeError as error:
            issues.append(f"{label}: {error}")
            continue

        missing_common = sorted(REQUIRED_COMMON_TAGS - tags)
        if missing_common:
            issues.append(f"{label}: missing required keyword(s) {missing_common}.")

        catalog_row = catalog_by_concept.get(concept_id)
        if catalog_row is None:
            issues.append(
                f"{label}: concept {concept_id} is absent from manifest/records.jsonl."
            )
            continue
        if catalog_row["current_record_id"] != rec_id or catalog_row["name"] != title:
            issues.append(
                f"{label}: manifest/records.jsonl has "
                f"name={catalog_row['name']!r}, "
                f"current_record_id={catalog_row['current_record_id']}."
            )

        if catalog_row["kind"] == "dataset":
            base_ds = datasets_by_name.get(catalog_row["name"])
            if base_ds is None:
                issues.append(
                    f"{label}: missing base dataset row in manifest/datasets.jsonl."
                )
                continue
            required, forbidden = expected_dataset_tags(
                catalog_row,
                base_ds,
                include_length_tags=check_length_tags,
                include_missing_tags=check_missing_tags,
            )
            missing_tags = sorted(required - tags)
            unexpected_tags = sorted(tags & forbidden)
            if missing_tags:
                issues.append(f"{label}: missing expected tag(s) {missing_tags}.")
            if unexpected_tags:
                issues.append(f"{label}: unexpected tag(s) {unexpected_tags}.")

    return issues


def create_argument_parser():
    """Create the command-line parser for the live Zenodo auditor."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit live public TSML Zenodo records against repository manifests."
        )
    )
    parser.add_argument(
        "--check-descriptions",
        action="store_true",
        help="also report records that do not yet have a non-empty description",
    )
    parser.add_argument(
        "--check-length-tags",
        action="store_true",
        help="also require 'equal length' / 'unequal length' tags on dataset records",
    )
    parser.add_argument(
        "--check-missing-tags",
        action="store_true",
        help="also require 'missing values' / 'no missing values' tags on dataset records",
    )
    return parser


def main(argv=None):
    """Run the read-only live Zenodo audit."""
    args = create_argument_parser().parse_args(argv)
    try:
        opener = record_lists.create_opener(None)
        raw_records = record_lists.fetch_records(opener, page_size=25)
        issues = audit_records_against_repository(
            raw_records,
            check_descriptions=args.check_descriptions,
            check_length_tags=args.check_length_tags,
            check_missing_tags=args.check_missing_tags,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if issues:
        print(f"Audit found {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f" - {issue}", file=sys.stderr)
        return 1

    current_count = sum(
        1
        for rec in record_lists.split_record_versions(raw_records)[0]
    )
    print(
        f"Audit passed: all {current_count} current Zenodo records and "
        f"{len(raw_records)} total versions match repository manifests and rules."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
