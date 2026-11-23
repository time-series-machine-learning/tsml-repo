"""Tests for the read-only live Zenodo auditor."""

from python_utilities import (
    audit_zenodo_records as audit,
    generate_catalog,
    generate_zenodo_record_lists as record_lists,
    generate_zenodo_snapshots as snapshots,
)


def make_compliant_record(
    record_id=101,
    concept_id=100,
    title="Adiac",
    keywords=None,
    creators=None,
    license_id="cc-by-4.0",
    access_right="open",
):
    """Build a synthetic Zenodo record matching TSML community conventions."""
    if keywords is None:
        keywords = ["classification", "time series", "tsml archive", "univariate"]
    if creators is None:
        creators = [{"name": "TSML Archive Maintainers"}]
    return {
        "id": record_id,
        "conceptrecid": str(concept_id),
        "created": "2026-01-01",
        "links": {"self_html": f"https://zenodo.org/records/{record_id}"},
        "metadata": {
            "title": title,
            "description": "<p>Dataset description.</p>",
            "access_right": access_right,
            "license": {"id": license_id},
            "resource_type": {"type": "dataset"},
            "communities": [{"id": "tsml"}],
            "creators": creators,
            "keywords": keywords,
        },
        "files": [
            {
                "key": f"{title}_TRAIN.ts",
                "size": 10,
                "checksum": "md5:0123456789abcdef0123456789abcdef",
                "links": {"self": f"https://zenodo.org/api/files/{title}_TRAIN.ts"},
            },
            {
                "key": f"{title}_TEST.ts",
                "size": 10,
                "checksum": "md5:0123456789abcdef0123456789abcdef",
                "links": {"self": f"https://zenodo.org/api/files/{title}_TEST.ts"},
            },
        ],
    }


def write_matching_repo_files(tmp_path, raw_records):
    """Populate matching ``zenodo_records/`` and ``manifest/`` files under ``tmp_path``."""
    records_dir = tmp_path / "zenodo_records"
    manifest_dir = tmp_path / "manifest"
    records_dir.mkdir(exist_ok=True)
    manifest_dir.mkdir(exist_ok=True)

    curr, prev = record_lists.split_record_versions(raw_records)
    record_lists.write_record_list(records_dir / "current-versions", curr)
    record_lists.write_record_list(records_dir / "previous-versions", prev)

    versions = snapshots.build_version_rows(raw_records, curr)
    files = snapshots.build_file_rows(raw_records)
    snap = snapshots.build_snapshot(versions, files, "2026-11-20T00:00:00Z")
    snapshots.write_jsonl(manifest_dir / "zenodo_versions.jsonl", versions)
    snapshots.write_jsonl(manifest_dir / "zenodo_files.jsonl", files)
    snapshots.write_jsonl(manifest_dir / "snapshot.jsonl", [snap])
    generate_catalog.write_jsonl(
        manifest_dir / "records.jsonl",
        [
            {
                "concept_id": 100,
                "current_record_id": 101,
                "kind": "dataset",
                "name": "Adiac",
                "record_lists": ["classification-uv"],
                "record_url": "https://zenodo.org/records/101",
            }
        ],
    )
    generate_catalog.write_jsonl(
        manifest_dir / "datasets.jsonl",
        [
            {
                "concept_id": 100,
                "dataset_name": "Adiac",
                "task": "classification",
                "statistics": {
                    "case_counts": {"train": 10, "test": 10},
                    "missing_value_count": 0,
                    "n_channels": 1,
                    "n_classes": 2,
                    "series_length": {"min": 50, "max": 50, "mean": 50.0},
                },
            }
        ],
    )
    return records_dir, manifest_dir


def test_audit_passes_when_records_and_metadata_match(tmp_path):
    """Report zero issues when live records match manifests and tag conventions."""
    raw = [make_compliant_record()]
    records_dir, manifest_dir = write_matching_repo_files(tmp_path, raw)

    issues = audit.audit_records_against_repository(
        raw, records_dir=records_dir, manifest_dir=manifest_dir
    )
    assert issues == []


def test_audit_detects_missing_and_forbidden_tags_and_license_drift(tmp_path):
    """Flag missing univariate tag, unexpected rehab tag, and wrong license."""
    raw = [make_compliant_record()]
    records_dir, manifest_dir = write_matching_repo_files(tmp_path, raw)

    drifted = [
        make_compliant_record(
            keywords=["classification", "rehab", "time series", "tsml archive"],
            license_id="cc0-1.0",
        )
    ]
    issues = audit.audit_records_against_repository(
        drifted, records_dir=records_dir, manifest_dir=manifest_dir
    )

    joined = "\n".join(issues)
    assert "license='cc0-1.0'" in joined
    assert "missing expected tag(s) ['univariate']" in joined
    assert "unexpected tag(s) ['rehab']" in joined

    # Optional length/missingness checks also flag missing optional tags when enabled
    opt_issues = audit.audit_records_against_repository(
        raw,
        records_dir=records_dir,
        manifest_dir=manifest_dir,
        check_length_tags=True,
        check_missing_tags=True,
    )
    opt_joined = "\n".join(opt_issues)
    assert "'equal length'" in opt_joined
    assert "'no missing values'" in opt_joined
