"""Tests for the TSML record catalogue generator."""

import json

import pytest

from python_utilities import generate_catalog as catalog


def make_version(concept_id, record_id, name):
    """Create a current Zenodo version snapshot row."""
    return {
        "concept_id": concept_id,
        "record_id": record_id,
        "record_url": f"https://zenodo.org/records/{record_id}",
        "title": name,
        "is_current": True,
    }


def test_preserves_previous_kinds_and_leaves_new_kinds_empty():
    """Keep reviewed kinds while leaving new records invalid until reviewed."""
    versions = [
        make_version(100, 101, "Existing"),
        make_version(200, 201, "New"),
    ]
    memberships = {
        "Existing": ["classification-uv"],
        "New": ["classification-mv"],
    }
    previous = [{"concept_id": 100, "kind": "dataset"}]

    rows = catalog.build_record_rows(versions, memberships, previous)

    assert rows[0]["kind"] == "dataset"
    assert rows[1]["kind"] == ""


def test_rejects_record_list_entries_without_current_zenodo_records():
    """Reject stale record-list entries rather than silently dropping them."""
    versions = [make_version(100, 101, "Current")]
    memberships = {
        "Current": ["classification-uv"],
        "Missing": ["classification-uv"],
    }

    with pytest.raises(ValueError, match="not found on Zenodo: Missing"):
        catalog.build_record_rows(versions, memberships, [])


def test_main_refreshes_derived_fields_and_preserves_kind(tmp_path, monkeypatch):
    """Regenerate derived fields without replacing a manually assigned kind."""
    manifest_dir = tmp_path / "manifest"
    record_lists_dir = tmp_path / "record_lists"
    manifest_dir.mkdir()
    record_lists_dir.mkdir()
    versions_path = manifest_dir / "zenodo_versions.jsonl"
    records_path = manifest_dir / "records.jsonl"
    catalog.write_jsonl(versions_path, [make_version(100, 102, "Updated")])
    catalog.write_jsonl(
        records_path,
        [{"concept_id": 100, "kind": "aggregate", "name": "Old"}],
    )
    (record_lists_dir / "aggregated-archives").write_text(
        "Updated\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(catalog, "VERSIONS_SNAPSHOT_PATH", versions_path)
    monkeypatch.setattr(catalog, "RECORDS_CATALOG_PATH", records_path)
    monkeypatch.setattr(catalog, "RECORD_LISTS_DIR", record_lists_dir)

    result = catalog.main()
    row = json.loads(records_path.read_text(encoding="utf-8"))

    assert result == 0
    assert row == {
        "concept_id": 100,
        "current_record_id": 102,
        "kind": "aggregate",
        "name": "Updated",
        "record_lists": ["aggregated-archives"],
        "record_url": "https://zenodo.org/records/102",
    }
