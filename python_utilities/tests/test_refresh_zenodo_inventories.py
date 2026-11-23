"""Tests for the single-pass Zenodo inventory refresher."""

from unittest import mock

from python_utilities import (
    generate_catalog,
    generate_zenodo_snapshots as snapshots,
    refresh_zenodo_inventories as refresh,
)


def make_raw_record(record_id, concept_id, created, title):
    """Build a synthetic Zenodo API search hit."""
    return {
        "id": record_id,
        "conceptrecid": str(concept_id),
        "created": created,
        "metadata": {"title": title},
        "links": {"self_html": f"https://zenodo.org/records/{record_id}"},
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


def test_single_pass_updates_all_inventories_and_reports_actions(
    tmp_path, monkeypatch, capsys
):
    """Fetch once, write all 5 outputs, and report missing kinds and bumped datasets."""
    records_dir = tmp_path / "zenodo_records"
    manifest_dir = tmp_path / "manifest"
    record_lists_dir = tmp_path / "record_lists"
    records_dir.mkdir()
    manifest_dir.mkdir()
    record_lists_dir.mkdir()

    # Pre-existing v1 of Adiac + datasets.jsonl entry
    old_v1 = make_raw_record(101, 100, "2025-01-01", "Adiac")
    old_versions = snapshots.build_version_rows([old_v1], [old_v1])
    snapshots.write_jsonl(manifest_dir / "zenodo_versions.jsonl", old_versions)
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
        [{"concept_id": 100, "dataset_name": "Adiac", "task": "classification"}],
    )
    (record_lists_dir / "classification-uv").write_text(
        "Adiac\nNewDataset\n", encoding="utf-8"
    )

    # Live Zenodo now has Adiac v2 (102) + NewDataset (201)
    live_records = [
        old_v1,
        make_raw_record(102, 100, "2026-01-01", "Adiac"),
        make_raw_record(201, 200, "2026-06-01", "NewDataset"),
    ]
    load_token = mock.Mock()
    fetch_records = mock.Mock(return_value=live_records)
    monkeypatch.setattr(refresh.record_lists, "load_token", load_token)
    monkeypatch.setattr(refresh.record_lists, "fetch_records", fetch_records)
    monkeypatch.setattr(
        refresh.snapshots,
        "utc_now",
        mock.Mock(return_value="2026-11-20T12:00:00Z"),
    )

    result = refresh.run_refresh(
        yes=True,
        records_dir=records_dir,
        manifest_dir=manifest_dir,
        record_lists_dir=record_lists_dir,
    )

    assert result == 0
    load_token.assert_not_called()
    fetch_records.assert_called_once()
    assert (records_dir / "current-versions").read_text(encoding="utf-8") == (
        "NewDataset: 201\nAdiac: 102\n"
    )
    assert (records_dir / "previous-versions").read_text(encoding="utf-8") == (
        "Adiac: 101\n"
    )

    rebuilt_records = generate_catalog.read_jsonl(manifest_dir / "records.jsonl")
    assert [row["current_record_id"] for row in rebuilt_records] == [102, 201]
    assert [row["kind"] for row in rebuilt_records] == ["dataset", ""]

    output = capsys.readouterr().out
    assert "NEW RECORD\tNewDataset" in output
    assert "NEW VERSION\tAdiac (concept 100: 101 -> 102)" in output
    assert "Assign 'kind'" in output and "NewDataset" in output
    assert "--replace-existing) for: Adiac" in output


def test_dry_run_and_missing_record_list_handling(tmp_path, monkeypatch, capsys):
    """Dry-run writes no files, and missing record_lists entries return non-zero."""
    records_dir = tmp_path / "zenodo_records"
    manifest_dir = tmp_path / "manifest"
    record_lists_dir = tmp_path / "record_lists"
    record_lists_dir.mkdir()
    (record_lists_dir / "classification-uv").write_text("", encoding="utf-8")

    live_records = [make_raw_record(101, 100, "2026-01-01", "UnlistedDataset")]
    monkeypatch.setattr(
        refresh.record_lists,
        "fetch_records",
        mock.Mock(return_value=live_records),
    )

    result = refresh.run_refresh(
        dry_run=True,
        yes=True,
        records_dir=records_dir,
        manifest_dir=manifest_dir,
        record_lists_dir=record_lists_dir,
    )

    assert result == 1
    assert not records_dir.exists()
    assert not manifest_dir.exists()
    captured = capsys.readouterr()
    assert "Dry run: 1 current records" in captured.out
    assert "UnlistedDataset" in captured.err
