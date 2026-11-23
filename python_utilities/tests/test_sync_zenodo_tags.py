"""Tests for catalogue-driven Zenodo tag synchronization."""

from unittest import mock

from python_utilities import generate_catalog, sync_zenodo_tags
from python_utilities import zenodo_maintenance as zm


def test_compute_synced_tags_preserves_unmanaged_and_avoids_noop_resorts():
    """Keep ``ucr archive`` intact, drop forbidden tags, and skip already-valid sets."""
    # Already valid (even if unsorted): preserve exact tuple to avoid unlocking
    unchanged = sync_zenodo_tags.compute_synced_tags(
        old_tags=("ucr archive", "time series", "classification", "tsml archive", "univariate"),
        required_tags={"classification", "time series", "tsml archive", "univariate"},
        forbidden_tags={"multivariate", "rehab"},
    )
    assert unchanged == (
        "ucr archive",
        "time series",
        "classification",
        "tsml archive",
        "univariate",
    )

    # Needs addition ('univariate') and removal ('multiverse-core'), preserving 'uea archive'
    updated = sync_zenodo_tags.compute_synced_tags(
        old_tags=("classification", "multiverse-core", "time series", "tsml archive", "uea archive"),
        required_tags={"classification", "multivariate", "multiverse", "time series", "tsml archive"},
        forbidden_tags={"univariate", "multiverse-core"},
    )
    assert updated == (
        "classification",
        "multivariate",
        "multiverse",
        "time series",
        "tsml archive",
        "uea archive",
    )


def test_cli_syncs_dataset_tags_including_optional_length_and_missing_flags(
    tmp_path, monkeypatch, capsys
):
    """Derive tags from manifests and preview deltas without touching credentials."""
    records_path = tmp_path / "records.jsonl"
    datasets_path = tmp_path / "datasets.jsonl"
    index_path = tmp_path / "current-versions"
    index_path.write_text("DodgerLoopDay: 11186618\n", encoding="utf-8")

    generate_catalog.write_jsonl(
        records_path,
        [
            {
                "concept_id": 11186617,
                "current_record_id": 11186618,
                "kind": "dataset",
                "name": "DodgerLoopDay",
                "record_lists": ["classification-uv"],
                "record_url": "https://zenodo.org/records/11186618",
            }
        ],
    )
    generate_catalog.write_jsonl(
        datasets_path,
        [
            {
                "concept_id": 11186617,
                "dataset_name": "DodgerLoopDay",
                "task": "classification",
                "statistics": {
                    "case_counts": {"train": 78, "test": 80},
                    "missing_value_count": 15,
                    "n_channels": 1,
                    "n_classes": 7,
                    "series_length": {"min": 288, "max": 288, "mean": 288.0},
                },
            }
        ],
    )

    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "DodgerLoopDay",
        {
            "title": "DodgerLoopDay",
            "keywords": [
                "classification",
                "time series",
                "tsml archive",
                "ucr archive",
                "univariate",
            ],
        },
    )
    monkeypatch.setattr(zm, "ZenodoClient", lambda *args, **kwargs: read_client)

    result = sync_zenodo_tags.main(
        [
            "--records-manifest",
            str(records_path),
            "--datasets-manifest",
            str(datasets_path),
            "--include-length-tags",
            "--include-missing-tags",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "CHANGE\t11186618\tDodgerLoopDay" in output
    assert '"equal length"' in output
    assert '"missing values"' in output
    assert '"ucr archive"' in output
