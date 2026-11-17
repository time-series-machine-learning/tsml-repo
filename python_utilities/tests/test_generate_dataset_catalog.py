"""Tests for generating dataset catalogue rows from local .ts files."""

import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pytest

from python_utilities import generate_dataset_catalog as catalog


def make_pair(directory, name):
    """Create placeholder train and test files for discovery tests."""
    pair = {}
    for split in catalog.SPLITS:
        path = directory / f"{name}_{split.upper()}.ts"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        pair[split] = path
    return pair


def file_row(path, record_id, concept_id):
    """Build a file inventory row from a small published test fixture."""
    data = path.read_bytes()
    return {
        "record_id": record_id,
        "concept_id": concept_id,
        "key": path.name,
        "size": len(data),
        "checksum": f"md5:{hashlib.md5(data, usedforsecurity=False).hexdigest()}",
    }


def make_loader(task="classification"):
    """Create an aeon-like loader for small in-memory collections."""
    metadata = {
        "classlabel": task == "classification",
        "targetlabel": task == "regression",
    }
    train = (
        [
            np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ],
        np.array(["a", "b"]),
        metadata,
    )
    test = (
        [np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])],
        np.array(["a"]),
        metadata,
    )
    return lambda path: train if path.name.endswith("_TRAIN.ts") else test


def test_discovers_base_and_available_variant_pairs(tmp_path):
    """Find complete base, equal-length, and no-missing-value pairs."""
    make_pair(tmp_path / "base", "Adiac")
    make_pair(tmp_path / "variants", "Adiac_eq")
    make_pair(tmp_path / "variants", "Adiac_nmv")
    (tmp_path / "Other_TRAIN.ts").touch()

    pairs = catalog.discover_dataset_pairs("Adiac", tmp_path)

    assert list(pairs) == ["Adiac", "Adiac_eq", "Adiac_nmv"]
    assert set(pairs["Adiac_eq"]) == {"train", "test"}


def test_rejects_an_incomplete_train_test_pair(tmp_path):
    """Reject a discovered variant when either its train or test file is absent."""
    make_pair(tmp_path, "Adiac")
    (tmp_path / "Adiac_eq_TRAIN.ts").touch()

    with pytest.raises(ValueError, match=r"Adiac_eq \(missing TEST\)"):
        catalog.discover_dataset_pairs("Adiac", tmp_path)


def test_builds_classification_statistics_from_both_splits(tmp_path):
    """Count cases, channels, classes, lengths, and missing values."""
    pair = make_pair(tmp_path, "Adiac")

    row = catalog.build_dataset_row(
        "Adiac",
        11179787,
        pair,
        loader=make_loader(),
    )

    assert row == {
        "concept_id": 11179787,
        "dataset_name": "Adiac",
        "statistics": {
            "case_counts": {"train": 2, "test": 1},
            "missing_value_count": 1,
            "n_channels": 2,
            "n_classes": 2,
            "series_length": {"min": 2, "max": 4, "mean": 3.0},
        },
        "task": "classification",
    }


def test_regression_row_does_not_include_a_class_count(tmp_path):
    """Infer regression from target labels and omit classification-only data."""
    pair = make_pair(tmp_path, "AppliancesEnergy")

    row = catalog.build_dataset_row(
        "AppliancesEnergy",
        100,
        pair,
        loader=make_loader(task="regression"),
    )

    assert row["task"] == "regression"
    assert "n_classes" not in row["statistics"]


def test_existing_family_requires_explicit_replacement():
    """Protect existing rows by default and replace the whole family explicitly."""
    existing = [
        {"dataset_name": "Adiac", "concept_id": 1},
        {"dataset_name": "Adiac_nmv", "concept_id": 1},
        {"dataset_name": "Other", "concept_id": 2},
    ]
    generated = [
        {"dataset_name": "Adiac", "concept_id": 1},
        {"dataset_name": "Adiac_eq", "concept_id": 1},
    ]

    with pytest.raises(ValueError, match="--replace-existing"):
        catalog.merge_dataset_rows(existing, generated, "Adiac")

    rows = catalog.merge_dataset_rows(
        existing,
        generated,
        "Adiac",
        replace=True,
    )
    assert [row["dataset_name"] for row in rows] == [
        "Adiac",
        "Adiac_eq",
        "Other",
    ]


def test_main_looks_up_concept_and_writes_discovered_rows(
    tmp_path, monkeypatch
):
    """Run the local workflow without requiring aeon or external requests."""
    data_dir = tmp_path / "data"
    pairs = [make_pair(data_dir, name) for name in ("Adiac", "Adiac_eq")]
    records_path = tmp_path / "records.jsonl"
    datasets_path = tmp_path / "datasets.jsonl"
    files_path = tmp_path / "zenodo_files.jsonl"
    versions_path = tmp_path / "zenodo_versions.jsonl"
    catalog.write_jsonl(
        records_path,
        [{
            "concept_id": 11179787,
            "current_record_id": 11179788,
            "kind": "dataset",
            "name": "Adiac",
        }],
    )
    catalog.write_jsonl(
        versions_path,
        [{"concept_id": 11179787, "record_id": 11179788, "is_current": True}],
    )
    catalog.write_jsonl(
        files_path,
        [file_row(path, 11179788, 11179787) for pair in pairs for path in pair.values()],
    )
    monkeypatch.setattr(catalog, "load_ts_file", make_loader())

    result = catalog.main(
        [
            "Adiac",
            str(data_dir),
            "--catalog",
            str(datasets_path),
            "--records",
            str(records_path),
            "--files",
            str(files_path),
            "--versions",
            str(versions_path),
        ]
    )
    rows = [json.loads(line) for line in datasets_path.read_text().splitlines()]

    assert result == 0
    assert [row["dataset_name"] for row in rows] == ["Adiac", "Adiac_eq"]
    assert {row["concept_id"] for row in rows} == {11179787}


@pytest.mark.parametrize(
    ("catalogue_record_id", "message"),
    [(101, "does not match the current Zenodo version snapshot"), (102, "Adiac_TRAIN.ts.*record 102")],
)
def test_main_rejects_cached_previous_version_without_writing(
    tmp_path, monkeypatch, catalogue_record_id, message
):
    """Reject stale data even if it has the old record ID, name, and size."""
    data = tmp_path / "data"
    pair = make_pair(data, "Adiac")
    record = {
        "concept_id": 100, "current_record_id": catalogue_record_id,
        "kind": "dataset", "name": "Adiac",
    }
    records_path = tmp_path / "records.jsonl"
    files_path = tmp_path / "zenodo_files.jsonl"
    versions_path = tmp_path / "zenodo_versions.jsonl"
    output = tmp_path / "datasets.jsonl"
    catalog.write_jsonl(records_path, [record])
    catalog.write_jsonl(versions_path, [{"concept_id": 100, "record_id": 102, "is_current": True}])
    output.write_bytes(b'{"dataset_name": "Adiac"}\n')
    original_output = output.read_bytes()

    train = pair["train"]
    train.write_bytes(b"v1\n")
    older = file_row(train, 101, 100)
    train.write_bytes(b"v2\n")
    catalog.write_jsonl(files_path, [older, file_row(train, 102, 100), file_row(pair["test"], 102, 100)])
    train.write_bytes(b"v1\n")

    def unexpected_loader(path):
        pytest.fail(f"Loaded {path} before verifying the source files")

    monkeypatch.setattr(catalog, "load_ts_file", unexpected_loader)
    with pytest.raises(ValueError, match=message):
        catalog.main([
            "Adiac", str(data),
            "--catalog", str(output),
            "--records", str(records_path),
            "--files", str(files_path),
            "--versions", str(versions_path),
            "--replace-existing",
        ])

    assert output.read_bytes() == original_output


def test_zip_only_records_verify_extracted_pairs(tmp_path):
    """Accept a published ZIP and detect changes in its extracted files."""
    pair = make_pair(tmp_path / "extracted", "S2Agri-17")
    pair["train"].write_bytes(b"train\n")
    pair["test"].write_bytes(b"test\n")
    archive = tmp_path / "S2Agri-17.zip"
    with ZipFile(archive, "w") as output:
        for path in pair.values():
            output.write(path, arcname=f"S2Agri-17/{path.name}")
    record = {"concept_id": 100, "current_record_id": 101}
    files = [file_row(archive, 101, 100)]
    pairs = {"S2Agri-17": pair}

    with pytest.raises(ValueError, match="Use --archive"):
        catalog.verify_dataset_files(pairs, record, files)
    catalog.verify_dataset_files(pairs, record, files, archive)

    pair["train"].write_bytes(b"stale\n")
    with pytest.raises(ValueError, match="does not match a dataset file in"):
        catalog.verify_dataset_files(pairs, record, files, archive)
