"""Tests for generating dataset catalogue rows from local .ts files."""

import json

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
    make_pair(data_dir, "Adiac")
    make_pair(data_dir, "Adiac_eq")
    records_path = tmp_path / "records.jsonl"
    datasets_path = tmp_path / "datasets.jsonl"
    catalog.write_jsonl(
        records_path,
        [{"concept_id": 11179787, "kind": "dataset", "name": "Adiac"}],
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
        ]
    )
    rows = [json.loads(line) for line in datasets_path.read_text().splitlines()]

    assert result == 0
    assert [row["dataset_name"] for row in rows] == ["Adiac", "Adiac_eq"]
    assert {row["concept_id"] for row in rows} == {11179787}
