"""Tests for the TSML record and dataset catalogue schemas."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from python_utilities.generate_catalog import RECORDS_CATALOG_PATH
from python_utilities.generate_zenodo_snapshots import SCHEMA_DIR


def load_validator(schema_name):
    """Load and check a catalogue schema used by the tests."""
    schema = json.loads(
        (SCHEMA_DIR / schema_name).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.mark.parametrize(
    "row",
    [
        {
            "name": "Adiac",
            "kind": "dataset",
            "concept_id": 11179787,
            "current_record_id": 11179788,
            "record_url": "https://zenodo.org/records/11179788",
            "record_lists": ["classification-uv"],
        },
        {
            "name": "UCR Time Series Classification Archive 2018",
            "kind": "aggregate",
            "concept_id": 100,
            "current_record_id": 101,
            "record_url": "https://zenodo.org/records/101",
            "record_lists": ["aggregated-archives"],
        },
    ],
)
def test_record_schema_accepts_dataset_and_aggregate_rows(row):
    """Accept complete record rows for both supported record kinds."""
    load_validator("record.schema.json").validate(row)


def test_record_schema_rejects_an_empty_kind():
    """Force new records to receive a manual kind before tests can pass."""
    row = {
        "name": "NewDataset",
        "kind": "",
        "concept_id": 100,
        "current_record_id": 101,
        "record_url": "https://zenodo.org/records/101",
        "record_lists": ["classification-uv"],
    }

    with pytest.raises(ValidationError):
        load_validator("record.schema.json").validate(row)


def test_generated_record_catalog_matches_the_record_schema():
    """Validate every generated record, including rows awaiting a kind."""
    rows = [
        json.loads(line)
        for line in RECORDS_CATALOG_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    validator = load_validator("record.schema.json")

    for row in rows:
        validator.validate(row)


def test_example_dataset_rows_match_the_dataset_schema():
    """Validate the base, equal-length, and no-missing-value examples."""
    examples = Path(__file__).parent / "fixtures" / "dataset-examples.jsonl"
    rows = [
        json.loads(line)
        for line in examples.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    validator = load_validator("dataset.schema.json")

    assert len(rows) == 3
    for row in rows:
        validator.validate(row)


def test_dataset_schema_rejects_fields_derived_from_other_manifests():
    """Reject derived fields that would duplicate another source of truth."""
    row = {
        "dataset_name": "Adiac",
        "concept_id": 11179787,
        "task": "classification",
        "files": ["Adiac_TRAIN.ts", "Adiac_TEST.ts"],
        "statistics": {
            "case_counts": {"train": 390, "test": 391},
            "missing_value_count": 0,
            "n_channels": 1,
            "n_classes": 37,
            "series_length": {"min": 176, "max": 176, "mean": 176},
        },
    }

    with pytest.raises(ValidationError):
        load_validator("dataset.schema.json").validate(row)
