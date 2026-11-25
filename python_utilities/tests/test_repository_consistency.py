"""Cross-file consistency and data-integrity checks for repository manifests and lists."""

import json
import re
from collections import Counter, defaultdict

import yaml

from python_utilities import (
    generate_catalog,
    generate_dataset_catalog,
    generate_zenodo_record_lists,
    generate_zenodo_snapshots,
)

REPOSITORY_ROOT = generate_catalog.REPOSITORY_ROOT
MANIFEST_DIR = REPOSITORY_ROOT / "manifest"
ARCHIVE_LISTS_DIR = REPOSITORY_ROOT / "archive_lists"
RECORD_LISTS_DIR = REPOSITORY_ROOT / "record_lists"
ZENODO_RECORDS_DIR = REPOSITORY_ROOT / "zenodo_records"
GITHUB_DIR = REPOSITORY_ROOT / ".github"


def load_canonical_jsonl(path):
    """Read a JSONL file and verify newline termination and canonical formatting."""
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n"), f"{path} must end with a newline."
    lines = raw.splitlines()
    assert lines and all(lines), f"{path} must not be empty or contain blank lines."

    rows = []
    for lineno, line in enumerate(lines, 1):
        row = json.loads(line)
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True)
        assert line == canonical, (
            f"{path.name}:{lineno} is not in canonical sorted-keys JSONL format."
        )
        rows.append(row)
    return rows


def load_clean_list(path):
    """Read a plain-text list file and verify newline, trimming, and uniqueness."""
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n"), f"{path} must end with a newline."
    lines = raw.splitlines()
    assert lines, f"{path} must not be empty."
    for lineno, line in enumerate(lines, 1):
        assert line and line == line.strip(), (
            f"{path}:{lineno} is blank or has leading/trailing whitespace."
        )
    duplicates = sorted(item for item, count in Counter(lines).items() if count > 1)
    assert not duplicates, f"{path} contains duplicate entries: {duplicates}."
    return lines


def parse_zenodo_version_list(path):
    """Parse a ``title: record_id`` inventory file under ``zenodo_records/``."""
    lines = load_clean_list(path)
    entries = []
    for lineno, line in enumerate(lines, 1):
        title, sep, record_id_text = line.rpartition(": ")
        assert sep == ": " and title and record_id_text.isdigit(), (
            f"{path}:{lineno} does not match 'title: record_id': {line!r}"
        )
        entries.append((title, int(record_id_text)))
    return entries


def test_manifests_use_canonical_jsonl_formatting_and_sort_order():
    """Verify canonical JSONL serialization and deterministic row ordering."""
    versions = load_canonical_jsonl(generate_zenodo_snapshots.VERSIONS_SNAPSHOT_PATH)
    files = load_canonical_jsonl(generate_zenodo_snapshots.FILES_SNAPSHOT_PATH)
    snapshots = load_canonical_jsonl(generate_zenodo_snapshots.SNAPSHOT_PATH)
    records = load_canonical_jsonl(generate_catalog.RECORDS_CATALOG_PATH)
    datasets = load_canonical_jsonl(generate_dataset_catalog.DATASET_CATALOG_PATH)

    assert snapshots
    assert versions == sorted(
        versions,
        key=lambda row: (row["concept_id"], row["record_id"]),
    )
    assert files == sorted(
        files,
        key=lambda row: (
            row["concept_id"],
            row["record_id"],
            row["key"].casefold(),
            row["key"],
        ),
    )
    assert records == sorted(
        records,
        key=lambda row: (row["name"].casefold(), row["name"], row["concept_id"]),
    )
    assert datasets == sorted(
        datasets,
        key=lambda row: (
            row["dataset_name"].casefold(),
            row["dataset_name"],
            row["concept_id"],
        ),
    )


def test_zenodo_record_lists_match_version_snapshot():
    """Ensure ``zenodo_records/*`` inventories agree with ``zenodo_versions.jsonl``."""
    current_entries = parse_zenodo_version_list(
        generate_zenodo_record_lists.CURRENT_VERSIONS_PATH
    )
    previous_entries = parse_zenodo_version_list(
        generate_zenodo_record_lists.PREVIOUS_VERSIONS_PATH
    )
    versions = load_canonical_jsonl(generate_zenodo_snapshots.VERSIONS_SNAPSHOT_PATH)

    current_titles = [title for title, _ in current_entries]
    assert len(current_titles) == len(set(current_titles)), (
        "Current Zenodo record titles must be unique."
    )

    expected_current = {
        (row["title"], row["record_id"]) for row in versions if row["is_current"]
    }
    expected_previous = {
        (row["title"], row["record_id"]) for row in versions if not row["is_current"]
    }
    assert set(current_entries) == expected_current
    assert set(previous_entries) == expected_previous


def test_record_lists_and_records_catalog_are_in_sync():
    """Check ``record_lists/*`` formatting, subsets, and ``records.jsonl`` sync."""
    list_entries = {}
    for path in sorted(RECORD_LISTS_DIR.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name.casefold() == "readme.md":
            continue
        list_entries[path.name] = set(load_clean_list(path))

    versions = load_canonical_jsonl(generate_zenodo_snapshots.VERSIONS_SNAPSHOT_PATH)
    records = load_canonical_jsonl(generate_catalog.RECORDS_CATALOG_PATH)
    memberships = generate_catalog.load_record_memberships(RECORD_LISTS_DIR)
    rebuilt = generate_catalog.build_record_rows(versions, memberships, records)
    assert rebuilt == records, (
        "manifest/records.jsonl is out of sync with zenodo_versions.jsonl or record_lists/."
    )

    for row in records:
        if row["kind"] == "aggregate":
            assert row["record_lists"] == ["aggregated-archives"]
        else:
            assert row["kind"] == "dataset"
            assert "aggregated-archives" not in row["record_lists"]
            assert row["record_lists"]

    assert list_entries["classification-uv-redux30"] <= list_entries["classification-uv"]
    assert list_entries["classification-mv-core65"] <= list_entries["classification-mv"]
    assert list_entries["classification-mv-uea30"] <= list_entries["classification-mv"]
    assert (
        list_entries["regression-uv"] | list_entries["regression-mv"]
        == list_entries["regression"]
    )
    assert not (list_entries["regression-uv"] & list_entries["regression-mv"])
    assert not (list_entries["classification-uv"] & list_entries["classification-mv"])
    assert not (
        (list_entries["classification-uv"] | list_entries["classification-mv"])
        & list_entries["regression"]
    )


def test_dataset_catalog_matches_records_and_published_files():
    """Verify dataset statistics, record linkages, and published file coverage."""
    records = load_canonical_jsonl(generate_catalog.RECORDS_CATALOG_PATH)
    datasets = load_canonical_jsonl(generate_dataset_catalog.DATASET_CATALOG_PATH)
    files = load_canonical_jsonl(generate_zenodo_snapshots.FILES_SNAPSHOT_PATH)
    versions = load_canonical_jsonl(generate_zenodo_snapshots.VERSIONS_SNAPSHOT_PATH)

    files_by_record = defaultdict(list)
    for file_row in files:
        files_by_record[file_row["record_id"]].append(file_row)

    for version in versions:
        assert version["record_url"] == f"https://zenodo.org/records/{version['record_id']}"
        assert files_by_record[version["record_id"]], (
            f"Zenodo record {version['record_id']} has no file entries."
        )

    record_by_concept = {row["concept_id"]: row for row in records}
    datasets_by_concept = defaultdict(list)
    dataset_by_name = {}

    for row in datasets:
        name = row["dataset_name"]
        assert name not in dataset_by_name, f"Duplicate dataset_name: {name}."
        dataset_by_name[name] = row

        concept_id = row["concept_id"]
        assert concept_id in record_by_concept, (
            f"{name} references unknown concept_id {concept_id}."
        )
        record = record_by_concept[concept_id]
        assert record["kind"] == "dataset", (
            f"{name} references non-dataset record {record['name']}."
        )
        assert name in generate_dataset_catalog.dataset_family_names(record["name"]), (
            f"{name} is not a valid variant of record {record['name']}."
        )
        datasets_by_concept[concept_id].append(row)

        generate_dataset_catalog.validate_length_flags(row)
        stats = row["statistics"]
        assert stats["case_counts"]["train"] >= 1
        assert stats["case_counts"]["test"] >= 1
        lengths = stats["series_length"]
        assert "mean" in lengths
        assert lengths["min"] <= lengths["mean"] <= lengths["max"]
        assert stats.get("unequal_case_lengths") is not False
        assert stats.get("unequal_channel_lengths") is not False

        if row["task"] == "classification":
            assert stats.get("n_classes", 0) >= 2
        else:
            assert "n_classes" not in stats

        if name.endswith("_eq") or "_eq_" in name:
            assert lengths["min"] == lengths["max"], f"{name} must have equal lengths."
        if name.endswith("_nmv") or "_nmv_" in name:
            assert stats["missing_value_count"] == 0, f"{name} must have no missing values."

    dataset_records = [row for row in records if row["kind"] == "dataset"]
    for record in dataset_records:
        concept_id = record["concept_id"]
        family_rows = datasets_by_concept[concept_id]
        family_names = {row["dataset_name"] for row in family_rows}
        assert record["name"] in family_names, (
            f"Dataset record {record['name']} has no base row in datasets.jsonl."
        )
        tasks = {row["task"] for row in family_rows}
        assert len(tasks) == 1, f"Record {record['name']} mixes tasks: {tasks}."

        task = next(iter(tasks))
        base_channels = dataset_by_name[record["name"]]["statistics"]["n_channels"]
        memberships = set(record["record_lists"])
        if memberships & {"classification-uv", "classification-uv-redux30"}:
            assert task == "classification" and base_channels == 1
        if memberships & {
            "classification-mv",
            "classification-mv-core65",
            "classification-mv-uea30",
        }:
            assert task == "classification" and base_channels > 1
        if "regression" in memberships:
            assert task == "regression"
        if "regression-uv" in memberships:
            assert task == "regression" and base_channels == 1
        if "regression-mv" in memberships:
            assert task == "regression" and base_channels > 1

        published_keys = {
            item["key"].casefold(): item["key"]
            for item in files_by_record[record["current_record_id"]]
        }
        has_zip = any(key.endswith(".zip") for key in published_keys)
        if not has_zip:
            for variant_name in family_names:
                assert f"{variant_name}_train.ts".casefold() in published_keys
                assert f"{variant_name}_test.ts".casefold() in published_keys

        for key in published_keys:
            if key.endswith("_train.ts"):
                prefix = key[: -len("_train.ts")]
                if f"{prefix}_test.ts" in published_keys:
                    assert any(name.casefold() == prefix for name in family_names), (
                        f"Record {record['name']} publishes {prefix} TRAIN/TEST "
                        "without a matching row in datasets.jsonl."
                    )


def test_archive_lists_and_readme_match_dataset_catalog():
    """Validate benchmark list counts, dataset properties, and subset relationships."""
    datasets = {
        row["dataset_name"]: row
        for row in load_canonical_jsonl(generate_dataset_catalog.DATASET_CATALOG_PATH)
    }
    records = {
        row["name"]: row
        for row in load_canonical_jsonl(generate_catalog.RECORDS_CATALOG_PATH)
    }
    readme_text = (ARCHIVE_LISTS_DIR / "README.md").read_text(encoding="utf-8")
    readme_counts = {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r"\|\s*\[`([^`]+\.txt)`\]\([^)]+\)\s*\|\s*(\d+)\s*\|",
            readme_text,
        )
    }

    txt_paths = sorted(ARCHIVE_LISTS_DIR.glob("*.txt"))
    assert set(readme_counts) == {path.name for path in txt_paths}

    lists_by_stem = {}
    for path in txt_paths:
        entries = load_clean_list(path)
        lists_by_stem[path.stem] = entries

        match = re.match(
            r"^(UnivariateClassification|MultivariateClassification|Regression)"
            r"(\d+)-(.+)\.txt$",
            path.name,
        )
        assert match is not None, f"Unexpected archive list filename: {path.name}."
        prefix, expected_count_text, label = match.groups()
        assert len(entries) == int(expected_count_text)
        assert readme_counts[path.name] == len(entries)

        expected_task = "regression" if prefix == "Regression" else "classification"
        is_clean_benchmark = "Clean" in label or label in {"MultiverseCore", "Bakeoff2024"}
        for name in entries:
            assert name in datasets, f"{path.name} references unknown dataset {name}."
            row = datasets[name]
            stats = row["statistics"]
            assert row["task"] == expected_task
            if prefix == "UnivariateClassification":
                assert stats["n_channels"] == 1
            elif prefix == "MultivariateClassification":
                assert stats["n_channels"] > 1
            if is_clean_benchmark:
                assert stats["missing_value_count"] == 0
                assert stats["series_length"]["min"] == stats["series_length"]["max"]

    assert set(lists_by_stem["MultivariateClassification26-UEA2018Clean"]) <= set(
        lists_by_stem["MultivariateClassification30-UEA2018"]
    )
    assert set(lists_by_stem["UnivariateClassification85-UCR2015"]) <= set(
        lists_by_stem["UnivariateClassification128-UCR2018"]
    )
    ucr112 = set(lists_by_stem["UnivariateClassification112-UCR2018Clean"])
    bakeoff30 = set(lists_by_stem["UnivariateClassification30-Bakeoff2024"])
    assert ucr112 <= set(lists_by_stem["UnivariateClassification128-UCR2018"])
    assert not (ucr112 & bakeoff30)
    assert set(lists_by_stem["UnivariateClassification142-Bakeoff2024"]) == (
        ucr112 | bakeoff30
    )
    assert set(lists_by_stem["Regression19-Monash"]) <= set(
        lists_by_stem["Regression63-MonashExtended"]
    )
    assert set(lists_by_stem["MultivariateClassification65-MultiverseCore"]) <= set(
        lists_by_stem["MultivariateClassification133-MultiverseClean"]
    )

    def concepts_for_archive(stem):
        return {datasets[name]["concept_id"] for name in lists_by_stem[stem]}

    def concepts_for_record_list(list_name):
        names = load_clean_list(RECORD_LISTS_DIR / list_name)
        return {records[name]["concept_id"] for name in names}

    assert concepts_for_archive("MultivariateClassification133-Multiverse") == (
        concepts_for_archive("MultivariateClassification133-MultiverseClean")
    )
    assert concepts_for_archive("Regression63-MonashExtended") == (
        concepts_for_archive("Regression63-MonashExtendedClean")
    )
    assert concepts_for_record_list("classification-mv-core65") == (
        concepts_for_archive("MultivariateClassification65-MultiverseCore")
    )
    assert concepts_for_record_list("classification-mv-uea30") == (
        concepts_for_archive("MultivariateClassification30-UEA2018")
    )
    assert concepts_for_record_list("classification-uv-redux30") == (
        concepts_for_archive("UnivariateClassification30-Bakeoff2024")
    )
    assert concepts_for_archive("MultivariateClassification133-Multiverse") <= (
        concepts_for_record_list("classification-mv")
    )
    assert (
        concepts_for_archive("UnivariateClassification128-UCR2018")
        | concepts_for_archive("UnivariateClassification30-Bakeoff2024")
    ) <= concepts_for_record_list("classification-uv")
    assert concepts_for_archive("Regression63-MonashExtended") <= (
        concepts_for_record_list("regression")
    )


def test_github_yaml_configs_are_valid():
    """Parse repository GitHub workflows and issue forms as valid YAML mappings."""
    workflow_paths = sorted((GITHUB_DIR / "workflows").glob("*.yml"))
    issue_form_paths = sorted((GITHUB_DIR / "ISSUE_TEMPLATE").glob("*.yml"))
    assert workflow_paths and issue_form_paths

    for path in workflow_paths:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"{path} did not parse as a YAML mapping."
        assert (parsed.get("on") or parsed.get(True)) and "jobs" in parsed

    for path in issue_form_paths:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"{path} did not parse as a YAML mapping."
        assert {"name", "description", "body"} <= set(parsed)

    rtd_path = REPOSITORY_ROOT / ".readthedocs.yaml"
    if rtd_path.is_file():
        rtd_config = yaml.safe_load(rtd_path.read_text(encoding="utf-8"))
        assert isinstance(rtd_config, dict)
        assert rtd_config.get("version") == 2
        assert "sphinx" in rtd_config
