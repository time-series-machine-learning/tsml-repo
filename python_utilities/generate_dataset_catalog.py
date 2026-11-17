"""Generate dataset catalogue rows from local .ts train/test pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
MANIFEST_DIR = REPOSITORY_ROOT / "manifest"
DATASET_CATALOG_PATH = MANIFEST_DIR / "datasets.jsonl"
RECORDS_CATALOG_PATH = MANIFEST_DIR / "records.jsonl"
VARIANT_SUFFIXES = ("", "_eq", "_nmv")
SPLITS = ("train", "test")


def read_jsonl(path):
    """Read non-empty JSONL rows, or return an empty list if absent."""
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path, rows):
    """Atomically write deterministic newline-delimited JSON."""
    contents = "".join(
        f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def dataset_family_names(dataset_name):
    """Return the supported catalogue names for one dataset record."""
    return [f"{dataset_name}{suffix}" for suffix in VARIANT_SUFFIXES]


def discover_dataset_pairs(dataset_name, directory):
    """Find the base, equal-length, and no-missing-value .ts pairs."""
    if not directory.is_dir():
        raise ValueError(f"Dataset path is not a directory: {directory}.")

    expected = {}
    for variant_name in dataset_family_names(dataset_name):
        for split in SPLITS:
            filename = f"{variant_name}_{split.upper()}.ts"
            expected[filename.casefold()] = (variant_name, split)

    pairs = {name: {} for name in dataset_family_names(dataset_name)}
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: (str(path).casefold(), str(path)),
    )
    for path in paths:
        identity = expected.get(path.name.casefold())
        if identity is None:
            continue

        variant_name, split = identity
        previous_path = pairs[variant_name].get(split)
        if previous_path is not None:
            raise ValueError(
                f"Duplicate {variant_name} {split.upper()} files: "
                f"{previous_path} and {path}."
            )
        pairs[variant_name][split] = path

    incomplete = []
    for variant_name, split_paths in pairs.items():
        if split_paths and len(split_paths) != len(SPLITS):
            missing = next(split for split in SPLITS if split not in split_paths)
            incomplete.append(f"{variant_name} (missing {missing.upper()})")
    if incomplete:
        raise ValueError(f"Incomplete dataset pairs: {', '.join(incomplete)}.")

    if not pairs[dataset_name]:
        raise ValueError(
            f"No {dataset_name}_TRAIN.ts and {dataset_name}_TEST.ts pair found "
            f"under {directory}."
        )

    return {
        variant_name: pairs[variant_name]
        for variant_name in dataset_family_names(dataset_name)
        if pairs[variant_name]
    }


def find_concept_id(dataset_name, record_rows):
    """Find the dataset's Zenodo concept ID in the record catalogue."""
    matches = [row for row in record_rows if row.get("name") == dataset_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one dataset record named {dataset_name}, found "
            f"{len(matches)}."
        )

    record = matches[0]
    if record.get("kind") != "dataset":
        raise ValueError(f"Record {dataset_name} is not marked as a dataset.")
    return int(record["concept_id"])


def load_ts_file(path):
    """Load one .ts file through aeon, including its header metadata."""
    try:
        from aeon.datasets import load_from_ts_file
    except ModuleNotFoundError as error:
        if error.name != "aeon":
            raise
        raise ModuleNotFoundError(
            "aeon is required to generate dataset catalogue rows."
        ) from error

    return load_from_ts_file(str(path), return_meta_data=True)


def metadata_task(metadata, path):
    """Infer classification or regression from one .ts header."""
    is_classification = bool(metadata.get("classlabel"))
    is_regression = bool(metadata.get("targetlabel"))
    if is_classification == is_regression:
        raise ValueError(
            f"{path} must declare exactly one of @classLabel or @targetLabel."
        )
    return "classification" if is_classification else "regression"


def load_split(path, loader=None):
    """Load and validate the three values returned for one .ts split."""
    if loader is None:
        loader = load_ts_file
    result = loader(path)
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(f"The aeon loader returned an invalid result for {path}.")
    return result


def collection_statistics(collections):
    """Calculate shape, length, and missing-value statistics across splits."""
    case_counts = {}
    channels = set()
    lengths = []
    missing_value_count = 0

    for split, (X, y, _) in collections.items():
        n_cases = len(X)
        if len(y) != n_cases:
            raise ValueError(
                f"{split.upper()} has {n_cases} cases but {len(y)} targets."
            )
        case_counts[split] = n_cases

        for case in X:
            values = np.asarray(case)
            if values.ndim != 2:
                raise ValueError(
                    "Expected each aeon case to have shape "
                    "(n_channels, n_timepoints)."
                )
            channels.add(int(values.shape[0]))
            lengths.append(int(values.shape[1]))
            missing_value_count += int(np.count_nonzero(np.isnan(values)))

    if not lengths:
        raise ValueError("The train/test pair contains no cases.")
    if len(channels) != 1:
        raise ValueError("Cases do not all have the same number of channels.")

    return {
        "case_counts": case_counts,
        "missing_value_count": missing_value_count,
        "n_channels": channels.pop(),
        "series_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def build_dataset_row(dataset_name, concept_id, pair, loader=None):
    """Load one pair and build its schema-ready dataset catalogue row."""
    collections = {
        split: load_split(pair[split], loader=loader) for split in SPLITS
    }
    tasks = {
        metadata_task(metadata, pair[split])
        for split, (_, _, metadata) in collections.items()
    }
    if len(tasks) != 1:
        raise ValueError(f"{dataset_name} TRAIN and TEST declare different tasks.")

    task = tasks.pop()
    statistics = collection_statistics(collections)
    if task == "classification":
        labels = np.concatenate(
            [np.asarray(collections[split][1]).reshape(-1) for split in SPLITS]
        )
        n_classes = int(np.unique(labels).size)
        if n_classes == 0:
            raise ValueError(f"{dataset_name} has no class labels.")
        statistics["n_classes"] = n_classes

    return {
        "concept_id": concept_id,
        "dataset_name": dataset_name,
        "statistics": statistics,
        "task": task,
    }


def merge_dataset_rows(existing_rows, generated_rows, dataset_name, replace=False):
    """Add one dataset family, optionally replacing all of its existing rows."""
    family = set(dataset_family_names(dataset_name))
    existing_family = [
        row for row in existing_rows if row.get("dataset_name") in family
    ]
    if existing_family and not replace:
        names = ", ".join(row["dataset_name"] for row in existing_family)
        raise ValueError(
            f"Dataset catalogue rows already exist for: {names}. Use "
            "--replace-existing to refresh the complete dataset family."
        )

    retained_rows = [
        row for row in existing_rows if row.get("dataset_name") not in family
    ]
    rows = retained_rows + generated_rows
    names = [row["dataset_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("The dataset catalogue contains duplicate dataset names.")

    return sorted(
        rows,
        key=lambda row: (
            row["dataset_name"].casefold(),
            row["dataset_name"],
            row["concept_id"],
        ),
    )


def parse_args(argv=None):
    """Parse command-line arguments for dataset catalogue generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="Base dataset name, such as Adiac.")
    parser.add_argument(
        "path",
        type=Path,
        help="Directory to scan recursively for matching .ts files.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DATASET_CATALOG_PATH,
        help="Dataset JSONL file to update.",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=RECORDS_CATALOG_PATH,
        help="Record JSONL file used to find the Zenodo concept ID.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace every existing base, _eq, and _nmv row for the dataset.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Scan local data and update the dataset catalogue atomically."""
    args = parse_args(argv)
    pairs = discover_dataset_pairs(args.dataset_name, args.path)
    record_rows = read_jsonl(args.records)
    concept_id = find_concept_id(args.dataset_name, record_rows)
    existing_rows = read_jsonl(args.catalog)

    if not args.replace_existing:
        merge_dataset_rows(existing_rows, [], args.dataset_name)

    generated_rows = [
        build_dataset_row(name, concept_id, pair)
        for name, pair in pairs.items()
    ]
    rows = merge_dataset_rows(
        existing_rows,
        generated_rows,
        args.dataset_name,
        replace=args.replace_existing,
    )

    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.catalog, rows)
    action = "Replaced" if args.replace_existing else "Added"
    names = ", ".join(row["dataset_name"] for row in generated_rows)
    print(f"{action} {len(generated_rows)} dataset rows: {names}.")
    print(f"Wrote {len(rows)} rows to {args.catalog}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
