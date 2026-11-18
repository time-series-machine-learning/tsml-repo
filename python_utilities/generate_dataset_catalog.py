"""Generate dataset catalogue rows from local .ts train/test pairs."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from filelock import FileLock

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
MANIFEST_DIR = REPOSITORY_ROOT / "manifest"
DATASET_CATALOG_PATH = MANIFEST_DIR / "datasets.jsonl"
RECORDS_CATALOG_PATH = MANIFEST_DIR / "records.jsonl"
FILES_SNAPSHOT_PATH = MANIFEST_DIR / "zenodo_files.jsonl"
VERSIONS_SNAPSHOT_PATH = MANIFEST_DIR / "zenodo_versions.jsonl"
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


def find_dataset_record(dataset_name, record_rows):
    """Find the current dataset record in the record catalogue."""
    matches = [row for row in record_rows if row.get("name") == dataset_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one dataset record named {dataset_name}, found "
            f"{len(matches)}."
        )

    record = matches[0]
    if record.get("kind") != "dataset":
        raise ValueError(f"Record {dataset_name} is not marked as a dataset.")
    return record


def md5_checksum(stream):
    """Compute the checksum format used in the Zenodo file inventory."""
    digest = hashlib.md5(usedforsecurity=False)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return f"md5:{digest.hexdigest()}"


def verify_current_record(record, version_rows):
    """Require the record catalogue to agree with the current version snapshot."""
    current_ids = [
        row["record_id"]
        for row in version_rows
        if row["concept_id"] == record["concept_id"] and row["is_current"]
    ]
    if current_ids != [record["current_record_id"]]:
        raise ValueError(
            f"The catalogue entry for {record['name']} does not match the current "
            "Zenodo version snapshot. Refresh the record catalogue first."
        )


def verify_published_file(path, expected, record_id):
    """Verify a local file against the current version's file inventory."""
    if expected is not None and path.stat().st_size == expected["size"]:
        with path.open("rb") as stream:
            if md5_checksum(stream) == expected["checksum"]:
                return
    raise ValueError(
        f"{path} does not match a published file for Zenodo record {record_id}."
    )


def verify_archive_members(archive_path, paths):
    """Compare extracted data to members of a verified ZIP without extracting it."""
    with ZipFile(archive_path) as archive:
        members = {}
        names = {path.name.casefold() for path in paths}
        for member in archive.infolist():
            name = PurePosixPath(member.filename).name.casefold()
            if member.is_dir() or name not in names:
                continue
            if name in members:
                raise ValueError(f"Multiple copies of {name} in {archive_path}.")
            members[name] = member

        for path in paths:
            member = members.get(path.name.casefold())
            if member is not None and path.stat().st_size == member.file_size:
                with path.open("rb") as local, archive.open(member) as published:
                    if md5_checksum(local) == md5_checksum(published):
                        continue
            raise ValueError(f"{path} does not match a dataset file in {archive_path}.")


def verify_dataset_files(pairs, record, file_rows, archive_path=None):
    """Check every discovered split against the dataset's current Zenodo version."""
    record_id = record["current_record_id"]
    files = {
        row["key"].casefold(): row
        for row in file_rows
        if row["record_id"] == record_id and row["concept_id"] == record["concept_id"]
    }
    paths = [path for pair in pairs.values() for path in pair.values()]
    if archive_path is not None:
        if archive_path.suffix.casefold() != ".zip":
            raise ValueError("--archive must point to the published ZIP file.")
        verify_published_file(
            archive_path, files.get(archive_path.name.casefold()), record_id
        )
        verify_archive_members(archive_path, paths)
        return

    if any(name.endswith(".zip") for name in files) and any(
        path.name.casefold() not in files for path in paths
    ):
        raise ValueError(f"Use --archive with the published ZIP for Zenodo record {record_id}.")
    for path in paths:
        verify_published_file(path, files.get(path.name.casefold()), record_id)


@contextmanager
def open_ts_split(path, loader=None):
    """Read the header once and decode one case at a time."""
    if loader is not None:
        cases, targets, metadata = loader(path)
        if len(cases) != len(targets):
            raise ValueError(f"{path} has {len(cases)} cases but {len(targets)} targets.")
        yield metadata, zip(cases, targets)
        return

    # Reuse the parsers behind load_from_ts_file without materializing its full array.
    from aeon.datasets._data_loaders import _load_data, _load_header_info

    with path.open(encoding="utf-8") as stream:
        metadata = _load_header_info(stream)
        allows_unequal_channels = not (metadata["equallength"] or metadata["timestamps"])

        def iter_cases():
            for line in stream:
                unequal_channels = False
                if allows_unequal_channels:
                    *channels, target = line.rstrip().split(":")
                    unequal_channels = len({channel.count(",") for channel in channels}) > 1
                if unequal_channels:
                    # aeon's case arrays require all channels to have the same length.
                    case = tuple(
                        np.asarray(channel.replace("?", "NaN").split(","), dtype=float)
                        for channel in channels
                    )
                    target = target.lower().replace("nan", "NaN").replace("?", "NaN")
                else:
                    cases, targets, _ = _load_data((line,), metadata)
                    case, target = cases[0], targets[0]
                yield case, float(target) if metadata["targetlabel"] else target

        yield metadata, iter_cases()


def metadata_task(metadata, path):
    """Infer classification or regression from one .ts header."""
    is_classification = bool(metadata.get("classlabel"))
    is_regression = bool(metadata.get("targetlabel"))
    if is_classification == is_regression:
        raise ValueError(
            f"{path} must declare exactly one of @classLabel or @targetLabel."
        )
    return "classification" if is_classification else "regression"


def summarize_case(case, path):
    """Return the channel lengths and missing-value count for one case."""
    if isinstance(case, tuple):
        channels = tuple(np.asarray(channel) for channel in case)
        if not channels or any(channel.ndim != 1 for channel in channels):
            raise ValueError(f"Expected one-dimensional channels in {path}.")
        lengths = tuple(channel.size for channel in channels)
        missing = sum(int(np.count_nonzero(np.isnan(channel))) for channel in channels)
        return lengths, missing

    values = np.asarray(case)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError(f"Expected a case with shape (n_channels, n_timepoints) in {path}.")
    channels, length = values.shape
    missing = int(np.count_nonzero(np.isnan(values)))
    return (length,) * channels, missing


def validate_length_flags(row):
    """Check that the length flags agree with the recorded range."""
    statistics = row["statistics"]
    lengths = statistics["series_length"]
    minimum, maximum = lengths["min"], lengths["max"]
    unequal = statistics.get("unequal_case_lengths", False) or statistics.get(
        "unequal_channel_lengths", False
    )
    if minimum > maximum or unequal != (minimum < maximum):
        raise ValueError(
            f"{row['dataset_name']} has unequal-length flags inconsistent "
            "with its series length range."
        )


def build_dataset_row(dataset_name, concept_id, pair, loader=None):
    """Accumulate dataset statistics without holding a full split in memory."""
    case_counts = {}
    tasks = set()
    classes = set()
    reference_lengths = None
    unequal_case_lengths = False
    unequal_channel_lengths = False
    min_length = None
    max_length = 0
    total_samples = 0
    missing_value_count = 0

    for split in SPLITS:
        path = pair[split]
        case_counts[split] = 0
        split_length = None
        with open_ts_split(path, loader) as (metadata, cases):
            task = metadata_task(metadata, path)
            tasks.add(task)
            for case, target in cases:
                channel_lengths, missing = summarize_case(case, path)
                if reference_lengths is None:
                    reference_lengths = channel_lengths
                channels = len(channel_lengths)
                if channels != len(reference_lengths) or (
                    metadata.get("univariate") and channels != 1
                ):
                    raise ValueError(f"Inconsistent channel count in {path}.")
                shortest, longest = min(channel_lengths), max(channel_lengths)
                if split_length is None:
                    split_length = shortest
                if metadata.get("equallength") and not shortest == longest == split_length:
                    raise ValueError(f"Cases have unequal lengths in equal-length split {path}.")

                unequal_case_lengths |= channel_lengths != reference_lengths
                unequal_channel_lengths |= shortest != longest
                case_counts[split] += 1
                min_length = shortest if min_length is None else min(min_length, shortest)
                max_length = max(max_length, longest)
                total_samples += sum(channel_lengths)
                missing_value_count += missing
                if task == "classification":
                    classes.add(target)

    if len(tasks) != 1:
        raise ValueError(f"{dataset_name} TRAIN and TEST declare different tasks.")
    if reference_lengths is None:
        raise ValueError("The train/test pair contains no cases.")

    task = tasks.pop()
    n_channels = len(reference_lengths)
    statistics = {
        "case_counts": case_counts,
        "missing_value_count": missing_value_count,
        "n_channels": n_channels,
        "series_length": {
            "min": min_length,
            "max": max_length,
            "mean": total_samples / (n_channels * sum(case_counts.values())),
        },
    }
    if task == "classification":
        statistics["n_classes"] = len(classes)
    if unequal_case_lengths:
        statistics["unequal_case_lengths"] = True
    if unequal_channel_lengths:
        statistics["unequal_channel_lengths"] = True

    row = {
        "concept_id": concept_id,
        "dataset_name": dataset_name,
        "statistics": statistics,
        "task": task,
    }
    validate_length_flags(row)
    return row


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
        help="Record JSONL file used to find the current Zenodo record.",
    )
    parser.add_argument(
        "--files",
        type=Path,
        default=FILES_SNAPSHOT_PATH,
        help="Zenodo file JSONL snapshot used to verify local inputs.",
    )
    parser.add_argument(
        "--versions",
        type=Path,
        default=VERSIONS_SNAPSHOT_PATH,
        help="Zenodo version JSONL snapshot used to check the current record ID.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Published ZIP to verify against when using extracted dataset files.",
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
    record = find_dataset_record(args.dataset_name, record_rows)
    catalog_path = args.catalog.resolve()
    lock_path = catalog_path.with_name(f".{catalog_path.name}.lock")
    lock = FileLock(str(lock_path))

    if not args.replace_existing and catalog_path.is_file():
        with lock:
            merge_dataset_rows(read_jsonl(catalog_path), [], args.dataset_name)

    verify_current_record(record, read_jsonl(args.versions))
    verify_dataset_files(pairs, record, read_jsonl(args.files), args.archive)
    generated_rows = [
        build_dataset_row(name, record["concept_id"], pair)
        for name, pair in pairs.items()
    ]
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        rows = merge_dataset_rows(
            read_jsonl(catalog_path),
            generated_rows,
            args.dataset_name,
            replace=args.replace_existing,
        )
        write_jsonl(catalog_path, rows)
    action = "Replaced" if args.replace_existing else "Added"
    names = ", ".join(row["dataset_name"] for row in generated_rows)
    print(f"{action} {len(generated_rows)} dataset rows: {names}.")
    print(f"Wrote {len(rows)} rows to {args.catalog}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
