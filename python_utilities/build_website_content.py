"""Generate Sphinx MyST pages and static JSON data from TSML manifests and lists."""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = REPOSITORY_ROOT / "docs"

# Structural/task record lists in record_lists/ (all other dataset lists are domain collections)
STRUCTURAL_RECORD_LIST_PREFIXES = (
    "aggregated-",
    "classification",
    "regression",
    "clustering",
    "forecasting",
)

# Optional display-label overrides; unknown collections/categories auto-format cleanly
COLLECTION_LABEL_OVERRIDES = {
    "rehab": "Rehab-Pile",
    "monster": "MONSTER",
    "eeg": "EEG",
}

BENCHMARK_CATEGORY_OVERRIDES = {
    "UnivariateClassification": "Univariate Classification",
    "MultivariateClassification": "Multivariate Classification",
    "Regression": "Extrinsic Regression",
}

PREFERRED_CATEGORY_ORDER = (
    "UnivariateClassification",
    "MultivariateClassification",
    "Regression",
)

TASK_LOADER_FUNCTIONS = {
    "classification": "load_classification",
    "regression": "load_regression",
    "clustering": "load_classification",
    "forecasting": "load_forecasting",
}

TASK_DISPLAY_LABELS = {
    "classification": "Classification",
    "regression": "Extrinsic Regression",
    "clustering": "Clustering",
    "forecasting": "Forecasting",
}


def _read_jsonl(path):
    """Read a UTF-8 JSONL file using only the Python standard library."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dataset_family_names(record_name):
    """Return canonical base and preprocessed variant names for a record."""
    return (
        record_name,
        f"{record_name}_eq",
        f"{record_name}_nmv",
        f"{record_name}_eq_nmv",
    )


def _write_if_changed(path, content):
    """Write UTF-8 text only when missing or modified to preserve Sphinx mtimes."""
    if not path.is_file() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def format_collection_label(key):
    """Format a ``record_lists/`` collection slug into a human-readable label."""
    if key in COLLECTION_LABEL_OVERRIDES:
        return COLLECTION_LABEL_OVERRIDES[key]
    if len(key) <= 4 and key.isalpha():
        return key.upper()
    return key.replace("-", " ").replace("_", " ").title()


def format_benchmark_category(category):
    """Format an ``archive_lists/`` CamelCase category prefix into a heading."""
    if category in BENCHMARK_CATEGORY_OVERRIDES:
        return BENCHMARK_CATEGORY_OVERRIDES[category]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", category)


def is_domain_collection(list_name):
    """Return True when a ``record_lists/`` name represents a thematic/domain collection."""
    return not list_name.startswith(STRUCTURAL_RECORD_LIST_PREFIXES)


def format_bytes(num_bytes):
    """Format an integer byte count into a compact human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value):,} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes:,} B"


def parse_archive_readme(readme_path):
    """Extract benchmark descriptions and ``## References`` from ``archive_lists/README.md``."""
    text = readme_path.read_text(encoding="utf-8")
    descriptions = {}
    pattern = re.compile(
        r"\|\s*\[`([^`]+\.txt)`\]\([^)]+\)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|"
    )
    for match in pattern.finditer(text):
        filename, count_text, description = match.groups()
        descriptions[filename] = (int(count_text), description.strip())

    ref_match = re.search(r"^##\s+References\b.*$", text, flags=re.MULTILINE)
    references_body = ""
    if ref_match:
        references_body = text[ref_match.end() :].strip()
    return descriptions, references_body


def variant_kind(dataset_name, record_name):
    """Classify a dataset identifier relative to its base record title."""
    if dataset_name == record_name:
        return "base"
    suffix = dataset_name[len(record_name) :]
    if suffix == "_eq":
        return "eq"
    if suffix == "_nmv":
        return "nmv"
    return suffix.lstrip("_") or "variant"


def format_length_summary(stats):
    """Format series length statistics concisely for Markdown tables."""
    lengths = stats["series_length"]
    min_len, max_len, mean_len = lengths["min"], lengths["max"], lengths["mean"]
    if min_len == max_len:
        return f"{min_len:,}"
    return f"{min_len:,}–{max_len:,} (mean {mean_len:,.1f})"


def build_aeon_snippet(record_name, variant_name, task, is_zip_only):
    """Generate a ready-to-copy Python loader snippet for one dataset variant."""
    if is_zip_only:
        return (
            "from aeon.datasets import load_from_ts_file\n\n"
            f"# {record_name} is distributed as a ZIP archive on Zenodo.\n"
            f"# Download and extract {record_name}.zip from Zenodo first:\n"
            f'X_train, y_train = load_from_ts_file("local/{record_name}/{variant_name}_TRAIN.ts")\n'
            f'X_test, y_test = load_from_ts_file("local/{record_name}/{variant_name}_TEST.ts")'
        )

    loader_fn = TASK_LOADER_FUNCTIONS.get(task, "load_from_ts_file")
    vkind = variant_kind(variant_name, record_name)
    extra_kwargs = ""
    if vkind == "eq":
        extra_kwargs = ", load_equal_length=True"
    elif vkind == "nmv":
        extra_kwargs = ", load_no_missing=True"

    return (
        f"from aeon.datasets import {loader_fn}\n\n"
        f'X_train, y_train = {loader_fn}("{record_name}", split="train"{extra_kwargs})\n'
        f'X_test, y_test = {loader_fn}("{record_name}", split="test"{extra_kwargs})'
    )


def _render_dataset_page(
    record,
    variants,
    files,
    versions,
    benchmarks_by_dataset,
    benchmark_meta_by_id,
    collection_labels,
    editorial_md=None,
):
    """Render one MyST Markdown page for a ``kind == 'dataset'`` Zenodo record."""
    record_name = record["name"]
    record_id = record["current_record_id"]
    record_url = record["record_url"]

    base_row = next(v for v in variants if v["dataset_name"] == record_name)
    base_stats = base_row["statistics"]
    task = base_row["task"]
    n_channels = base_stats["n_channels"]
    is_zip_only = any(f["key"].lower().endswith(".zip") for f in files) and not any(
        f["key"].lower().endswith("_train.ts") for f in files
    )

    badges = []
    task_label = TASK_DISPLAY_LABELS.get(task, task.replace("_", " ").title())
    if task == "classification":
        badges.append(f"{{bdg-primary}}`{task_label}`")
    else:
        badges.append(f"{{bdg-secondary}}`{task_label}`")

    if n_channels == 1:
        badges.append("{bdg-info}`Univariate`")
    else:
        badges.append(f"{{bdg-info}}`Multivariate ({n_channels:,} channels)`")

    if base_stats["series_length"]["min"] == base_stats["series_length"]["max"]:
        badges.append(
            f"{{bdg-success}}`Equal length ({base_stats['series_length']['min']:,})`"
        )
    else:
        badges.append("{bdg-warning}`Unequal length`")
        if base_stats.get("unequal_channel_lengths"):
            badges.append("{bdg-warning}`Unequal channels`")

    if base_stats["missing_value_count"] == 0:
        badges.append("{bdg-success}`No missing values`")
    else:
        badges.append("{bdg-warning}`Missing values`")

    for v in variants:
        vkind = variant_kind(v["dataset_name"], record_name)
        if vkind != "base":
            badges.append(f"{{bdg-light}}`+{vkind} variant`")

    lines = [
        "---",
        "orphan: true",
        "---",
        "",
        "{.tsml-back-link}",
        "[← Back to Dataset Catalogue](../catalogue.md)",
        "",
        f"# {record_name}",
        "",
        "{.tsml-badge-row}",
        " ".join(badges),
        "",
        f"**Zenodo Record:** [{record_id}]({record_url})",
        "",
    ]

    if editorial_md:
        lines.extend([editorial_md.strip(), ""])

    lines.extend(["## Dataset Statistics", ""])
    has_classes = any(v["statistics"].get("n_classes") is not None for v in variants)
    any_unequal_ch = any(
        v["statistics"].get("unequal_channel_lengths") for v in variants
    )
    any_missing = any(v["statistics"]["missing_value_count"] > 0 for v in variants)

    headers = ["Variant", "Train Cases", "Test Cases", "Total Cases", "Channels"]
    aligns = ["---", "---:", "---:", "---:", "---:"]
    if has_classes:
        headers.append("Classes")
        aligns.append("---:")
    headers.append("Series Length")
    aligns.append("---")
    if any_unequal_ch:
        headers.append("Unequal Channels")
        aligns.append("---")
    if any_missing:
        headers.append("Missing Values")
        aligns.append("---")

    lines.extend(
        [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(aligns)} |",
        ]
    )

    for v in variants:
        st = v["statistics"]
        train_c = st["case_counts"]["train"]
        test_c = st["case_counts"]["test"]
        total_c = train_c + test_c
        cells = [
            f"`{v['dataset_name']}`",
            f"{train_c:,}",
            f"{test_c:,}",
            f"{total_c:,}",
            f"{st['n_channels']:,}",
        ]
        if has_classes:
            n_cls = st.get("n_classes")
            cells.append(f"{n_cls:,}" if n_cls is not None else "—")
        cells.append(format_length_summary(st))
        if any_unequal_ch:
            cells.append("Yes" if st.get("unequal_channel_lengths") else "No")
        if any_missing:
            cells.append("Yes" if st["missing_value_count"] > 0 else "No")
        lines.append(f"| {' | '.join(cells)} |")
    lines.append("")

    # Load with aeon section
    lines.extend(["## Load in Python with `aeon`", ""])
    if len(variants) == 1:
        snippet = build_aeon_snippet(record_name, record_name, task, is_zip_only)
        lines.extend(["```python", snippet, "```", ""])
    else:
        lines.append("::::{tab-set}")
        for v in variants:
            vname = v["dataset_name"]
            vkind = variant_kind(vname, record_name)
            tab_title = (
                f"{vname} (base)"
                if vkind == "base"
                else f"{vname} (equal length)"
                if vkind == "eq"
                else f"{vname} (no missing values)"
                if vkind == "nmv"
                else vname
            )
            snippet = build_aeon_snippet(record_name, vname, task, is_zip_only)
            lines.extend(
                [
                    f":::{{tab-item}} {tab_title}",
                    "```python",
                    snippet,
                    "```",
                    ":::",
                ]
            )
        lines.extend(["::::", ""])

    # Published Zenodo files section
    lines.extend(
        [
            "## Published Files",
            "",
            "| File | Size | Checksum | Download |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for f in files:
        size_label = f"{format_bytes(f['size'])} ({f['size']:,} B)"
        lines.append(
            f"| `{f['key']}` | {size_label} | `{f['checksum']}` | "
            f"[Download]({f['download_url']}) |"
        )
    lines.append("")

    # Benchmarks & collections
    bench_items = []
    for v in variants:
        vname = v["dataset_name"]
        for bid in benchmarks_by_dataset.get(vname, ()):
            bmeta = benchmark_meta_by_id[bid]
            variant_note = f" *(uses `{vname}`)*" if vname != record_name else ""
            bench_items.append(
                f'- <a href="../catalogue.html?benchmark={bid}">{bmeta["title"]}</a>{variant_note}'
            )

    collections = [
        (key, collection_labels[key])
        for key in collection_labels
        if key in record["record_lists"]
    ]

    if bench_items or collections:
        lines.extend(["## Benchmarks & Collections", ""])
        for item in bench_items:
            lines.append(item)
        for key, label in collections:
            lines.append(
                f'- <a href="../catalogue.html?collection={key}">{label} Collection</a>'
            )
        lines.append("")

    # Version history if multiple versions exist
    if len(versions) > 1:
        lines.extend(
            [
                "## Version History",
                "",
                "| Record ID | Title | Status | Link |",
                "| --- | --- | --- | --- |",
            ]
        )
        for ver in sorted(versions, key=lambda r: r["record_id"], reverse=True):
            status = "Current" if ver["is_current"] else "Previous version"
            lines.append(
                f"| `{ver['record_id']}` | `{ver['title']}` | {status} | "
                f"[Zenodo]({ver['record_url']}) |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_benchmarks_content(
    benchmarks,
    collections_info,
    aggregate_records,
    files_by_record,
    references_body,
):
    """Render the generated Markdown body included by ``docs/benchmarks.md``."""
    lines = []
    grouped = defaultdict(list)
    for b in benchmarks:
        grouped[b["category"]].append(b)

    ordered_categories = [c for c in PREFERRED_CATEGORY_ORDER if c in grouped] + sorted(
        c for c in grouped if c not in PREFERRED_CATEGORY_ORDER
    )

    for category_key in ordered_categories:
        items = grouped[category_key]
        lines.extend(
            [
                f"## {format_benchmark_category(category_key)}",
                "",
                "| Benchmark | Datasets | Description | Explore & Download |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for b in items:
            lines.append(
                f"| **`{b['id']}`** | {b['count']:,} | {b['description']} | "
                f'<a href="catalogue.html?benchmark={b["id"]}">View in Catalogue</a> · '
                f'<a href="{b["download_path"]}" download><code>.txt</code> list</a> |'
            )
        lines.append("")

    if collections_info:
        lines.extend(
            [
                "## Domain & Thematic Collections",
                "",
                "| Collection | Records | Explore |",
                "| --- | ---: | --- |",
            ]
        )
        for col in collections_info:
            lines.append(
                f"| **{col['label']}** | {col['count']:,} | "
                f'<a href="catalogue.html?collection={col["id"]}">View in Catalogue</a> |'
            )
        lines.append("")

    if aggregate_records:
        lines.extend(
            [
                "## Bulk Aggregate Zenodo Releases",
                "",
                "Published multi-dataset ZIP archives in the "
                "[TSML Zenodo community](https://zenodo.org/communities/tsml/records):",
                "",
                "| Archive Release | Zenodo Record | Files & Sizes | Download |",
                "| --- | --- | --- | --- |",
            ]
        )
        for rec in aggregate_records:
            rec_id = rec["current_record_id"]
            rec_files = files_by_record.get(rec_id, [])
            file_summaries = ", ".join(
                f"`{f['key']}` ({format_bytes(f['size'])})" for f in rec_files
            )
            dl_links = " · ".join(
                f"[Download `{f['key']}`]({f['download_url']})" for f in rec_files
            )
            lines.append(
                f"| **{rec['name']}** | [`{rec_id}`]({rec['record_url']}) | "
                f"{file_summaries} | {dl_links} |"
            )
        lines.append("")

    if references_body:
        lines.extend(
            [
                "(references-bibtex)=",
                "## References & BibTeX",
                "",
                references_body,
                "",
            ]
        )

    return "\n".join(lines)


def build_website_content(repo_root=REPOSITORY_ROOT, docs_dir=DEFAULT_DOCS_DIR):
    """Build ``catalogue.json``, benchmark downloads, and per-dataset MyST pages."""
    repo_root = Path(repo_root)
    docs_dir = Path(docs_dir)
    manifest_dir = repo_root / "manifest"
    archive_lists_dir = repo_root / "archive_lists"
    descriptions_dir = docs_dir / "descriptions"

    records = _read_jsonl(manifest_dir / "records.jsonl")
    datasets = _read_jsonl(manifest_dir / "datasets.jsonl")
    versions = _read_jsonl(manifest_dir / "zenodo_versions.jsonl")
    files = _read_jsonl(manifest_dir / "zenodo_files.jsonl")
    snapshots = _read_jsonl(manifest_dir / "snapshot.jsonl")

    latest_snapshot = snapshots[-1] if snapshots else {}
    retrieved_at = latest_snapshot.get("retrieved_at", "")

    files_by_record = defaultdict(list)
    for f in files:
        files_by_record[f["record_id"]].append(f)

    versions_by_concept = defaultdict(list)
    for v in versions:
        versions_by_concept[v["concept_id"]].append(v)

    variants_by_concept = defaultdict(list)
    for d in datasets:
        variants_by_concept[d["concept_id"]].append(d)

    # Parse archive_lists/*.txt and README.md dynamically
    readme_descriptions, references_body = parse_archive_readme(
        archive_lists_dir / "README.md"
    )
    static_data_dir = docs_dir / "_static" / "data"
    static_lists_dir = static_data_dir / "archive_lists"
    static_lists_dir.mkdir(parents=True, exist_ok=True)

    expected_txt_names = {p.name for p in archive_lists_dir.glob("*.txt")}
    for stale_txt in static_lists_dir.glob("*.txt"):
        if stale_txt.name not in expected_txt_names:
            stale_txt.unlink()

    benchmarks = []
    benchmark_meta_by_id = {}
    benchmarks_by_dataset = defaultdict(list)

    for txt_path in sorted(archive_lists_dir.glob("*.txt")):
        shutil.copy2(txt_path, static_lists_dir / txt_path.name)
        entries = [
            ln.strip()
            for ln in txt_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        match = re.match(r"^([A-Za-z]+?)(\d+)-(.+)\.txt$", txt_path.name)
        category = match.group(1) if match else "Other"
        short_name = match.group(3) if match else txt_path.stem
        _, description = readme_descriptions.get(txt_path.name, (len(entries), ""))
        bid = txt_path.stem
        bmeta = {
            "id": bid,
            "filename": txt_path.name,
            "title": f"{short_name} ({len(entries)})",
            "category": category,
            "category_label": format_benchmark_category(category),
            "count": len(entries),
            "description": description,
            "download_path": f"_static/data/archive_lists/{txt_path.name}",
        }
        benchmarks.append(bmeta)
        benchmark_meta_by_id[bid] = bmeta
        for ds_name in entries:
            benchmarks_by_dataset[ds_name].append(bid)

    # Prepare output directories without resetting unchanged mtimes
    datasets_docs_dir = docs_dir / "datasets"
    datasets_docs_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = docs_dir / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    record_by_concept = {r["concept_id"]: r for r in records}
    dataset_records = [r for r in records if r["kind"] == "dataset"]
    aggregate_records = [r for r in records if r["kind"] == "aggregate"]

    # Auto-discover all domain/thematic collections from record_lists memberships
    collection_counts = defaultdict(int)
    for rec in dataset_records:
        for list_name in rec["record_lists"]:
            if is_domain_collection(list_name):
                collection_counts[list_name] += 1

    collection_keys = sorted(
        collection_counts,
        key=lambda k: (-collection_counts[k], format_collection_label(k).casefold()),
    )
    collection_labels = {k: format_collection_label(k) for k in collection_keys}
    collections_info = [
        {"id": k, "label": collection_labels[k], "count": collection_counts[k]}
        for k in collection_keys
    ]

    expected_dataset_files = {f"{r['name']}.md" for r in dataset_records}
    for existing_md in datasets_docs_dir.glob("*.md"):
        if existing_md.name not in expected_dataset_files:
            existing_md.unlink()

    # Write per-dataset Markdown pages
    for rec in dataset_records:
        cid = rec["concept_id"]
        rec_name = rec["name"]
        order_map = {
            name: idx
            for idx, name in enumerate(_dataset_family_names(rec_name))
        }
        rec_variants = sorted(
            variants_by_concept[cid],
            key=lambda row: (order_map.get(row["dataset_name"], 99), row["dataset_name"]),
        )
        editorial_path = descriptions_dir / f"{rec_name}.md"
        editorial_md = (
            editorial_path.read_text(encoding="utf-8")
            if editorial_path.is_file()
            else None
        )
        page_md = _render_dataset_page(
            rec,
            rec_variants,
            files_by_record[rec["current_record_id"]],
            versions_by_concept[cid],
            benchmarks_by_dataset,
            benchmark_meta_by_id,
            collection_labels,
            editorial_md=editorial_md,
        )
        _write_if_changed(datasets_docs_dir / f"{rec_name}.md", page_md)

    # Build enriched catalogue JSON rows for all dataset variants
    catalogue_rows = []
    for d in datasets:
        ds_name = d["dataset_name"]
        cid = d["concept_id"]
        rec = record_by_concept[cid]
        rec_name = rec["name"]
        rec_id = rec["current_record_id"]
        rec_files = files_by_record[rec_id]
        files_cf = {f["key"].casefold(): f for f in rec_files}

        train_file = files_cf.get(f"{ds_name}_train.ts".casefold())
        test_file = files_cf.get(f"{ds_name}_test.ts".casefold())
        zip_files = [f for f in rec_files if f["key"].lower().endswith(".zip")]
        if train_file and test_file:
            packaging = "ts"
            total_bytes = train_file["size"] + test_file["size"]
        elif zip_files:
            packaging = "zip"
            total_bytes = sum(f["size"] for f in zip_files)
        else:
            packaging = "other"
            total_bytes = sum(f["size"] for f in rec_files)

        st = d["statistics"]
        train_c = st["case_counts"]["train"]
        test_c = st["case_counts"]["test"]
        lengths = st["series_length"]
        collections = [k for k in collection_keys if k in rec["record_lists"]]

        catalogue_rows.append(
            {
                "dataset_name": ds_name,
                "record_name": rec_name,
                "variant": variant_kind(ds_name, rec_name),
                "record_id": rec_id,
                "record_url": rec["record_url"],
                "page_url": f"datasets/{rec_name}.html",
                "task": d["task"],
                "train_cases": train_c,
                "test_cases": test_c,
                "total_cases": train_c + test_c,
                "n_channels": st["n_channels"],
                "n_classes": st.get("n_classes"),
                "min_length": lengths["min"],
                "max_length": lengths["max"],
                "mean_length": round(lengths["mean"], 2),
                "equal_length": lengths["min"] == lengths["max"],
                "unequal_case_lengths": bool(st.get("unequal_case_lengths", False)),
                "unequal_channel_lengths": bool(
                    st.get("unequal_channel_lengths", False)
                ),
                "has_missing": st["missing_value_count"] > 0,
                "packaging": packaging,
                "total_bytes": total_bytes,
                "benchmarks": benchmarks_by_dataset.get(ds_name, []),
                "collections": collections,
            }
        )

    summary = {
        "current_record_count": len(records),
        "dataset_record_count": len(dataset_records),
        "aggregate_record_count": len(aggregate_records),
        "dataset_variant_count": len(datasets),
        "preprocessed_variant_count": len(datasets) - len(dataset_records),
        "classification_variant_count": sum(
            d["task"] == "classification" for d in datasets
        ),
        "regression_variant_count": sum(d["task"] == "regression" for d in datasets),
        "benchmark_list_count": len(benchmarks),
        "retrieved_at": retrieved_at,
        "retrieved_date": retrieved_at[:10] if retrieved_at else "",
    }

    payload = {
        "summary": summary,
        "benchmarks": benchmarks,
        "collections": collections_info,
        "datasets": catalogue_rows,
    }
    _write_if_changed(
        static_data_dir / "catalogue.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )

    benchmarks_md = _render_benchmarks_content(
        benchmarks,
        collections_info,
        aggregate_records,
        files_by_record,
        references_body,
    )
    _write_if_changed(generated_dir / "benchmarks_content.md", benchmarks_md)

    home_hero_html = f"""<div class="tsml-hero tsml-hero-centered">
  <img class="tsml-hero-backdrop dark-light" src="_static/images/tsml-hero-waves.svg" alt="" aria-hidden="true" />
  <div class="tsml-hero-inner">
    <div class="tsml-hero-title">Time Series Machine Learning Archive</div>
    <p class="tsml-hero-lead">
      Standardized benchmark datasets for <strong>classification</strong>, <strong>extrinsic regression</strong>,
      and <strong>clustering</strong> — published openly as predefined <code>TRAIN</code> and <code>TEST</code>
      splits in <code>.ts</code> format.
    </p>
    <div class="tsml-hero-actions">
      <a class="tsml-cta tsml-cta-primary" href="catalogue.html">Explore Dataset Catalogue →</a>
      <a class="tsml-cta tsml-cta-secondary" href="https://zenodo.org/communities/tsml/records" target="_blank" rel="noopener">TSML on Zenodo ↗</a>
    </div>
    <div class="tsml-hero-metrics" aria-label="Archive summary statistics">
      <div class="tsml-hero-metric">
        <span class="tsml-hero-metric-num">{summary['dataset_record_count']:,}</span>
        <span class="tsml-hero-metric-label">Dataset Records</span>
      </div>
      <div class="tsml-hero-metric">
        <span class="tsml-hero-metric-num">{summary['dataset_variant_count']:,}</span>
        <span class="tsml-hero-metric-label">Total Variants</span>
      </div>
      <div class="tsml-hero-metric">
        <span class="tsml-hero-metric-num">{summary['benchmark_list_count']:,}</span>
        <span class="tsml-hero-metric-label">Benchmark Suites</span>
      </div>
      <div class="tsml-hero-metric">
        <span class="tsml-hero-metric-num">{summary['aggregate_record_count']:,}</span>
        <span class="tsml-hero-metric-label">Bulk ZIP Releases</span>
      </div>
    </div>
  </div>
</div>
"""
    _write_if_changed(generated_dir / "home_hero.md", home_hero_html)

    return summary


if __name__ == "__main__":
    stats = build_website_content()
    print(
        f"Generated {stats['dataset_record_count']} dataset pages and "
        f"{stats['dataset_variant_count']} catalogue rows."
    )
