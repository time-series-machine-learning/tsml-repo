"""Tests for the TSML website content and static JSON generator."""

import json

from python_utilities import build_website_content as web


def test_builds_catalogue_json_benchmarks_and_dataset_pages(tmp_path):
    """Generate complete website data and verify key dataset variants and pages."""
    docs_dir = tmp_path / "docs"
    descriptions_dir = docs_dir / "descriptions"
    descriptions_dir.mkdir(parents=True)
    (descriptions_dir / "GunPoint.md").write_text(
        "Curated GunPoint editorial note.\n", encoding="utf-8"
    )

    summary = web.build_website_content(docs_dir=docs_dir)

    assert summary["dataset_record_count"] >= 352
    assert summary["dataset_variant_count"] >= 397
    assert summary["benchmark_list_count"] == 13

    catalogue_path = docs_dir / "_static" / "data" / "catalogue.json"
    payload = json.loads(catalogue_path.read_text(encoding="utf-8"))
    assert len(payload["datasets"]) == summary["dataset_variant_count"]
    assert len(payload["benchmarks"]) == 13

    by_name = {row["dataset_name"]: row for row in payload["datasets"]}
    assert by_name["GunPoint"]["task"] == "classification"
    assert by_name["GunPoint"]["equal_length"] is True
    assert by_name["GunPoint"]["has_missing"] is False
    assert "missing_value_count" not in by_name["GunPoint"]
    assert by_name["DodgerLoopDay"]["has_missing"] is True
    assert by_name["DodgerLoopDay_nmv"]["has_missing"] is False
    assert by_name["PPGDalia_disc"]["unequal_channel_lengths"] is True
    assert by_name["PPGDalia_disc_eq"]["equal_length"] is True
    assert by_name["S2Agri-17"]["packaging"] == "zip"

    gunpoint_md = (docs_dir / "datasets" / "GunPoint.md").read_text(encoding="utf-8")
    assert "Curated GunPoint editorial note." in gunpoint_md
    assert 'load_classification("GunPoint"' in gunpoint_md
    assert "DOI:" not in gunpoint_md
    assert "Concept ID:" not in gunpoint_md
    assert "CC-BY-4.0" not in gunpoint_md
    assert "| Missing Values |" not in gunpoint_md

    dodger_md = (docs_dir / "datasets" / "DodgerLoopDay.md").read_text(encoding="utf-8")
    assert "{bdg-warning}`Missing values`" in dodger_md
    assert "| Missing Values |" in dodger_md
    assert "| Yes |" in dodger_md
    assert "| No |" in dodger_md

    ppg_md = (docs_dir / "datasets" / "PPGDalia_disc.md").read_text(encoding="utf-8")
    assert "::::{tab-set}" in ppg_md
    assert "load_equal_length=True" in ppg_md
    assert "## Version History" in ppg_md
    assert "{bdg-warning}`Unequal channels`" in ppg_md
    assert "(ch)" not in ppg_md

    gesture_md = (docs_dir / "datasets" / "AllGestureWiimoteX.md").read_text(
        encoding="utf-8"
    )
    assert "(ch)" not in gesture_md
    assert "cases)" not in gesture_md

    s2agri_md = (docs_dir / "datasets" / "S2Agri-17.md").read_text(encoding="utf-8")
    assert "load_from_ts_file" in s2agri_md

    benchmarks_md = (docs_dir / "_generated" / "benchmarks_content.md").read_text(
        encoding="utf-8"
    )
    assert "UnivariateClassification142-Bakeoff2024" in benchmarks_md
    assert "## Bulk Aggregate Zenodo Releases" in benchmarks_md
    assert "## References & BibTeX" in benchmarks_md
    assert "@article{dau2019ucr," in benchmarks_md
    assert "@article{middlehurst2024bake," in benchmarks_md
    assert ": M. Middlehurst et al." not in benchmarks_md
    assert ": D. Guijo-Rubio et al." not in benchmarks_md

    home_hero_md = (docs_dir / "_generated" / "home_hero.md").read_text(
        encoding="utf-8"
    )
    assert "{{" not in home_hero_md
    assert "tsml-hero-centered" in home_hero_md
    assert f">{summary['dataset_record_count']:,}<" in home_hero_md
    assert f">{summary['dataset_variant_count']:,}<" in home_hero_md
    assert f">{summary['aggregate_record_count']:,}<" in home_hero_md


def test_auto_discovers_collections_and_benchmark_categories():
    """Verify new collection slugs and CamelCase benchmark prefixes format automatically."""
    assert web.is_domain_collection("rehab") is True
    assert web.is_domain_collection("audio-speech") is True
    assert web.is_domain_collection("classification-uv") is False
    assert web.is_domain_collection("aggregated-archives") is False
    assert web.format_collection_label("audio-speech") == "Audio Speech"
    assert web.format_collection_label("ecg") == "ECG"
    assert web.format_benchmark_category("UnivariateClustering") == "Univariate Clustering"


def test_website_generator_requires_only_stdlib(monkeypatch):
    """Ensure docs builds on Read the Docs and CI never require numpy or aeon."""
    import importlib
    import sys

    for mod in ("numpy", "aeon", "jsonschema", "filelock"):
        monkeypatch.setitem(sys.modules, mod, None)
    reloaded = importlib.reload(web)
    assert reloaded.REPOSITORY_ROOT.is_dir()
