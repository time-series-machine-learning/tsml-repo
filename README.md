# Time Series Machine Learning Archive

**[TSML Archive Website](https://tsml-archive.readthedocs.io/)** · **[timeseriesclassification.com](https://www.timeseriesclassification.com/)** · **[TSML on Zenodo](https://zenodo.org/communities/tsml/records)** · **[Issue Tracker](https://github.com/time-series-machine-learning/tsml-repo/issues/new/choose)**

Catalogue, benchmark dataset lists, and issue tracker for the **Time Series Machine Learning Archive** — standardized datasets for **classification**, **extrinsic regression**, and **clustering** published openly in the [TSML Zenodo community](https://zenodo.org/communities/tsml/records) as predefined `TRAIN` and `TEST` splits in `.ts` format.

- Browse, filter, and export dataset metadata in the **[Dataset Catalogue](https://tsml-archive.readthedocs.io/en/latest/catalogue.html)** or on **[timeseriesclassification.com](https://www.timeseriesclassification.com/)**.
- Download standard benchmark identifier lists and bulk ZIP links on **[Benchmarks & Bulk Archives](https://tsml-archive.readthedocs.io/en/latest/benchmarks.html)** (maintained in [`archive_lists/`](archive_lists/README.md)).
- Suggest a dataset, report a data or metadata problem, or share feedback on the website via the **[issue tracker](https://github.com/time-series-machine-learning/tsml-repo/issues/new/choose)**.

## Repository Structure

| Directory | Contents |
| --- | --- |
| [`archive_lists/`](archive_lists/README.md) | Plain-text dataset identifier lists (`.txt`) for standard evaluation benchmarks, plus benchmark references and BibTeX. |
| [`record_lists/`](record_lists/README.md) | Record membership lists grouping Zenodo records by task, subset, and thematic domain collection. |
| [`manifest/`](manifest/) | Canonical JSONL catalogues and Zenodo version/file snapshots (`records.jsonl`, `datasets.jsonl`, `zenodo_versions.jsonl`, `zenodo_files.jsonl`). |
| [`docs/`](docs/) | Sphinx source for the [TSML Archive website](https://tsml-archive.readthedocs.io/). |
| [`python_utilities/`](python_utilities/README.md) | Maintainer CLI utilities for Zenodo inventory refreshes, dataset statistics, read-only audits, and website generation. |

## Quick Start with `aeon`

Install [`aeon`](https://www.aeon-toolkit.org/) and evaluate HIVE-COTE v2 on the predefined `GunPoint` splits (fetched automatically from Zenodo and cached in `extract_path`):

```sh
python -m pip install "aeon>=1.4.0"
```

```python
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets import load_classification

data_dir = "local/tsml-data"
X_train, y_train = load_classification("GunPoint", split="train", extract_path=data_dir)
X_test, y_test = load_classification("GunPoint", split="test", extract_path=data_dir)

classifier = HIVECOTEV2(random_state=0)
classifier.fit(X_train, y_train)

print(f"Training data shape (cases, channels, time points): {X_train.shape}")
print(f"Test accuracy: {classifier.score(X_test, y_test):.1%}")
```

See **[Loading & Citing Data](https://tsml-archive.readthedocs.io/en/latest/usage.html)** for loading extrinsic regression problems (`load_regression`), preprocessed variants (`load_equal_length=True` / `load_no_missing=True`), and ZIP-packaged datasets.

## Citing & Licences

- If you use datasets or benchmark lists from the archive in your research, please cite the relevant [benchmark or collection publication](archive_lists/README.md#references) and/or original dataset source.
- Repository code, manifests, and documentation are distributed under the [BSD 3-Clause License](LICENSE). Consult each Zenodo record for the terms that apply to its data.
