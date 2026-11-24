---
html_theme.sidebar_secondary.remove: true
---

{.tsml-home-page}
# Time Series Machine Learning Archive

```{include} _generated/home_hero.md
```

::::{grid} 1 1 3 3
:gutter: 4

:::{grid-item-card} Benchmarks & Bulk Releases
:link: benchmarks
:link-type: doc

Standard benchmark suites, domain collections, downloadable `.txt` dataset lists, BibTeX references, and multi-dataset Zenodo ZIP releases.
:::

:::{grid-item-card} Loading & Citing Data
:link: usage
:link-type: doc

Load predefined `TRAIN` and `TEST` splits and `_eq` / `_nmv` variants in Python, verify local `.ts` archives, and review general citation, BibTeX, and licence notes.
:::

:::{grid-item-card} Contribute or Report Issues
:link: https://github.com/time-series-machine-learning/tsml-repo/issues/new/choose

Donate a new dataset, propose a benchmark list addition, report a data or metadata issue, or share feedback on GitHub.
:::
::::

## Quick Start with `aeon`

Install [`aeon`](https://www.aeon-toolkit.org/) and evaluate HIVE-COTE v2 on the predefined `GunPoint` splits (fetched automatically from Zenodo and cached locally):

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

```{toctree}
:hidden:
:maxdepth: 2

catalogue
benchmarks
usage
```
