# Loading & Citing Data

All datasets in the archive are hosted openly in the [TSML Zenodo community](https://zenodo.org/communities/tsml/records) as predefined `TRAIN` and `TEST` splits in `.ts` format.

## Loading Datasets with `aeon`

Install [`aeon`](https://www.aeon-toolkit.org/):

```sh
python -m pip install "aeon>=1.4.0"
```

### Classification & Extrinsic Regression

Use `load_classification` for classification problems and `load_regression` for extrinsic regression problems:

```python
from aeon.datasets import load_classification, load_regression

# Classification (e.g. GunPoint)
X_train, y_train = load_classification("GunPoint", split="train", extract_path="local/tsml-data")
X_test, y_test = load_classification("GunPoint", split="test", extract_path="local/tsml-data")

# Extrinsic regression (e.g. AppliancesEnergy)
X_reg_train, y_reg_train = load_regression("AppliancesEnergy", split="train", extract_path="local/tsml-data")
X_reg_test, y_reg_test = load_regression("AppliancesEnergy", split="test", extract_path="local/tsml-data")
```

### Preprocessed Variants (`_eq`, `_nmv`, `_disc`)

Benchmark lists in [Benchmarks & Bulk Archives](benchmarks.md) use explicit suffixes to identify exact dataset variants:

- **`_eq`** — preprocessed equal-length version of an unequal-length dataset (`load_equal_length=True` in `aeon`).
- **`_nmv`** — preprocessed version with missing values removed (`load_no_missing=True` in `aeon`).
- **`_disc`** — classification dataset formed by discretising a continuous regression target (published under its own Zenodo record, such as `AppliancesEnergy_disc`).

`_eq` and `_nmv` `.ts` pairs are stored inside the **same Zenodo record** as their base dataset:

```python
from aeon.datasets import load_classification

# Load AllGestureWiimoteX_eq from the AllGestureWiimoteX Zenodo record
X_train, y_train = load_classification("AllGestureWiimoteX", split="train", load_equal_length=True)

# Load DodgerLoopDay_nmv from the DodgerLoopDay Zenodo record
X_nmv, y_nmv = load_classification("DodgerLoopDay", split="train", load_no_missing=True)
```

### ZIP-Packaged Records

Large datasets distributed as a `.zip` on Zenodo (such as `S2Agri-17` and `S2Agri-34`, or the [bulk archive ZIP releases](benchmarks.md#bulk-aggregate-zenodo-releases)) can be downloaded from Zenodo, extracted locally, and loaded with `load_from_ts_file`:

```python
from aeon.datasets import load_from_ts_file

X_train, y_train = load_from_ts_file("local/S2Agri-17/S2Agri-17_TRAIN.ts")
X_test, y_test = load_from_ts_file("local/S2Agri-17/S2Agri-17_TEST.ts")
```

## Citing & Licences

- **Benchmark suites & collections:** If you use datasets or benchmark lists from the archive in your research, please cite the relevant archive or collection publication and/or original dataset source. Formatted citations and copyable Google Scholar BibTeX entries for all benchmark suites and collections are listed under [References & BibTeX](benchmarks.md#references-bibtex).
- **Toolkit citation:** If you load or evaluate datasets with `aeon`, please also consider citing the [JMLR toolkit paper](http://jmlr.org/papers/v25/23-1444.html):

```bibtex
@article{middlehurst2024aeon,
  title={aeon: a Python toolkit for learning from time series},
  author={Middlehurst, Matthew and Ismail-Fawaz, Ali and Guillaume, Antoine and Holder, Christopher and Guijo-Rubio, David and Bulatova, Guzal and Tsaprounis, Leonidas and Mentel, Lukasz and Walter, Martin and Sch{\"a}fer, Patrick and Bagnall, Anthony},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={289},
  pages={1--10},
  year={2024}
}
```

- **Licences:** Repository code, manifests, and documentation are distributed under the [BSD 3-Clause License](https://github.com/time-series-machine-learning/tsml-repo/blob/main/LICENSE). Consult each Zenodo record for the terms that apply to its data.
