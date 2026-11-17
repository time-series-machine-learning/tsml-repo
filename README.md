# Time Series Machine Learning Archive

Browse and download published datasets from the [TSML community on Zenodo](https://zenodo.org/communities/tsml/records). Use the [issue tracker](https://github.com/time-series-machine-learning/tsml-repo/issues/new/choose) to suggest a dataset, report an archive problem, or give feedback on the website.

The [catalogue manifests](manifest/) list records, versions, and files. The [Python utilities](python_utilities/) generate these inventories and compute dataset statistics from local files.

## Load and classify a dataset with aeon

Install [aeon](https://www.aeon-toolkit.org/).

```sh
python -m pip install "aeon>=1.4.0"
```

Use the predefined TRAIN and TEST splits of GunPoint to train a ROCKET classifier and measure its accuracy on the test split. The loader can fetch missing archive files from Zenodo and reuse the copies in `extract_path`.

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_classification

data_dir = "local/tsml-data"
X_train, y_train = load_classification("GunPoint", split="train", extract_path=data_dir)
X_test, y_test = load_classification("GunPoint", split="test", extract_path=data_dir)

classifier = RocketClassifier(n_kernels=500, random_state=0)
classifier.fit(X_train, y_train)

print(f"Training data shape (cases, channels, time points): {X_train.shape}")
print(f"Test accuracy: {classifier.score(X_test, y_test):.1%}")
```

## Citing the archive

If you use the archive in your research, please cite the relevant archive paper and/or dataset.

## Licence

The code and documentation in this repository are distributed under the [BSD 3-Clause License](LICENSE). Consult each Zenodo record for the terms that apply to its data.
