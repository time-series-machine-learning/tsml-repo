# Time Series Machine Learning Archive

Browse and download published datasets from the [TSML community on Zenodo](https://zenodo.org/communities/tsml/records). Use the [issue tracker](https://github.com/time-series-machine-learning/tsml-repo/issues/new/choose) to suggest a dataset, report an archive problem, or give feedback on the website.

See the [dataset catalogue guide](Datasets.md) for the record and file inventories and instructions for refreshing them.

## Load a dataset with aeon

Install [aeon](https://www.aeon-toolkit.org/) and load the predefined splits by dataset name.

```python
from aeon.datasets import load_classification

X_train, y_train = load_classification("Adiac", split="train")
X_test, y_test = load_classification("Adiac", split="test")
```

Cite the Zenodo version and the original dataset publication when using archive data in research.

## Licence

The code and documentation in this repository are distributed under the [BSD 3-Clause License](LICENSE). Consult each Zenodo record for the terms that apply to its data.
