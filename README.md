# Time Series Machine Learning Repository

This is a place to discuss the TSML time series classification and clustering archives hosted at <a href="https://www.timeseriesclassification.com/"> tsc.com</a>. If you use the archives, please star this repo. If you want to donate data or have any problems with data in the archive, please raise an issue.

You can download the data directly from tsc.com, or use aeon, whichlassifiers and clusterers that can be used on these data, and code to automatically download it. You can do

```python
from aeon.datasets._data_loaders import download_dataset
download_dataset("ArrowHead", save_path="C:\Temp")
```
or you can automatically download and load into memory, and use it directly with a classifier or clusterer. 
```python
import numpy as np
from aeon.datasets import load_classification
X, y, meta_data = load_classification("GunPoint", split="TRAIN")
testX, testy, meta_data = load_classification("GunPoint", split="TEST")
print(" Shape of X = ", X.shape)
print(" Meta data = ", meta_data)
from aeon.classification.feature_based import FreshPRINCEClassifier
fp = FreshPRINCEClassifier()
fp.fit(X,y)
pred = fp.predict(testX)
print("Number correct = ",np.sum(pred==testy))
print("Accuracy = ",np.sum(pred==testy)/len(testy))
```
Java TSC code is in the tsml repo

https://github.com/uea-machine-learning/tsml

Note:
22/11/23: The CharacterTrajectories dataset has been reformatted by Philip Darke (@philipdarke). The problem was that stripping out padding made channels unequal length, which is not a format supported by aeon. Philip has

•	Dropped initial and trailing zeros from all channels.
•	For the 39 cases with irregular channel lengths - padded the shorter channels with trailing zeros for the reasons discussed.

source code and instructions are here

here https://github.com/philipdarke/chartrajs-data.

20/11/23: New data donated by Paul Rabich, see  https://github.com/time-series-machine-learning/tsml-repo/issues/93





