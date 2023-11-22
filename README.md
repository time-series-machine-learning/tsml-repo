# Time Series Machine Learning Repository

This is a place to discuss the UCR and UEA time series classification archives hosted at https://www.timeseriesclassification.com/. If you use the archives, please star this repo. If you want to donate data or have any problems with data in the archive, please raise an issue.

We will document all changes on the Wiki page, and update this repo with more info when we get time

aeon contains equivalent python classifiers and clusterers that can be used on these data, and code to automatically download the data

```python
from aeon.datasets import load_classification
X, y, meta_data = load_classification("GunPoint")
print(" Shape of X = ", X.shape)
print(" Meta data = ", meta_data)
```
Java TSC code is in the tsml repo

https://github.com/uea-machine-learning/tsml

Note:
22/11/23: The CharacterTrajectories dataset has been reformatted by Philip Darke (@philipdarke). The problem was that stripping out padding made channels unequal length, which is not a format supported by aeon. Philip has

•	Dropped initial and trailing zeros from all channels.
•	For the 39 cases with irregular channel lengths - padded the shorter channels with trailing zeros for the reasons discussed.

source code and instructions are here

here https://github.com/philipdarke/chartrajs-data.





