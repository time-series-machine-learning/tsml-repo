# Dataset Lists

This directory contains the dataset lists used for time series machine learning
benchmarks. Each `.txt` file contains one dataset identifier per line. The identifiers
correspond to datasets in the [Time Series Machine Learning community on Zenodo](https://zenodo.org/communities/tsml/records).

## Dataset Variants

- `_eq` identifies a preprocessed equal-length version of an unequal-length dataset.
- `_nmv` identifies a preprocessed version with missing values removed.
- `_disc` identifies a classification problem produced by discretising the continuous
  target of a time-series extrinsic regression dataset.
- Suffixes can be combined, as in `_disc_nmv`. The `_eq` and `_nmv` files are stored
  in the same Zenodo record as the corresponding dataset without that suffix.

For aeon’s Zenodo loaders, use the record name and pass
`load_equal_length=True` for `_eq` or `load_no_missing=True` for `_nmv`.
For records that publish a ZIP, download it from Zenodo and load the
extracted `.ts` files.

## Available Lists

| List | Entries | Description |
| --- | ---: | --- |
| [`MultivariateClassification26-UEA2018Clean.txt`](MultivariateClassification26-UEA2018Clean.txt) | 26 | The equal-length, no-missing-values subset of the 30-dataset UEA 2018 archive used in standard multivariate benchmarks. |
| [`MultivariateClassification30-UEA2018.txt`](MultivariateClassification30-UEA2018.txt) | 30 | The complete UEA 2018 multivariate time-series classification archive. |
| [`MultivariateClassification65-MultiverseCore.txt`](MultivariateClassification65-MultiverseCore.txt) | 65 | Multiverse-core (MV-core), the recommended computationally manageable subset of the Multiverse archive using cleaned variants. |
| [`MultivariateClassification133-Multiverse.txt`](MultivariateClassification133-Multiverse.txt) | 133 | The full 2026 Multiverse multivariate time-series classification archive, using each problem's original form. |
| [`MultivariateClassification133-MultiverseClean.txt`](MultivariateClassification133-MultiverseClean.txt) | 133 | The same 133 Multiverse problems, with 14 unequal-length or missing-value datasets replaced by their cleaned variants. |
| [`Regression19-Monash.txt`](Regression19-Monash.txt) | 19 | The original Monash University, UEA, and UCR time-series extrinsic regression (TSER) archive. |
| [`Regression63-MonashExtended.txt`](Regression63-MonashExtended.txt) | 63 | The 2024 extended TSER archive: the original 19 regression problems plus 44 additional datasets. |
| [`Regression63-MonashExtendedClean.txt`](Regression63-MonashExtendedClean.txt) | 63 | The same 63 extended TSER problems, selecting equal-length and no-missing-values variants. |
| [`UnivariateClassification30-Bakeoff2024.txt`](UnivariateClassification30-Bakeoff2024.txt) | 30 | The 30 new equal-length univariate datasets and cleaned variants introduced in the 2024 bake-off redux. |
| [`UnivariateClassification85-UCR2015.txt`](UnivariateClassification85-UCR2015.txt) | 85 | The complete 2015 release of the UCR univariate time-series classification archive. |
| [`UnivariateClassification112-UCR2018Clean.txt`](UnivariateClassification112-UCR2018Clean.txt) | 112 | The equal-length, no-missing-values UCR 2018 subset commonly used for clean univariate benchmarking. |
| [`UnivariateClassification128-UCR2018.txt`](UnivariateClassification128-UCR2018.txt) | 128 | The complete 2018 release of the UCR univariate time-series classification archive. |
| [`UnivariateClassification142-Bakeoff2024.txt`](UnivariateClassification142-Bakeoff2024.txt) | 142 | The complete 2024 bake-off benchmark, formed by combining the 112 clean UCR problems with the 30 bake-off redux datasets. |

## References

- [The UCR Time Series Archive](https://doi.org/10.1109/JAS.2019.1911747)
  - Dau, H.A., Bagnall, A., Kamgar, K., Yeh, C.C.M., Zhu, Y., Gharghabi, S., Ratanamahatana, C.A. and Keogh, E., 2019. The UCR time series archive. IEEE/CAA Journal of Automatica Sinica, 6(6), pp.1293-1305.
- [The UEA Multivariate Time Series Classification Archive, 2018](https://arxiv.org/abs/1811.00075)
  - Bagnall, A., Dau, H.A., Lines, J., Flynn, M., Large, J., Bostrom, A., Southam, P. and Keogh, E., 2018. The UEA multivariate time series classification archive, 2018. arXiv preprint arXiv:1811.00075.
- [Bake Off Redux](https://doi.org/10.1007/s10618-024-01022-1)
  - Middlehurst, M., Schäfer, P. and Bagnall, A., 2024. Bake off redux: a review and experimental evaluation of recent time series classification algorithms: M. Middlehurst et al. Data Mining and Knowledge Discovery, 38(4), pp.1958-2031.
- [Monash University, UEA, UCR Time Series Extrinsic Regression Archive](https://arxiv.org/abs/2006.10996)
  - Tan, C.W., Bergmeir, C., Petitjean, F. and Webb, G.I., 2020. Monash university, uea, ucr time series extrinsic regression archive. arXiv preprint arXiv:2006.10996.
- [Extended TSER archive](https://doi.org/10.1007/s10618-024-01027-w)
  - Guijo-Rubio, D., Middlehurst, M., Arcencio, G., Silva, D.F. and Bagnall, A., 2024. Unsupervised feature based algorithms for time series extrinsic regression: D. Guijo-Rubio et al. Data Mining and Knowledge Discovery, 38(4), pp.2141-2185.
- [The Multiverse of Time Series Machine Learning](https://arxiv.org/abs/2603.20352)
  - Middlehurst, M., Rushbrooke, A., Ismail-Fawaz, A., Devanne, M., Forestier, G., Dempster, A., Webb, G.I., Holder, C. and Bagnall, A., 2026. The multiverse of time series machine learning: an archive for multivariate time series classification. arXiv preprint arXiv:2603.20352.
