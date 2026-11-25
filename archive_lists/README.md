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

- [The UCR Time Series Archive](https://doi.org/10.1109/JAS.2019.1911747) — Dau, H.A., Bagnall, A., Kamgar, K., Yeh, C.C.M., Zhu, Y., Gharghabi, S., Ratanamahatana, C.A. and Keogh, E., 2019. The UCR time series archive. *IEEE/CAA Journal of Automatica Sinica*, 6(6), pp.1293–1305.

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{dau2019ucr,
    title={The UCR time series archive},
    author={Dau, Hoang Anh and Bagnall, Anthony and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Keogh, Eamonn},
    journal={IEEE/CAA Journal of Automatica Sinica},
    volume={6},
    number={6},
    pages={1293--1305},
    year={2019},
    publisher={IEEE}
  }
  ```

  </details>

- [The UEA Multivariate Time Series Classification Archive, 2018](https://arxiv.org/abs/1811.00075) — Bagnall, A., Dau, H.A., Lines, J., Flynn, M., Large, J., Bostrom, A., Southam, P. and Keogh, E., 2018. The UEA multivariate time series classification archive, 2018. *arXiv preprint arXiv:1811.00075*.

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{bagnall2018uea,
    title={The UEA multivariate time series classification archive, 2018},
    author={Bagnall, Anthony and Dau, Hoang Anh and Lines, Jason and Flynn, Michael and Large, James and Bostrom, Aaron and Southam, Paul and Keogh, Eamonn},
    journal={arXiv preprint arXiv:1811.00075},
    year={2018}
  }
  ```

  </details>

- [Bake Off Redux](https://doi.org/10.1007/s10618-024-01022-1) — Middlehurst, M., Schäfer, P. and Bagnall, A., 2024. Bake off redux: a review and experimental evaluation of recent time series classification algorithms. *Data Mining and Knowledge Discovery*, 38(4), pp.1958–2031.

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{middlehurst2024bake,
    title={Bake off redux: a review and experimental evaluation of recent time series classification algorithms},
    author={Middlehurst, Matthew and Sch{\"a}fer, Patrick and Bagnall, Anthony},
    journal={Data Mining and Knowledge Discovery},
    volume={38},
    number={4},
    pages={1958--2031},
    year={2024},
    publisher={Springer}
  }
  ```

  </details>

- [Time Series Extrinsic Regression (Monash, UEA & UCR TSER Archive)](https://doi.org/10.1007/s10618-021-00745-9) — Tan, C.W., Bergmeir, C., Petitjean, F. and Webb, G.I., 2021. Time series extrinsic regression: Predicting numeric values from time series data. *Data Mining and Knowledge Discovery*, 35(3), pp.1032–1060. (Archive technical report: [arXiv:2006.10996](https://arxiv.org/abs/2006.10996).)

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{tan2021time,
    title={Time series extrinsic regression: Predicting numeric values from time series data},
    author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Fran{\c{c}}ois and Webb, Geoffrey I},
    journal={Data Mining and Knowledge Discovery},
    volume={35},
    number={3},
    pages={1032--1060},
    year={2021},
    publisher={Springer}
  }
  ```

  </details>

- [Extended TSER Archive](https://doi.org/10.1007/s10618-024-01027-w) — Guijo-Rubio, D., Middlehurst, M., Arcencio, G., Silva, D.F. and Bagnall, A., 2024. Unsupervised feature based algorithms for time series extrinsic regression. *Data Mining and Knowledge Discovery*, 38(4), pp.2141–2185.

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{guijo2024unsupervised,
    title={Unsupervised feature based algorithms for time series extrinsic regression},
    author={Guijo-Rubio, David and Middlehurst, Matthew and Arcencio, Guilherme and Silva, Diego Furtado and Bagnall, Anthony},
    journal={Data Mining and Knowledge Discovery},
    volume={38},
    number={4},
    pages={2141--2185},
    year={2024},
    publisher={Springer}
  }
  ```

  </details>

- [The Multiverse of Time Series Machine Learning](https://arxiv.org/abs/2603.20352) — Middlehurst, M., Rushbrooke, A., Ismail-Fawaz, A., Devanne, M., Forestier, G., Dempster, A., Webb, G.I., Holder, C. and Bagnall, A., 2026. The multiverse of time series machine learning: an archive for multivariate time series classification. *arXiv preprint arXiv:2603.20352*.

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{middlehurst2026multiverse,
    title={The multiverse of time series machine learning: an archive for multivariate time series classification},
    author={Middlehurst, Matthew and Rushbrooke, Aiden and Ismail-Fawaz, Ali and Devanne, Maxime and Forestier, Germain and Dempster, Angus and Webb, Geoffrey I and Holder, Christopher and Bagnall, Anthony},
    journal={arXiv preprint arXiv:2603.20352},
    year={2026}
  }
  ```

  </details>

- [MONSTER: Monash Scalable Time Series Evaluation Repository](https://arxiv.org/abs/2502.15122) — Dempster, A., Foumani, N.M., Tan, C.W., Miller, L., Mishra, A., Salehi, M., Pelletier, C., Schmidt, D.F. and Webb, G.I., 2025. MONSTER: Monash scalable time series evaluation repository. *arXiv preprint arXiv:2502.15122*.

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @article{dempster2025monster,
    title={MONSTER: Monash scalable time series evaluation repository},
    author={Dempster, Angus and Foumani, Navid Mohammadi and Tan, Chang Wei and Miller, Lynn and Mishra, Amish and Salehi, Mahsa and Pelletier, Charlotte and Schmidt, Daniel F and Webb, Geoffrey I},
    journal={arXiv preprint arXiv:2502.15122},
    year={2025}
  }
  ```

  </details>

- [Rehab-Pile](https://arxiv.org/abs/2507.21018) — Ismail-Fawaz, A., Devanne, M., Berretti, S., Weber, J. and Forestier, G., 2026. A standardized benchmark for skeleton-based rehabilitation assessment using deep learning. *IEEE International Conference on Automatic Face and Gesture Recognition (FG)*, pp.1–9. (Preprint: [arXiv:2507.21018](https://arxiv.org/abs/2507.21018).)

  <details class="tsml-bibtex">
  <summary>Show BibTeX</summary>

  ```bibtex
  @inproceedings{ismail2026rehabpile,
    title={A standardized benchmark for skeleton-based rehabilitation assessment using deep learning},
    author={Ismail-Fawaz, Ali and Devanne, Maxime and Berretti, Stefano and Weber, Jonathan and Forestier, Germain},
    booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
    pages={1--9},
    year={2026}
  }
  ```

  </details>
