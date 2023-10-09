# On optimizing mass spectral library search and spectral alignment by weighting low-intensity peaks and m/z frequency

## Table of Contents

- General info
- Datasets
- Reproducibility
- Code structure

## General Info
This repository contains code for evaluating weighting methods for spectral similarity metrics. 

## Datasets
- [GNPS library](https://gnps.ucsd.edu/)
- [NIST library](https://chemdata.nist.gov/)

Links to the specific files are given in the notebooks
The dump of the processed files are available at https://doi.org/10.5281/zenodo.8417612 and the zip files should be unzipped at
placed ./data with the following structure:

```
-- data
    |-- modified_cosine_queries
    |-- network_method
    |-- nist
    |-- Wout_data
```

## Reproducibility [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8417612.svg)](https://doi.org/10.5281/zenodo.8417612)
To reproduce the results, run the notebooks in the given order.
1. The Python virtual environment can be installed using [Poetry](https://python-poetry.org/).
2. Install the spectral_entropy package from https://github.com/YuanyueLi/SpectralEntropy (this package is not available in PyPi).

## Code
The project has the following structure:

```
-- notebooks
    |-- 1_spectral_aligment
        |-- 1_benchmark_other_similarity_metrics.ipynb
        |-- 1.1_benchmark_stats.ipynb
        |-- 2_weighted_modified_cosine.ipynb
        |-- 3_create_figures.ipynb
        |-- 4_compare_scores.ipynb
        |-- 5_some_weighted.ipynb
        |-- benchmark.py
    |-- 2_library_search
        |-- 1_gnps_ppm.ipynb
        |-- 2_query_library_example.ipynb
        |-- 3_results_benchmark.ipynb
        |-- 4_results_comparing_ppm_windows.ipynb
        |-- query_library.py

-- src
    |-- ms_similarity_metrics
        ├── __init__.py
        ├── create_spectrum.py
        ├── frequency.py
        ├── hash_utils.py
        ├── plot.py
        ├── query_pool.py
        ├── reformat_columns.py
        ├── siamese_query.py
        ├── similarity_utils.py
        ├── similarity_weighted.py
        └── version.py

```
