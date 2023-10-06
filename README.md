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

Links to the specific files are given in the notebooks and data input should be located at ./data

## Reproducibility
To reproduce the results, run the notebooks in the given order.
1. The Python virtual environment can be installed using [Poetry](https://python-poetry.org/).
2. Install the spectral_entropy package from https://github.com/YuanyueLi/SpectralEntropy (this package is not available in PyPi).

## Code
The project has the following structure:

```
`-- notebooks
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
        |-- 3.1_modified_cosine_no_ppm_window.ipynb
        |-- 3.2_spectral_entropy_no_ppm_window.ipynb
        |-- 4.1_modified_cosine_queries_stats.ipynb
        |-- 4.2_spectral_entropy_query_stats.ipynb
        |-- 5.1_modified_cosine_query_graphs.ipynb
        |-- 5.2_spectral_entropy_query_graphs.ipynb
        |-- 6_compare_weights.ipynb
        |-- 7_compare_ppm_windows.ipynb
        |-- 8_nist_vs_nist.ipynb
        |-- 9_siamese_query.ipynb
        |-- query_library.py
    |-- helper_notebooks
        |-- Get_siamese_vectors.ipynb
        |-- MS2DeepScore.ipynb
        |-- check_for_cations.ipynb
        |-- create_figures_for weights.ipynb
        |-- create_siamese_network_pq_files.ipynb
        |-- matching_peaks_plot.ipynb
`-- src
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
