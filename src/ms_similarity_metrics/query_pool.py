import importlib
import logging
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from random import sample

sys.path.append('./')
from .hash_utils import hash_spectrum
from .similarity_weighted import weighted_modified_cosine
from spectral_entropy.spectral_similarity import multiple_similarity

logger = logging.getLogger(__name__)


def query(
    query_spectra,
    library_spectra,
    library_hash,
    metric_type,
    nist_inchi_dict,
    threshold=0.7,
    top_n=None,
    fragment_mz_tolerance=0.1,
    ppm_window=5
):
    """
    Query a library with a set of spectra.

    Parameters
    ----------
    query_spectra : Msms.Spectrum
        A spectrum to query the library with.
    library_spectra : list of Msms.Spectrum
        A set of spectra in the library.
    similarity_func : function
        A function that takes two spectra and returns a similarity score.
    threshold : float, optional
        The minimum similarity score to consider a match. The default is 0.7.
    top_n : int, optional
        The maximum number of matches to return. The default is None.
    fragment_mz_tolerance : float, optional
        The m/z tolerance to use when matching peaks. The default is 0.1.
    mz_weights : pandas DataFrame, optional
        A dataframe of m/z weights for each m/z value. The default is None.
    intensity_weights : function, optional
        A function that takes an intensity value and returns a weight. The default is None.
    library_percentage: float
        A float between 0 and 1 that gives the percentage of the spectral library to query.

    Returns
    -------
    np.ndarray
        The id's of the library spectra that match the query spectra and the corresponding similarity scores.
    """

    # Set initial variables
    num_spectra = len(library_spectra)
    indexes_to_keep = []
    num_matches = 0

    # Filter library spectra by precursor m/z
    library_precursor = np.array([s.precursor_mz for s in library_spectra])
    ppm = 1e6 * ((library_precursor - query_spectra.precursor_mz) / query_spectra.precursor_mz)
    indexes_to_keep = np.nonzero(np.abs(ppm) <= ppm_window)[0]

    # Get exact matches
    exact_matches = nist_inchi_dict[query_spectra.partial_inchikey]
    num_matches = len(exact_matches)
    exact_matches = list(set(exact_matches).intersection(set(indexes_to_keep)))
    num_matches_in_query = len(exact_matches)

    # Filter library spectra by precursor m/z
    library_spectra = np.array(library_spectra)[indexes_to_keep]

    # Get best matches for query spectra
    if isinstance(metric_type, str) and 'modified_cosine' in metric_type:
        return query_modified_cosine(query_spectra, library_spectra, library_hash,
                                     threshold=threshold, top_n=top_n,
                                     fragment_mz_tolerance=fragment_mz_tolerance), num_matches, num_matches_in_query

    return query_spectralEntropy(query_spectra, library_spectra, library_hash,
                                 metric_name=metric_type, threshold=threshold, top_n=top_n,
                                 fragment_mz_tolerance=fragment_mz_tolerance), num_matches, num_matches_in_query


def call_spectraEntropy(
    query_spectra,
    metric_name,
    fragment_mz_tolerance,
    #   mz_weights=None,
    #   intensity_weights=lambda x: x,
    library_spectra=None):
    """
    This function calls the spectral entropy package to calculate the similarity score between
    a query spectrum and a library spectrum.

    Parameters
    ----------
    library_spectra : Msms.Spectrum
        A spectrum in the library.
    query_spectra : Msms.Spectrum
        A spectrum to query the library with.
    metric_name : list of strings
        A list of similarity metrics to calculate.
    ms2da : float
        The m/z tolerance to use when matching peaks.
    mz_weights : pandas DataFrame, optional
        A dataframe of m/z weights for each m/z value. The default is None.
    intensity_weights : function, optional
        A function that takes an intensity value and returns a weight.

    Returns
    -------
    score : dict
        A dictionary of similarity scores for each metric.
    """

    # Check if library spectra is provided
    if library_spectra is None:
        raise ValueError("library_spectra must be provided")

    # # Get similarity scores
    # if mz_weights is not None:
    #     score = multiple_similarity(
    #                 np.array(list(zip(list(query_spectra.mz*np.sqrt(mz_weights.loc[query_spectra.mz, 'weight'])), 
    #                                 list(intensity_weights(query_spectra.intensity)))),
    #                                 dtype=np.float32),
    #                 np.array(list(zip(list(library_spectra.mz*np.sqrt(mz_weights.loc[library_spectra.mz, 'weight'])), 
    #                                 list(intensity_weights(library_spectra.intensity)))), 
    #                                 dtype=np.float32),
    #                 methods=metric_name,
    #                 ms2_da=fragment_mz_tolerance)
    # elif mz_weights is None:
    score = multiple_similarity(
        np.array(list(zip(list(query_spectra.mz),
                          list(query_spectra.intensity))),
                 dtype=np.float32),
        np.array(list(zip(list(library_spectra.mz),
                          list(library_spectra.intensity))),
                 dtype=np.float32),
        methods=metric_name,
        ms2_da=fragment_mz_tolerance)

    return score


def query_spectralEntropy(
    query_spectra,
    library_spectra,
    library_hash,
    metric_name,
    threshold=0.7,
    top_n=None,
    fragment_mz_tolerance=0.1,
    mz_weights=None,
    intensity_weights=lambda x: x,
    n_workers=None,
):
    """
    Query a library with a set of spectra.

    Parameters
    ----------
    query_spectra : Msms.Spectrum
        A spectrum to query the library with.
    library_spectra : list of Msms.Spectrum
        A set of spectra in the library.
    library_hash : dict
        A dictionary of hashes for the library spectra.
    metric_name : string
        A name of a similarity metric from the spectralEntropy repo.
        (https://github.com/YuanyueLi/SpectralEntropy)
    threshold : float, optional
        The minimum similarity score to consider a match. The default is 0.7.
    top_n : int, optional

    Returns
    -------
    best_matches: list of tuples
        The library spectra that match the query spectra and the corresponding similarity scores.
    identical_spectra: list of strings
        The library spectra that are identical to the query spectra (these are not included in 
        best_matches)
    """

    # Get hash for query spectra
    query_hash = hash_spectrum(query_spectra.mz, query_spectra.intensity, precision=2, iterative=True,
                               sort=True)

    # If only one single metric name is passed, make a list
    if isinstance(metric_name, str):
        metric_name = [metric_name]

    # Set initial variables
    best_matches = defaultdict(list)
    # n_workers = min(n_workers, cpu_count()) if n_workers else cpu_count()

    # Get similarity scores for each metric
    identical_spectra = []
    scores = []
    for spectra in library_spectra:
        if query_hash == library_hash[spectra.identifier]:
            identical_spectra.append(spectra.identifier)
        else:
            score = call_spectraEntropy(query_spectra, metric_name, fragment_mz_tolerance, spectra)
            scores.append(score)

    # Get matches above threshold
    for metric in metric_name:
        best_matches[metric] = [(spectra.identifier, score[metric]) for spectra, score in zip(library_spectra, scores)
                                if score[metric] > threshold and query_hash != library_hash[spectra.identifier]]
        best_matches[metric].sort(key=lambda x: x[1], reverse=True)

        # Get top_n matches
        if top_n is not None:
            if len(best_matches[metric]) > top_n:
                best_matches[metric] = best_matches[metric][:top_n]
            else:
                logger.warning(
                    f"Less matches  ({len(best_matches[metric])}) were found for metric {metric} than the given topN, returning all matches.")

    # Get identical spectra using hash
    # identical_spectra = [spectra.identifier for spectra in library_spectra if query_hash == library_hash[spectra.identifier]]

    return best_matches, identical_spectra


def call_modified_cosine(
    query_spectra,
    fragment_mz_tolerance,
    library_spectra
):
    """
    This function calls the modified_cosine package to calculate the similarity score between
    a query spectrum and a library spectrum.

    Parameters
    ----------
    library_spectra : Msms.Spectrum
        A spectrum in the library.
    query_spectra : Msms.Spectrum
        A spectrum to query the library with.
    ms2da : float
        The m/z tolerance to use when matching peaks.

    Returns
    -------
    score : float
        The similarity score.
    """

    # Get modified cosine score
    score = weighted_modified_cosine(
        query_spectra,
        library_spectra,
        fragment_mz_tolerance=fragment_mz_tolerance,
    ).score

    return score


def query_modified_cosine(
    query_spectra,
    library_spectra,
    library_hash,
    threshold=0.7,
    top_n=None,
    fragment_mz_tolerance=0.1,
    mz_weights=None,
    intensity_weights=lambda x: x,
    n_workers=None,
):
    """
    Query a library with a set of spectra.

    Parameters
    ----------
    query_spectra : Msms.Spectrum
        A spectrum to query the library with.
    library_spectra : list of Msms.Spectrum
        A set of spectra in the library.
    library_hash : dict
        A dictionary of hashes for the library spectra.
    metric_func : function
        A function that takes two spectra and returns a similarity score.
    threshold : float, optional
        The minimum similarity score to consider a match. The default is 0.7.
    top_n : int, optional

    Returns
    -------
    best_matches: list of tuples
        The library spectra that match the query spectra and the corresponding similarity scores.
    identical_spectra: list of strings
        The library spectra that are identical to the query spectra (these are not included in 
        best_matches)
    """

    # Set initial variables
    # n_workers = min(n_workers, cpu_count()) if n_workers else cpu_count()

    # Get hash for query spectra
    query_hash = hash_spectrum(query_spectra.mz, query_spectra.intensity, precision=2, iterative=True,
                               sort=True)

    # Get similarity scores for each metric
    identical_spectra = []
    best_matches = []
    for spectra in library_spectra:
        if query_hash == library_hash[spectra.identifier]:
            identical_spectra.append(spectra.identifier)
        else:
            score = call_modified_cosine(query_spectra, fragment_mz_tolerance, spectra)
            if score > threshold:
                best_matches.append((spectra.identifier, score))

    # Get matches above threshold
    # best_matches = [(spectra.identifier, score) for spectra, score in zip(library_spectra, scores) 
    #                             if score > threshold and query_hash != library_hash[spectra.identifier]]
    best_matches.sort(key=lambda x: x[1], reverse=True)

    # Return top_n matches
    if top_n is not None:
        if len(best_matches) > top_n:
            best_matches = best_matches[:top_n]
        else:
            logger.warning(
                f"Less matches  ({len(best_matches)}) were found for modified cosine than the given topN, returning all matches.")

    # Get identical spectra using hash
    # identical_spectra = [spectra.identifier for spectra in library_spectra if query_hash == library_hash[spectra.identifier]]

    return best_matches, identical_spectra
