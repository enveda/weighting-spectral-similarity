import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('./')
from .hash_utils import hash_spectrum

logger = logging.getLogger(__name__)


def query(
        query_spectra,
        library_spectra,
        library_hash,
        siamese_query_df,
        siamese_library_df,
        nist_inchi_dict,
        threshold=0.7,
        top_n=None,
        ppm_window=5000
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
        A dictionary of hashes for each spectrum in the library.
    siamese_query_df : pd.DataFrame
        A dataframe of siamese vectors for query spectra.
    siamese_library_df : pd.DataFrame
        A dataframe of siamese vectors for library spectra.
    similarity_func : function
        A function that takes two spectra and returns a similarity score.
    threshold : float, optional
        The minimum similarity score to consider a match. The default is 0.7.
    top_n : int, optional
        The maximum number of matches to return. The default is None.
    ppm_window : float, optional
        The maximum precursor m/z difference to consider a match. The default is 5000.

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
    ppm = abs(1e6 * ((library_precursor - query_spectra.precursor_mz) / query_spectra.precursor_mz))
    indexes_to_keep = np.nonzero(np.abs(ppm) <= ppm_window)[0]

    # Get exact matches
    exact_matches = nist_inchi_dict[query_spectra.partial_inchikey]
    exact_matches = list(set(exact_matches).intersection(set(indexes_to_keep)))
    num_matches = len(exact_matches)

    # Filter library spectra by precursor m/z
    if num_matches != 0:
        library_spectra = np.array(library_spectra)[indexes_to_keep]

    # If there are no exact matches then return empty list
    else:
        library_spectra = []

    # Get identifiers
    library_identifiers = [s.identifer for s in library_spectra]
    query_identifier = query_spectra.identifier

    # Get hash for query spectra
    query_hash = hash_spectrum(query_spectra.mz, query_spectra.intensity, precision=2, iterative=True,
                               sort=True)

    return query_siamese(query_identifier, library_identifiers, library_hash, query_hash,
                         siamese_query_df, siamese_library_df, threshold=threshold, top_n=top_n), num_matches


def query_siamese(query_identifier,
                  library_identifiers,
                  library_hash,
                  query_hash,
                  siamese_query_df,
                  siamese_library_df,
                  threshold=0.5,
                  top_n=None):
    """
    Get query results using siamese network

    Parameters:
    -----------
    query_identifer (str): identifer for query spectra
    library_identifiers (list): identifier for library spectra
    library_hash (dict): dictionary of hashes for library spectra
    query_hash (str): hash for query spectra
    siamese_query_df (pd.DataFrame): dataframe of siamese vectors for query spectra
    siamese_library_df (pd.DataFrame): dataframe of siamese vectors for library spectra
    threshold : float, optional
        The minimum similarity score to consider a match. The default is 0.7.
    top_n : int, optional
        number of results to return. The default is all results.

    Returns:
    --------
    best_matches (list): list of tuples of best matches and scores
    identical_spectra (list): list of identical spectra

    """

    # Get similarity scores for each metric
    query_vector = siamese_query_df.loc[query_identifier].siamese_vector
    library_vectors = np.array(list(siamese_library_df.loc[library_identifiers].siamese_vector))

    scores = cosine_similarity(query_vector.reshape(1, -1), library_vectors)[0]

    # Get matches above threshold
    best_matches = [(spectra, score) for spectra, score in zip(library_identifiers, scores)
                    if score > threshold and query_hash != library_hash[spectra]]
    best_matches.sort(key=lambda x: x[1], reverse=True)

    # Return top_n matches
    if top_n is not None:
        if len(best_matches) > top_n:
            best_matches = best_matches[:top_n]
        else:
            logger.warning(
                f"Less matches  ({len(best_matches)}) were found for modified cosine than the given topN, returning all matches.")

    # Get identical spectra using hash
    identical_spectra = [spectra for spectra in library_identifiers if query_hash == library_hash[spectra]]

    return best_matches, identical_spectra
