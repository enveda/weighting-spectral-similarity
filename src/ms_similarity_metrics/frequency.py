import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm


def get_frequency_df(spectra, pairs):
    """
    Compute the frequency dataframe of the given spectra.
    
    Parameters:
    -----------
    spectra : list
        The spectra.
    num_spectra : int
        The number of spectra.
        
    Returns:
    --------
    frequency_df : pd.DataFrame
        The frequency dataframe.
    """

    # Set initial variables
    frequency_count = []
    unique_spectrum_indexes = np.unique(np.concatenate((np.unique(pairs[:, 0]), np.unique(pairs[:, 1]))))

    # Count the frequency of mz values in the pairs
    for i in tqdm(unique_spectrum_indexes):
        for mz in spectra[i].round(1).mz:
            frequency_count.append(mz)

    # Create a dataframe with the frequency count
    # Sort by mz 
    frequency_df = pd.DataFrame.from_dict(
        Counter(frequency_count),
        orient='index',
        columns=['frequency'],
    ).sort_index()

    # make index a column called m/z
    frequency_df.reset_index(inplace=True)
    frequency_df.rename(columns={'index': 'mz'}, inplace=True)
    num_spectra = len(unique_spectrum_indexes)

    return frequency_df, num_spectra, frequency_count


# def tf(mz_values):
#     """
#     Compute the term frequency of the given m/z values.

#     Parameters:
#     -----------
#     mz_values : np.ndarray
#         The m/z values.

#     Returns:
#     --------
#     term_frequency : dict
#         The term frequency.
#     """

#     term_frequency = {mz:1/len(mz_values) for mz in mz_values}

#     return term_frequency

def idf(frequency_df, num_spectra, frequency_col='frequency'):
    """
    Compute the inverse document frequency of the given frequency dataframe.
    
    Parameters:
    -----------
    frequency_df : pd.DataFrame
        The frequency dataframe.
    num_spectra : int
        The number of spectra.
    
    Returns:
    --------
    inverse_document_frequency : dict
        The inverse document frequency.
    """

    # Get idf
    frequency_df = frequency_df.set_index('mz')
    inverse_document_frequency = {mz: np.log(num_spectra / frequency_df.loc[mz, frequency_col]) for mz in
                                  frequency_df.index}

    return inverse_document_frequency


def get_weights(frequency_df, weight_func, weight_col='prob', idf=False):
    """
    Compute the weights of the given frequency dataframe.
    
    Parameters:
    -----------
    idf_df : pd.DataFrame
        The frequency dataframe with idf scores.
    weight_func : function
        The weight function.

    Returns:
    --------
    weights : dataframe
        The weights.
    """

    # Get weights
    weights = frequency_df.copy()
    if idf:
        weights['weight'] = weights['mz'].map(weight_func)
    else:
        weights['weight'] = weights[weight_col].map(weight_func)

    return weights
