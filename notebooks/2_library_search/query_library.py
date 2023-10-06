import pandas as pd
from tqdm import tqdm
from ms_entropy.file_io.msp_file import read_one_spectrum

import numpy as np

from rdkit import RDLogger
import importlib

import sys

sys.path.append('../src/')

importlib.reload(sys.modules['ms_similarity_metrics.hash_utils'])
importlib.reload(sys.modules['ms_similarity_metrics.create_spectrum'])
importlib.reload(sys.modules['ms_similarity_metrics.query_pool'])
importlib.reload(sys.modules['ms_similarity_metrics.frequency'])
from ms_similarity_metrics.hash_utils import hash_spectrum
from ms_similarity_metrics.create_spectrum import create_spectra_NIST23, create_spectra_wout
from ms_similarity_metrics.query_pool import query
from ms_similarity_metrics.frequency import get_weights
from ms_similarity_metrics.create_spectrum import weight_NIST23_spectra, weight_wout_spectra

RDLogger.DisableLog('rdApp.*')


def get_frequency_df(frequency_df_path):
    """
    Get frequency df and extend to other m/z values
    
    Parameters
    ----------
    frequency_df_path : str
        Path to frequency df
    
    Returns
    -------
    new_frequency_df : pd.DataFrame
        Frequency df with extended m/z values
    """

    # Get frequency df
    frequency_df = pd.read_csv(frequency_df_path, index_col=0)
    frequency_df = frequency_df.set_index('mz')
    min_frequency = min(frequency_df.index.values)
    max_frequency = max(frequency_df.index.values)
    new_frequency_df = frequency_df.copy()

    # Extend weights to other m/z values
    for mz in tqdm(np.arange(min_frequency, max_frequency, 0.1)):
        mz = np.round(mz, 1)
        if mz not in frequency_df.index.values:
            lower_bound = frequency_df[frequency_df.index < mz].index.max()
            upper_bound = frequency_df[frequency_df.index > mz].index.min()
            prob_value = (frequency_df.loc[lower_bound, 'prob'] + frequency_df.loc[upper_bound, 'prob']) / 2
            new_frequency_df.loc[mz] = [0, prob_value]
    for mz in tqdm(np.arange(0, min_frequency, 0.1)):
        mz = np.round(mz, 1)
        new_frequency_df.loc[mz] = [0, frequency_df.loc[min_frequency]['prob']]
    for mz in tqdm(np.arange(max_frequency, 5000, 0.1)):
        mz = np.round(mz, 1)
        new_frequency_df.loc[mz] = [0, frequency_df.loc[max_frequency]['prob']]

    # Get indexes for new frequency df
    new_frequency_df = new_frequency_df.sort_index()
    new_frequency_df['mz'] = np.round(new_frequency_df.index.values, 1)
    new_frequency_df = new_frequency_df.set_index('mz', drop=True)

    return new_frequency_df


def get_NIST23_spectra(normalize=True):
    """
    Get NIST23 spectra and convert to right format.

    Parameters:
        normalize: bool, whether to normalize the spectra intensities

    Returns:
        nist_spectra: list of Spectrum objects
        info_dict: dict with info about spectra
        nist_inchi_dict: dict with inchikey to index mapping
    """

    # Get NIST23 library
    spectra_list = []

    # Read spectra from file 
    for i, spectrum in tqdm(enumerate(read_one_spectrum('../../data/NIST23-HR-MSMS.MSP'))):
        spectra_list.append(spectrum)

    #  make a dict from inchikey to smiles
    inchikey_nist23_to_smiles = pd.read_csv(
        '../../data//nist/nist23_resolver_inchikeys_to_smiles.tsv.gz',
        sep='\t',
        compression='gzip',
    )
    inchikey_nist23_to_smiles = inchikey_nist23_to_smiles.dropna(subset=['smiles'])
    inchikey_to_smiles = dict(zip(inchikey_nist23_to_smiles['inchikey'], inchikey_nist23_to_smiles['smiles']))

    # Convert spectra to right format and filter
    # See create_spectrum.py for more filtering information
    nist_spectra, info_dict, nist_inchi_dict = create_spectra_NIST23(spectra_list,
                                                                     inchikey_to_smiles,
                                                                     min_n_peaks=6,
                                                                     normalize=normalize)

    return nist_spectra, info_dict, nist_inchi_dict


def get_matching_inchis(nist_spectra, wout_spectra):
    """
    Get the inchikeys that are present in both NIST23 and wout.

    Parameters:
        nist_spectra: list of Spectrum objects
        wout_spectra: list of Spectrum objects

    Returns:
        matching_inchis: list of inchikeys
    """

    # Get set with all inchis from nist spectra
    nist_inchis = {
        spectra.partial_inchikey
        for spectra in tqdm(nist_spectra)
    }

    # Check for matching spectra in NIST23 and wout
    matching_inchis = []
    for i, spectra in enumerate(tqdm(wout_spectra)):
        current_inchi = spectra.partial_inchikey
        if current_inchi != None and current_inchi in nist_inchis:
            matching_inchis.append(current_inchi)

    return matching_inchis


def query_all_spectra(save_dir,
                      mz_weight_func=None,
                      intensity_weight_func=None,
                      weights=False,
                      ppm_window=5,
                      threshold=0.5,
                      metric_type='modified_cosine',
                      Stein_weights=False,
                      normalize=True
                      ):
    """
    This function queries all spectra in the wout library with inchikey 
    overlap against the NIST23 library.

    Parameters:
        mz_weight_func: function
            Function that takes the frequency of an m/z value and returns a weight
        intensity_weight_func: function
            Function that takes a intensity value and returns a weight
        save_dir: str
            Directory to save results
        weights: bool
            Whether to use weights or not
        ppm_window: int
            ppm window to use for NIST23 library search
        threshold: float
            Threshold to use for NIST23 library search
        metric_type: str or list
            Metric to use for NIST23 library search. Should be a list if you are using metrics from the
            SpectralEntropy paper (anything except 'modified_cosine').
        Stein_weights: bool
            Whether to use Stein weights or not
        normalize: bool
            Whether to normalize the spectra intensities
    """

    # Check if Stein weights and normalize are both True
    if Stein_weights and normalize:
        raise ValueError('Stein weights doesn\'t use normalized intensities')

    # Check for mz weights
    if weights and mz_weight_func == None:
        raise ValueError('You need to specify a mz_weight_func')

    # Check for intensity weights
    if weights and intensity_weight_func == None:
        raise ValueError('You need to specify a intensity_weight_func')

    # Get NIST23 spectra
    nist_spectra, info_dict, nist_inchi_dict = get_NIST23_spectra(normalize=True)

    # Covert spectra to right format and filter
    wout_spectra, wout_info_dict = create_spectra_wout(
        '../../data/ALL_GNPS_NO_PROPOGATED.mgf',
        min_n_peaks=6,
        normalize=True
    )

    # Get wout metadata
    metadata = pd.read_csv(
        'https://zenodo.org/record/6829249/files/gnps_libraries_metadata.csv?download=1'
    )
    metadata.set_index('id', inplace=True)

    # dict of all hashes for all spectra in NIST23
    all_hashes = {
        s.identifier: hash_spectrum(s.mz, s.intensity, precision=2, iterative=True, sort=True)
        for s in tqdm(nist_spectra)
    }

    # Get weights if needed
    if weights:
        frequency_df = get_frequency_df(
            '../../data/Wout_data/frequency_df.csv')
        weight_df = get_weights(frequency_df, mz_weight_func, weight_col='prob')

        # Weight intensities (using m/z frequency values)
        if not Stein_weights:
            # Weight intensities
            weighted_nist_spectra = weight_NIST23_spectra(nist_spectra, intensity_weight_func, weight_df)
            weighted_wout_spectra = weight_wout_spectra(wout_spectra, intensity_weight_func, weight_df)

        # Weight intensities using Steins weights (using m/z values)
        if Stein_weights:
            Stein_weight_df = pd.DataFrame(index=weight_df.index, columns=['weight'])
            Stein_weight_df['weight'] = mz_weight_func(weight_df.index.values)

            # Weight intensities
            weighted_nist_spectra = weight_NIST23_spectra(nist_spectra, intensity_weight_func, Stein_weight_df)
            weighted_wout_spectra = weight_wout_spectra(wout_spectra, intensity_weight_func, Stein_weight_df)
    else:
        weighted_nist_spectra = nist_spectra
        weighted_wout_spectra = wout_spectra

    # Filter query spectra to only those with matching inchikey
    matching_inchis = get_matching_inchis(nist_spectra, wout_spectra)
    filtered_query_spectra = [i for i in range(len(weighted_wout_spectra)) if
                              weighted_wout_spectra[i].partial_inchikey in matching_inchis]

    # Filter query spectra to only those that are not in the NIST library
    filtered_query_spectra = [i for i in filtered_query_spectra if
                              metadata.loc[weighted_wout_spectra[i].identifier, 'library'] != 'GNPS-NIST14-MATCHES']

    # Filter query spectra to only those that have a ppm value between -100 and 100
    wout_ppm_corrected = pd.read_csv(
        '../../data/Wout_data/wout_ppm_corrected.csv', index_col=0).set_index(
        'spectrumid')
    filtered_query_spectra = [i for i in filtered_query_spectra if weighted_wout_spectra[i].identifier
                              in wout_ppm_corrected.index.values]
    filtered_query_spectra = [i for i in filtered_query_spectra if
                              abs(wout_ppm_corrected.loc[weighted_wout_spectra[i].identifier, 'ppm']) < 100]

    # Create dataframe to store results
    query_df = pd.DataFrame(columns=['wout_identifier', 'library_spectra_matches',
                                     'identical_pairs', 'num_inchi_matches', 'num_matches_in_query'])

    # Query NIST23
    for i in tqdm(filtered_query_spectra):
        query_spectra = weighted_wout_spectra[i]
        (best_matches, identical_matches), num_matches, num_matches_in_query = query(
            query_spectra,
            weighted_nist_spectra,
            all_hashes,
            nist_inchi_dict=nist_inchi_dict,
            metric_type=metric_type,
            threshold=threshold,
            top_n=None,
            ppm_window=ppm_window
        )

        # Add results to dataframe
        query_df.loc[i] = {'wout_identifier': query_spectra.identifier,
                           'library_spectra_matches': best_matches,
                           'identical_pairs': identical_matches,
                           'num_inchi_matches': num_matches,
                           'num_matches_in_query': num_matches_in_query}

        # Save results
        if i % 100 == 99:
            query_df.to_csv(save_dir)


if __name__ == "__main__":
    # Set parameters
    save_dir = '../../data/mod_cosine_queries/filtered_5000_ppm.csv'
    weights = False
    ppm_window = 5000
    threshold = 0.5
    metric_type = 'modified_cosine'
    normalize = True
    mz_weight_func = None
    intensity_weight_func = None

    # # Set Stein weight parameters
    # mz_weight_func = lambda x: x**(1)
    # intensity_weight_func = lambda x: x**(0.5)
    # save_dir = '../data/mod_cosine_queries/Stein_weights_0.5_1.csv'
    # weights=True
    # ppm_window=10
    # threshold=0.5
    # metric_type='modified_cosine'
    # normalize=False
    # Stein_weights=True

    # # Set weighted parameters
    # mz_weight_func = lambda x: x**(1/4)
    # intensity_weight_func = lambda x: x**(1/4)
    # save_dir = '../data/mod_cosine_queries/weighted_filtered_5_ppm.csv'
    # weights=True
    # ppm_window=5
    # threshold=0.5
    # metric_type='modified_cosine'
    # normalize=True

    # Run query
    query_all_spectra(
        save_dir=save_dir,
        mz_weight_func=mz_weight_func,
        intensity_weight_func=intensity_weight_func,
        weights=weights,
        ppm_window=ppm_window,
        threshold=threshold,
        metric_type=metric_type,
        normalize=normalize
    )
