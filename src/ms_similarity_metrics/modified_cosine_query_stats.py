import pandas as pd
import re
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import itertools

from ms_entropy.file_io.msp_file import read_one_spectrum
import numba as nb
import pyteomics.mgf
import sys
import importlib

sys.path.append("../../src/ms_similarity_metrics/")
from create_spectrum import smile2inchi
importlib.reload(sys.modules['create_spectrum'])
from create_spectrum import smile2inchi

QUERY_DIR = '../../data/modified_cosine_queries/'
QUERY_PATH = 's3://enveda-data-user/chloe.engler/cosine_similarity/modified_cosine_queries/'
DATA_DIR = '../../data/'

# Get NIST23 library
# Get data from s3://enveda-data-user/chloe.engler/cosine_similarity/NIST_data/NIST23-HR-MSMS.MSP
spectra_list = []
for i,spectrum in tqdm(enumerate(read_one_spectrum(f'{DATA_DIR}NIST23-HR-MSMS.MSP'))):
    spectra_list.append(spectrum)

# Create NIST dataframe
nist_df = pd.DataFrame(spectra_list)
nist_df = nist_df[nist_df['precursor_type'] == '[M+H]+' ]

# Profile spectra contain 0 intensity values.
@nb.njit
def is_centroid(intensity_array):
    return np.all(intensity_array > 0)

# Read all spectra from the MGF.
spectra = []

# Download from https://zenodo.org/record/6829249/files/ALL_GNPS_NO_PROPOGATED.mgf?download=1
filename = (f"{DATA_DIR}ALL_GNPS_NO_PROPOGATED.mgf")

# Get wout spectra
with pyteomics.mgf.MGF(filename) as f_in:
    for spectrum_dict in tqdm(f_in):
        spectra.append(spectrum_dict)

# Create wout dataframe
wout_df = pd.DataFrame(spectra)
wout_df = pd.concat([wout_df.drop(['params'], axis=1), wout_df['params'].apply(pd.Series)], axis=1)
wout_df = wout_df.set_index('spectrumid')

# Get wout metadata
metadata = pd.read_csv(
    'https://zenodo.org/record/6829249/files/gnps_libraries_metadata.csv?download=1'
)
metadata.set_index('id', inplace=True)

# Get nist smiles dict
nist_smiles_dict = {}
for index in tqdm(nist_df.index.values):
    nist_smiles_dict[index] = nist_df.loc[index,'smiles']

# Get wout smiles dict
wout_smiles_dict = {}
for current_id in tqdm(wout_df.index.values):
    wout_smiles_dict[current_id] = wout_df.loc[current_id,'smiles']

file_names = [
    '0.5_filtered_10_ppm'
    ]

for name in tqdm(file_names):

    # Get the NIST23 queries
    # queries = pd.read_csv(f'{QUERY_PATH}{FILE}.csv', index_col=0)
    queries = pd.read_csv(f'{QUERY_DIR}{name}.csv', index_col=0)

    # Reformat the library_spectra_matches column to a list of tuples
    all_matches = {}
    for query in tqdm(queries.index.values):
        matches = []
        non_decimal = re.compile(r'[^\d.]+')

        test = queries.loc[query]['library_spectra_matches'].replace("'", "").replace(')', '').split('(')[1:]
        for pair in test:
            matches.append((pair.split(',')[0], float(non_decimal.sub('', pair.split(',')[1]))))

        all_matches[query] = matches
    queries['library_spectra_matches'] = all_matches

    # Remove any query spectra that came from the NIST library
    queries['wout_library'] = list(metadata.loc[queries['wout_identifier'],'library'])
    queries = queries[queries['wout_library'] != 'GNPS-NIST14-MATCHES']

    # Get wout smiles for unweighted queries
    for i in tqdm(queries.index.values):
        wout_id = queries.loc[i, 'wout_identifier']
        queries.loc[i, 'wout_smiles'] = wout_smiles_dict[wout_id]

    # Get NIST23 partial inchikeys
    inchi_dict = {}
    for i in tqdm(queries.index.values):
        inchi_list = []
        for pair in queries.loc[i, 'library_spectra_matches']:
            index = int(pair[0].split('_')[0])
            inchi_list.append(nist_df.loc[int(pair[0].split('_')[0]),'inchikey'][:14])
        inchi_dict[i] = inchi_list
    queries['nist_inchis'] = queries.index.map(inchi_dict)

    # Get wout partial inchikeys for queries
    for i in tqdm(queries.index.values):
        inchi = smile2inchi(queries.loc[i, 'wout_smiles'])
        queries.loc[i, 'wout_inchi'] = inchi[:14]

    # Get indexes of exact matches for queries
    all_matches = {}
    for i in tqdm(queries.index.values):
        exact_matches = np.where(np.array(list(queries.loc[i, 'nist_inchis'])) == queries.loc[i, 'wout_inchi'])[0]
        all_matches[i] = exact_matches
    queries['exact_matches'] = queries.index.map(all_matches)

    from rdkit import Chem, DataStructs
    import functools

    @functools.lru_cache
    def _smiles_to_mol(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None
    @functools.lru_cache
    def tanimoto(smiles1, smiles2):
        mol1, mol2 = _smiles_to_mol(smiles1), _smiles_to_mol(smiles2)
        if mol1 is None or mol2 is None:
            return np.nan
        fp1, fp2 = Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    # Get NIST23 smiles for queries
    all_smiles_dict = {}
    for i in tqdm(queries.index.values):
        current_matches = queries.loc[i, 'library_spectra_matches']
        if len(current_matches) == 0:
            all_smiles_dict[i] = []
        else:
            nist_indexes = [int(x.split('_')[0]) for x in np.array(queries.loc[i, 'library_spectra_matches'])[:,0]]
            smiles_list = nist_df.loc[nist_indexes, 'smiles'].values
            all_smiles_dict[i] = smiles_list
    queries[f'smiles'] = queries.index.map(all_smiles_dict)

    # Get dictionary of tanimoto scores
    all_tanimotos = {}

    # Get tanimoto scores for queries
    for i in tqdm(queries.index.values):
        tanimotos = []
        query_smiles = queries.loc[i, 'wout_smiles']
        if len(queries.loc[i, f'smiles']) != 0:
            for library_smiles in queries.loc[i, f'smiles']:
                tanimotos.append(tanimoto(query_smiles, library_smiles))
            all_tanimotos[i] = [x for x in tanimotos if not pd.isna(x)]
        else:
            all_tanimotos[i] = []
    queries[f'tanimoto'] = queries.index.map(all_tanimotos)

    from sklearn.metrics import roc_curve, auc

    no_matches = 0

    # Get AUC scores for queries
    for index in tqdm(queries.index.values):
        if len(np.array(list(queries.loc[index,'library_spectra_matches']))) != 0:
            prob = np.array(list(queries.loc[index,'library_spectra_matches']))[:,1].astype('float')
            y_true = np.zeros(len(prob))
            y_true[queries.loc[index,f'exact_matches']] = 1

            # check if there arent any 1.0s in y_true
            if np.sum(y_true) == 0:
                queries.loc[index, f'auc'] = 0
            # check if all values are 1.0
            elif np.sum(y_true) == len(y_true):
                queries.loc[index, f'auc'] = 1
            else:
                fpr, tpr, thresholds = roc_curve(y_true, prob)
                queries.loc[index, f'auc'] = auc(fpr, tpr)
        else:
            no_matches += 1
            queries.loc[index, f'auc'] = np.nan

    #queries.to_csv(f'{QUERY_PATH}{name}_with_stats.csv')
    queries.to_csv(f'{QUERY_DIR}{name}_with_stats.csv')