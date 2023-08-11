import numba as nb
import numpy as np
import numpy as np
import pandas as pd
import pyteomics.mgf
import seaborn as sns
from matplotlib import pyplot as plt
import spectrum_utils.spectrum as sus
import sys
import importlib
import spectral_entropy

from tqdm import tqdm

sys.path.append("../src/ms_similarity_metrics")
import frequency
import similarity_weighted as similarity
importlib.reload(frequency)
importlib.reload(similarity)

tqdm.pandas()


# Spectra and spectrum pairs to include with the following settings.
charges = 0, 1
min_n_peaks = 6
fragment_mz_tolerance = 0.1
min_mass_diff = 1    # Da
max_mass_diff = 200    # Da

# Profile spectra contain 0 intensity values.
@nb.njit
def is_centroid(intensity_array):
    return np.all(intensity_array > 0)

# Assumes that the spectra are sorted by ascending precusor m/z.
@nb.njit
def generate_pairs(
    spectrum_indexes, masses, min_mass_diff, max_mass_diff
):
    for i in range(len(spectrum_indexes)):
        j = i + 1
        while (
            j < len(spectrum_indexes) and
            masses[j] - masses[i] < min_mass_diff
        ):
            j += 1
        while (
            j < len(spectrum_indexes) and
            masses[j] - masses[i] < max_mass_diff
        ):
            yield spectrum_indexes[i]
            yield spectrum_indexes[j]
            j += 1

spectra = []

# Download from https://zenodo.org/record/6829249/files/ALL_GNPS_NO_PROPOGATED.mgf?download=1
filename = ("../data/ALL_GNPS_NO_PROPOGATED.mgf")

with pyteomics.mgf.MGF(filename) as f_in:
    for spectrum_dict in tqdm(f_in):
        if (
            int(spectrum_dict["params"]["libraryquality"]) <= 3 and
            int(spectrum_dict["params"]["charge"][0]) in charges and
            float(spectrum_dict["params"]["pepmass"][0]) > 0 and
            len(spectrum_dict["m/z array"]) >= min_n_peaks and
            spectrum_dict["params"]["ionmode"] == "Positive" and
            spectrum_dict["params"]["name"].rstrip().endswith(" M+H") and
            is_centroid(spectrum_dict["intensity array"]) and
            (
                spectrum_dict["params"]["inchi"] != "N/A" or
                spectrum_dict["params"]["smiles"] != "N/A"
            )
        ):
            spec = sus.MsmsSpectrum(
                spectrum_dict["params"]["spectrumid"],
                float(spectrum_dict["params"]["pepmass"][0]),
                # Re-assign charge 0 to 1.
                max(int(spectrum_dict["params"]["charge"][0]), 1),
                spectrum_dict["m/z array"],
                spectrum_dict["intensity array"]/max(spectrum_dict["intensity array"]),
            )
            spec.library = spectrum_dict["params"]["organism"]
            spec.inchi = spectrum_dict["params"]["inchi"]
            spec.smiles = spectrum_dict["params"]["smiles"]
            spec.remove_precursor_peak(0.1, "Da")
            spec.filter_intensity(0.01)
            spectra.append(spec)

# Round spectra
for s in spectra:
    s.mz.round(1, out=s.mz)

metadata = pd.read_csv(
    'https://zenodo.org/record/6829249/files/gnps_libraries_metadata.csv?download=1'
)

with open('../data/pairs_subset.txt', 'r') as f:
    pairs_subset = f.read().splitlines()
    pairs_subset = pairs_subset[1:]
    pairs_subset = [np.array(pair.split(' ')).astype(int) for pair in pairs_subset]

pairs_subset = np.array(pairs_subset)

frequency_df = pd.read_csv('../data/frequency_df.csv')
with open('../data/num_spectra.txt', 'r') as f:
    num_spectra = int(f.read())


with open('../data/frequency_count.txt', 'r') as f:
    frequency_count = f.read().splitlines()
    frequency_count = frequency_count[1:]
    frequency_count = [float(x) for x in frequency_count]

#pairs = pairs_subset[np.random.choice(pairs_subset.shape[0], 1_000, replace=False)]
pairs = pairs_subset.copy()

# Save core information about the pairs.
similarities_df = pd.DataFrame(
    {
        "pair1": pairs[:, 0],
        "pair2": pairs[:, 1],
        "id1": metadata.loc[pairs[:, 0], "id"].values,
        "id2": metadata.loc[pairs[:, 1], "id"].values,
    }
)

# Define weights
weight_func = lambda x: x**(1/4)
intensity_weight_func = lambda x: x**(1/4)
weight_df = frequency.get_weights(frequency_df, weight_func, weight_col='prob')
weight_df = weight_df.set_index('mz', drop=True)

metrics = []
weighted_metrics = []

for i, j in tqdm(pairs, total=pairs.shape[0]):

    # Add unweighted results
    metrics.append(
        spectral_entropy.all_similarity(
            np.array(list(zip(list(spectra[i].mz), list(spectra[i].intensity))), dtype=np.float32),
            np.array(list(zip(list(spectra[j].mz), list(spectra[j].intensity))), dtype=np.float32),
            ms2_da=0.1,  # Same as Wout's paper
        )
    )
    
    # Add rewaited results
    weighted_intensity1 = weight_df.loc[spectra[i].mz, 'weight']*intensity_weight_func(spectra[i].intensity)
    weighted_intensity2 = weight_df.loc[spectra[j].mz, 'weight']*intensity_weight_func(spectra[j].intensity)
    current_dict = spectral_entropy.all_similarity(
            np.array(list(zip(list(spectra[i].mz), 
                            list(weighted_intensity1))), dtype=np.float32),
            np.array(list(zip(list(spectra[j].mz), 
                            list(weighted_intensity2))), dtype=np.float32),
            ms2_da=0.1,  # Same as Wout's paper 
        )
    weighted_metrics.append(
        current_dict
    )

final_df = pd.concat(
    [
        similarities_df,
        pd.DataFrame(metrics),
    ],
    axis=1,
)

final_weighted_df = pd.concat(
    [
        similarities_df,
        pd.DataFrame(weighted_metrics)
    ],
    axis=1,
)

final_df.to_parquet("../data/benchmark_metrics_corrected_2.parquet")
final_weighted_df.to_parquet('../data/weighted_benchmark_metrics_corrected_2.parquet')
