import numpy as np
import pyteomics.mgf
import spectrum_utils.spectrum as sus
from rdkit import Chem
from tqdm import tqdm


def is_centroid(intensity_array):
    return np.all(intensity_array > 0)


def is_valid_smiles(sm):
    """
    Return true if the input is a string that is nonempty and maps to a non-null
    rdkit Mol, false otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(sm)
        if sm == "" or mol is None:
            return False
        else:
            return True
    except Exception:
        return False


def smile2inchi(smile):
    if is_valid_smiles(smile):
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smile)).split("-")[0]
    else:
        return None


def create_spectra_NIST23(spectra_list, inchikey_dict, min_n_peaks=6, normalize=True):
    """
    This funciton creates Msms spectrum for each spectrum in the NIST23 dataset.
    It also checks if the spectrum is centroided and if it has a smiles and inchikey.

    Spectrum quality filtering:
      - Don't include multiply charged molecules.
      - Don't include spectra with invalid precursor m/z (0).
      - Don't include spectra with too few peaks (minimum 6).
      - Only include positive ion mode spectra.
      - Only include spectra with [M+H]+ adducts.
      - Only include centroid data (does not contain zero intensity values).
      - Only include spectra with InChI and/or SMILES specified.

    Parameters:
    -----------
    spectra_list : list of dicts
        A list of spectra dictionaries.
    inchikey_dict : dict
        A dictionary of inchikeys and smiles.
    min_n_peaks : int
        The minimum number of peaks a spectrum should have.
    normalize : bool
        If True, normalize the spectrum.

    Returns:
    --------
    spectra : list of Msms.Spectrum
        A list of Msms.Spectrum objects.
    info_dict : dict
        A dictionary of information about the spectra.
    inchi_dict: dict
        A dictionary mapping spectra to their inchikeys
    """

    # Set initial values
    not_smiles = 0
    not_inchikey = 0
    not_centroid = 0
    inchikeys_to_resolve = set()
    spectra = []
    inchi_dict = {}
    precursor_dict = {}

    # Create Msms Spectrum
    for i, spectrum_dict in tqdm(enumerate(spectra_list)):

        if (
            float(spectrum_dict["precursormz"][0]) > 0 and
            len(spectrum_dict["peaks"]) >= min_n_peaks and
            spectrum_dict["ion_mode"] == "P" and
            spectrum_dict['precursor_type'] == '[M+H]+'
        ):

            # They should all be positive.
            charge = spectrum_dict["precursor_type"].split(']')[1]

            if charge != '+':
                raise ValueError('Charge is not +')

            # check it has smiles
            if "smiles" not in spectrum_dict and "inchikey" in spectrum_dict:
                if spectrum_dict["inchikey"] in inchikey_dict:
                    spectrum_dict["smiles"] = inchikey_dict[spectrum_dict["inchikey"]]
                else:
                    inchikeys_to_resolve.add(spectrum_dict["inchikey"])

                continue

            elif "smiles" not in spectrum_dict and "inchikey" not in spectrum_dict:
                not_smiles += 1
                continue

            # check it has inchikey
            if "inchikey" not in spectrum_dict:
                not_inchikey += 1
                continue

            # Iterate through the list of tuples from the 'peaks' and get two lists for m/z and intensity.
            spectrum_dict["m/z array"] = spectrum_dict['peaks'][:, 0]
            spectrum_dict["intensity array"] = spectrum_dict['peaks'][:, 1]

            # Check if the spectrum is centroided.
            if not is_centroid(spectrum_dict["intensity array"]):
                not_centroid += 1
                continue

            # Normalize the intensity array.
            if normalize:
                spectrum_dict["intensity array"] = spectrum_dict["intensity array"] / max(
                    spectrum_dict["intensity array"])

            spec = sus.MsmsSpectrum(
                identifier=str(i) + '_' + spectrum_dict["cas#"] if "cas#" in spectrum_dict else str(i) + '_' + str(
                    spectrum_dict['nist#']),
                precursor_mz=float(spectrum_dict["precursormz"]),
                precursor_charge=1,
                mz=spectrum_dict["m/z array"],
                intensity=spectrum_dict["intensity array"],
            )
            spec.inchi = spectrum_dict["inchikey"]
            spec.partial_inchikey = spectrum_dict["inchikey"][:14]
            spec.smiles = spectrum_dict["smiles"]
            spec.remove_precursor_peak(0.1, "Da")
            spec.filter_intensity(0.01, max_num_peaks=200)

            # Add inchikey to inchi dict
            if spec.partial_inchikey in inchi_dict.keys():
                inchi_dict[spec.partial_inchikey].append(len(spectra))
            else:
                inchi_dict[spec.partial_inchikey] = [len(spectra)]

            # Add spectra to list 
            spectra.append(spec)

    # Create info_dict
    info_dict = {'not_smiles': not_smiles,
                 'not_inchikey': not_inchikey,
                 'not_centroid': not_centroid,
                 'inchikeys_to_resolve': inchikeys_to_resolve}

    return spectra, info_dict, inchi_dict


def weight_NIST23_spectra(spectra, intensity_weights, mz_weight_df):
    """
    This function returns a list of weighted NIST23 spectra.

    Parameters:
    -----------
    spectra : list of Msms.Spectrum
        A list of Msms.Spectrum objects.
    intensity_weights : function
        A function that weights the intensities of a spectrum
    mz_weight_df : pandas.DataFrame
        A dataframe with weights for each m/z value
    
    
    Returns:
    --------
    weighted_spectra: list of Msms.Spectrum
        A list of weightedMsms.Spectrum objects.
    """

    # Set initial values
    weighted_spectra = []

    # Create Msms Spectrum
    for i, spectrum in tqdm(enumerate(spectra)):
        # Get weighted intensity
        rounded_mz = np.round(spectrum.mz, 1)
        weighted_intensity = intensity_weights(spectrum.intensity) * mz_weight_df.loc[rounded_mz, 'weight']

        spec = sus.MsmsSpectrum(
            identifier=spectrum.identifier,
            precursor_mz=spectrum.precursor_mz,
            precursor_charge=spectrum.precursor_charge,
            mz=spectrum.mz,
            intensity=weighted_intensity,
        )
        spec.inchi = spectrum.inchi
        spec.partial_inchikey = spectrum.partial_inchikey
        spec.smiles = spectrum.smiles

        # Add spectra to list 
        weighted_spectra.append(spec)

    return weighted_spectra


def create_spectra_wout(filename="../data/ALL_GNPS_NO_PROPOGATED.mgf",
                        charges=(0, 1),
                        min_n_peaks=6,
                        normalize=True):
    """
    This funciton creates Msms spectrum for each spectrum in the wout dataset.

    Read all spectra from the MGF.
    ALL_GNPS_NO_PROPOGATED (retrieved on 2022-05-12) downloaded from
    https://gnps-external.ucsd.edu/gnpslibrary

    Spectrum quality filtering:
      - Don't include propagated spectra (LIBRARYQUALITY==4).
      - Don't include multiply charged molecules.
      - Don't include spectra with invalid precursor m/z (0).
      - Don't include spectra with too few peaks (minimum 6).
      - Only include positive ion mode spectra.
      - Only include spectra with [M+H]+ adducts.
      - Only include centroid data (does not contain zero intensity values).
      - Only include spectra with InChI and/or SMILES specified.

    Parameters:
    -----------
    filename : str
        The path to the wout dataset.
    charges : tuple
        The charges to keep.
    min_n_peaks : int
        The minimum number of peaks a spectrum should have.
    normalize : bool
        Whether to normalize the intensities of the spectra.
    
    Returns:
    --------
    spectra : list of Msms.Spectrum
        A list of Msms.Spectrum objects.
    info_dict : dict
        A dictionary of information about the spectra.
    """
    # Set initial values
    spectra = []
    not_inchi = 0

    # Get Msms spectra
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

                # Normalize intensity array
                if normalize:
                    spectrum_dict["intensity array"] = spectrum_dict["intensity array"] / max(
                        spectrum_dict["intensity array"])

                # Create Msms Spectrum
                spec = sus.MsmsSpectrum(
                    spectrum_dict["params"]["spectrumid"],
                    float(spectrum_dict["params"]["pepmass"][0]),
                    # Re-assign charge 0 to 1.
                    max(int(spectrum_dict["params"]["charge"][0]), 1),
                    spectrum_dict["m/z array"],
                    spectrum_dict["intensity array"] / max(spectrum_dict["intensity array"]),
                )

                # Add metadata
                spec.library = spectrum_dict["params"]["organism"]
                spec.inchi = spectrum_dict["params"]["inchi"]
                spec.smiles = spectrum_dict["params"]["smiles"]
                spec.partial_inchikey = smile2inchi(spectrum_dict["params"]["smiles"])
                if spec.partial_inchikey is None:
                    not_inchi += 1
                spec.remove_precursor_peak(0.1, "Da")
                spec.filter_intensity(0.01, max_num_peaks=200)
                spectra.append(spec)

        # Create info_dict
        info_dict = {'not_inchi': not_inchi}

    return spectra, info_dict


def weight_wout_spectra(spectra, intensity_weights, mz_weight_df):
    """
    This function returns a list of weighted wout spectra.

    Parameters:
    -----------
    spectra : list of Msms.Spectrum
        A list of Msms.Spectrum objects.
    intensity_weights : function
        A function that weights the intensities of a spectrum
    mz_weight_df : pandas.DataFrame
        A dataframe with weights for each m/z value
    
    Returns:
    --------
    weighted_spectra: list of Msms.Spectrum
        A list of weightedMsms.Spectrum objects.
    """

    # Set initial values
    weighted_spectra = []

    # Get Msms spectra
    for spectrum in tqdm(spectra):
        rounded_mz = np.round(spectrum.mz, 1)
        weighted_intensity = intensity_weights(spectrum.intensity) * mz_weight_df.loc[rounded_mz, 'weight']
        spec = sus.MsmsSpectrum(
            spectrum.identifier,
            spectrum.precursor_mz,
            spectrum.precursor_charge,
            spectrum.mz,
            weighted_intensity,
        )
        spec.library = spectrum.library
        spec.inchi = spectrum.inchi
        spec.smiles = spectrum.smiles
        spec.partial_inchikey = spectrum.partial_inchikey
        weighted_spectra.append(spec)

    return weighted_spectra
