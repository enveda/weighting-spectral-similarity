import hashlib

import numpy as np


def _string_to_hash(input_string):
    """Hash input string deterministically."""
    m = hashlib.sha256()
    m.update(bytes(input_string, "utf-8"))
    return m.hexdigest()


def _spectrum_to_string(mzs, intensities):
    """Convert mzs and intensities string arrays to single string."""
    spectrum_string = "({0}),({1})".format(" ".join(mzs), " ".join(intensities))
    return spectrum_string


def _array_to_rounded_string_list_iterative(arr, max_precision, min_precision):
    """Round all values in array iteratively from max_precision to min_precision."""
    rounded_array = [str(x) for x in arr]
    rounded_arrays = [rounded_array]

    for offset in range(max_precision - min_precision + 1):
        precision = max_precision - offset
        round_string = f"{{:.{precision}f}}"
        rounded_array = [round_string.format(float(x)) for x in rounded_array]
        rounded_arrays.append(rounded_array)

    return rounded_arrays


def _array_to_rounded_string_list_direct(arr, precision=None):
    """Round all values in array directly to specified number of decimal places."""
    if precision is None:
        return [str(x) for x in arr]
    else:
        return [f"{{:.{precision}f}}".format(x) for x in arr]


def hash_spectrum(
        mzs,
        intensities,
        precision=None,
        iterative=False,
        sort=False,
        normalize_intensity=False,
):
    """
    Hash the input mzs and intensities by first (optionally) truncating
    to specified number of decimals, sorting lists, creating a single
    string and hashing it.

    Parameters
    ----------
    mzs : flat ndarray of floats
    intensities : flat ndarray of floats
    precision : int or None
        Specifies how many decimal places to keep; all if None
    sort : bool
        Indicates whether to sort arrays by mz -- do this if not already sorted
    normalize_intensity : bool
        Indicates whether to normalized intensities to max of 1. Do this if not
        already done

    Returns
    -------
    str, the hashed value as hexidecimal string
    """
    if sort:
        sorted_idx = np.argsort(mzs)
        mzs = mzs[sorted_idx]
        intensities = intensities[sorted_idx]

    if normalize_intensity:
        intensities /= intensities.max()

    if iterative and (precision is not None):
        rounded_mzs = _array_to_rounded_string_list_iterative(mzs, 6, precision)[-1]
        rounded_intensities = _array_to_rounded_string_list_iterative(
            intensities, 6, precision
        )[-1]
    else:
        rounded_mzs = _array_to_rounded_string_list_direct(mzs, precision=precision)
        rounded_intensities = _array_to_rounded_string_list_direct(
            intensities, precision=precision
        )

    spectrum_string = _spectrum_to_string(rounded_mzs, rounded_intensities)
    return _string_to_hash(spectrum_string)
