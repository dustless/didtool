import pandas as pd
import numpy as np


def to_ndarray(s, dtype=None):
    """
    Convert array-like input to numpy.ndarray
    """
    if isinstance(s, np.ndarray):
        arr = np.copy(s)
    elif isinstance(s, pd.DataFrame):
        arr = np.copy(s.values)
    else:
        arr = np.array(s)

    if dtype is not None:
        arr = arr.astype(dtype)
    # covert object type to str
    elif arr.dtype.type is np.object_:
        arr = arr.astype(np.str)

    return arr


def fillna(data, by=-1):
    """
    Return a new array with NA/NaN value replaced by `by`

    Parameters
    ----------
    data: array-like
    by: scalar
        Value to use to fill NA/NaN values

    Returns
    -------
    out : numpy.ndarray
        a new array
    """
    out = np.copy(data)
    mask = pd.isna(out)
    out[mask] = by
    return out
