import math

import pandas as pd
import numpy as np
import lightgbm
from sklearn.tree import DecisionTreeClassifier, _tree

from .utils import fillna, to_ndarray

DEFAULT_BINS = 10


def step_cut(x, n_bins=DEFAULT_BINS, nan=None, return_bins=False):
    """
    Cut values into discrete intervals by step.
    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended by .1% on each side to include the minimum
        and maximum values of `x`.
    nan: Replace NA values with `nan` in the result if `nan` is not None.
    return_bins : bool, default False
        Whether to return the bins or not.

    Returns
    -------
    out : numpy.ndarray
        An array-like object representing the respective bin for each value
         of `x`.

    bins : numpy.ndarray
        The computed or specified bins. Only returned when `return_bins=True`.
    """
    out, bins = pd.cut(x, n_bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        return out, bins
    else:
        return out


def quantile_cut(x, n_bins=DEFAULT_BINS, nan=None, return_bins=False):
    """
    Cut values into discrete intervals by quantile.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended by .1% on each side to include the minimum
        and maximum values of `x`.
    nan: Replace NA values with `nan` in the result if `nan` is not None.
    return_bins : bool, default False
        Whether to return the bins or not.

    Returns
    -------
    out : numpy.ndarray
        An array-like object representing the respective bin for each value
         of `x`.

    bins : numpy.ndarray
        The computed or specified bins. Only returned when `return_bins=True`.
    """
    out, bins = pd.qcut(x, n_bins, labels=False, retbins=True,
                        duplicates='drop')
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        return out, bins
    else:
        return out


def dt_cut(x, target, n_bins=DEFAULT_BINS, min_bin=0.01):
    """
    Cut values into discrete intervals by decision tree.
    NA value will be merged into a bucket with other values.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    target: array-like
        target will be used to fit decision tree
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended by .1% on each side to include the minimum
        and maximum values of `x`.
    min_bin : float, optional (default=0.01)
        The minimum fraction of samples required to be in a bin.

    Returns
    -------
    out : numpy.ndarray
        An array-like object representing the respective bin for each value
         of `x`.
    """
    x = to_ndarray(x).reshape(-1, 1)
    min_child_samples = math.ceil(min_bin * len(x))

    tree = lightgbm.LGBMClassifier(
        n_estimators=1,
        num_leaves=n_bins,
        min_child_samples=min_child_samples,
        random_state=27
    )
    tree.fit(x, target)
    out = tree.predict(x, pred_leaf=True)
    return out
