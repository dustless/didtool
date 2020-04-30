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


def dt_cut(x, target, n_bins=DEFAULT_BINS, nan=-1, min_bin=0.01,
           return_bins=False):
    """
    Cut values into discrete intervals by decision tree.
    NA value will be put into an independent bucket without other values.

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
    nan: Replace NA values with `nan` in the result if `nan` is not None.
    min_bin : float, optional (default=0.01)
        The minimum fraction of samples required to be in a bin.
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
    x = to_ndarray(x)
    target = to_ndarray(target)
    mask = np.isnan(x)

    tree = DecisionTreeClassifier(
        min_samples_leaf=min_bin,
        max_leaf_nodes=n_bins,
    )
    # only use non-nan values to fit decision tree
    tree.fit(x[~mask].reshape((-1, 1)), target[~mask])

    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
    bins = np.sort(thresholds)
    min_val = np.nanmin(x)
    max_val = np.nanmax(x)
    min_val = min_val - max(np.abs(min_val) * 0.001, 0.001)
    max_val = max_val + max(np.abs(max_val) * 0.001, 0.001)
    bins = np.array([min_val] + list(bins) + [max_val])

    out, bins = pd.cut(x, bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        return out, bins
    else:
        return out


def lgb_cut(x, target, n_bins=DEFAULT_BINS, min_bin=0.01,
            return_bins=False):
    """
    Cut values into discrete intervals by lightgbm.
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
    if not return_bins:
        return out

    model = tree.booster_.dump_model()
    tree_infos = model['tree_info']
    nodes = [tree_infos[0]['tree_structure']]
    thresholds = []
    i = 0
    while i < len(nodes):
        if 'threshold' in nodes[i]:
            thresholds.append(nodes[i]['threshold'])
            if 'left_child' in nodes[i]:
                nodes.append(nodes[i]['left_child'])
            if 'right_child' in nodes[i]:
                nodes.append(nodes[i]['right_child'])
        i += 1
    bins = np.sort(thresholds)
    min_val = np.nanmin(x)
    max_val = np.nanmax(x)
    min_val = min_val - max(np.abs(min_val) * 0.001, 0.001)
    max_val = max_val + max(np.abs(max_val) * 0.001, 0.001)
    bins = np.array([min_val] + list(bins) + [max_val])
    return out, bins


def cut(x, target=None, method='dt', n_bins=DEFAULT_BINS,
        return_bins=False, **kwargs):
    """
    Cut values into discrete intervals.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    target: array-like
        target will be used to fit decision tree or others.
        Only used when method is 'dt'/'lgb'.
    method : str, optional (default='dt')
        - 'dt': cut values by decision tree
        - 'lgb': cut values by lightgbm
        - 'step': cut values by step
        - 'quantile': cut values by quantile
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended by .1% on each side to include the minimum
        and maximum values of `x`.
    return_bins : bool, default False
        Whether to return the bins or not.
    **kwargs
        Other parameters for sub functions.

    Returns
    -------
    out : numpy.ndarray
        An array-like object representing the respective bin for each value
         of `x`.

    bins : numpy.ndarray
        The computed or specified bins. Only returned when `return_bins=True`.
    """
    if method == 'dt':
        res = dt_cut(x, target, n_bins=n_bins, return_bins=return_bins,
                     **kwargs)
    elif method == 'lgb':
        res = lgb_cut(x, target, n_bins=n_bins, return_bins=return_bins,
                      **kwargs)
    elif method == 'step':
        res = step_cut(x, n_bins=n_bins, return_bins=return_bins, **kwargs)
    elif method == 'quantile':
        res = quantile_cut(x, n_bins=n_bins, return_bins=return_bins, **kwargs)
    else:
        raise Exception("unsupported method `%s`" % method)
    return res

