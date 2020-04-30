import numpy as np

from .utils import fillna
from .cut import DEFAULT_BINS, cut


def iv_discrete(x, y):
    """
    Compute IV for discrete feature.
    :param x: numpy.array
    :param y: numpy.array
    :return: iv
    """
    if np.any(np.isnan(x)):
        x = fillna(x, -999)
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    n0_group = np.zeros(np.unique(x).shape)
    n1_group = np.zeros(np.unique(x).shape)
    for i in range(len(np.unique(x))):
        n0_group[i] = max(len(y[(x == np.unique(x)[i]) & (y == 0)]), 0.5)
        n1_group[i] = max(len(y[(x == np.unique(x)[i]) & (y == 1)]), 0.5)
    iv = np.sum((n0_group / n0 - n1_group / n1) *
                np.log((n0_group / n0) / (n1_group / n1)))
    return iv


def iv_continuous(x, y, n_bins=DEFAULT_BINS, cut_method='dt'):
    """
    Compute IV for continuous feature.
    Parameters
    ----------
    x : array-like
    y: array-like
    cut_method : str, optional (default='dt')
        see didtool.cut
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`.

    Returns
    -------
    iv : IV of feature x
    """
    x_bin = cut(x, y, method=cut_method, n_bins=n_bins)
    return iv_discrete(x_bin, y)
