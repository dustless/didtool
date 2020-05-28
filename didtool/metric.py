import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score
import seaborn as sns

from .cut import DEFAULT_BINS, cut, step_cut, cut_with_bins

sns.set(rc={"figure.figsize": (10, 8)})


def probability(y, group_mask=None):
    """
    get probability of target by mask of a group

    Parameters
    ----------
    y: array-like
        binary labels

    group_mask : array of bool
        mask of a group

    Returns
    -------
    prob1 : float
        counts of 1 in group / counts of 1 in total
    prob0 : float
        counts of 0 in group / counts of 0 in total
    """
    if group_mask is None:
        return 1, 1

    total_0 = max((y == 0).sum(), 0.5)
    total_1 = max((y == 1).sum(), 0.5)

    group_y = y[group_mask]
    group_0 = max((group_y == 0).sum(), 0.5)
    group_1 = max((group_y == 1).sum(), 0.5)

    prob1 = group_1 / total_1
    prob0 = group_0 / total_0

    return prob1, prob0


def woe(prob1, prob0):
    """
    get WOE of a group

    Args:
        prob1: the probability of grouped 1 in total 1
        prob0: the probability of grouped 0 in total 0

    Returns:
        number: woe value
    """
    return np.log(prob1 / prob0)


def _iv_discrete(x, y):
    """
    Compute IV for discrete feature.

    Parameters
    ----------
    x : array-like
    y: array-like

    Returns
    -------
    iv : IV of feature x
    """
    iv_value = 0
    for v in np.unique(x):
        prob1, prob0 = probability(y, group_mask=(x == v))
        iv_value += (prob1 - prob0) * woe(prob1, prob0)
    return iv_value


def _iv_continuous(x, y, n_bins=DEFAULT_BINS, cut_method='dt', **kwargs):
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
    x_bin = cut(x, y, method=cut_method, n_bins=n_bins, **kwargs)
    return _iv_discrete(x_bin, y)


def iv(x, y, is_continuous=True, **kwargs):
    """
    Compute IV for continuous feature.

    Parameters
    ----------
    x : array-like
    y: array-like
    is_continuous : whether x is continuous, optional (default=True)

    Returns
    -------
    (name, iv) : IV of feature x
    """
    if is_continuous or len(np.unique(x)) / len(x) > 0.5:
        return _iv_continuous(x, y, **kwargs)
    return _iv_discrete(x, y)


def _psi_discrete(expected_array, actual_array, detail=False):
    """
    Compute PSI for discrete var.

    Parameters
    ----------
    expected_array : array-like
    actual_array: array-like
    detail : bool
        whether return expect and actual distribution DataFrame

    Returns
    -------
    psi_value : float
    psi_df : DataFrame, only returned when `detail` is True
        detailed distribution of expect and actual arrays
    """
    groups = np.sort(np.union1d(expected_array.unique(), actual_array.unique()))

    def _get_group_cnt(arr, groups):
        group_cnt = []
        for group in groups:
            group_cnt.append((arr == group).sum())
        return np.array(group_cnt)

    expected_cnt = _get_group_cnt(expected_array, groups)
    expected_rate = expected_cnt / len(expected_array)

    actual_cnt = _get_group_cnt(actual_array, groups)
    actual_rate = actual_cnt / len(actual_array)

    # avoid zero values
    actual_rate[actual_rate == 0] = 1e-10
    expected_rate[expected_rate == 0] = 1e-10

    psi_value = np.sum((actual_rate - expected_rate) *
                       np.log(actual_rate / expected_rate))

    if detail:
        df = pd.DataFrame({"expect": expected_rate, "actual": actual_rate},
                          index=groups)
        return psi_value, df
    return psi_value


def _psi_continuous(expected_array, actual_array, n_bins=DEFAULT_BINS,
                    detail=False):
    """
    Compute PSI for continuous var.

    Parameters
    ----------
    expected_array : array-like
    actual_array: array-like
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`.
    detail : bool
        whether return expect and actual distribution DataFrame

    Returns
    -------
    psi_value : float
    psi_df : DataFrame, only returned when `detail` is True
        detailed distribution of expect and actual arrays
    """
    expected_bin_values, bins = step_cut(expected_array, n_bins,
                                         return_bins=True)
    # cut actual array with the same bins as expected array
    actual_bin_values = cut_with_bins(actual_array, bins)
    has_nan = any(expected_bin_values == -1) or any(actual_bin_values == -1)

    def _get_bins_cnt(bin_values):
        bins_cnt = []
        if has_nan:
            bins_cnt.append((bin_values == -1).sum())
        for i in range(0, len(bins) - 1):
            bins_cnt.append((bin_values == i).sum())
        return np.array(bins_cnt)

    expected_cnt = _get_bins_cnt(expected_bin_values)
    expected_rate = expected_cnt / len(expected_array)

    actual_cnt = _get_bins_cnt(actual_bin_values)
    actual_rate = actual_cnt / len(actual_array)

    # avoid zero values
    actual_rate[actual_rate == 0] = 1e-10
    expected_rate[expected_rate == 0] = 1e-10

    psi_value = np.sum((actual_rate - expected_rate) *
                       np.log(actual_rate / expected_rate))

    if detail:
        bin_ranges = []
        if has_nan:
            bin_ranges.append('NA')
        for i in range(len(bins) - 1):
            bin_ranges.append('(%.4f, %.4f]' % (bins[i], bins[i + 1]))
        df = pd.DataFrame({"expect": expected_rate, "actual": actual_rate},
                          index=bin_ranges)
        return psi_value, df
    return psi_value


def psi(expected_array, actual_array, n_bins=DEFAULT_BINS, plot=False,
        is_continuous=True):
    """
    Compute PSI for continuous var.

    Parameters
    ----------
    expected_array : array-like
    actual_array: array-like
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`.
    plot : bool
        whether plot expect and actual distributions
    is_continuous : bool, default=True
        whether the input array is continuous

    Returns
    -------
    psi_value : float
    """
    if is_continuous:
        psi_value, df = _psi_continuous(expected_array, actual_array, n_bins,
                                        detail=True)
    else:
        psi_value, df = _psi_discrete(expected_array, actual_array, detail=True)

    if plot:
        df.plot(kind="bar")
        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc="best")
        plt.title("psi={}".format(psi_value))
        plt.show()
    return psi_value


def distribution(x, bins=None, out_path=None, file_name='distribution.png'):
    """
    plot distribution of x.

    Parameters
    ----------
    x: array-like, shape = [n_samples]

    bins: int or None, num of bins

    out_path : str or None
        if out_path specified, save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    plt.figure()
    sns.distplot(x, bins=bins, kde=True, rug=False)
    if out_path:
        plt.savefig(os.path.join(out_path, file_name))
    else:
        plt.show()


def distributions(x_list, bins=None, out_path=None,
                  file_name='distributions.png'):
    """
    compare distributions of x in x_list.

    Parameters
    ----------
    x_list: array of array-like, shape = [n_input, n_samples]

    bins: int or None, num of bins

    out_path : str or None
        if out_path specified, save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    plt.figure()
    for i, x in enumerate(x_list):
        sns.distplot(x, bins=bins, kde=True, rug=False)
    if out_path:
        plt.savefig(os.path.join(out_path, file_name))
    else:
        plt.show()


def plot_roc(y_true, y_pred, out_path=None, file_name='roc.png'):
    """
    Compute receiver operating characteristic (ROC) and save the figure.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        target scores, predicted by estimator

    out_path : str or None
        if out_path specified, save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks_value = np.max(tpr - fpr)
    auc_value = auc(fpr, tpr)

    # roc curve
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, lw=1, label='ROC')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (AUC=%.3f,KS=%.3f)' % (auc_value, ks_value))
    if out_path:
        plt.savefig(os.path.join(out_path, file_name))
    else:
        plt.show()


def compare_roc(y_true_list, y_pred_list, model_name_list, out_path=None,
                file_name='roc_cmp.png'):
    """
    Plot multi ROC of different input and save the figure.

    Parameters
    ----------

    y_true_list : list of array, shape = [n_curve, n_samples]
        True binary labels.

    y_pred_list : list of array, shape = [n_curve, n_samples]
        target scores, predicted by estimator

    model_name_list : array of str
        curve labels

    out_path : str or None
        if out_path specified, save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    plt.figure(figsize=(8, 8))
    for i in range(len(y_true_list)):
        fpr, tpr, _ = roc_curve(y_true_list[i], y_pred_list[i])
        ks_value = np.max(tpr - fpr)
        auc_value = auc(fpr, tpr)
        label = '%s-AUC(%.3f)-KS(%.3f)' % \
                (model_name_list[i], auc_value, ks_value)
        plt.plot(fpr, tpr, lw=1, label=label)

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if out_path:
        plt.savefig(os.path.join(out_path, file_name))
    else:
        plt.show()


def plot_pr_curve(y_true, y_pred, out_path=None, file_name='pr.png'):
    """
    Compute Precision-Recall Curve (PRC) and save the figure.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        target scores, predicted by estimator

    out_path : str or None
        if out_path specified, save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    plt.figure(figsize=(8, 8))
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
    if out_path:
        plt.savefig(os.path.join(out_path, file_name))
    else:
        plt.show()


def plot_pr_threshold(y_true, y_pred, out_path=None,
                      file_name='pr_threshold.png'):
    """
    Compute precision&recall curve changed by threshold and save the figure.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        target scores, predicted by estimator

    out_path : str or None
        if out_path specified, save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    plt.figure(figsize=(8, 8))
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    thresholds = np.append(thresholds, 1.0)
    plt.plot(thresholds, precision, lw=1, label='Precision')
    plt.plot(thresholds, recall, lw=1, label='Recall')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Thresholds')
    plt.ylabel('Rate')
    plt.title('Precision and Recall Rate')
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(linestyle='-')
    plt.legend()
    if out_path:
        plt.savefig(os.path.join(out_path, file_name))
    else:
        plt.show()
