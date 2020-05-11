import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score
import seaborn as sns

from .cut import DEFAULT_BINS, cut

sns.set(rc={"figure.figsize": (8, 4)})


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


def iv_discrete(x, y):
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


def iv_continuous(x, y, n_bins=DEFAULT_BINS, cut_method='dt', **kwargs):
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
    return iv_discrete(x_bin, y)


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
        return iv_continuous(x, y, **kwargs)
    return iv_discrete(x, y)


def psi(expect_score, actual_score, n_bins=DEFAULT_BINS, plot=False):
    """
    Compute IV for continuous feature.

    Parameters
    ----------
    expect_score : array-like
    actual_score: array-like
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`.
    plot : bool
        whether plot expect and actual distributions

    Returns
    -------
    psi_value : float
    """
    expect_cut, cut_bins = pd.cut(expect_score, n_bins, retbins=True)
    expect = expect_cut.value_counts() / np.sum(expect_cut.value_counts())
    cut_bins = expect_cut.unique().categories
    actual_cut = pd.cut(actual_score, bins=cut_bins)
    actual = actual_cut.value_counts() / np.sum(actual_cut.value_counts())

    actual[actual == 0] = 1e-10
    expect[expect == 0] = 1e-10

    psi_value = np.sum((actual - expect) * np.log(actual / expect))
    if plot:
        df = pd.DataFrame({"expect": expect, "actual": actual})
        df.plot(kind="bar")
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
    plt.figure(figsize=(5, 5))
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
    plt.figure(figsize=(5, 5))
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
    plt.figure(figsize=(5, 5))
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
    plt.figure(figsize=(5, 5))
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
