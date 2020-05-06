import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score

from .utils import fillna
from .cut import DEFAULT_BINS, cut


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
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    n0_group = np.zeros(np.unique(x).shape)
    n1_group = np.zeros(np.unique(x).shape)
    for i in range(len(np.unique(x))):
        n0_group[i] = np.maximum(len(y[(x == np.unique(x)[i]) & (y == 0)]), 0.5)
        n1_group[i] = np.maximum(len(y[(x == np.unique(x)[i]) & (y == 1)]), 0.5)
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


def iv(x, y, is_continuous=True):
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
        return iv_continuous(x, y)
    return iv_discrete(x, y)


def psi(expect_score, actual_score, n_bins=DEFAULT_BINS):
    """
    Compute IV for continuous feature.

    Parameters
    ----------
    expect_score : array-like
    actual_score: array-like
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`.

    Returns
    -------
    psi : float
    """
    expect_cut, cut_bins = pd.cut(expect_score, n_bins, retbins=True)
    expect = expect_cut.value_counts() / np.sum(expect_cut.value_counts())
    cut_bins = expect_cut.unique().categories
    actual_cut = pd.cut(actual_score, bins=cut_bins)
    actual = actual_cut.value_counts() / np.sum(actual_cut.value_counts())

    actual[actual == 0] = 1e-10
    expect[expect == 0] = 1e-10

    psi = np.sum((actual - expect) * np.log(actual / expect))
    return psi


def save_roc(y_true, y_pred, out_path, file_name='roc.png'):
    """
    Compute receiver operating characteristic (ROC) and save the figure.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        target scores, predicted by estimator

    out_path : str
        save figure to `out_path`

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
    plt.savefig(os.path.join(out_path, file_name))


def compare_roc(y_true_list, y_pred_list, model_name_list, out_path,
                file_name='roc_cmp.png'):
    """
    Plot multi ROC of different input and save the figure.

    Parameters
    ----------

    y_true_list : array of array, shape = [n_curve, n_samples]
        True binary labels.

    y_pred_list : array of array, shape = [n_curve, n_samples]
        target scores, predicted by estimator

    model_name_list : array of str
        curve labels

    out_path : str
        save figure to `out_path`

    file_name : str
        save figure as `file_name`
    """
    plt.figure(figsize=(5, 5))
    for i in range(len(y_true_list)):
        fpr, tpr, _ = roc_curve(y_true_list[i], y_pred_list[i])
        ks_value = np.max(tpr - fpr)
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='%s-AUC(%.3f)-KS(%.3f)' %
                 (model_name_list[i], auc_value, ks_value))

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(os.path.join(out_path, file_name))


def save_pr(y_true, y_pred, out_path, file_name='pr.png'):
    """
    Compute Precision-Recall Curve (PRC) and save the figure.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        target scores, predicted by estimator

    out_path : str
        save figure to `out_path`

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
    plt.savefig(os.path.join(out_path, file_name))


def save_pr_threshold(y_true, y_pred, out_path, file_name='pr_threshold.png'):
    """
    Compute precision&recall curve changed by threshold and save the figure.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        target scores, predicted by estimator

    out_path : str
        save figure to `out_path`

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
    plt.savefig(os.path.join(out_path, file_name))
