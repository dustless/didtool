# coding: utf-8
import math

import pandas as pd
import numpy as np
import lightgbm
from sklearn.tree import DecisionTreeClassifier, _tree

from .utils import fillna, to_ndarray
from scipy.stats import chi2

DEFAULT_BINS = 10


def step_cut(x, n_bins=DEFAULT_BINS, nan=-1, return_bins=False,
             remove_empty_bins=True):
    """
    Cut values into discrete intervals by step.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended to -inf/inf on each side to cover the whole
        range of `x`.
    nan: Replace NA values with `nan` in the result if `nan` is not None.
    return_bins : bool, default False
        Whether to return the bins or not.
    remove_empty_bins : bool, default True
        Whether remove empty bins

    Returns
    -------
    out : numpy.ndarray
        An array-like object representing the respective bin for each value
         of `x`.

    bins : numpy.ndarray
        The computed or specified bins. Only returned when `return_bins=True`.
    """
    out, bins = pd.cut(x, n_bins, labels=False, retbins=True)

    if remove_empty_bins:
        # merge empty bins
        cut_bins = []
        unique_bins = np.sort(np.unique(out))
        for i in range(1, n_bins):
            if i in unique_bins:
                cut_bins.append(bins[i])
        cut_bins = [-np.inf] + cut_bins + [np.inf]
        # cut again with merged bins
        out, bins = pd.cut(x, cut_bins, labels=False, retbins=True)
    else:
        cut_bins = bins
        cut_bins[0] = -np.inf
        cut_bins[-1] = np.inf

    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        return out, cut_bins
    else:
        return out


def quantile_cut(x, n_bins=DEFAULT_BINS, nan=-1, return_bins=False):
    """
    Cut values into discrete intervals by quantile.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended to -inf/inf on each side to cover the whole
        range of `x`.
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
        bins[0] = -np.inf
        bins[-1] = np.inf
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
        range of `x` is extended to -inf/inf on each side to cover the whole
        range of `x`.
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
    bins = np.array([-np.inf] + list(bins) + [np.inf])

    out, bins = pd.cut(x, bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        return out, bins
    else:
        return out


def lgb_cut(x, target, n_bins=DEFAULT_BINS, nan=-1, min_bin=0.01,
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
        range of `x` is extended to -inf/inf on each side to cover the whole
        range of `x`.
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
    min_child_samples = math.ceil(min_bin * len(x))

    tree = lightgbm.LGBMClassifier(
        n_estimators=1,
        num_leaves=n_bins,
        min_child_samples=min_child_samples,
        random_state=27
    )
    tree.fit(x[~mask].reshape((-1, 1)), target[~mask])

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
    bins = np.array([-np.inf] + list(bins) + [np.inf])

    out, bins = pd.cut(x, bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        return out, bins
    else:
        return out


def _get_chi_square_distribution(d_free=4, c_f=0.1):
    """
    根据自由度和置信度得到卡方分布和阈值
    params:
        d_free: 自由度, 最大分箱数-1, default 4
        cf: 显著性水平, default 10%
    return:
        卡方阈值
    """
    percents = [0.95, 0.90, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(
        np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index + 1
    # 显示小数点后面数字
    pd.set_option('precision', 3)
    return df.loc[d_free, c_f]


def chi_square_cut(x, target, n_bins=DEFAULT_BINS, cf=0.1, nan=-1,
                   return_bins=False):
    """
    计算某个特征每种属性值的卡方统计量
    params: 
        x: 待分箱特征
            array-like
                The input array to be binned. Must be 1-dimensional.
        target: 目标Y值 (0或1) Y值为二分类变量
            array-like
                The input array to be binned. Must be 1-dimensional.
        n_bins: 期望分箱个数
        cf: 显著性水平, default 10%
        nan: 指定特征为空的分箱结果
        return_bins：是否返回分箱划分界限
    return:
        out:分箱后结果
        bins:分箱界限
    """
    # 去除nan，单独分箱
    x = to_ndarray(x)
    target = to_ndarray(target)
    mask = np.isnan(x)
    df = pd.DataFrame({'feature': x[~mask], 'label': target[~mask]})

    # 对变量按属性值从小到大排序
    df.sort_values(by='feature', ascending=True, inplace=True)
    # 计算每一个属性值对应的卡方统计量等信息
    df['count_1'] = df['label']
    df['count_0'] = 1 - df['label']
    df['max_value'] = df['feature']
    feature_min = df['feature'].min()
    df.drop(['feature', 'label'], axis=1, inplace=True)

    # 获取卡方分箱阀值，大于此值则不合并
    threshold = _get_chi_square_distribution(d_free=n_bins - 1, c_f=cf)

    while df.shape[0] > n_bins:
        min_index = None
        min_chi_score = None
        for i in range(df.shape[0] - 1):
            # 计算相邻两个分箱的卡方值
            i_value = df.loc[
                i, ['count_0', 'count_1', 'max_value']].values
            i_value_next = df.loc[
                i + 1, ['count_0', 'count_1', 'max_value']].values
            # 如果两个区域值一样，则直接合并，忽略卡方值
            if i_value[2] == i_value_next[2]:
                chi_score = 0
            else:
                label_total = i_value[0] + i_value_next[0] + i_value[1] + \
                              i_value_next[1]
                label_0_ratio = (i_value[0] + i_value_next[0]) / label_total
                label_1_ratio = (i_value[1] + i_value_next[1]) / label_total
                i_1_cal = (i_value[0] + i_value[1]) * label_1_ratio
                i_0_cal = (i_value[0] + i_value[1]) * label_0_ratio
                i_1_next_cal = (i_value_next[0] + i_value_next[
                    1]) * label_1_ratio
                i_0_next_cal = (i_value_next[0] + i_value_next[
                    1]) * label_0_ratio

                chi_part1 = 0 if i_0_cal == 0 \
                    else (i_value[0] - i_0_cal) ** 2 / i_0_cal
                chi_part2 = 0 if i_1_cal == 0 \
                    else (i_value[1] - i_1_cal) ** 2 / i_1_cal
                chi_part3 = 0 if i_0_next_cal == 0 \
                    else (i_value_next[0] - i_0_next_cal) ** 2 / i_0_next_cal
                chi_part4 = 0 if i_1_next_cal == 0 \
                    else (i_value_next[1] - i_1_next_cal) ** 2 / i_1_next_cal

                chi_score = chi_part1 + chi_part2 + chi_part3 + chi_part4
            # 找到合并后最小的卡方值
            if min_index is None or min_chi_score > chi_score:
                min_index = i
                min_chi_score = chi_score

        if min_chi_score < threshold:
            # 将差异最小的相邻分箱，进行合并
            df.loc[min_index, 'count_0'] = \
                df.loc[min_index, 'count_0'] + df.loc[min_index + 1, 'count_0']
            df.loc[min_index, 'count_1'] = \
                df.loc[min_index, 'count_1'] + df.loc[min_index + 1, 'count_1']
            df.loc[min_index, 'max_value'] = df.loc[min_index + 1, 'max_value']
            df.drop([min_index + 1], inplace=True)
            df.reset_index(inplace=True, drop=True)
        else:
            break

    # 获取分箱结果
    bins = [feature_min - 0.0001]
    for i in df['max_value'].values:
        if i > bins[-1]:
            bins.append(i)
    bins[-1] = bins[-1] + 0.0001
    out, bins = pd.cut(x, bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)
    if return_bins:
        bins[0] = -np.inf
        bins[-1] = np.inf
        return out, bins
    else:
        return out


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
        - 'chi': cut values by chi square
    n_bins : int, default DEFAULT_BINS
        Defines the number of equal-width bins in the range of `x`. The
        range of `x` is extended to -inf/inf on each side to cover the whole
        range of `x`.
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
        return dt_cut(x, target, n_bins=n_bins, return_bins=return_bins,
                      **kwargs)
    elif method == 'lgb':
        return lgb_cut(x, target, n_bins=n_bins, return_bins=return_bins,
                       **kwargs)
    elif method == 'step':
        return step_cut(x, n_bins=n_bins, return_bins=return_bins, **kwargs)
    elif method == 'quantile':
        return quantile_cut(x, n_bins=n_bins, return_bins=return_bins, **kwargs)
    elif method == 'chi':
        return chi_square_cut(x, target, n_bins=n_bins, return_bins=return_bins,
                              **kwargs)
    else:
        raise Exception("unsupported method `%s`" % method)


def cut_with_bins(x, bins, nan=-1, right=True):
    """
    Cut values into discrete intervals.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins: array-like
        sequence of scalars : Defines the bin edges allowing non-uniform width.
    nan: Replace NA values with `nan` in the result if `nan` is not None.
    right : bool(default=True)
        Indicates whether `bins` includes the rightmost edge or not.

    Returns
    -------
    out : numpy.ndarray
        An array-like object representing the respective bin for each value
         of `x`.
    """
    out = pd.cut(x, bins, right=right, labels=False)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)
    return out
