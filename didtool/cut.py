import math

import pandas as pd
import numpy as np
import lightgbm
from sklearn.tree import DecisionTreeClassifier, _tree

from .utils import fillna, to_ndarray
from scipy.stats import chi2

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
    out, bins = pd.cut(x, n_bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
        bins[0] = -np.inf
        bins[-1] = np.inf
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


def chi_square_cut(x, target, n_bins=DEFAULT_BINS, cf=0.1, nan=None, return_bins=False):
    '''
    计算某个特征每种属性值的卡方统计量
    params: 
        x: 待分箱特征
        target: 目标Y值 (0或1) Y值为二分类变量
    return:
        out:分箱后结果
        bins:分箱界限
        卡方统计量dataframe
        feature: 特征名称
        act_target_cnt: 实际坏样本数
        expected_target_cnt：期望坏样本数
        chi_square：卡方统计量
    '''
    # 对变量按属性值从小到大排序
    x = to_ndarray(x)
    target = to_ndarray(target)
    mask = np.isnan(x)
    df = pd.DataFrame({'feature':x[~mask],'label':target[~mask]})
    df.sort_index(axis = 0,ascending = True,by = 'feature',inplace=True)
    # 计算每一个属性值对应的卡方统计量等信息
    df['label_1_count'] = df['label']
    df['label_0_count'] = 1 - df['label']
    df['max_value'] = df['feature']
    feature_min = df['feature'].min()
    df.drop(['feature','label'],axis=1,inplace=True)

    dfree = n_bins - 1

    def get_chiSquare_distuibution(dfree=4, cf=0.1):
        '''
        根据自由度和置信度得到卡方分布和阈值
        params:
            dfree: 自由度, 最大分箱数-1, default 4
            cf: 显著性水平, default 10%
        return:
            卡方阈值
        '''
        percents = [0.95, 0.90, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
        df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
        df.columns = percents
        df.index = df.index + 1
        # 显示小数点后面数字
        pd.set_option('precision', 3)
        return df.loc[dfree, cf]

    threshold = get_chiSquare_distuibution(dfree=dfree, cf=cf)

    while df.shape[0] > n_bins:
        min_index = None
        min_chi_score = None
        for i in range(df.shape[0] - 1):
            i_value = df.loc[i,['label_0_count','label_1_count']].values
            i_value_next = df.loc[i + 1,['label_0_count','label_1_count']].values
            label_total = i_value[0] + i_value_next[0] + i_value[1] + i_value_next[1]
            label_0_ratio = (i_value[0] + i_value_next[0]) / label_total
            label_1_ratio = (i_value[1] + i_value_next[1]) / label_total
            i_1_should = (i_value[0] + i_value[1]) * label_1_ratio
            i_0_should = (i_value[0] + i_value[1]) * label_0_ratio
            i_1_next_should = (i_value_next[0] + i_value_next[1]) * label_1_ratio
            i_0_next_should = (i_value_next[0] + i_value_next[1]) * label_0_ratio

            chi_part1 = 0 if i_0_should == 0 else (i_value[0] - i_0_should)**2 / i_0_should
            chi_part2 = 0 if i_1_should == 0 else (i_value[1] - i_1_should) ** 2 / i_1_should
            chi_part3 = 0 if i_0_next_should == 0 else (i_value_next[0] - i_0_next_should)**2 / i_0_next_should
            chi_part4 = 0 if i_1_next_should == 0 else (i_value_next[1] - i_1_next_should)**2 / i_1_next_should

            chi_score = chi_part1 + chi_part2 + chi_part3 + chi_part4
            if min_index is None or min_chi_score > chi_score:
                min_index = i
                min_chi_score = chi_score

        if min_chi_score < threshold:
            df.loc[min_index,'label_0_count'] = df.loc[min_index,'label_0_count'] + df.loc[min_index + 1,'label_0_count']
            df.loc[min_index, 'label_1_count'] = df.loc[min_index, 'label_1_count'] + df.loc[min_index + 1, 'label_1_count']
            df.loc[min_index,'max_value'] = df.loc[min_index + 1, 'max_value']
            df.drop([min_index + 1],inplace=True)
            df.reset_index(inplace=True,drop=True)
        else:
            break
    bins = [feature_min - 0.0001]
    bins.extend([i for i in df['max_value'].values])
    bins[-1] = bins[-1] + 0.0001
    out, bins = pd.cut(x, bins, labels=False, retbins=True)
    if nan is not None:
        out = fillna(out, nan).astype(np.int)

    if return_bins:
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
    elif method == 'chi_square':
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