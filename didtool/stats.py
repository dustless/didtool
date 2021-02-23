# coding: utf-8
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

from .metric import iv, psi
from .utils import is_categorical, handle_categorical_value


def _iv_with_name(x, y, name='feature', **kwargs):
    """
    Compute IV for continuous feature.
    Parameters
    ----------
    x : array-like
    y: array-like
    name: feature name

    Returns
    -------
    [name, iv] : feature name and IV of feature x
    """
    is_continuous = not is_categorical(x)
    if not is_continuous:
        x = handle_categorical_value(x)

    iv_value = iv(x, y, is_continuous, **kwargs)
    return [name, iv_value]


def iv_all(frame, y, exclude_cols=None, **kwargs):
    """
    Compute IV of features in frame

    Parameters
    ----------
    frame : DataFrame
        frame that will be calculate iv
    y : array-like
        the target's value
    exclude_cols: list, optional(default None)
        columns that do not need to calculate iv

    Returns
    -------
    DataFrame: iv of features with the features' name as row index
    """
    res = []
    pool = Pool(cpu_count())

    for name, x in frame.iteritems():
        if not (exclude_cols and name in exclude_cols):
            kwds = kwargs.copy()
            kwds['name'] = name
            r = pool.apply_async(_iv_with_name, args=(x, y), kwds=kwds)
            res.append(r)

    pool.close()
    pool.join()

    rows = [r.get() for r in res]

    return pd.DataFrame(rows, columns=["feature", "iv"]).sort_values(
        by='iv',
        ascending=False,
    ).set_index('feature')


def psi_all(data, features=None, group_col='month',
            expected_data=None):
    """
    Compute psi of features group by group if `expected_data` is None.
    If `expected_data` specified, compute psi for each group compared to
    expected data.

    Parameters
    ----------
    data : DataFrame
        frame that will be calculate psi
    features : array-like, default=None
        features that need to be analyzed. All features should be in `data`.
        If `feature_names` not specified, all feature
    group_col: str, default='month'
        group column name used to group data. This colume should be in `data`.
    expected_data: DataFrame, default=None
        expected data specified, same columns as data

    Returns
    -------
    DataFrame: psi of features in each group with the feature_names as row index
    """
    if group_col not in data:
        raise Exception("%s is not in `data`" % group_col)

    if not features:
        features = [col for col in data.columns.values if col != group_col]
    else:
        for feature in features:
            if feature not in data:
                raise Exception("%s is not in `data`" % feature)

    if expected_data is not None:
        for col in features:
            if col not in expected_data:
                raise Exception("%s is not in `expected_data`" % col)

    group_by_group = expected_data is None  # compute method
    groups = np.sort(data[group_col].unique())
    psi_df = pd.DataFrame(columns=groups, index=features)

    for i in range(0, len(groups)):
        if group_by_group:
            if i == 0:
                continue
            expected_data = data[data[group_col] == groups[i - 1]]
        actual = data[data[group_col] == groups[i]]
        psi_df.loc[:, groups[i]] = \
            [psi(expected_data[col], actual[col],
                 is_continuous=not is_categorical(data[col]))
             for col in features]

    return psi_df
