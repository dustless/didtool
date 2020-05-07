from multiprocessing import Pool, cpu_count

import pandas as pd

from .metric import iv
from .utils import is_categorical


def iv_with_name(x, y, name='feature', **kwargs):
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
            r = pool.apply_async(iv_with_name, args=(x, y), kwds=kwds)
            res.append(r)

    pool.close()
    pool.join()

    rows = [r.get() for r in res]

    return pd.DataFrame(rows, columns=["feature", "iv"]).sort_values(
        by='iv',
        ascending=False,
    ).set_index('feature')
