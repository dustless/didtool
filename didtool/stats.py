from multiprocessing import Pool, cpu_count

import pandas as pd

from .metric import iv
from .utils import is_categorical


def iv_with_name(x, y, name='feature'):
    """
    Compute IV for continuous feature.
    Parameters
    ----------
    x : array-like
    y: array-like
    name: feature name
    is_continuous : whether x is continuous, optional (default=True)

    Returns
    -------
    [name, iv] : feature name and IV of feature x
    """
    is_continuous = not is_categorical(x)
    iv_value = iv(x, y, is_continuous)
    return [name, iv_value]


def iv_all(frame, target='target', exclude_cols=None):
    """get iv of features in frame

    Args:
        frame (DataFrame): frame that will be calculate quality
        target (str): the target's name in frame
        exclude_cols

    Returns:
        DataFrame: quality of features with the features' name as row name
    """
    res = []
    pool = Pool(cpu_count())

    y = frame[target]
    for name, x in frame.iteritems():
        if name != target or (exclude_cols and name not in exclude_cols):
            r = pool.apply_async(iv_with_name, args=(x, y), kwds={'name': name})
            res.append(r)

    pool.close()
    pool.join()

    rows = [r.get() for r in res]

    return pd.DataFrame(rows, columns=["feature", "iv"]).sort_values(
        by='iv',
        ascending=False,
    )
