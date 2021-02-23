# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split


def split_data(data, train_mask, val_mask, group_col='group'):
    """
    Split data set into train/val/test part by specified masks.
    Add a `group_col` column into `data`:
        - 0: training data set
        - 1: validation data set
        - -1: testing data set

    Parameters
    --------
    data : DataFrame
        data to split

    train_mask : array or series of bool
        used to split train data set

    val_mask : array or series of bool
        used to split validation data set

    group_col : str(default='group')
        group column name

    Returns
    --------
    data : DataFrame
        data with `group_col` column
    """
    data[group_col] = -1
    data.loc[train_mask, group_col] = 0
    data.loc[val_mask, group_col] = 1
    return data


def split_data_random(data, train_size=0.6, val_size=0.2, group_col='group',
                      random_state=None):
    """
    Split data set into train/val/test part randomly

    Add a group column into data:
        - 0: training data set
        - 1: validation data set
        - -1: testing data set

    Parameters
    --------
    data : DataFrame
        data to split

    train_size : float, 0.0~1.0
        the proportion of the dataset to include in the train split

    val_size : float, 0.0~1.0
        the proportion of the dataset to include in the validation split

    group_col : str(default='group')
        group column name

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    --------
    data : DataFrame
        data with `group_col` column
    """
    if not 0 < train_size < 1:
        raise Exception("train_size should be in range (0.0, 1.0)")
    if not 0 < val_size < 1:
        raise Exception("val_size should be in range (0.0, 1.0)")
    if train_size + val_size > 1.0:
        raise Exception("train_size + val_size should not be greater than 1.0")

    train, val = train_test_split(data, train_size=train_size,
                                  random_state=random_state)
    # if train_size + val_size < 1.0,
    # then test_size = 1.0 - train_size - val_size
    if train_size + val_size < 1.0:
        val, _ = train_test_split(val, train_size=val_size / (1 - train_size),
                                  random_state=random_state)
    train_mask = np.zeros(data.shape[0], dtype=bool)
    train_mask[train.index] = True
    val_mask = np.zeros(data.shape[0], dtype=bool)
    val_mask[val.index] = True
    return split_data(data, train_mask, val_mask, group_col)


def split_data_stacking(data, oot_mask, n_fold=5, random_state=None,
                        group_col='group'):
    """
    Split data into train/oot part, and then split training dataset into
    `n_fold` part

    After split, a column named `group_col` will be append to `data`.
        - -1: oot data set
        - [0, n_fold): k-fold of training data set

    Parameters
    --------
    data : DataFrame
        data to split

    oot_mask : array or series of bool
        used to split oot dataset

    n_fold : int(default=5)
        split training data into `n_fold` folds

    random_state : int or None
        used for KFold

    group_col : str(default='group')
        group column name

    Returns
    --------
    data : DataFrame
        data with `group_col` column:
        - -1: oot data set
        - [0, n_fold): k-fold of training data set
    """
    train_data = data[~oot_mask]
    k_fold = KFold(n_splits=n_fold, shuffle=True,
                   random_state=random_state)
    k_fold_index = []
    for _, indexes in k_fold.split(train_data):
        k_fold_index.append(indexes)

    train_data.reset_index(inplace=True)
    train_data.loc[:, group_col] = -1
    for k in range(0, n_fold):
        train_data.loc[k_fold_index[k], group_col] = k

    data.reset_index(inplace=True)
    data = pd.merge(data, train_data[["index", group_col]],
                    how="left", on="index")
    data[group_col] = data[group_col].fillna(-1).astype(np.int)
    return data.drop("index", axis=1)
