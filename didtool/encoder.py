# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from sklearn.utils import column_or_1d

from .metric import woe, probability


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    WOE Encoder
    Encode categorical features as woe values.

    Attributes
    ----------
    _woe_map : dict of woe values.
    _nan_value : key for NaN value
    """

    def __init__(self):
        # init here and updated when `fit` called
        self._woe_map = {}
        self._nan_value = 'NA'

    def fit(self, x, y):
        """
        Fit the WOEEncoder to x and y.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        y : numpy.ndarray
            the target's value

        Returns
        -------
        self
        """
        self._woe_map = {}
        uniques = np.unique(x[~pd.isnull(x)])
        for value in uniques:
            prob1, prob0 = probability(y, group_mask=(x == value))
            self._woe_map[value] = woe(prob1, prob0)

        if np.any(pd.isnull(x)):
            prob1, prob0 = probability(y, group_mask=(pd.isnull(x)))
            self._woe_map[self._nan_value] = woe(prob1, prob0)
        return self

    def transform(self, x, default=0.0):
        """
        transform function for single feature

        Parameters
        ----------
        x : numpy.ndarray
            value to transform

        default : float
            the default woe value for unknown group

        Returns
        ----------
        res : 2d array
        """
        res = np.zeros(len(x))

        # replace unknown group to default value
        res[np.isin(x, self._woe_map.keys(), invert=True)] = default

        for key in self._woe_map.keys():
            if key == self._nan_value:
                res[pd.isnull(x)] = self._woe_map[self._nan_value]
            else:
                res[x == key] = self._woe_map[key]

        return res.reshape(-1, 1)


class WrappedLabelEncoder(PMMLLabelEncoder):
    """
    overwrite PMMLLabelEncoder
    在刷分时，如果类别特征出现新的类别，可以当做缺失值处理，防止崩溃
    """
    def transform(self, x):
        x = column_or_1d(x, warn=True)
        index = list(self.classes_)
        xt = np.array(
            [self.missing_values if pd.isnull(v) or v not in index
             else index.index(v) for v in x])

        return xt.reshape(-1, 1)
