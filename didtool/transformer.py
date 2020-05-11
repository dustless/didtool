from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt

from .utils import is_categorical
from .cut import cut, DEFAULT_BINS, cut_with_bins
from .metric import woe, probability


class SingleWOETransformer(TransformerMixin):
    """
    Single WOE transformer

    Parameters
    ----------
    cut_method : str, optional (default='dt')
        Cut values into different buckets with specific method.
        Only used for continuous feature.
    n_bins : int, default DEFAULT_BINS
        Max num of buckets. Only used for continuous feature.

    Attributes
    --------
    bins : list of thresholds
        After fitted, `bins` used to cut values when transform func is called

    woe_map : dict
        map of bins_num to woe value

    is_continuous : bool
        whether the feature fitted is continuous

    var_name : str
        feature name

    woe_df: DataFrame
        detail info of buckets
    """

    def __init__(self, cut_method='dt', n_bins=DEFAULT_BINS):
        self.cut_method = cut_method
        self.n_bins = n_bins

        # init here and updated when `fit` called
        self.bins = []
        self.woe_map = {}
        self.is_continuous = True
        self.var_name = 'x'

        # DataFrame store detail of bins
        self.woe_df = None

    def fit(self, x, y, var_name='x'):
        """
        fit WOE transformer

        Parameters
        ----------
        x : numpy.ndarray
            the feature value for training
        y : numpy.ndarray
            the target's value
        var_name : str
            feature name of x
        """
        self.var_name = var_name
        self.is_continuous = not is_categorical(x)
        self.bins = []
        self.woe_map = {}

        woe_bins = []
        if self.is_continuous:
            x, bins = cut(x, y, n_bins=self.n_bins, method=self.cut_method,
                          return_bins=True)
            bins[0] = -np.inf
            bins[-1] = np.inf
            self.bins = bins
            if any(x == -1):
                woe_bins.append('NA')
            for i in range(len(bins) - 1):
                woe_bins.append('(%.4f, %.4f]' % (bins[i], bins[i + 1]))
        else:
            woe_bins = ['[%s]' % v for v in np.sort(np.unique(x))]

        value = np.sort(np.unique(x))
        num_bins = len(value)
        group_count = np.zeros(num_bins)
        group_rate = np.zeros(num_bins)
        positive_count = np.zeros(num_bins)
        positive_rate = np.zeros(num_bins)
        woe_list = np.zeros(num_bins)
        iv_list = np.zeros(num_bins)

        total = len(y)
        for i in range(num_bins):
            group_y = y[(x == value[i])]
            group_count[i] = len(group_y)
            group_rate[i] = group_count[i] / total
            positive_count[i] = (group_y == 1).sum()
            positive_rate[i] = positive_count[i] / group_count[i]

            prob1, prob0 = probability(y, group_mask=(x == value[i]))
            woe_list[i] = woe(prob1, prob0)
            iv_list[i] = (prob1 - prob0) * woe_list[i]
            self.woe_map[value[i]] = woe_list[i]

        self.woe_df = pd.DataFrame({
            'var_name': var_name,
            'bin_value': value,
            'bin_range': woe_bins,
            'group_count': group_count,
            'group_rate': group_rate,
            'positive_count': positive_count,
            'positive_rate': positive_rate,
            'woe': woe_list,
            'iv_list': iv_list,
            'var_iv': np.sum(iv_list)
        })
        return self

    def transform(self, x, default=0.0):
        """
        transform function for single feature

        Parameters
        ----------
        x : numpy.ndarray
            value to transform
        default : numpy.ndarray
            the default woe value for unknown group

        Returns
        ----------
        res : array-like
        """
        if self.is_continuous:
            x = cut_with_bins(x, self.bins)

        res = np.zeros(len(x))

        # replace unknown group to default value
        res[np.isin(x, self.woe_map.keys(), invert=True)] = default

        for key in self.woe_map.keys():
            res[x == key] = self.woe_map[key]

        return res

    def plot_woe(self):
        """
        plot details of bins
        """
        n_bins = self.woe_df.shape[0]

        fig = plt.figure()
        plt.xticks(range(n_bins), self.woe_df['bin_range'], rotation=90)
        plt.subplots_adjust(bottom=0.3)

        ax1 = fig.add_subplot(111)
        ax1.plot(range(n_bins), self.woe_df['woe'], 'og-', label='woe')
        ax1.plot(range(n_bins), self.woe_df['iv_list'], 'oy-', label='iv')
        ax1.axhline(y=0, ls=":", c="grey")
        ax1.legend(loc=1)
        ax1.set_ylabel('woe')
        ax1.set_ylim([self.woe_df['woe'].min() - 1,
                      self.woe_df['woe'].max() + 1])

        # Create a twin of Axes with a shared x-axis but independent y-axis.
        ax2 = ax1.twinx()
        ax2.bar([i - 0.2 for i in range(n_bins)], self.woe_df['group_rate'],
                alpha=0.5, color='blue', width=0.4, label='group_rate')
        ax2.bar([i + 0.2 for i in range(n_bins)], self.woe_df['positive_rate'],
                alpha=0.5, color='red', width=0.4, label='positive_rate')
        ax2.legend(loc=2)
        ax2.set_ylim([0, 1])

        plt.show()


def _create_and_fit_transformer(cut_method, n_bins, x, y, name):
    transformer = SingleWOETransformer(cut_method, n_bins)
    transformer.fit(x, y, name)
    return transformer


class WOETransformer(TransformerMixin):
    """
    WOE transformer

    Parameters
    ----------
    cut_method : str, optional (default='dt')
        Cut values into different buckets with specific method.
        Only used for continuous feature.
    n_bins : int, default DEFAULT_BINS
        Max num of buckets. Only used for continuous feature.

    Attributes
    --------
    transformers : dict, feature name -> object of SingleWOETransformer

    woe_df: DataFrame
        detail info of buckets
    """

    def __init__(self, cut_method='dt', n_bins=DEFAULT_BINS):
        self.cut_method = cut_method
        self.n_bins = n_bins
        self.transformers = {}
        self.woe_df = None

    def fit(self, x, y):
        """
        fit WOE transformer

        Parameters
        ----------
        x : DataFrame
            frame for training
        y : array-like
            the target's value
        """
        self.transformers = {}
        self.woe_df = None

        # use multi-process
        res = []
        pool = Pool(cpu_count())

        for name, v in x.iteritems():
            r = pool.apply_async(
                _create_and_fit_transformer,
                args=(self.cut_method, self.n_bins, v, y, name))
            res.append(r)

        pool.close()
        pool.join()

        transformers = [r.get() for r in res]
        for transformer in transformers:
            self.transformers[transformer.var_name] = transformer

        self.woe_df = pd.concat([t.woe_df for t in self.transformers.values()])
        return self

    def transform(self, x, default=0.0):
        """
        transform function for all features

        Parameters
        ----------
        x : DataFrame
            frame to transform
        default : float(default=0.0)
            the default woe value for unknown group

        Returns
        ----------
        res : DataFrame
        """
        res = {}
        for name, v in x.iteritems():
            if name in self.transformers:
                tmp = self.transformers[name].transform(v, default)
                res[name] = tmp
            else:
                raise Exception("column `%s` has not been fitted" % name)

        return pd.DataFrame(res)
