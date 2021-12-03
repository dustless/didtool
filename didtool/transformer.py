# coding: utf-8
from multiprocessing import Pool, cpu_count
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
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
    is_continuous : bool
        whether the feature fitted is continuous
    Attributes
    --------
    bins : list of thresholds
        After fitted, `bins` used to cut values when transform func is called

    woe_map : dict
        map of bins_num to woe value

    var_name : str
        feature name

    woe_df: DataFrame
        detail info of buckets

    value2code : dict
        when data is not continuous, encode the data

    code2value : dict
        when "plot_woe", decode the data
    """

    def __init__(self, cut_method='dt', n_bins=DEFAULT_BINS, is_continuous=True):
        self.cut_method = cut_method
        self.n_bins = n_bins

        # init here and updated when `fit` called
        self.bins = []
        self.woe_map = {}
        self.is_continuous = is_continuous
        self.var_name = 'x'

        # DataFrame store detail of bins
        self.woe_df = None
        self.value2code = {}
        self.code2value = {}

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
        if not self.is_continuous or is_categorical(x):

            tmp_df = pd.DataFrame({var_name: x, 'label': y})
            tmp_df.loc[:, var_name] = tmp_df[var_name].astype(str).replace(['nan', ''], np.nan)

            self.value2code = {k: v for v, k in
                               enumerate(tmp_df.groupby(var_name).label.mean().sort_values().index)}
            # 该map用来解析编码的
            self.code2value = {v: k for k, v in self.value2code.items()}
            # 对原数据进行编码
            x = np.array([self.value2code.get(str(item), np.nan) for item in x])

        self.var_name = var_name
        self.bins = []
        self.woe_map = {}

        woe_bins = []
        x, bins = cut(x, y, n_bins=self.n_bins, method=self.cut_method,
                      return_bins=True)

        if self.is_continuous:
            bins[0] = -np.inf
            bins[-1] = np.inf
        else:
            bins[0] = -99
            bins[-1] = max(self.code2value.keys())
        self.bins = bins
        if any(x == -1):
            woe_bins.append('NA')
        for i in range(len(bins) - 1):
            val = []
            if self.is_continuous:
                woe_bins.append('(%.4f, %.4f]' % (bins[i], bins[i + 1]))
            else:
                for item in range(max(0, int(bins[i] + 1)), int(bins[i + 1]) + 1):
                    val.append(self.code2value.get(item))
                woe_bins.append(val)

        value = np.sort(np.unique(x))
        num_bins = len(value)
        group_count = np.zeros(num_bins, dtype=np.int64)
        group_rate = np.zeros(num_bins)
        positive_count = np.zeros(num_bins, dtype=np.int64)
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
        if not self.is_continuous or is_categorical(x):
            x = np.array([self.value2code.get(str(item), np.nan) for item in x])

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


def _create_and_fit_transformer(cut_method, n_bins, x, y, name, is_continuous):
    transformer = SingleWOETransformer(cut_method, n_bins, is_continuous)
    transformer.fit(x, y, name)
    return transformer


class WOETransformer(TransformerMixin):
    """
    WOE transformer

    Parameters
    ----------
    features : list of str
        columns to woe transformer

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

    def __init__(self, cut_method='dt', n_bins=DEFAULT_BINS, features=None):
        self.features = features
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
        # 若不指定特征，则对全部列进行编码
        if not self.features:
            self.features = x.columns.to_list()

        for feature in self.features:
            if feature not in x.columns:
                raise Exception('{} not in x'.format(feature))

        # use multi-process
        res = []
        pool = Pool(cpu_count())

        for feature in self.features:
            r = pool.apply_async(
                _create_and_fit_transformer,
                args=(self.cut_method, self.n_bins, x[feature], y, feature, False))
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
        for feature in self.features:
            if feature not in x.columns:
                raise Exception('{} not in x'.format(feature))

        res = {}
        for col in x.columns:
            if col in self.features:
                tmp = self.transformers[col].transform(x[col], default)
            else:
                tmp = x[col].to_list()
            res[col] = tmp

        return pd.DataFrame(res)


class CategoryTransformer(TransformerMixin, BaseEstimator):
    """
    Category Transformer

    Attributes
    --------
    map_encoder : dict of encoder
        After fitted, `map_encoder` used to encode oot data

    df_encoder : pd.DataFrame
        Easy for consumers to persist the encoding table

    nan_value : 'nan'
        The default fill value for empty values
    """

    def __init__(self):
        self.map_encoder = {}
        self.df_encoder = pd.DataFrame()
        self.nan_value = 'nan'

    def fit(self, x, y=None, max_bins=None, min_coverage=None):
        """
        fit category transformer

        Parameters
        ----------
        x: pd.DataFrame
            data to fit transformer

        y: placeholder to support pipeline

        max_bins: None or int
            max category of every encoded column
            if 'max_bins' is None,
            'min_coverage' will determines the numbers of category

        min_coverage: None or float
            min coverage of every encoded column
            if 'max_bins' is not None,
            'max_bins' will determines the numbers of category
            when max_bins and min_coverage are both None,
            numbers of category no limit
        """
        for col in x.columns:
            has_nan = x[col].isnull().any()

            df_tmp = pd.DataFrame(x[col].value_counts())
            df_tmp.reset_index(inplace=True)
            df_tmp.columns = [col, 'cnt']
            n_bins = df_tmp.shape[0]
            if max_bins:
                n_bins = min(n_bins, max_bins)
            elif min_coverage:
                cnt = 0
                for i, cnt_tmp in enumerate(df_tmp.cnt.tolist()):
                    cnt += cnt_tmp
                    if cnt >= x.shape[0] * min_coverage:
                        n_bins = i + 1
                        break
            map_encoder = {
                key: val + 1 for val, key in enumerate(
                    df_tmp.iloc[:n_bins][col].tolist())
            }
            # 为什么others编码和最后一个分箱编码相同？主要是为了避免others不存在的情况
            map_encoder.update({'others': n_bins})

            # encode np.nan if this column has np.nan
            if has_nan:
                map_encoder.update({self.nan_value: 0})

            df_encoder = pd.DataFrame(pd.Series(map_encoder))
            df_encoder.reset_index(inplace=True)
            df_encoder.columns = [col, col + '_encoder']
            self.df_encoder = pd.concat([self.df_encoder, df_encoder],
                                        axis=1)
            # self.df_encoder.replace([self.nan_value], np.nan, inplace=True)
            self.map_encoder.update({col: map_encoder})

        return self

    def transform(self, x) -> pd.DataFrame:
        """
        transform function for all columns needed category encode

        Parameters
        ----------
        x: pd.DataFrame
            data to transform

        Returns
        -------
        pd.DataFrame with category encoded
        """
        x = x.copy()
        for key in self.map_encoder:
            if key not in x.columns:
                raise Exception('%s not in x' % key)
        for col in self.map_encoder:
            default_val = self.map_encoder.get(col).get('others')
            x.loc[:, col] = x[col].fillna(
                self.nan_value).apply(
                lambda s: self.map_encoder.get(
                    col).get(s, default_val)).astype('category')

        return x


class OneHotTransformer(TransformerMixin, BaseEstimator):
    """
    OneHot Transformer

    Attributes
    --------
    map_encoder : dict of encoder
        After fitted, `map_encoder` used to encode oot data

    _features_length : int
        length of fitted train data

    nan_value : 'nan'
        The default fill value for empty values
    """

    def __init__(self):
        self.map_encoder = {}
        self._features_length = 0
        self.nan_value = 'nan'

    def fit(self, x, y=None, max_bins=None, min_coverage=None):
        """
        fit oneHot transformer

        Parameters
        ----------
        x: pd.DataFrame
            data to fit transformer

        y: placeholder to support pipeline

        max_bins: None or int
            max category of every encoded column
            if 'max_bins' is None,
            'min_coverage' will determines the numbers of category

        min_coverage: None or float
            min coverage of every encoded column
            if 'max_bins' is not None,
            'max_bins' will determines the numbers of category
            when max_bins and min_coverage are both None,
            numbers of category no limit
        """
        for col in x.columns:
            has_nan = x[col].isnull().any()

            df_tmp = pd.DataFrame(x[col].value_counts())
            df_tmp.reset_index(inplace=True)
            df_tmp.columns = [col, 'cnt']
            n_bins = df_tmp.shape[0]
            if max_bins:
                n_bins = min(n_bins, max_bins)
            elif min_coverage:
                cnt = 0
                for i, cnt_tmp in enumerate(df_tmp.cnt.tolist()):
                    cnt += cnt_tmp
                    if cnt >= x.shape[0] * min_coverage:
                        n_bins = i + 1
                        break
            col_vals = df_tmp.iloc[:n_bins][col].tolist() + ['others']

            # encode np.nan if this column has np.nan
            if has_nan:
                col_vals.append(self.nan_value)

            self.map_encoder.update({col: col_vals})
            self._features_length += len(col_vals)

        return self

    def transform(self, x) -> pd.DataFrame:
        """
        transform function for all columns needed oneHot encode

        Parameters
        ----------
        x: pd.DataFrame
            data to transform

        Returns
        -------
        pd.DataFrame with oneHot encoded
        """
        x = x.fillna(self.nan_value)
        for key in self.map_encoder:
            if key not in x.columns:
                raise Exception('%s not in x' % key)

        zero_matrix = np.zeros(shape=(x.shape[0], self._features_length),
                               dtype=np.int8)

        cols = []
        i = 0
        for col in self.map_encoder:
            x[col] = x[col].apply(
                lambda s: 'others' if s not in self.map_encoder[col] else s)
            for val in self.map_encoder.get(col):
                cols.append(col + '_' + str(val))
                zero_matrix[:, i] = [
                    np.int8(1) if str(val) == str(item) else np.int8(0) for item
                    in x[col]]
                i += 1
            del x[col]
        dummies = pd.DataFrame(zero_matrix, columns=cols)
        return dummies


class ListTransformer(TransformerMixin, BaseEstimator):
    """
    List Transformer
    Expand list values to columns

    example:
      "0,1,4" -> [1,1,0,0,1]
      "0:10,1:5,4:20" -> [10,5,0,0,20]

    Attributes
    --------
    map_encoder : dict of encoder
        After fitted, `map_encoder` used to encode oot data

    sep : str
        separator of list value

    sub_sep : str, default None
        sub-separator of list value
    """

    def __init__(self, sep=',', sub_sep=None):
        self.map_encoder = {}
        self.sep = sep
        self.sub_sep = sub_sep

    def fit(self, x, y=None, max_bins=None):
        """
        fit List Transformer

        Parameters
        ----------
        x: pd.DataFrame
            data to fit transformer

        y: placeholder to support pipeline

        max_bins: None or int
            max number of encoding bins
        """
        for col in x.columns:
            if col not in x.columns:
                raise Exception("%s not in x" % col)

        for col in x.columns:
            x_col = x[col].dropna().tolist()

            feat_counter = Counter()
            for item_list_str in x_col:
                item_list = item_list_str.split(self.sep)
                # 如果有二级分隔符，则列名取二级分隔符左值
                if self.sub_sep:
                    feat_counter.update([item.split(self.sub_sep)[0]
                                         for item in item_list])
                else:
                    feat_counter.update(item_list)

            if max_bins:
                feat_names = \
                    [item[0] for item in feat_counter.most_common(max_bins)]
            else:
                feat_names = feat_counter.keys()

            feat_names = sorted(feat_names)
            self.map_encoder.update({col: feat_names})

        return self

    def transform(self, x) -> pd.DataFrame:
        """
        transform function for all columns needed oneHot encode

        Parameters
        ----------
        x: pd.DataFrame
            data to transform

        Returns
        -------
        pd.DataFrame with list encoded
        """
        for key in self.map_encoder:
            if key not in x.columns:
                raise Exception('%s not in x' % key)

        res = []
        for i in range(x.shape[0]):
            row_res = {}
            for col in self.map_encoder:
                if pd.isna(x.loc[i, col]):
                    continue
                item_list = x.loc[i, col].split(self.sep)
                if self.sub_sep:
                    item_dict = {
                        item.split(self.sub_sep)[0]: item.split(self.sub_sep)[1]
                        for item in item_list
                    }
                    for feat in self.map_encoder[col]:
                        row_res.update({
                            "%s_%s" % (col, feat): float(item_dict.get(feat, 0))
                        })
                else:
                    for feat in self.map_encoder[col]:
                        row_res.update({
                            "%s_%s" % (col, feat): int(feat in item_list)
                        })
            res.append(row_res)

        feat_names = ["%s_%s" % (c, f) for c in self.map_encoder
                      for f in self.map_encoder[c]]
        res_df = pd.DataFrame(res, columns=sorted(feat_names))
        return res_df
