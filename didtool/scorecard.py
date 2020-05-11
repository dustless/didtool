# -*- coding:utf-8 -*-
import os
import math

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ScoreCardTransformer(BaseEstimator, TransformerMixin):
    """
    Transform 0~1 probability value to integer score

    Parameters
    ----------
    n_bins : int(default=20)
        Defines the number of equal-width bins in the range of [0.0, 1.0).

    standard_score : int(default=500)
        standard score when odds is equal to `standard_odds`

    standard_odds : int or float(default=0.01)
        equals to real odds. Odds means `good_rate / bad_rate`

    pdo : int(default=20)
        when odds doubles, the score should increase `pdo`

    bad_flag : bool(default=True)
        Whether label=1 indicates a good user. In credit model, label=1
        usually indicates that user is bad because of overdue order.

    Attributes
    --------
    _step: float
        equals to 1.0 / `self._n_bins`

    mapping_df : DataFrame
        records slope and intercept of each bin
    """

    def __init__(self, n_bins=20, standard_score=500, standard_odds=0.01,
                 pdo=20, bad_flag=True):
        self._n_bins = n_bins
        self._standard_score = standard_score
        self._standard_odds = standard_odds
        self._pdo = pdo
        self._bad_flag = bad_flag

        self._step = 1.0 / self._n_bins
        self.binning_df = None
        self.mapping_df = None

    def fit(self, x, y):
        """
        fit ScoreCardTransformer

        Parameters
        ----------
        x : numpy.array, 1-d array
            array of probability(0.0~1.0)

        y : numpy.array, 1-d array
            array of real binary labels
        """
        if not self.__is_one_dim_array(x) and not self.__is_one_dim_array(y):
            raise ValueError(r"x and y should be 1-dim array.")

        if x.shape[0] != y.shape[0]:
            raise ValueError(r"Row size of x(%s) and y(%s) do not match."
                             % (x.shape[0], y.shape[0]))

        self.binning_df = self.__calc_bins(x, y)
        self.mapping_df = self.__calc_mapping_df()
        return self

    def transform(self, x):
        """
        transform probability value to integer score

        Parameters
        ----------
        x : numpy.array, 1-d array
            array of probability(0.0~1.0)

        Returns
        ----------
        y : numpy.array, 1-d array
            integer scores with same shape as x.
        """
        data = pd.DataFrame({'prob': x})
        data['bin'] = data['prob'].apply(
            lambda x: int((x + self._step / 2) / self._step))
        data = data.merge(self.mapping_df[['slope', 'intercept']], how='left',
                          left_on='bin', right_index=True)
        res = (data['slope'] * data['prob'] + data['intercept']).apply(
            lambda x: int(round(x)))
        return res.values

    @staticmethod
    def __is_one_dim_array(arr):
        return isinstance(arr, np.ndarray) and arr.ndim == 1

    def __calc_bins(self, x, y):
        """
        execution of pre-binning and setting odds (good/bad) of each bin
        corresponding to the samples
        """
        score_df = pd.DataFrame({'prob': x, 'label': y})
        # if bad_flag=True, adjust prob
        if self._bad_flag:
            score_df['prob'] = 1.0 - score_df['prob']
        score_df['bin'] = score_df['prob'].apply(lambda x: int(x / self._step))

        binning_df = pd.DataFrame(
            columns=['prob_l', 'prob_r', 'mean_prob', 'score', 'bad_hits',
                     'good_hits', 'hits', 'odds'],
            index=range(self._n_bins)
        )

        binning_df['hits'] = score_df['label'].groupby(score_df['bin']).count()
        if self._bad_flag:  # if 1 represent bad label
            binning_df['bad_hits'] = \
                score_df['label'].groupby(score_df['bin']).sum()
            binning_df['good_hits'] = \
                binning_df['hits'] - binning_df['bad_hits']
        else:  # if 1 represent good label
            binning_df['good_hits'] = \
                score_df['label'].groupby(score_df['bin']).sum()
            binning_df['bad_hits'] = \
                binning_df['hits'] - binning_df['good_hits']

        binning_df['odds'] = binning_df['good_hits'] / (binning_df['bad_hits'])
        binning_df = self.__adjust_odds(binning_df)

        binning_df['prob_l'] = np.arange(0, 1, self._step)
        binning_df['prob_r'] = binning_df['prob_l'] + self._step
        # adjust prob should be done after odds adjusted
        if self._bad_flag:
            binning_df.sort_values(by='prob_l', ascending=False, inplace=True)
            binning_df['prob_l'] = np.arange(0, 1, self._step)
            binning_df['prob_r'] = binning_df['prob_l'] + self._step
            binning_df.reset_index(drop=True, inplace=True)

        # score = standard_score + pdo * ln(odds / standard_odds) / ln(2)
        # reference: (https://zhuanlan.zhihu.com/p/82670834)
        binning_df['score'] = binning_df['adjusted_odds'].apply(
            lambda x: int(self._standard_score +
                          self._pdo * math.log2(x / self._standard_odds)))
        binning_df['mean_prob'] = \
            (binning_df['prob_l'] + binning_df['prob_r']) / 2
        return binning_df

    def __adjust_odds(self, df):
        """
        adjust odds of each bin when it face to the following scenarios:
            (1) bad hits is zero in the high p-value bins
            (2) good hits is zero in the low p-value bins
            (3) good/bad hits is zero in the middle bins
        """
        df['adjusted_odds'] = df['odds']
        df['adjusted_odds'][np.isinf(df['adjusted_odds'])] = 0
        df['adjusted_odds'].fillna(0, inplace=True)
        max_odds = max(df['adjusted_odds'])
        max_odds_index = df['adjusted_odds'].argmax()
        min_odds = min(df[df['adjusted_odds'] != 0]['adjusted_odds'])
        min_odds_index = df[df['adjusted_odds'] != 0]['adjusted_odds'].argmin()

        # adjust odds in bins with zero good hits from min_odds_index to 0
        is_zero_good = False
        for i in range(min_odds_index - 1, -1, -1):
            if df['good_hits'][i] == 0.0:
                is_zero_good = True
            if is_zero_good:
                min_odds /= 2
                df['adjusted_odds'][i] = min_odds

        # adjust odds in bins with zero bad hits from max_odds_index to n_bins
        is_zero_bad = False
        for i in range(max_odds_index + 1, self._n_bins):
            if df['bad_hits'][i] == 0.0:
                is_zero_bad = True
            if is_zero_bad:
                max_odds *= 2
                df['adjusted_odds'][i] = max_odds

        # adjust odds in bins from min_odds_index to max_odds_index
        for i in range(min_odds_index + 1, max_odds_index - 1):
            if df['adjusted_odds'][i] == 0.0:
                if df['adjusted_odds'][i + 1] != 0.0:
                    df['adjusted_odds'][i] = (df['adjusted_odds'][i - 1] +
                                              df['adjusted_odds'][i + 1]) / 2
                else:
                    df['adjusted_odds'][i] = df['adjusted_odds'][i - 1]
        return df

    def __calc_mapping_df(self):
        """
        Use (mean_prob, score) in all bins as anchor nodes, link all anchor
        nodes, then we get a sectional-continuous function from prob value
        to integer score.
        """
        anchor = pd.DataFrame(
            columns=['prob_l', 'score_l', 'prob_r', 'score_r'],
            index=range(self._n_bins + 1))
        anchor['prob_l'][1:self._n_bins + 1] = self.binning_df['mean_prob']
        anchor['score_l'][1:self._n_bins + 1] = self.binning_df['score']
        anchor['prob_r'][0:self._n_bins] = self.binning_df['mean_prob']
        anchor['score_r'][0:self._n_bins] = self.binning_df['score']
        anchor['prob_l'][0] = 0
        anchor['prob_r'][self._n_bins] = 1
        if self._bad_flag:
            anchor['score_l'][0] = max(self.binning_df['score']) + self._pdo
            anchor['score_r'][self._n_bins] = \
                min(self.binning_df['score']) - self._pdo / 2
        else:
            anchor['score_l'][0] = min(self.binning_df['score']) - self._pdo
            anchor['score_r'][self._n_bins] = \
                max(self.binning_df['score']) + self._pdo / 2

        den = anchor['prob_r'] - anchor['prob_l']

        mapping_df = pd.DataFrame(index=range(self._n_bins + 1),
                                  columns=['slope', 'intercept'])
        mapping_df['slope'] = (anchor['score_r'] - anchor['score_l']) / den
        mapping_df['intercept'] = (anchor['prob_r'] * anchor['score_l'] -
                                   anchor['prob_l'] * anchor['score_r']) / den
        return mapping_df

    def plot_bins(self):
        """
        plot hit_rate, pos_rate and score of bins
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(range(self._n_bins), self.binning_df['score'], 'og-',
                 label='score')
        for i in range(self._n_bins):
            ax1.text(i, self.binning_df['score'][i],
                     self.binning_df['score'][i], color='black', fontsize=10)
        ax1.legend(loc=1)
        ax1.set_ylabel('score')
        ax1.set_ylim([self.binning_df['score'].min() - 50,
                      self.binning_df['score'].max() + 50])

        # Create a twin of Axes with a shared x-axis but independent y-axis.
        ax2 = ax1.twinx()
        hit_rate = self.binning_df['hits'] / self.binning_df['hits'].sum()
        pos_rate = (self.binning_df['bad_hits'] if self._bad_flag else
                    self.binning_df['good_hits']) / self.binning_df['hits']

        ax2.bar([i - 0.2 for i in range(self._n_bins)], hit_rate, alpha=0.5,
                color='blue', width=0.4, label='hit_rate')
        ax2.bar([i + 0.2 for i in range(self._n_bins)], pos_rate, alpha=0.5,
                color='red', width=0.4, label='pos_rate')
        ax2.legend(loc=2)
        ax2.set_ylim([0, 1])
        plt.xticks(range(self._n_bins), range(self._n_bins))
        plt.show()

    def export_mapping(self, out_path, file_name='map_score_card.csv'):
        """
        save mapping_df to file
        """
        self.mapping_df.to_csv(os.path.join(out_path, file_name), index=False)
