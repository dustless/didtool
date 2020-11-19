# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：didtool -> encoder
@IDE    ：PyCharm
@Author ：ysjin
@Date   ：2020/11/17 15:30
@Desc   ：
=================================================="""
import pandas as pd
import numpy as np
from typing import Union


class Encoder:
    """
    """

    def __init__(self, data):
        self.data = data
        self.df_encoder = pd.DataFrame()

    def one_hot_encode(self, columns: Union[list, str]) -> pd.DataFrame:
        if isinstance(columns, str):
            # columns = [columns]
            # todo
            pass
        return self.data

    def category_encode(self, columns: Union[list, str], max_bins=None,
                        min_coverage=None, nan_value='-999') -> pd.DataFrame:

        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            # 判断是否有空缺值
            flag = self.data[col].isnull().any()
            if flag:
                # 填充空缺值
                self.data.loc[:, col] = self.data[col].fillna(nan_value)
            # 统计类别频率
            df_tmp = pd.DataFrame(self.data[col].value_counts())
            df_tmp.reset_index(inplace=True)
            df_tmp.columns = [col, 'cnt']
            n_bins = df_tmp.shape[0]
            if max_bins:
                # 以最大分箱数为限制编码
                n_bins = min(n_bins, max_bins)
            elif min_coverage:
                # 以最大分箱数为限制编码
                cnt = 0
                for i, cnt_tmp in enumerate(df_tmp.cnt.to_list()):
                    cnt += cnt_tmp
                    if cnt >= self.data.shape[0] * min_coverage:
                        n_bins = i + 1
                        break
            else:
                # 以全部类别全部编码（参考sklearn的OrdinalEncoder方法）
                n_bins = n_bins
            map_encode = {
                key: val for val, key in enumerate(
                    df_tmp.iloc[:n_bins - 1][col].to_list())
            }

            map_encode.update({'others': n_bins - 1})

            if nan_value not in map_encode and flag:
                map_encode.update({nan_value: n_bins})

            self.data.loc[:, col + '_encoder'] = self.data.loc[:, col].apply(
                lambda x: map_encode.get(x, n_bins - 1))
            del self.data[col]

            df_encoder = pd.DataFrame(pd.Series(map_encode))
            df_encoder.reset_index(inplace=True)
            df_encoder.columns = [col, col + '_encoder']
            self.df_encoder = pd.concat([self.df_encoder, df_encoder],
                                        axis=1)
            self.df_encoder.replace([nan_value], np.nan, inplace=True)

        return self.data
