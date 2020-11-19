# -*- coding: UTF-8 -*-
import unittest
import pandas as pd
import numpy as np
from didtool.encoder import Encoder


class TestEncoder(unittest.TestCase):
    def test_category_encode(self):
        df = pd.DataFrame({
            'x1': [1, 2, 1, 2, 1, 7.3, np.nan,
                   np.nan, np.nan, 0, np.nan],
            'x2': ['北京', '上海', '上海', '山东', '北京', '北京',
                   np.nan, np.nan, np.nan, np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        df_except = pd.DataFrame({
            'x1_encoder': [1, 2, 1, 2, 1, 3, 0, 0, 0, 4, 0],
            'x2_encoder': [1, 2, 2, 3, 1, 1, 0, 0, 0, 0, 0],
            'x3_encoder': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })
        df_encoder = pd.DataFrame({
            'x1': [np.nan, 1.0, 2.0, 7.3, 'others'],
            'x1_encoder': [0, 1, 2, 3, 4],
            'x2': [np.nan, '北京', '上海', 'others', np.nan],
            'x2_encoder': [0.0, 1.0, 2.0, 3.0, np.nan],
            'x3': ['others', np.nan, np.nan, np.nan, np.nan],
            'x3_encoder': [0.0, 1.0, np.nan, np.nan, np.nan]
        })

        e = Encoder(df)
        df = e.category_encode(columns=['x1', 'x2', 'x3'])

        self.assertListEqual(df.x1_encoder.to_list(),
                             df_except.x1_encoder.to_list())
        self.assertListEqual(df.x2_encoder.to_list(),
                             df_except.x2_encoder.to_list())
        self.assertListEqual(df.x3_encoder.to_list(),
                             df_except.x3_encoder.to_list())

        np.testing.assert_array_equal(e.df_encoder.x1.to_list(),
                                      df_encoder.x1.to_list())
        np.testing.assert_array_equal(e.df_encoder.x2.to_list(),
                                      df_encoder.x2.to_list())
        np.testing.assert_array_equal(e.df_encoder.x3.to_list(),
                                      df_encoder.x3.to_list())

        np.testing.assert_array_equal(e.df_encoder.x1_encoder.to_list(),
                                      df_encoder.x1_encoder.to_list())
        np.testing.assert_array_equal(e.df_encoder.x2_encoder.to_list(),
                                      df_encoder.x2_encoder.to_list())
        np.testing.assert_array_equal(e.df_encoder.x3_encoder.to_list(),
                                      df_encoder.x3_encoder.to_list())
