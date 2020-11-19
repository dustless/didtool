import os
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from didtool.transformer import SingleWOETransformer, WOETransformer, \
    CategoryTransformer


class TestTransformer(unittest.TestCase):
    def test_single_woe_transformer(self):
        df = pd.read_csv('samples.csv')
        x = df['v1']
        y = df['target']

        transformer = SingleWOETransformer()
        transformer.fit(x, y, 'v1')
        self.assertListEqual(
            list(np.round(transformer.bins, 5)),
            [-np.inf, 0.00455, 0.00485, 0.0072, 0.01415, 0.01485, 0.0212,
             0.02815, 0.03165, 0.04235, np.inf]
        )
        self.assertDictEqual(
            transformer.woe_map,
            {-1: -1.0171553366121715, 0: -0.10844300821451114,
             1: 2.825413861621392, 2: 0.5741220630148971, 3: 2.621814906380153,
             4: 3.924026150289502, 5: 1.4391195005015018, 6: 2.7384024846317625,
             7: 0.340507211833392, 8: 2.1322666810614472,
             9: -0.6403220411783341}
        )
        self.assertAlmostEqual(transformer.woe_df['var_iv'][0], 1.878709, 6)
        self.assertEqual(transformer.woe_df.shape[0], 11)
        self.assertEqual(transformer.var_name, 'v1')

        res = transformer.transform(np.array([0.02, 0.05, np.nan]))
        self.assertAlmostEqual(res[0], 1.439120, 6)
        self.assertAlmostEqual(res[1], -0.640322, 6)
        self.assertAlmostEqual(res[2], -1.017155, 6)

        # transformer.plot_woe()

        # fit another categorical value
        x = df['v5'].astype('category')
        transformer.fit(x, y, 'v5')
        self.assertEqual(transformer.var_name, 'v5')
        self.assertEqual(transformer.is_continuous, False)
        self.assertListEqual(transformer.bins, [])
        self.assertDictEqual(transformer.woe_map,
                             {0: -0.21690835519242824, 1: 0.48454658205632983})

        res = transformer.transform(np.array([0, 1, -1]))
        self.assertAlmostEqual(res[0], -0.216908, 6)
        self.assertAlmostEqual(res[1], 0.484547, 6)
        self.assertAlmostEqual(res[2], 0)

    def test_woe_transformer(self):
        df = pd.read_csv('samples.csv')
        x = df[['v1', 'v5']]
        y = df['target']
        x['v5'] = x['v5'].astype('category')

        transformer = WOETransformer()
        transformer.fit(x, y)
        print(transformer.woe_df)

        test_x = pd.DataFrame({'v1': [0.02, 0.05, np.nan], 'v5': [0, 1, -1]})
        res = transformer.transform(test_x)
        self.assertAlmostEqual(res['v1'][0], 1.439120, 6)
        self.assertAlmostEqual(res['v1'][1], -0.640322, 6)
        self.assertAlmostEqual(res['v1'][2], -1.017155, 6)
        self.assertAlmostEqual(res['v5'][0], -0.216908, 6)
        self.assertAlmostEqual(res['v5'][1], 0.484547, 6)
        self.assertAlmostEqual(res['v5'][2], 0)

    def test_category_encode(self):
        df = pd.DataFrame({
            'x1': [1, 2, 1, 2, 1, 7.3, 0, 0, 0, 0, np.nan],
            'x2': ['北京', '上海', '上海', '山东', '北京', '北京',
                   np.nan, np.nan, np.nan, np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'x4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        df_except = pd.DataFrame({
            'x1_encoder': [1, 2, 1, 2, 1, 3, 0, 0, 0, 0, 4],
            'x2_encoder': [0, 1, 1, 2, 0, 0, 3, 3, 3, 3, 3],
            'x3_encoder': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'x4_encoder': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
        for col in df_except.columns:
            df_except[col] = df_except[col].astype('category')

        df_encoder = pd.DataFrame({
            'x1': [0.0, 1.0, 2.0, 7.3, 'others', np.nan],
            'x1_encoder': [0, 1, 2, 3, 3, 4],
            'x2': ['北京', '上海', '山东', 'others', np.nan, np.nan],
            'x2_encoder': [0.0, 1.0, 2.0, 2.0, 3.0, np.nan],
            'x3': ['others', np.nan, np.nan, np.nan, np.nan, np.nan],
            'x3_encoder': [0.0, 0.0, np.nan, np.nan, np.nan, np.nan],
            'x4': [1, 'others', np.nan, np.nan, np.nan, np.nan],
            'x4_encoder': [0.0, 0.0, np.nan, np.nan, np.nan, np.nan]
        })

        for col in df_except.columns:
            df_encoder[col] = df_encoder[col].astype('category')

        df_te = pd.DataFrame({
            'x1': [1, 2, 1, 0, np.nan],
            'x2': ['北京', '上海', '山东', np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan, ],
            'x4': [1, 1, 1, 1, 1]
        })

        df_te_except = pd.DataFrame({
            'x1_encoder': [1, 2, 1, 0, 4],
            'x2_encoder': [0, 1, 2, 3, 3],
            'x3_encoder': [0, 0, 0, 0, 0],
            'x4_encoder': [0, 0, 0, 0, 0]
        })

        ct = CategoryTransformer()
        ct.fit_transform(df, columns=df.columns, max_bins=64)

        df = ct.transform(df)
        df_te = ct.transform(df_te)

        self.assertListEqual(df.x1_encoder.to_list(),
                             df_except.x1_encoder.to_list())
        self.assertListEqual(df.x2_encoder.to_list(),
                             df_except.x2_encoder.to_list())
        self.assertListEqual(df.x3_encoder.to_list(),
                             df_except.x3_encoder.to_list())
        self.assertListEqual(df.x4_encoder.to_list(),
                             df_except.x4_encoder.to_list())

        np.testing.assert_array_equal(ct.df_encoder.x1.to_list(),
                                      df_encoder.x1.to_list())
        np.testing.assert_array_equal(ct.df_encoder.x2.to_list(),
                                      df_encoder.x2.to_list())
        np.testing.assert_array_equal(ct.df_encoder.x3.to_list(),
                                      df_encoder.x3.to_list())
        np.testing.assert_array_equal(ct.df_encoder.x4.to_list(),
                                      df_encoder.x4.to_list())

        np.testing.assert_array_equal(ct.df_encoder.x1_encoder.to_list(),
                                      df_encoder.x1_encoder.to_list())
        np.testing.assert_array_equal(ct.df_encoder.x2_encoder.to_list(),
                                      df_encoder.x2_encoder.to_list())
        np.testing.assert_array_equal(ct.df_encoder.x3_encoder.to_list(),
                                      df_encoder.x3_encoder.to_list())
        np.testing.assert_array_equal(ct.df_encoder.x4_encoder.to_list(),
                                      df_encoder.x4_encoder.to_list())

        self.assertListEqual(df_te.x1_encoder.to_list(),
                             df_te_except.x1_encoder.to_list())
        self.assertListEqual(df_te.x2_encoder.to_list(),
                             df_te_except.x2_encoder.to_list())
        self.assertListEqual(df_te.x3_encoder.to_list(),
                             df_te_except.x3_encoder.to_list())
        self.assertListEqual(df_te.x4_encoder.to_list(),
                             df_te_except.x4_encoder.to_list())
