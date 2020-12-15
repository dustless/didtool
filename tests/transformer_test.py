import unittest
import pandas as pd
import numpy as np

from didtool.transformer import SingleWOETransformer, WOETransformer, \
    CategoryTransformer, OneHotTransformer


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

    def test_category_transformer(self):
        df = pd.DataFrame({
            'x1': [1, 2, 1, 2, 1, 7.3, 0, 0, 0, 0, np.nan],
            'x2': ['北京', '上海', '上海', '山东', '北京', '北京',
                   np.nan, np.nan, np.nan, np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'x4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        df_except = pd.DataFrame({
            'x1': [2, 3, 2, 3, 2, 4, 1, 1, 1, 1, 0],
            'x2': [1, 2, 2, 3, 1, 1, 0, 0, 0, 0, 0],
            'x3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'x4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        for col in df_except.columns:
            df_except[col] = df_except[col].astype('category')

        df_encoder = pd.DataFrame({
            'x1': [0.0, 1.0, 2.0, 7.3, 'others', np.nan],
            'x1_encoder': [1, 2, 3, 4, 4, 0],
            'x2': ['北京', '上海', '山东', 'others', np.nan, np.nan],
            'x2_encoder': [1.0, 2.0, 3.0, 3.0, 0.0, np.nan],
            'x3': ['others', np.nan, np.nan, np.nan, np.nan, np.nan],
            'x3_encoder': [0.0, 0.0, np.nan, np.nan, np.nan, np.nan],
            'x4': [1, 'others', np.nan, np.nan, np.nan, np.nan],
            'x4_encoder': [1.0, 1.0, np.nan, np.nan, np.nan, np.nan]
        })

        for col in df_except.columns:
            df_encoder[col] = df_encoder[col].astype('category')

        df_test = pd.DataFrame({
            'x1': [1, 2, 1, 0, np.nan],
            'x2': ['北京', '上海', '山东', np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan, ],
            'x4': [1, 1, 1, 1, 1]
        })

        df_test_except = pd.DataFrame({
            'x1': [2, 3, 2, 1, 0],
            'x2': [1, 2, 3, 0, 0],
            'x3': [0, 0, 0, 0, 0],
            'x4': [1, 1, 1, 1, 1]
        })

        ct = CategoryTransformer()
        ct.fit(df, columns=df.columns.to_list(), max_bins=64)
        df = ct.transform(df)
        df_test = ct.transform(df_test)

        self.assertListEqual(df.x1.to_list(),
                             df_except.x1.to_list())
        self.assertListEqual(df.x2.to_list(),
                             df_except.x2.to_list())
        self.assertListEqual(df.x3.to_list(),
                             df_except.x3.to_list())
        self.assertListEqual(df.x4.to_list(),
                             df_except.x4.to_list())

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

        self.assertListEqual(df_test.x1.to_list(),
                             df_test_except.x1.to_list())
        self.assertListEqual(df_test.x2.to_list(),
                             df_test_except.x2.to_list())
        self.assertListEqual(df_test.x3.to_list(),
                             df_test_except.x3.to_list())
        self.assertListEqual(df_test.x4.to_list(),
                             df_test_except.x4.to_list())

    def test_onehot_transformer(self):
        df_train = pd.DataFrame({
            'x1': [1, 2, 1, 1, np.nan],
            'x2': ['河南省', np.nan, '浙江省', '福建省', np.nan]
        })

        df_train_encode = pd.DataFrame({'x1_1.0': [1, 0, 1, 1, 0],
                                        'x1_2.0': [0, 1, 0, 0, 0],
                                        'x1_nan': [0, 0, 0, 0, 1],
                                        'x2_河南省': [1, 0, 0, 0, 0],
                                        'x2_nan': [0, 1, 0, 0, 1],
                                        'x2_浙江省': [0, 0, 1, 0, 0],
                                        'x2_福建省': [0, 0, 0, 1, 0]
                                        })

        df_test = pd.DataFrame({
            'x1': [1, 2, 2, np.nan],
            'x2': ['河南省', '湖南省', '北京市', np.nan]
        })

        df_test_encode = pd.DataFrame({'x1_1.0': [1, 0, 0, 0],
                                       'x1_2.0': [0, 1, 1, 0],
                                       'x1_nan': [0, 0, 0, 1],
                                       'x2_河南省': [1, 0, 0, 0],
                                       'x2_nan': [0, 0, 0, 1],
                                       'x2_浙江省': [0, 0, 0, 0],
                                       'x2_福建省': [0, 0, 0, 0]
                                       })

        oht = OneHotTransformer()
        oht.fit(df_train, columns=['x1', 'x2'])
        df_train_except = oht.transform(df_train)
        df_test_except = oht.transform(df_test)

        self.assertListEqual(df_train_encode['x1_1.0'].to_list(),
                             df_train_except['x1_1.0'].to_list())
        self.assertListEqual(df_train_encode['x1_2.0'].to_list(),
                             df_train_except['x1_2.0'].to_list())
        self.assertListEqual(df_train_encode['x1_nan'].to_list(),
                             df_train_except['x1_nan'].to_list())
        self.assertListEqual(df_train_encode['x2_河南省'].to_list(),
                             df_train_except['x2_河南省'].to_list())
        self.assertListEqual(df_train_encode['x2_nan'].to_list(),
                             df_train_except['x2_nan'].to_list())
        self.assertListEqual(df_train_encode['x2_浙江省'].to_list(),
                             df_train_except['x2_浙江省'].to_list())
        self.assertListEqual(df_train_encode['x2_福建省'].to_list(),
                             df_train_except['x2_福建省'].to_list())

        self.assertListEqual(df_test_encode['x1_1.0'].to_list(),
                             df_test_except['x1_1.0'].to_list())
        self.assertListEqual(df_test_encode['x1_2.0'].to_list(),
                             df_test_except['x1_2.0'].to_list())
        self.assertListEqual(df_test_encode['x1_nan'].to_list(),
                             df_test_except['x1_nan'].to_list())
        self.assertListEqual(df_test_encode['x2_河南省'].to_list(),
                             df_test_except['x2_河南省'].to_list())
        self.assertListEqual(df_test_encode['x2_nan'].to_list(),
                             df_test_except['x2_nan'].to_list())
        self.assertListEqual(df_test_encode['x2_浙江省'].to_list(),
                             df_test_except['x2_浙江省'].to_list())
        self.assertListEqual(df_test_encode['x2_福建省'].to_list(),
                             df_test_except['x2_福建省'].to_list())
