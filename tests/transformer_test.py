import unittest
from pathlib import Path

import pandas as pd
import numpy as np

import didtool


class TestTransformer(unittest.TestCase):
    def test_single_woe_transformer(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        x = df['v1']
        y = df['target']

        transformer = didtool.SingleWOETransformer()
        transformer.fit(x, y, 'v1')

        self.assertListEqual(
            list(np.round(transformer.bins, 5)),
            [-np.inf, 0.00455, 0.00485, 0.0072, 0.01415, 0.01485, 0.0212,
             0.02815, 0.03165, 0.04235, np.inf]
        )
        expect_dict = {
            -1: -1.0171553366121715, 0: -0.10844300821451114, 1: 2.825413861621392, 2: 0.5741220630148971,
            3: 2.621814906380153, 4: 3.924026150289502, 5: 1.4391195005015018, 6: 2.7384024846317625,
            7: 0.340507211833392, 8: 2.1322666810614472, 9: -0.6403220411783341}
        for key in expect_dict:
            self.assertAlmostEqual(expect_dict[key], transformer.woe_map.get(key, 0), 6)
        self.assertAlmostEqual(transformer.woe_df['var_iv'][0], 1.878709, 6)
        self.assertEqual(transformer.woe_df.shape[0], 11)
        self.assertEqual(transformer.var_name, 'v1')

        res = transformer.transform(np.array([0.02, 0.05, np.nan]))

        self.assertAlmostEqual(res[0], 1.439120, 6)
        self.assertAlmostEqual(res[1], -0.640322, 6)
        self.assertAlmostEqual(res[2], -1.017155, 6)

        x = np.array(['heh', '哈哈', np.nan, '1'])
        y = np.array([1, 0, 1, 1])
        transformer = didtool.SingleWOETransformer(is_continuous=False)
        transformer.fit(x, y, 'Chinese')

        self.assertListEqual(
            list(np.round(transformer.transform(x), 8)),
            [0.28768207, -1.79175947, -0.40546511, 0.28768207]
        )
        print(transformer.woe_df)

    def test_woe_transformer(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        y = df['target']

        transformer = didtool.WOETransformer(features=['v1', 'v2', 'v5'])
        transformer.fit(df[['v1', 'v2', 'v5', 'target']], y)
        print(transformer.woe_df)

        train_x = pd.DataFrame({'v1': [0.02, 0.02, 0.1, np.nan],
                                'v2': ['0.05', '1', '1', np.nan],
                                'v3': ['cc', np.nan, 'f', np.nan],
                                'v4': ['张三', np.nan, '王五', '王五'],
                                'label': [0, 1, 0, 1]})

        test_x = pd.DataFrame({'v1': [0.02, 0.05, 0.1, np.nan],
                               'v2': ['0.02', '0.05', '1', np.nan],
                               'v3': ['a', 'cc', 'f', np.nan],
                               'v4': ['张三', '李四', '王五', np.nan],
                               'label': [0, 1, 0, 0]})
        transformer = didtool.WOETransformer(features=['v1', 'v2', 'v3', 'v4'])
        transformer.fit(train_x, train_x['label'])
        res = transformer.transform(test_x)

        np.testing.assert_array_equal(np.round(res['v1'].to_list(), 6), [0., 0.693147, -0.693147, 0.693147])
        np.testing.assert_array_equal(np.round(res['v2'].to_list(), 6), [0.693147, -0.693147, 0., 0.693147])
        np.testing.assert_array_equal(np.round(res['v3'].to_list(), 6), [1.386294, -1.386294, -1.386294, 1.386294])
        np.testing.assert_array_equal(np.round(res['v4'].to_list(), 6), [-0.693147, 0.693147, 0., 0.693147])

    def test_category_transformer(self):
        df = pd.DataFrame({
            'x1': [1, 2, 1, 2, 1, 7.3, 0, 0, 0, 0, np.nan],
            'x2': ['北京', '上海', '上海', '山东', '北京', '北京',
                   np.nan, np.nan, np.nan, np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'x4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        df_expect = pd.DataFrame({
            'x1': [2, 3, 2, 3, 2, 4, 1, 1, 1, 1, 0],
            'x2': [1, 2, 2, 3, 1, 1, 0, 0, 0, 0, 0],
            'x3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'x4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        for col in df_expect.columns:
            df_expect[col] = df_expect[col].astype('category')

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

        for col in df_expect.columns:
            df_encoder[col] = df_encoder[col].astype('category')

        df_test = pd.DataFrame({
            'x1': [1, 2, 1, 0, np.nan],
            'x2': ['北京', '上海', '山东', np.nan, np.nan],
            'x3': [np.nan, np.nan, np.nan, np.nan, np.nan, ],
            'x4': [1, 1, 1, 1, 1]
        })

        df_test_expect = pd.DataFrame({
            'x1': [2, 3, 2, 1, 0],
            'x2': [1, 2, 3, 0, 0],
            'x3': [0, 0, 0, 0, 0],
            'x4': [1, 1, 1, 1, 1]
        })

        ct = didtool.CategoryTransformer()
        ct.fit(df, max_bins=64)
        df = ct.transform(df)
        df_test = ct.transform(df_test)

        self.assertListEqual(df.x1.tolist(),
                             df_expect.x1.tolist())
        self.assertListEqual(df.x2.tolist(),
                             df_expect.x2.tolist())
        self.assertListEqual(df.x3.tolist(),
                             df_expect.x3.tolist())
        self.assertListEqual(df.x4.tolist(),
                             df_expect.x4.tolist())

        np.testing.assert_array_equal(ct.df_encoder.x1.tolist(),
                                      df_encoder.x1.tolist())
        np.testing.assert_array_equal(ct.df_encoder.x2.tolist(),
                                      df_encoder.x2.tolist())
        np.testing.assert_array_equal(ct.df_encoder.x3.tolist(),
                                      df_encoder.x3.tolist())
        np.testing.assert_array_equal(ct.df_encoder.x4.tolist(),
                                      df_encoder.x4.tolist())

        np.testing.assert_array_equal(ct.df_encoder.x1_encoder.tolist(),
                                      df_encoder.x1_encoder.tolist())
        np.testing.assert_array_equal(ct.df_encoder.x2_encoder.tolist(),
                                      df_encoder.x2_encoder.tolist())
        np.testing.assert_array_equal(ct.df_encoder.x3_encoder.tolist(),
                                      df_encoder.x3_encoder.tolist())
        np.testing.assert_array_equal(ct.df_encoder.x4_encoder.tolist(),
                                      df_encoder.x4_encoder.tolist())

        self.assertListEqual(df_test.x1.tolist(),
                             df_test_expect.x1.tolist())
        self.assertListEqual(df_test.x2.tolist(),
                             df_test_expect.x2.tolist())
        self.assertListEqual(df_test.x3.tolist(),
                             df_test_expect.x3.tolist())
        self.assertListEqual(df_test.x4.tolist(),
                             df_test_expect.x4.tolist())

    def test_onehot_transformer(self):
        df_train = pd.DataFrame({
            'x1': [1, 2, 1, 1, np.nan],
            'x2': ['河南省', np.nan, '浙江省', '福建省', np.nan]
        })

        df_train_expect = pd.DataFrame({'x1_1.0': [1, 0, 1, 1, 0],
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

        df_test_expect = pd.DataFrame({'x1_1.0': [1, 0, 0, 0],
                                       'x1_2.0': [0, 1, 1, 0],
                                       'x1_nan': [0, 0, 0, 1],
                                       'x2_河南省': [1, 0, 0, 0],
                                       'x2_nan': [0, 0, 0, 1],
                                       'x2_浙江省': [0, 0, 0, 0],
                                       'x2_福建省': [0, 0, 0, 0]
                                       })

        oht = didtool.OneHotTransformer()
        oht.fit(df_train)
        df_train_encode = oht.transform(df_train)
        df_test_encode = oht.transform(df_test)

        self.assertListEqual(df_train_expect['x1_1.0'].tolist(),
                             df_train_encode['x1_1.0'].tolist())
        self.assertListEqual(df_train_expect['x1_2.0'].tolist(),
                             df_train_encode['x1_2.0'].tolist())
        self.assertListEqual(df_train_expect['x1_nan'].tolist(),
                             df_train_encode['x1_nan'].tolist())
        self.assertListEqual(df_train_expect['x2_河南省'].tolist(),
                             df_train_encode['x2_河南省'].tolist())
        self.assertListEqual(df_train_expect['x2_nan'].tolist(),
                             df_train_encode['x2_nan'].tolist())
        self.assertListEqual(df_train_expect['x2_浙江省'].tolist(),
                             df_train_encode['x2_浙江省'].tolist())
        self.assertListEqual(df_train_expect['x2_福建省'].tolist(),
                             df_train_encode['x2_福建省'].tolist())

        self.assertListEqual(df_test_expect['x1_1.0'].tolist(),
                             df_test_encode['x1_1.0'].tolist())
        self.assertListEqual(df_test_expect['x1_2.0'].tolist(),
                             df_test_encode['x1_2.0'].tolist())
        self.assertListEqual(df_test_expect['x1_nan'].tolist(),
                             df_test_encode['x1_nan'].tolist())
        self.assertListEqual(df_test_expect['x2_河南省'].tolist(),
                             df_test_encode['x2_河南省'].tolist())
        self.assertListEqual(df_test_expect['x2_nan'].tolist(),
                             df_test_encode['x2_nan'].tolist())
        self.assertListEqual(df_test_expect['x2_浙江省'].tolist(),
                             df_test_encode['x2_浙江省'].tolist())
        self.assertListEqual(df_test_expect['x2_福建省'].tolist(),
                             df_test_encode['x2_福建省'].tolist())

    def test_list_transformer(self):
        df_train = pd.DataFrame({
            'x1': ['1,2,5', '2,3,4', '2,4,5', np.nan],
            'x2': ['a,c', 'a,b', 'b', 'a']
        })

        df_test = pd.DataFrame({
            'x1': [np.nan, '1,2,6'],
            'x2': ['a,c,d', np.nan]
        })

        transformer = didtool.ListTransformer()
        transformer.fit(df_train)
        df_train_encode = transformer.transform(df_train)
        df_train_expect = pd.DataFrame({
            'x1_1': [1, 0, 0, np.nan],
            'x1_2': [1, 1, 1, np.nan],
            'x1_3': [0, 1, 0, np.nan],
            'x1_4': [0, 1, 1, np.nan],
            'x1_5': [1, 0, 1, np.nan],
            'x2_a': [1, 1, 0, 1],
            'x2_b': [0, 1, 1, 0],
            'x2_c': [1, 0, 0, 0],
        })

        self.assertListEqual(df_train_encode.loc[0, :].dropna().tolist(),
                             df_train_expect.loc[0, :].dropna().tolist())
        self.assertListEqual(df_train_encode.loc[1, :].dropna().tolist(),
                             df_train_expect.loc[1, :].dropna().tolist())
        self.assertListEqual(df_train_encode.loc[2, :].dropna().tolist(),
                             df_train_expect.loc[2, :].dropna().tolist())
        self.assertListEqual(df_train_encode.loc[3, :].dropna().tolist(),
                             df_train_expect.loc[3, :].dropna().tolist())

        df_test_encode = transformer.transform(df_test)
        df_test_expect = pd.DataFrame({
            'x1_1': [np.nan, 1],
            'x1_2': [np.nan, 1],
            'x1_3': [np.nan, 0],
            'x1_4': [np.nan, 0],
            'x1_5': [np.nan, 0],
            'x2_a': [1, np.nan],
            'x2_b': [0, np.nan],
            'x2_c': [1, np.nan],
        })
        self.assertListEqual(df_test_encode.loc[0, :].dropna().tolist(),
                             df_test_expect.loc[0, :].dropna().tolist())
        self.assertListEqual(df_test_encode.loc[1, :].dropna().tolist(),
                             df_test_expect.loc[1, :].dropna().tolist())

    def test_list_transformer2(self):
        df_train = pd.DataFrame({
            'x1': ['1:0.9,2:0.85,4:0.8', '2:0.7,3:0.6,4:0.5',
                   '2:0.4,4:0.3', np.nan],
            'x2': ['a:0.11,c:0.22', 'a:0.33,b:0.44', 'b:0.99', 'a:0.55']
        })

        df_test = pd.DataFrame({
            'x1': [np.nan, '1:0.25,2:0.35,6:0.45'],
            'x2': ['a:0.66,c:0.77,d:0.88', 'b:0.99']
        })

        transformer = didtool.ListTransformer(sub_sep=':')
        transformer.fit(df_train)
        df_train_encode = transformer.transform(df_train)
        df_train_expect = pd.DataFrame({
            'x1_1': [0.9, 0, 0, np.nan],
            'x1_2': [0.85, 0.7, 0.4, np.nan],
            'x1_3': [0, 0.6, 0, np.nan],
            'x1_4': [0.8, 0.5, 0.3, np.nan],
            'x2_a': [0.11, 0.33, 0, 0.55],
            'x2_b': [0, 0.44, 0.99, 0],
            'x2_c': [0.22, 0, 0, 0],
        })

        self.assertListEqual(df_train_encode.loc[0, :].dropna().tolist(),
                             df_train_expect.loc[0, :].dropna().tolist())
        self.assertListEqual(df_train_encode.loc[1, :].dropna().tolist(),
                             df_train_expect.loc[1, :].dropna().tolist())
        self.assertListEqual(df_train_encode.loc[2, :].dropna().tolist(),
                             df_train_expect.loc[2, :].dropna().tolist())
        self.assertListEqual(df_train_encode.loc[3, :].dropna().tolist(),
                             df_train_expect.loc[3, :].dropna().tolist())

        df_test_encode = transformer.transform(df_test)
        df_test_expect = pd.DataFrame({
            'x1_1': [np.nan, 0.25],
            'x1_2': [np.nan, 0.35],
            'x1_3': [np.nan, 0],
            'x1_4': [np.nan, 0],
            'x2_a': [0.66, 0],
            'x2_b': [0, 0.99],
            'x2_c': [0.77, 0],
        })
        self.assertListEqual(df_test_encode.loc[0, :].dropna().tolist(),
                             df_test_expect.loc[0, :].dropna().tolist())
        self.assertListEqual(df_test_encode.loc[1, :].dropna().tolist(),
                             df_test_expect.loc[1, :].dropna().tolist())
