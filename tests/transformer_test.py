import os
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from didtool.transformer import SingleWOETransformer, WOETransformer


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

        test_x = pd.DataFrame({'v1': [0.02, 0.05, np.nan], 'v5': [0, 1, -1]})
        res = transformer.transform(test_x)
        self.assertAlmostEqual(res['v1'][0], 1.439120, 6)
        self.assertAlmostEqual(res['v1'][1], -0.640322, 6)
        self.assertAlmostEqual(res['v1'][2], -1.017155, 6)
        self.assertAlmostEqual(res['v5'][0], -0.216908, 6)
        self.assertAlmostEqual(res['v5'][1], 0.484547, 6)
        self.assertAlmostEqual(res['v5'][2], 0)
