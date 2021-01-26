import unittest
import pandas as pd
import numpy as np

import didtool


class TestEncoder(unittest.TestCase):
    def test_woe_encoder(self):
        df = pd.read_csv('samples.csv')
        x = df['v5'].astype('category')
        y = df['target']

        encoder = didtool.WOEEncoder()
        encoder.fit(x, y)
        self.assertDictEqual(encoder._woe_map,
                             {0: -0.21690835519242824, 1: 0.48454658205632983})

        res = encoder.transform(np.array([0, 1, -1]))
        self.assertAlmostEqual(res[0, 0], -0.216908, 6)
        self.assertAlmostEqual(res[1, 0], 0.484547, 6)
        self.assertAlmostEqual(res[2, 0], 0)

        # test nan value
        x = df['v5']
        x[:100] = np.nan
        encoder.fit(x, y)
        self.assertDictEqual(encoder._woe_map,
                             {0.0: -0.2511705085616937, 1.0: 0.5387442239332461,
                              'NA': 0.04152558412767761})
        res = encoder.transform(np.array([0, 1, -1, np.nan]))
        self.assertAlmostEqual(res[0, 0], -0.251171, 6)
        self.assertAlmostEqual(res[1, 0], 0.538744, 6)
        self.assertAlmostEqual(res[2, 0], 0)
        self.assertAlmostEqual(res[3, 0], 0.041526, 6)
