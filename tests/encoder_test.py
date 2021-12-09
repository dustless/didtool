import unittest
import pandas as pd
import numpy as np
from pathlib import Path

import didtool


class TestEncoder(unittest.TestCase):
    def test_woe_encoder(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        x = df['v5'].astype('category')
        y = df['target']

        encoder = didtool.WOEEncoder()
        encoder.fit(x, y)
        expect_dict = {0: -0.21690835519242824, 1: 0.48454658205632983}
        for key in expect_dict:
            self.assertAlmostEqual(expect_dict[key], encoder._woe_map.get(key, 0), 6)

        res = encoder.transform(np.array([0, 1, -1]))
        self.assertAlmostEqual(res[0, 0], -0.216908, 6)
        self.assertAlmostEqual(res[1, 0], 0.484547, 6)
        self.assertAlmostEqual(res[2, 0], 0)

        # test nan value
        x = df['v5']
        x[:100] = np.nan
        encoder.fit(x, y)
        expect_dict = {0.0: -0.2511705085616937, 1.0: 0.5387442239332461, 'NA': 0.04152558412767761}
        for key in expect_dict:
            self.assertAlmostEqual(expect_dict[key], encoder._woe_map.get(key, 0), 6)
        res = encoder.transform(np.array([0, 1, -1, np.nan]))
        self.assertAlmostEqual(res[0, 0], -0.251171, 6)
        self.assertAlmostEqual(res[1, 0], 0.538744, 6)
        self.assertAlmostEqual(res[2, 0], 0)
        self.assertAlmostEqual(res[3, 0], 0.041526, 6)

    def test_wrapped_label_encoder(self):
        x = pd.DataFrame({'x': [1, 5, 6, 2, 3, 5, 6, np.nan]})
        encoder = didtool.encoder.WrappedLabelEncoder(missing_values=-1)
        xt = encoder.fit_transform(x).reshape(-1)

        expected_xt = [0, 3, 4, 1, 2, 3, 4, -1]
        self.assertListEqual(expected_xt, list(xt))

        new_x = [1, 5, 999, np.nan]
        new_xt = encoder.transform(new_x).reshape(-1)
        self.assertListEqual(list(new_xt), [0, 3, -1, -1])
