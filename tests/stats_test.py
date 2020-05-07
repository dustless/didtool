import unittest
import pandas as pd
import numpy as np

import didtool


class TestStats(unittest.TestCase):
    def test_iv_all(self):
        df = pd.DataFrame({
            "x1": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
                   10, 10, 10, 11, 11, 11, np.nan, np.nan, np.nan],
            "x2": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
                   10, 10, 10, 11, 11, 11, -1, -1, -1],
            "target": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 0, 1, 0, 1, 1]
        })
        df["x2"] = df["x2"].astype("category")
        iv_df = didtool.iv_all(df[['x1', 'x2']], df["target"])
        self.assertAlmostEqual(iv_df["iv"]['x1'], 1.405716, places=6)
        self.assertAlmostEqual(iv_df["iv"]['x2'], 1.398188, places=6)

        iv_df = didtool.iv_all(df[['x1', 'x2']], df["target"],
                               cut_method='step')
        self.assertAlmostEqual(iv_df["iv"]['x1'], 1.434012, places=6)
        self.assertAlmostEqual(iv_df["iv"]['x2'], 1.398188, places=6)