import unittest
from pathlib import Path

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
                   10, 10, 10, 11, 11, 11, np.nan, np.nan, np.nan],
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
        self.assertAlmostEqual(iv_df["iv"]['x1'], 1.497495, places=6)
        self.assertAlmostEqual(iv_df["iv"]['x2'], 1.398188, places=6)

    def test_psi_all(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        df['v5'] = df['v5'].astype('category')
        df['month'] = df.index % 6
        df['month'] = df['month'].apply(lambda x: "m%02d" % (x + 1))

        features = ['v%d' % i for i in range(1, 21)]
        psi_df = didtool.psi_all(df, features, group_col='month')
        self.assertAlmostEqual(psi_df.loc['v5', 'm02'], 0.000192, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm03'], 0.000192, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm04'], 0.000192, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm05'], 0.015468, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm06'], 0.036753, places=6)
        self.assertAlmostEqual(psi_df.loc['v2', 'm04'], 0.578705, places=6)

        expected_data = df[df.month < 'm04']
        psi_df = didtool.psi_all(df, features, expected_data=expected_data)
        self.assertAlmostEqual(psi_df.loc['v5', 'm01'], 0.000021, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm02'], 0.000085, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm03'], 0.000021, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm04'], 0.000085, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm05'], 0.017849, places=6)
        self.assertAlmostEqual(psi_df.loc['v5', 'm06'], 0.003370, places=6)
        self.assertAlmostEqual(psi_df.loc['v2', 'm04'], 0.254352, places=6)
