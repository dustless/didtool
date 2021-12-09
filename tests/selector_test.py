import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from didtool.selector import Selector


class TestSelector(unittest.TestCase):
    def test_drop_missing(self):
        df = pd.DataFrame({
            "x1": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
            "x2": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan],
            "x3": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "x4": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   -1, -1, -1, -1, -1, -1, 9, 9, 9]
        })
        selector = Selector(df)
        selector.drop_missing(0.5, missing_value=-1)
        self.assertAlmostEqual(selector.missing_stats['missing_rate'][0], 1.0)
        self.assertAlmostEqual(selector.missing_stats['missing_rate'][1], 0.5)
        self.assertAlmostEqual(selector.missing_stats['missing_rate'][2], 0.4)
        self.assertAlmostEqual(selector.missing_stats['missing_rate'][3], 0)
        self.assertListEqual(selector.drop_cols, ["x3"])
        self.assertEqual(selector.data.shape[1], 3)

    def test_drop_iv(self):
        df = pd.DataFrame({
            "x1": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
                   10, 10, 10, 11, 11, 11, np.nan, np.nan, np.nan],
            "x2": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
                   10, 10, 10, 11, 11, 11, -1, -1, -1],
            "x3": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                   0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                   0, 1, 0, 1, 0, 1, 1, 0, 1],
            "target": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 0, 1, 0, 1, 1]
        })
        df["x2"] = df["x2"].astype("category")
        selector = Selector(df.drop("target", axis=1), df["target"])
        selector.drop_low_iv(0.5)
        self.assertAlmostEqual(selector.iv_stats.loc['x1', 'iv'], 1.405716, 6)
        self.assertAlmostEqual(selector.iv_stats.loc['x2', 'iv'], 1.398188, 6)
        self.assertAlmostEqual(selector.iv_stats.loc['x3', 'iv'], 0.129772, 6)
        self.assertListEqual(selector.drop_cols, ['x3'])
        self.assertEqual(selector.data.shape[1], 2)

    def test_drop_correlated(self):
        df = pd.DataFrame({
            "x0": [12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9,
                   8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4,
                   3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0],
            "x1": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                   5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
                   10, 10, 10, 11, 11, 11, np.nan, np.nan, np.nan],
            "x2": [12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9,
                   8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4,
                   3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0],
            "x3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                   12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "target": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                       0, 0, 0, 0, 0, 1, 0, 1, 1]
        })
        selector = Selector(df.drop("target", axis=1), df["target"])
        selector.drop_correlated(0.5)
        self.assertAlmostEqual(selector.iv_stats.loc['x0', 'iv'], 1.455369, 6)
        self.assertAlmostEqual(selector.iv_stats.loc['x1', 'iv'], 1.405716, 6)
        self.assertAlmostEqual(selector.iv_stats.loc['x2', 'iv'], 1.455369, 6)
        self.assertAlmostEqual(selector.iv_stats.loc['x3', 'iv'], 0.929362, 6)
        self.assertAlmostEqual(selector.corr_matrix.loc['x0', 'x2'], 1)
        self.assertAlmostEqual(selector.corr_matrix.loc['x1', 'x2'], -1)
        self.assertAlmostEqual(selector.corr_matrix.loc['x1', 'x3'], -0.0887, 4)
        self.assertListEqual(selector.drop_cols, ['x1', 'x2'])
        self.assertEqual(selector.data.shape[1], 2)

    def test_drop_all(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        selector = Selector(df.drop("target", axis=1), df["target"])
        selector.drop_missing(0.8).drop_low_iv(0.1).drop_correlated(0.8) \
            .drop_low_importance(0.85)

        print(selector.missing_stats)
        print(selector.iv_stats)
        print(selector.record_correlated)
        print(selector.importance_stats)

        drop_cols = selector.drop_cols
        self.assertSetEqual(
            set(drop_cols),
            {'v13', 'v3', 'v12', 'v19', 'v7', 'v10', 'v11', 'v4', 'v16', 'v20',
             'v17', 'v9', 'v6', 'v8', 'v5'}
        )
        # selector.plot_missing()
        # selector.plot_iv()
        # selector.plot_correlated(plot_all=True)
        # selector.plot_importance()
