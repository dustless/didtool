import unittest
import numpy as np
import didtool


class TestCut(unittest.TestCase):
    def test_step_cut(self):
        x = [0, 1, 2, 2, 3, 6, 8, 10, np.nan]
        out, bins = didtool.step_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 0, 0, 1, 2, 3, 3, -1]
        self.assertListEqual(list(out), expect_out)

    def test_quantile_cut(self):
        x = [0, 1, 2, 2, 3, 5, 6, 10, np.nan]
        out, bins = didtool.quantile_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 1, 1, 2, 2, 3, 3, -1]
        self.assertListEqual(list(out), expect_out)

    def test_dt_cut(self):
        x = [0, 1, 2, 2, 3, 5, 6, 10, np.nan, np.nan]
        target = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
        out, bins = didtool.dt_cut(x, target, 4, return_bins=True)
        expect_out = [0, 0, 1, 1, 1, 2, 3, 3, -1, -1]
        self.assertListEqual(list(out), expect_out)

    def test_lgb_cut(self):
        x = [0, 1, 2, 2, 3, 5, 6, 10, np.nan, np.nan]
        target = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
        out, bins = didtool.lgb_cut(x, target, 4, return_bins=True)
        expect_out = [0, 1, 1, 1, 2, 2, 2, 3, -1, -1]
        self.assertListEqual(list(out), expect_out)

