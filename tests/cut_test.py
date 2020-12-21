import unittest
import numpy as np
import didtool


class TestCut(unittest.TestCase):
    def test_step_cut(self):
        x = [0, 1, 2, 2, 3, 6, 8, 10, np.nan]
        out, bins = didtool.step_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 0, 0, 1, 2, 3, 3, -1]
        self.assertListEqual(list(out), expect_out)

    def test_step_cut_unbalance(self):
        x = [0, 1, 2, 2, 3, 3, 3, 10, np.nan]
        out, bins = didtool.step_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 0, 0, 1, 1, 1, 2, -1]
        expect_bins = [-np.inf, 2.5, 7.5, np.inf]
        self.assertListEqual(list(out), expect_out)
        self.assertListEqual(list(bins), expect_bins)

        x = [0, 6, 8, 8, 8, 9, 10, 10, np.nan]
        out, bins = didtool.step_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 1, 2, 2, 2, 2, 2, 2, -1]
        expect_bins = [-np.inf, 5.0, 7.5, np.inf]
        self.assertListEqual(list(out), expect_out)
        self.assertListEqual(list(bins), expect_bins)

    def test_quantile_cut(self):
        x = [0, 1, 2, 2, 3, 5, 6, 10, np.nan]
        out, bins = didtool.quantile_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 1, 1, 2, 2, 3, 3, -1]
        self.assertListEqual(list(out), expect_out)

    def test_quantile_cut_unbalance(self):
        x = [1, 1, 1, 1, 1, 2, 3, 3, 3, 10, np.nan]
        out, bins = didtool.quantile_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, -1]
        expect_bins = [-np.inf, 1.5, 3, np.inf]
        self.assertListEqual(list(out), expect_out)
        self.assertListEqual(list(bins), expect_bins)

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

    def test_chi_square_cut(self):
        x = [0, 1, 2, 2, 3, 5, 6, 10, np.nan, np.nan]
        target = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
        out, bins = didtool.chi_square_cut(x, target, 4, return_bins=True)
        expect_out = [0, 0, 1, 1, 1, 2, 3, 3, -1, -1]
        self.assertListEqual(list(out), expect_out)
        self.assertListEqual(list(bins), [-np.inf, 1.0, 3.0, 5.0, np.inf])

    def test_cut_with_bins(self):
        x = [0, 1, 2, 2, 3, 6, 8, 10, np.nan]
        out, bins = didtool.step_cut(x, 4, nan=-1, return_bins=True)
        expect_out = [0, 0, 0, 0, 1, 2, 3, 3, -1]
        self.assertListEqual(list(out), expect_out)
        self.assertListEqual(list(bins), [-np.inf, 2.5, 5., 7.5, np.inf])

        x2 = [0, 4, 11, np.nan]
        out2 = didtool.cut_with_bins(x2, bins)
        self.assertListEqual(list(out2), [0, 1, 3, -1])
