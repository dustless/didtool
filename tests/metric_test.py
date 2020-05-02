import unittest
import numpy as np
import didtool


class TestMetric(unittest.TestCase):
    def test_iv(self):
        x = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
             5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
             10, 10, 10, 11, 11, 11, np.nan, np.nan, np.nan])
        y = np.array(
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 1, 1])
        iv1 = didtool.iv(x, y, False)
        iv2 = didtool.iv(x, y, True)

        self.assertAlmostEqual(iv1, 1.334705486550453)
        self.assertAlmostEqual(iv2, 1.4057157347824798)

    def test_psi(self):
        x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        psi = didtool.psi(x, y, 3)
        self.assertAlmostEqual(psi, 0.07701635339554946)
