import unittest
import numpy as np
import didtool


class TestMetric(unittest.TestCase):
    def test_iv_discrete(self):
        x = np.array([0, 1, 5, 5, 3, 4, 4, 4, 4, np.nan])
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
        iv = didtool.iv_discrete(x, y)
        self.assertAlmostEqual(iv, 0.5972531564093516)

    def test_iv_continuous(self):
        x = np.array([0, 1, 5, 5, 3, 4, 4, 4, 4, np.nan])
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
        iv = didtool.iv_continuous(x, y)
        self.assertAlmostEqual(iv, 0.9776155056983381)

    def test_psi(self):
        x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        psi = didtool.psi(x, y, 3)
        self.assertAlmostEqual(psi, 0.07701635339554946)
