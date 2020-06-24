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
        iv3 = didtool.iv(x, y, True, cut_method='step')
        iv4 = didtool.iv(x, y, True, cut_method='quantile')
        iv5 = didtool.iv(x, y, True, cut_method='lgb')

        self.assertAlmostEqual(iv1, 1.334705486550453)
        self.assertAlmostEqual(iv2, 1.4057157347824798)
        self.assertAlmostEqual(iv3, 1.4340120285033071)
        self.assertAlmostEqual(iv4, 1.3120787039390784)
        self.assertAlmostEqual(iv5, 1.4057157347824798)

    def test_psi(self):
        x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        psi = didtool.psi(x, y, 3)
        self.assertAlmostEqual(psi, 0.07701635339554946)

    def test_ks(self):
        import pandas as pd
        import random
        data = pd.read_csv("samples.csv")[['target']]

        data['pro'] = data['target'].apply(lambda x: random.uniform(0, 0.8) if x < 1 else random.uniform(0.2, 1))
        y_pre = data['pro']
        y_real = data['target']

        #ks = didtool.plot_ks_in_cum(y_pre, y_real)
        #ks_compare = didtool.plot_ks_in_tpr_fpr(y_pre, y_real)