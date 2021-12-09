import unittest
from pathlib import Path

import pandas as pd

from didtool.scorecard import ScoreCardTransformer
import random


class TestScoreCard(unittest.TestCase):
    def test_score_card_transformer(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))[['target']]

        random.seed(1)
        df['prob'] = df['target'].apply(
            lambda x: random.uniform(0, 0.8) if x < 1 else random.uniform(0.2,
                                                                          1))

        transformer = ScoreCardTransformer(bad_flag=True)
        transformer.fit(df['prob'].values, df['target'].values)
        # print(transformer.binning_df)
        # print(transformer.mapping_df)
        # transformer.plot_bins()
        scores = transformer.transform([0.05, 0.5, 0.8])
        self.assertEqual(scores[0], 815)
        self.assertEqual(scores[1], 677)
        self.assertEqual(scores[2], 666)

        slope = transformer.mapping_df['slope'][1]
        intercept = transformer.mapping_df['intercept'][1]
        self.assertAlmostEqual(slope, -400)
        self.assertAlmostEqual(intercept, 835)
        self.assertEqual(int(slope * 0.05 + intercept), 815)
