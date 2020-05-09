import os
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from didtool import split_data_random
from didtool.scorecard import ScoreCardTransformer
from didtool.model import LGBModelSingle


class TestScoreCard(unittest.TestCase):
    def test_score_card_transformer(self):
        df = pd.read_csv('samples.csv')
        df['v5'] = df['v5'].astype('category')

        features = [col for col in df.columns.values if col != 'target']
        data = split_data_random(df, 0.6, 0.2)

        model_params = dict(
            boosting_type='gbdt', n_estimators=100, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=20, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelSingle(data, features, 'target', out_path='./test_out',
                           model_params=model_params)
        m.train(early_stopping_rounds=10)
        result = m.evaluate()

        transformer = ScoreCardTransformer(bad_flag=True)
        transformer.fit(result['prob'].values, result['target'].values)
        print(transformer.binning_df)
        print(transformer.mapping_df)
        # transformer.plot_bins()
        scores = transformer.transform([0.05, 0.5, 0.8])
        self.assertEqual(scores[0], 753)
        self.assertEqual(scores[1], 679)
        self.assertEqual(scores[2], 610)

        slope = transformer.mapping_df['slope'][1]
        intercept = transformer.mapping_df['intercept'][1]
        self.assertAlmostEqual(slope, -200)
        self.assertAlmostEqual(intercept, 763)
        self.assertEqual(int(slope * 0.05 + intercept), 753)
