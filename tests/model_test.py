import os
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from didtool.model import LGBModelSingle, LGBModelStacking


class TestModel(unittest.TestCase):
    def setUp(self):
        for root, dirs, files in os.walk('./test_out', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def test_model_single(self):
        df = pd.read_csv('samples.csv')
        df['v5'] = df['v5'].astype('category')

        features = [col for col in df.columns.values if col != 'target']
        model_param = dict(
            boosting_type='gbdt', n_estimators=10, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=10, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelSingle(df, features, 'target', './test_out', model_param)

        # test split_data
        m.split_data_random(0.6, 0.2)
        self.assertEqual(m.data[m.data.group == 0].shape[0], 600)
        self.assertEqual(m.data[m.data.group == 1].shape[0], 200)
        self.assertEqual(m.data[m.data.group == 2].shape[0], 200)
        self.assertEqual(m.data[m.data.group == 0]['target'].sum(), 64)
        self.assertEqual(m.data[m.data.group == 1]['target'].sum(), 21)
        self.assertEqual(m.data[m.data.group == 1]['target'].sum(), 21)

        # test train
        m.train(save_learn_curve=False)
        m.evaluate()

        # test update model params
        m.update_model_params({'n_estimators': 100})
        m.train(save_learn_curve=True)

        # test evaluate
        result = m.evaluate()
        print(result)

        # test save_feature_importance
        m.save_feature_importance()

        # test export
        m.export()

    def test_model_stacking(self):
        df = pd.read_csv('samples.csv')
        df['v5'] = df['v5'].astype('category')

        features = [col for col in df.columns.values if col != 'target']
        model_param = dict(
            boosting_type='gbdt', n_estimators=10, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=10, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelStacking(df, features, 'target', './test_out', model_param,
                             n_fold=3)

        # test split_data
        m.split_data(df.index >= 900)
        self.assertEqual(m.data[m.data.fold == 0].shape[0], 300)
        self.assertEqual(m.data[m.data.fold == 1].shape[0], 300)
        self.assertEqual(m.data[m.data.fold == 2].shape[0], 300)
        self.assertEqual(m.data[m.data.fold == -1].shape[0], 100)

        # test train
        m.train(save_learn_curve=False)

        # test evaluate
        result = m.evaluate()
        print(result)

        # test update model param
        m.update_model_params({'n_estimators': 100})
        m.train(save_learn_curve=True)
        result = m.evaluate()
        print(result)

        # test save_feature_importance
        m.save_feature_importance()

        # test export
        m.export()
