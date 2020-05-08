import os
import unittest
import pandas as pd

from didtool.model import LGBModelSingle, LGBModelStacking
from didtool.split import split_data_random, split_data_stacking


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
        # split_data
        data = split_data_random(df, 0.6, 0.2)

        model_params = dict(
            boosting_type='gbdt', n_estimators=10, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=20, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelSingle(data, features, 'target', out_path='./test_out',
                           model_params=model_params)

        # test train
        m.train(early_stopping_rounds=10, save_learn_curve=False)
        m.evaluate()

        # test update model params
        m.update_model_params({'n_estimators': 100})
        m.train(early_stopping_rounds=10, save_learn_curve=False)

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

        n_fold = 5
        # split_data
        df = split_data_stacking(df, df.index >= 900, n_fold)

        model_params = dict(
            boosting_type='gbdt', n_estimators=10, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=10, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelStacking(df, features, 'target', out_path='./test_out',
                             model_params=model_params, n_fold=n_fold)

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