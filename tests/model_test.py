import os
import random
import unittest
import joblib
import time
import pandas as pd
import numpy as np
import sys

from didtool.model import LGBModelSingle, LGBModelStacking
from didtool.split import split_data_random, split_data_stacking
from didtool.logger import Logger
from pathlib import Path


class TestModel(unittest.TestCase):
    def setUp(self):
        self.out_path = Path(__file__).parent / 'test_out'
        for root, dirs, files in os.walk(str(self.out_path), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def test_model_single(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        df['v5'] = df['v5'].apply(lambda x: (x + 1) * 2).astype('category')

        features = [col for col in df.columns.values if col != 'target']
        # split_data
        data = split_data_random(df, 0.6, 0.2)

        model_params = dict(
            boosting_type='gbdt', n_estimators=10, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=20, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelSingle(data, features, 'target', out_path=str(self.out_path),
                           model_params=model_params)

        # test train
        m.train(early_stopping_rounds=10, save_learn_curve=False)
        m.evaluate()

        # test update model params
        m.update_model_params({'n_estimators': 100})
        m.train(early_stopping_rounds=10, save_learn_curve=True)

        # test evaluate
        result = m.evaluate()
        print(result)

        # test save_feature_importance
        m.save_feature_importance()

        # test export
        m.export(export_pkl=True)

        date_str = time.strftime("%Y%m%d")
        m_load = joblib.load(str(self.out_path / ('model_%s.pkl' % date_str)))
        data['v5'] = data['v5'].astype(int)
        new_res = np.around(m_load.predict_proba(data[features][:10])[:, 1], 6)
        expect_res = round(result['prob'], 6)[:10].tolist()
        self.assertListEqual(list(new_res), expect_res)

    def test_model_single_with_woe_encoder(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        df['v5'] = df['v5'].apply(lambda x: (x + 1) * 2).astype('category')
        df['v7'] = df['target'].apply(
            lambda x: random.randint(0, 1) * x + random.randint(0, 2) * (1 - x))

        features = [col for col in df.columns.values if col != 'target']
        # split_data
        data = split_data_random(df, 0.6, 0.2)

        model_params = dict(
            boosting_type='gbdt', n_estimators=100, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=20, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelSingle(data, features, 'target', out_path=str(self.out_path),
                           model_params=model_params, woe_features=['v7'],
                           need_pmml=False)

        # test train
        m.train(early_stopping_rounds=10, eval_metric='auc',
                save_learn_curve=True)
        result = m.evaluate()
        m.save_feature_importance()
        m.export(export_pkl=True)

        date_str = time.strftime("%Y%m%d")
        m_load = joblib.load(str(self.out_path / ('model_%s.pkl' % date_str)))
        data['v5'] = data['v5'].astype('category')
        new_res = np.around(m_load.predict_proba(data[features][:10])[:, 1], 6)
        expect_res = round(result['prob'], 6)[:10].tolist()
        self.assertListEqual(list(new_res), expect_res)

        # test new category value
        data['v5'] = data['v5'].astype(int)
        data.loc[0, 'v5'] = 999
        print(m_load.predict_proba(data[features][:1])[:, 1])

    def test_model_stacking(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
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
        m = LGBModelStacking(df, features, 'target', out_path=str(self.out_path),
                             model_params=model_params, n_fold=n_fold)

        # test train
        m.train(save_learn_curve=False, eval_metric='auc')

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
        m.export(export_pkl=True)

    def test_model_stacking_with_woe_encoder(self):
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        df['v5'] = df['v5'].astype('category')
        df['v7'] = df['target'].apply(
            lambda x: random.randint(0, 1) * x + random.randint(0, 2) * (1 - x))

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
        m = LGBModelStacking(df, features, 'target', out_path=str(self.out_path),
                             model_params=model_params, n_fold=n_fold,
                             woe_features=['v7'], need_pmml=False)

        # test train
        m.train(save_learn_curve=False, eval_metric='auc')

        # test evaluate
        result = m.evaluate()
        print(result)

        # test save_feature_importance
        m.save_feature_importance()

        # test export
        m.export()

    def test_optimize_model_param(self):
        # log config
        log_file_dir = Path(__file__).parent / "test_out/bayes_parameter_search.txt"
        if log_file_dir.exists():
            sys.stdout = Logger(log_file_dir)
        else:
            open(log_file_dir, "w").close()
            sys.stdout = Logger(log_file_dir)
        # data read and split
        df = pd.read_csv(str(Path(__file__).parent / 'samples.csv'))
        df['v5'] = df['v5'].astype('category')
        features = [col for col in df.columns.values if col != 'target']
        data = split_data_random(df, 0.6, 0.2)

        model_params = dict(
            boosting_type='gbdt', n_estimators=10, learning_rate=0.05,
            max_depth=5, feature_fraction=1, bagging_fraction=1, reg_alpha=1,
            reg_lambda=1, min_data_in_leaf=20, random_state=27,
            class_weight='balanced'
        )
        m = LGBModelSingle(data, features, 'target', out_path=str(self.out_path),
                           model_params=model_params)

        searching_space = {
            'n_estimators': (100, 1500),
            'num_leaves': (32, 64),
            'learning_rate': (0.001, 0.1),
            'scale_pos_weight': (5, 20),
            'max_depth': (4, 7),
            'reg_lambda': (0, 1),
            'reg_alpha': (0, 1),
        }
        # test parameters searching
        m.optimize_model_param(searching_space)
