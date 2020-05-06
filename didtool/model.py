import time
import os

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection import KFold
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_column_transformer
import matplotlib.pyplot as plt


class LGBModelSingle:
    """
    Class for build LGBMClassifier single model.

    Implements several different methods to build & export model:

    Parameters
    --------
    data : DataFrame
        the whole data for training and testing

    target : str(dafault='target')
        label name in data

    out_path: str, output path

    model_name: str, name of model

    feature_names: str or list(default='auto')
        feature_names of dateset. If set to 'auto', use all features in x_train

    model_params : dict
        params for LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/Parameters.html)

    Attributes
    --------
    model : object of LGBMClassifier
        model for training & predicting

    pipeline : object of PMMLPipeline
        pipeline for exporting PMML model file

    used_features : list of features used in model
    """

    def __init__(self, data, feature_names, target='target', out_path='out',
                 model_params={}, model_name='model'):
        self.data = data
        self.target = target
        self.model_name = model_name

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        if feature_names == 'auto':
            self.feature_names = list(data.columns.values)
        elif type(feature_names) == list:
            self.feature_names = feature_names
        else:
            raise Exception("param `feature_names` must be a list or 'auto'")

        dtypes = {feat: data[feat].dtype for feat in self.feature_names}
        mapper, categorical_feature = make_lightgbm_column_transformer(
            dtypes, missing_value_aware=True)
        model_params['categorical_feature'] = categorical_feature
        model_params['feature_name'] = self.feature_names
        self.model = lgb.LGBMClassifier(**model_params)
        self.pipeline = PMMLPipeline([("mapper", mapper),
                                      ("model", self.model)])

        self.used_features = None

    def split_data(self, train_mask, val_mask):
        """
        Split data set into train/val/test part.
        Add a group column into `self.data`:
            - 0: training data set
            - 1: validation data set
            - 2: testing data set

        Parameters
        --------
        train_mask : array or series of bool
            used to split train data set

        val_mask : array or series of bool
            used to split validation data set
        """
        self.data["group"] = -1
        self.data.loc[train_mask, "group"] = 0
        self.data.loc[val_mask, "group"] = 1
        self.data.loc[~(train_mask | val_mask), "group"] = 2

    def train(self, early_stopping_rounds=20, eval_metric="binary_logloss",
              save_learn_curve=False):
        """
        train model

        Parameters
        --------
        early_stopping_rounds : int(default=20)
            Activates early stopping. The model will train until the validation
            score stops improving in recent `early_stopping_rounds` round(s).
        eval_metric: str(default='binary_logloss')
            usually use 'binary_logloss' or 'auc'
        save_learn_curve : bool(default=False)
            whether save learn curve
        """

        train_data = self.data[self.data.group == 0]
        val_data = self.data[self.data.group == 1]

        eval_set = [
            (train_data[self.feature_names], train_data[self.target]),
            (val_data[self.feature_names], val_data[self.target])
        ]
        self.pipeline.fit(train_data[self.feature_names],
                          train_data[self.target],
                          model__early_stopping_rounds=early_stopping_rounds,
                          model__eval_set=eval_set,
                          model__eval_metric=eval_metric)

        if save_learn_curve:
            result = self.model.evals_result_
            train_res = result.get("valid_0")
            val_res = result.get("valid_1")
            epochs = len(train_res["binary_logloss"])

            plt.figure()
            plt.plot(list(range(epochs)), train_res["binary_logloss"],
                     label="train", color='red')
            plt.plot(list(range(epochs)), val_res["binary_logloss"],
                     label="validation", color='blue')

            plt.xlabel('epoch')
            plt.ylabel('logloss')
            plt.legend()
            plt.title('learning curve')
            plt.savefig(os.path.join(self.out_path, 'learn_curve.png'))

    def evaluate(self):
        """
        Evaluate model and get prediction for every sample.

        Returns
        -------
        result : DataFrame
            keep columns from `self.data` except features,
            then append prediction columns.
        """
        result = self.data.drop(self.feature_names, axis=1)
        result['prob'] = self.model.predict_proba(
            self.data[self.feature_names])[:, -1]

        print('train AUC: %.5f' % roc_auc_score(
            result[result.group == 0][self.target],
            result[result.group == 0]['prob']))
        print('val AUC: %.5f' % roc_auc_score(
            result[result.group == 1][self.target],
            result[result.group == 1]['prob']))
        print('test AUC: %.5f' % roc_auc_score(
            result[result.group == 2][self.target],
            result[result.group == 2]['prob']))
        return result

    def save_feature_importance(self):
        """
        Save feature importance
        """
        imp_score = self.model.feature_importances_
        df_imp = pd.DataFrame(
            {'feature': self.feature_names, 'importance': imp_score})
        self.used_features = list(df_imp[df_imp.importance > 0].feature.values)

        df_imp = df_imp.sort_values(by='importance', ascending=False)
        df_imp.to_csv(os.path.join(self.out_path, "feature_importance.csv"),
                      index=False)

        feature_file = open(os.path.join(self.out_path, 'feature.txt'), 'w')
        feature_file.writelines([col + '\n' for col in self.used_features])
        feature_file.close()

        plt.figure()
        df_imp[:20].plot.barh(x='feature', y='importance', legend=False,
                              figsize=(18, 10))
        plt.title('Model Feature Importances')
        plt.xlabel('Feature Importance')
        plt.savefig(os.path.join(self.out_path, 'feature_importance.png'))

    def export(self, export_pmml=True, export_pkl=False):
        """
        Export trained model

        Parameters
        --------
        export_pmml : bool(default=True)
            export model as PMML file
        export_pkl: bool(default=False)
            export model as pkl file
        """
        date_str = time.strftime("%Y%m%d")

        if export_pmml:
            pmml_file = "%s_%s.pmml" % (self.model_name, date_str)
            sklearn2pmml(self.pipeline, os.path.join(self.out_path, pmml_file),
                         with_repr=False)

        if export_pkl:
            from sklearn.externals import joblib
            pkl_file = "%s_%s.pkl" % (self.model_name, date_str)
            joblib.dump(self.model, os.path.join(self.out_path, pkl_file))


class LGBModelStacking:
    """
    Class for build LGBMClassifier stacking models.

    Implements several different methods to build & export model:

    Parameters
    --------
    data : DataFrame
        the whole data for training and testing

    target : str(dafault='target')
        label name in data

    n_fold : int
        split num of training dataset

    out_path: str, output path

    model_name: str, name of model

    feature_names: list of features
        feature_names of dateset.

    model_params : dict
        params for LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/Parameters.html)

    Attributes
    --------
    models : list of LGBMClassifier, length equals to `n_fold`
        models for training & predicting

    pipelines : list of PMMLPipeline, length equals to `n_fold`
        pipelines for exporting PMML model file

    used_features : lists of features used in models
    """

    def __init__(self, data, feature_names, target='target', out_path='out',
                 model_params={}, n_fold=5, model_name='model'):
        self.data = data
        self.target = target
        self.n_fold = n_fold
        self.model_name = model_name

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        self.feature_names = feature_names

        dtypes = {feat: data[feat].dtype for feat in self.feature_names}
        mapper, categorical_feature = make_lightgbm_column_transformer(
            dtypes, missing_value_aware=True)
        model_params['categorical_feature'] = categorical_feature
        model_params['feature_name'] = self.feature_names
        self.models = []
        self.pipelines = []
        for _ in range(n_fold):
            model = lgb.LGBMClassifier(**model_params)
            pipeline = PMMLPipeline([("mapper", mapper),
                                     ("model", model)])
            self.models.append(model)
            self.pipelines.append(pipeline)
        self.used_features = None

    def split_data(self, oot_mask, random_state=None):
        """
        Split data into train/oot part, and then split train dataset into
        `self.k_fold` part

        After split, a column named 'fold' will be append to `self.data`.
            - -1: oot data set
            - [0, n_fold): k-fold of training data set

        Parameters
        --------
        oot_mask : array or series of bool
            used to split oot dataset
        random_state : int or None
            used for KFold
        """
        train_data = self.data[~oot_mask]
        k_fold = KFold(n_splits=self.n_fold, shuffle=True,
                       random_state=random_state)
        k_fold_index = []
        for _, indexes in k_fold.split(train_data[self.target]):
            k_fold_index.append(indexes)

        train_data.reset_index(inplace=True)
        train_data.loc[:, "fold"] = self.n_fold
        for k in range(0, self.n_fold):
            train_data.loc[k_fold_index[k], 'fold'] = k

        self.data.reset_index(inplace=True)
        self.data = pd.merge(self.data, train_data[["index", "fold"]],
                             how="left", on="index")
        self.data["fold"].fillna(-1, inplace=True)
        self.data.drop("index", axis=1, inplace=True)

    def train(self, early_stopping_rounds=20, eval_metric="binary_logloss",
              save_learn_curve=False):
        """
        train model

        Parameters
        --------
        early_stopping_rounds : int(default=20)
            Activates early stopping. The model will train until the validation
            score stops improving in recent `early_stopping_rounds` round(s).
        eval_metric: str(default='binary_logloss')
            usually use 'binary_logloss' or 'auc'
        save_learn_curve : bool
            whether save learning curve of models
        """
        for k in range(0, self.n_fold):
            train_k = self.data[
                (self.data.fold >= 0) & (self.data.fold != k)]
            val_k = self.data[self.data.fold == k]
            eval_set = [(train_k[self.feature_names], train_k[self.target]),
                        (val_k[self.feature_names], val_k[self.target])]
            self.pipelines[k].fit(
                train_k[self.feature_names], train_k[self.target],
                model__early_stopping_rounds=early_stopping_rounds,
                model__eval_set=eval_set, model__eval_metric=eval_metric
            )
            if save_learn_curve:
                result = self.models[k].evals_result_
                train_res = result.get("valid_0")
                val_res = result.get("valid_1")
                epochs = len(train_res["binary_logloss"])

                plt.figure()
                plt.plot(list(range(epochs)), train_res["binary_logloss"],
                         label="train", color='red')
                plt.plot(list(range(epochs)), val_res["binary_logloss"],
                         label="validation", color='blue')

                plt.xlabel('epoch')
                plt.ylabel('logloss')
                plt.legend()
                plt.title('learning curve')
                plt.savefig(
                    os.path.join(self.out_path, 'learn_curve_%d.png' % k))

    def save_feature_importance(self):
        """
        Save feature importance
        """
        self.used_features = []
        for i in range(self.n_fold):
            imp_score = self.models[i].feature_importances_
            df_imp = pd.DataFrame(
                {'feature': self.feature_names, 'importance': imp_score})
            used_cols = list(df_imp[df_imp.importance > 0].feature.values)
            self.used_features.append(used_cols)
            # save importance stats
            df_imp.sort_values(by='importance', ascending=False).to_csv(
                os.path.join(self.out_path, "feature_importance_%d.csv" % i),
                index=False
            )

            # save feature names used in model
            feature_file = open(
                os.path.join(self.out_path, 'feature_%d.txt' % i), 'w')
            feature_file.writelines([col + '\n' for col in used_cols])
            feature_file.close()

            plt.figure()
            df_imp[:20].plot.barh(x='feature', y='importance', legend=False,
                                  figsize=(18, 10))
            plt.title('Model(%d) Feature Importances' % i)
            plt.xlabel('Feature Importance')
            plt.savefig(
                os.path.join(self.out_path, 'feature_importance_%d.png' % i))

    def evaluate(self):
        """
        Evaluate models and get final prediction of every sample.

        Returns
        -------
        result : DataFrame
            keep columns from `self.data` except features,
            then append prediction columns.
        """
        result = self.data.drop(self.feature_names, axis=1)
        for k in range(0, self.n_fold):
            result["prob_%d" % k] = self.models[k].predict_proba(
                self.data[self.feature_names])[:, -1]

        def _get_final_prob(probs, fold):
            if fold >= 0:
                return probs[fold]
            return np.mean(probs)

        result['final_prob'] = result.apply(
            lambda x: _get_final_prob(
                [x["prob_%d" % i] for i in range(0, self.n_fold)],
                int(x["fold"])), axis=1)
        return result

    def export(self, export_pmml=True, export_pkl=False):
        """
        Export trained models

        Parameters
        --------
        export_pmml : bool(default=True)
            export model as PMML file
        export_pkl: bool(default=False)
            export model as pkl file
        """
        date_str = time.strftime("%Y%m%d")

        for i in range(self.n_fold):
            if export_pmml:
                pmml_file = "%s_%d_%s.pmml" % (self.model_name, i, date_str)
                sklearn2pmml(self.pipelines[i],
                             os.path.join(self.out_path, pmml_file),
                             with_repr=False)

            if export_pkl:
                from sklearn.externals import joblib
                pkl_file = "%s_%d_%s.pkl" % (self.model_name, i, date_str)
                joblib.dump(self.models[i],
                            os.path.join(self.out_path, pkl_file))
