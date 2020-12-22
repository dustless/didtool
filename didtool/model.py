import time
import os

import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.ranking import roc_auc_score
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_column_transformer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from bayes_opt import BayesianOptimization


class LGBModelSingle:
    """
    Class for build LGBMClassifier single model.

    Implements several different methods to build & export model:

    Parameters
    --------
    data : DataFrame
        The whole data for training and evaluating. There should be a column
        named `group_col` indicate data group:
        - 0: training data set
        - 1: validation data set
        - 2: testing data set

    target : str(dafault='target')
        label name in data

    group_col : str(default='group')
        group column name

    out_path: str, output path

    model_name: str, name of model

    feature_names: list
        used to filter features from data. All features should be in data.

    model_params : dict
        params for LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/Parameters.html)

    Attributes
    --------
    model : object of LGBMClassifier
        model for training & predicting

    pipeline : object of PMMLPipeline
        pipeline for exporting PMML model file

    importance_df : DataFrame
        records feature importance of model

    _model_params : dict
        params for LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/Parameters.html)

    _mapper : object of ColumnTransformer
    """

    def __init__(self, data, feature_names, target='target', group_col='group',
                 out_path='out', model_params=None, model_name='model'):
        # check data columns
        if target not in data:
            raise Exception("column '%s' is missing from data" % target)
        if group_col not in data:
            raise Exception("column '%s' is missing from data" % group_col)
        for col in feature_names:
            if col not in data:
                raise Exception("column %s not in `data`, "
                                "but appears in `feature_names`" % col)

        self.data = data
        self.target = target
        self.model_name = model_name
        self.group_col = group_col

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        self.feature_names = feature_names

        self.importance_df = None
        self.model = None
        self.pipeline = None

        dtypes = {feat: self.data[feat].dtype for feat in self.feature_names}
        mapper, _ = make_lightgbm_column_transformer(dtypes,
                                                     missing_value_aware=True)
        self._model_params = {}
        self._mapper = mapper
        self.update_model_params(model_params)

    def update_model_params(self, model_params):
        """
        update model params and create new model
        """
        if model_params:
            self._model_params.update(model_params)

        # create new model by updated model params
        self.model = lgb.LGBMClassifier(**self._model_params)
        self.pipeline = PMMLPipeline([("mapper", self._mapper),
                                      ("model", self.model)])

    def train(self, early_stopping_rounds=20, eval_metric="binary_logloss",
              verbose=-1, save_learn_curve=False):
        """
        train model

        Parameters
        --------
        early_stopping_rounds : int(default=20)
            Activates early stopping. The model will train until the validation
            score stops improving in recent `early_stopping_rounds` round(s).
        eval_metric: str(default='binary_logloss')
            usually use 'binary_logloss' or 'auc'
        verbose : bool or int, optional (default=True)
            Requires at least one evaluation data.
            If True, the eval metric on the eval set is printed at each
             boosting stage.
            If int, the eval metric on the eval set is printed at every
             ``verbose`` boosting stage.
        save_learn_curve : bool(default=False)
            whether save learn curve
        """
        train_data = self.data[self.data[self.group_col] == 0]
        val_data = self.data[self.data[self.group_col] == 1]

        eval_set = [
            (train_data[self.feature_names], train_data[self.target]),
            (val_data[self.feature_names], val_data[self.target])
        ]
        self.pipeline.fit(train_data[self.feature_names],
                          train_data[self.target],
                          model__early_stopping_rounds=early_stopping_rounds,
                          model__eval_set=eval_set,
                          model__eval_metric=eval_metric,
                          model__verbose=verbose)

        imp_score = self.model.feature_importances_
        self.importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp_score}
        ).sort_values(by='importance', ascending=False)

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
            result[result[self.group_col] == 0][self.target],
            result[result[self.group_col] == 0]['prob']))
        print('val AUC: %.5f' % roc_auc_score(
            result[result[self.group_col] == 1][self.target],
            result[result[self.group_col] == 1]['prob']))
        if any(result[self.group_col] == -1):
            print('test AUC: %.5f' % roc_auc_score(
                result[result[self.group_col] == -1][self.target],
                result[result[self.group_col] == -1]['prob']))
        return result

    def _save_used_features(self):
        """
        save used features of model as txt file
        """
        used_cols = list(self.importance_df[self.importance_df.importance > 0]
                         .feature.values)
        # keep the same order as `self.feature_names`
        used_cols = [col for col in self.feature_names if col in used_cols]

        # save feature names used in model
        feature_file = open(os.path.join(self.out_path, 'feature.txt'), 'w')
        feature_file.writelines([col + '\n' for col in used_cols])
        feature_file.close()

    def save_feature_importance(self, plot=True):
        """
        Save feature importance
        """
        self.importance_df.to_csv(
            os.path.join(self.out_path, "feature_importance.csv"), index=False)

        if plot:
            plt.figure()
            self.importance_df[:20].plot.barh(
                x='feature', y='importance', legend=False, figsize=(18, 10))
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
        # save used features
        self._save_used_features()

        date_str = time.strftime("%Y%m%d")

        if export_pmml:
            pmml_file = "%s_%s.pmml" % (self.model_name, date_str)
            sklearn2pmml(self.pipeline, os.path.join(self.out_path, pmml_file),
                         with_repr=False)

        if export_pkl:
            pkl_file = "%s_%s.pkl" % (self.model_name, date_str)
            joblib.dump(self.model, os.path.join(self.out_path, pkl_file))

    def optimize_model_param(self, searching_space, n_iter=10):
        """
        start the  bayes search

        Parameters
        --------
        n_iter: the number of the total finds loop
        searching_space: the searching space of parameters
        """

        def _convert_param_types(model_param):
            """
            convert model param types to correct types
            """
            int_sets = (
                "n_estimators", "num_leaves", "max_depth", "subsample_for_bin",
                "min_child_samples", "max_bin"
            )
            float_sets = (
                "learning_rate", "reg_lambda", "reg_alpha", "subsample",
                "min_child_weight", "min_split_gain", "scale_pos_weight"
            )
            for k, v in model_param.items():
                if k in int_sets:
                    model_param[k] = int(v)
                elif k in float_sets:
                    model_param[k] = float(v)
            return model_param

        def _model_cv(**model_param):
            """
            define the process of paramaters searching and the feedback
            indicators
            """
            model_param = _convert_param_types(model_param)
            # setting the indicator to loop
            val = np.mean(cross_val_score(
                lgb.LGBMClassifier(**model_param), x_train, y_train,
                scoring='roc_auc', cv=5
            ))

            return val

        # init parameters
        tran_data = self.data[self.data[self.group_col] == 0]
        x_train = tran_data[self.feature_names]
        y_train = tran_data[self.target]

        model_bo = BayesianOptimization(
            _model_cv,
            searching_space
        )
        # start optimize
        model_bo.maximize(n_iter=n_iter)
        optimized_param = _convert_param_types(model_bo.max["params"])
        print("the optimized searching params are : ", optimized_param)

        self.update_model_params(optimized_param)
        print("the latest model params are updated to: ", self._model_params)


class LGBModelStacking:
    """
    Class for build LGBMClassifier stacking models.

    Implements several different methods to build & export model:

    Parameters
    --------
    data : DataFrame
        The whole data for training and evaluating. There should be a column
        named `group` indicate data group:
        - -1: oot data set
        - [0, n_fold): k-fold of training data set

    target : str(dafault='target')
        label name in data

    group_col : str(default='group')
        group column name

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

    importance_dfs : list of DataFrame
        records feature importance of all models

    _model_params : dict
        params for LGBMClassifier
        (https://lightgbm.readthedocs.io/en/latest/Parameters.html)

    _mapper : object of ColumnTransformer
    """

    def __init__(self, data, feature_names, target='target', group_col='group',
                 out_path='out', model_params=None, n_fold=5,
                 model_name='model'):
        # check data columns
        if target not in data:
            raise Exception("column '%s' is missing from data" % target)
        if group_col not in data:
            raise Exception("column '%s' is missing from data" % group_col)
        for col in feature_names:
            if col not in data:
                raise Exception("column %s not in `data`, "
                                "but appears in `feature_names`" % col)
        if data[group_col].max() != n_fold - 1:
            raise Exception("training data groups(%d) does not match the fold"
                            " num(%d)" % (data[group_col].max() + 1, n_fold))

        self.data = data
        self.target = target
        self.group_col = group_col
        self.n_fold = n_fold
        self.model_name = model_name
        self.feature_names = feature_names

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        self.models = None
        self.pipelines = None
        self.importance_dfs = None

        dtypes = {feat: self.data[feat].dtype for feat in self.feature_names}
        mapper, _ = make_lightgbm_column_transformer(dtypes,
                                                     missing_value_aware=True)
        self._model_params = {}
        self._mapper = mapper
        self.update_model_params(model_params)

    def update_model_params(self, model_params):
        """
        update model params and create new models
        """
        if model_params:
            self._model_params.update(model_params)

        self.models = []
        self.pipelines = []
        for _ in range(self.n_fold):
            model = lgb.LGBMClassifier(**self._model_params)
            pipeline = PMMLPipeline([("mapper", self._mapper),
                                     ("model", model)])
            self.models.append(model)
            self.pipelines.append(pipeline)

    def train(self, early_stopping_rounds=20, eval_metric="binary_logloss",
              verbose=-1, save_learn_curve=False):
        """
        train model

        Parameters
        --------
        early_stopping_rounds : int(default=20)
            Activates early stopping. The model will train until the validation
            score stops improving in recent `early_stopping_rounds` round(s).
        eval_metric: str(default='binary_logloss')
            usually use 'binary_logloss' or 'auc'
        verbose : bool or int, optional (default=True)
            Requires at least one evaluation data.
            If True, the eval metric on the eval set is printed at each
             boosting stage.
            If int, the eval metric on the eval set is printed at every
             ``verbose`` boosting stage.
        save_learn_curve : bool
            whether save learning curve of models
        """
        self.importance_dfs = []
        for k in range(0, self.n_fold):
            train_k = self.data[(self.data[self.group_col] >= 0) &
                                (self.data[self.group_col] != k)]
            val_k = self.data[self.data[self.group_col] == k]
            eval_set = [(train_k[self.feature_names], train_k[self.target]),
                        (val_k[self.feature_names], val_k[self.target])]
            self.pipelines[k].fit(
                train_k[self.feature_names], train_k[self.target],
                model__early_stopping_rounds=early_stopping_rounds,
                model__eval_set=eval_set, model__eval_metric=eval_metric,
                model__verbose=verbose
            )

            # append feature importance of model k
            imp_score = self.models[k].feature_importances_
            df_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': imp_score}
            ).sort_values(by='importance', ascending=False)
            self.importance_dfs.append(df_imp)

            # if needed, plot learn curve
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

    def _save_used_features(self):
        """
        save used features of models as txt file
        """
        for i in range(self.n_fold):
            df_imp = self.importance_dfs[i]
            used_cols = list(df_imp[df_imp.importance > 0].feature.values)
            # keep the same order as `self.feature_names`
            used_cols = [col for col in self.feature_names if col in used_cols]

            # save feature names used in model
            feature_file = open(
                os.path.join(self.out_path, 'feature_%d.txt' % i), 'w')
            feature_file.writelines([col + '\n' for col in used_cols])
            feature_file.close()

    def save_feature_importance(self, plot=True):
        """
        Save feature importance
        """
        for i in range(self.n_fold):
            # save importance stats
            df_imp = self.importance_dfs[i]
            df_imp.to_csv(
                os.path.join(self.out_path, "feature_importance_%d.csv" % i),
                index=False
            )

            if plot:
                plt.figure()
                df_imp[:20].plot.barh(x='feature', y='importance', legend=False,
                                      figsize=(18, 10))
                plt.title('Model(%d) Feature Importances' % i)
                plt.xlabel('Feature Importance')
                plt.savefig(os.path.join(self.out_path,
                                         'feature_importance_%d.png' % i))

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

        result['prob'] = result.apply(
            lambda x: _get_final_prob(
                [x["prob_%d" % i] for i in range(0, self.n_fold)],
                int(x[self.group_col])), axis=1)

        train_res = result[result[self.group_col] >= 0]
        for k in range(self.n_fold):
            print("\n**** model_%d ****" % k)
            print('train AUC: %.5f' % roc_auc_score(
                train_res[train_res[self.group_col] != k][self.target],
                train_res[train_res[self.group_col] != k]['prob']))
            print('val AUC: %.5f' % roc_auc_score(
                train_res[train_res[self.group_col] == k][self.target],
                train_res[train_res[self.group_col] == k]['prob']))

        print("\n**** model_stacking ****")
        print('total train AUC: %.5f' % roc_auc_score(
            result[result[self.group_col] >= 0][self.target],
            result[result[self.group_col] >= 0]['prob']))
        print('total test AUC: %.5f' % roc_auc_score(
            result[result[self.group_col] == -1][self.target],
            result[result[self.group_col] == -1]['prob']))
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
        # save used features
        self._save_used_features()

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
