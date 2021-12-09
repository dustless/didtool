# coding: utf-8
import time
import os

import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from bayes_opt import BayesianOptimization


from .encoder import WOEEncoder, WrappedLabelEncoder
from .utils import is_categorical


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

    woe_features: list or None
        features need to be transformed to woe value

    need_pmml: bool, True/False
        If pmml file is needed, customized encoders cannot be used

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

    mapper : object of ColumnTransformer
    """

    def __init__(self, data, feature_names, target='target', group_col='group',
                 out_path='.', model_params=None, model_name='model',
                 woe_features=None, need_pmml=True):
        # check data columns
        if target not in data:
            raise Exception("column '%s' is missing from data" % target)
        if group_col not in data:
            raise Exception("column '%s' is missing from data" % group_col)
        if need_pmml and woe_features:
            raise Exception("if `woe_features` is specified, then pmml is not "
                            "supported, `need_pmml` should be set to False")
        for col in feature_names:
            if col not in data:
                raise Exception("column %s not in `data`, "
                                "but appears in `feature_names`" % col)

        self.data = data
        self.target = target
        self.model_name = model_name
        self.group_col = group_col
        self.woe_features = woe_features or []
        self.need_pmml = need_pmml

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        self.feature_names = feature_names

        self.importance_df = None
        self.model = None
        self.pipeline = None

        transformers = list()
        cat_features = list()
        i = 0
        for feature in feature_names:
            # 如果特征需要进行WOE编码，则需要使用WOEEncoder转换
            if feature in self.woe_features:
                transformers.append([feature, WOEEncoder(), feature])
            # 普通类别型特征，使用PMMLLabelEncoder进行编码转换
            elif is_categorical(self.data[feature]):
                label_enc = PMMLLabelEncoder(missing_values=-1) \
                    if self.need_pmml else \
                    WrappedLabelEncoder(missing_values=-1)
                transformers.append((feature, label_enc, [feature]))
                cat_features.append(i)
            else:
                transformers.append((feature, "passthrough", [feature]))
            i += 1

        self._model_params = {"categorical_feature": cat_features}
        self.mapper = ColumnTransformer(transformers)
        self.update_model_params(model_params)

    def update_model_params(self, model_params):
        """
        update model params and create new model
        """
        if model_params:
            self._model_params.update(model_params)

        # create new model by updated model params
        self.model = lgb.LGBMClassifier(**self._model_params)
        self.pipeline = PMMLPipeline([("mapper", self.mapper),
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

        # step0: fit mapper
        self.pipeline[0].fit(train_data[self.feature_names],
                             train_data[self.target])

        # step1: fit model
        trans_train_data = self.pipeline[0].transform(
            train_data[self.feature_names])
        trans_val_data = self.pipeline[0].transform(
            val_data[self.feature_names])
        eval_set = [
            (trans_train_data, train_data[self.target]),
            (trans_val_data, val_data[self.target])
        ]
        self.pipeline[-1].fit(
            trans_train_data,
            train_data[self.target],
            early_stopping_rounds=early_stopping_rounds,
            eval_set=eval_set,
            eval_metric=eval_metric,
            verbose=verbose
        )

        imp_score = self.model.feature_importances_
        self.importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp_score}
        ).sort_values(by='importance', ascending=False)

        if save_learn_curve:
            result = self.model.evals_result_
            train_res = result.get("training")
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
        result['prob'] = self.pipeline.predict_proba(
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
        with open(os.path.join(self.out_path, 'used_feature.txt'), 'w') as f:
            f.writelines([col + '\n' for col in used_cols])

    def _save_feature_list(self):
        """
        save all features of input as a txt file
        """
        dtypes = self.data[self.feature_names].dtypes
        with open(os.path.join(self.out_path, 'feature_list.txt'), 'w') as f:
            for item in dtypes.items():
                f.writelines("%s\t%s\n" % (item[0], item[1]))

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

    def export(self, export_pkl=True):
        """
        Export trained model

        Parameters
        --------
        export_pkl: bool(default=False)
            export model as pkl file
        """
        # save used features
        self._save_used_features()
        self._save_feature_list()

        date_str = time.strftime("%Y%m%d")

        if self.need_pmml:
            pmml_file = "%s_%s.pmml" % (self.model_name, date_str)
            sklearn2pmml(self.pipeline, os.path.join(self.out_path, pmml_file),
                         with_repr=False)

        if export_pkl:
            pkl_file = "%s_%s.pkl" % (self.model_name, date_str)
            joblib.dump(self.pipeline, os.path.join(self.out_path, pkl_file))

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
            base_param = self._model_params.copy()
            base_param.update(_convert_param_types(model_param))
            # setting the indicator to loop
            val = np.mean(cross_val_score(
                lgb.LGBMClassifier(**base_param), x_train, y_train,
                scoring='roc_auc', cv=5
            ))

            return val

        # init parameters
        train_data = self.data[self.data[self.group_col] == 0]

        # fit mapper first
        self.pipeline[0].fit(train_data[self.feature_names],
                             train_data[self.target])

        x_train = self.pipeline[0].transform(train_data[self.feature_names])
        y_train = train_data[self.target]

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

    woe_features: list or None
        features need to be transformed to woe value

    need_pmml: bool, True/False
        If pmml file is needed, customized encoders cannot be used

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
    """

    def __init__(self, data, feature_names, target='target', group_col='group',
                 out_path='.', model_params=None, n_fold=5,
                 model_name='model', woe_features=None, need_pmml=True):
        # check data columns
        if target not in data:
            raise Exception("column '%s' is missing from data" % target)
        if group_col not in data:
            raise Exception("column '%s' is missing from data" % group_col)
        if need_pmml and woe_features:
            raise Exception("if `woe_features` is specified, then pmml is not "
                            "supported, `need_pmml` should be set to False")
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
        self.woe_features = woe_features or []
        self.need_pmml = need_pmml and not woe_features

        self.out_path = os.path.abspath(out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print('Create directory %s' % self.out_path)

        self.mappers = []
        self.models = None
        self.pipelines = None
        self.importance_dfs = None

        cat_features = []
        for i, column in enumerate(self.feature_names):
            if is_categorical(self.data[column]) and \
                    column not in self.woe_features:
                cat_features.append(i)

        for _ in range(self.n_fold):
            transformers = list()
            cat_features = list()
            i = 0
            for feature in feature_names:
                # 如果特征需要进行WOE编码，则需要使用WOEEncoder转换
                if feature in self.woe_features:
                    transformers.append([feature, WOEEncoder(), feature])
                # 普通类别型特征，使用PMMLLabelEncoder进行编码转换
                elif is_categorical(self.data[feature]):
                    label_enc = PMMLLabelEncoder(missing_values=-1) \
                        if self.need_pmml else \
                        WrappedLabelEncoder(missing_values=-1)
                    transformers.append((feature, label_enc, [feature]))
                    cat_features.append(i)
                else:
                    transformers.append((feature, "passthrough", [feature]))
                i += 1
            self.mappers.append(ColumnTransformer(transformers))

        self._model_params = {"categorical_feature": cat_features}
        self.update_model_params(model_params)

    def update_model_params(self, model_params):
        """
        update model params and create new models
        """
        if model_params:
            self._model_params.update(model_params)

        self.models = []
        self.pipelines = []
        for i in range(self.n_fold):
            model = lgb.LGBMClassifier(**self._model_params)
            pipeline = PMMLPipeline([("mapper", self.mappers[i]),
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
            train_k = self.data[(self.data[self.group_col] >= 0) & (self.data[self.group_col] != k)]
            val_k = self.data[self.data[self.group_col] == k]

            # step0: fit mapper
            self.pipelines[k][0].fit(train_k[self.feature_names],
                                     train_k[self.target])

            # step1: fit model
            trans_train_k = self.pipelines[k][0].transform(
                train_k[self.feature_names])
            trans_val_k = self.pipelines[k][0].transform(
                val_k[self.feature_names])
            eval_set = [
                (trans_train_k, train_k[self.target]),
                (trans_val_k, val_k[self.target])
            ]
            self.pipelines[k][-1].fit(
                trans_train_k,
                train_k[self.target],
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=verbose
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
                train_res = result.get("training")
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
            with open(os.path.join(self.out_path, 'used_feature_%d.txt' % i),
                      'w') as f:
                f.writelines([col + '\n' for col in used_cols])

    def _save_feature_list(self):
        """
        save all features of input as a txt file
        """
        dtypes = self.data[self.feature_names].dtypes
        with open(os.path.join(self.out_path, 'feature_list.txt'), 'w') as f:
            for item in dtypes.items():
                f.writelines("%s\t%s\n" % (item[0], item[1]))

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
            result["prob_%d" % k] = self.pipelines[k].predict_proba(
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
                train_res[train_res[self.group_col] != k]['prob_%d' % k]))
            print('val AUC: %.5f' % roc_auc_score(
                train_res[train_res[self.group_col] == k][self.target],
                train_res[train_res[self.group_col] == k]['prob_%d' % k]))

        print("\n**** model_stacking ****")
        print('total train AUC: %.5f' % roc_auc_score(
            result[result[self.group_col] >= 0][self.target],
            result[result[self.group_col] >= 0]['prob']))
        print('total test AUC: %.5f' % roc_auc_score(
            result[result[self.group_col] == -1][self.target],
            result[result[self.group_col] == -1]['prob']))
        return result

    def export(self, export_pkl=True):
        """
        Export trained models

        Parameters
        --------
        export_pkl: bool(default=False)
            export model as pkl file
        """
        # save used features
        self._save_used_features()
        self._save_feature_list()

        date_str = time.strftime("%Y%m%d")

        for i in range(self.n_fold):
            if self.need_pmml:
                pmml_file = "%s_%d_%s.pmml" % (self.model_name, i, date_str)
                sklearn2pmml(self.pipelines[i],
                             os.path.join(self.out_path, pmml_file),
                             with_repr=False)

            if export_pkl:
                pkl_file = "%s_%d_%s.pkl" % (self.model_name, i, date_str)
                joblib.dump(self.pipelines[i],
                            os.path.join(self.out_path, pkl_file))
