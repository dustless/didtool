import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from .stats import iv_all


class Selector:
    """
    Class for performing feature selection for machine learning.

    Implements several different methods to identify features to remove

    1. Find columns with a missing percentage greater than a specified threshold
    2. Find features with IV less than a specified threshold
    3. Find collinear variables with a correlation greater than a specified
       correlation coefficient
    4. Find low importance features that do not contribute to a specified
       cumulative feature importance from the gbm

    Parameters
    --------
    data : DataFrame
        A dataset with observations in the rows and features in the columns
    label : array or series, default = None
        Array of labels for training the machine learning model to calculate
        iv and feature importance. These should be binary labels.
        If no labels are provided, then the feature importance and iv based
        methods are not available.

    Attributes
    --------

    drop_cols : dict
        # Hold all columns dropped

    missing_stats : DataFrame
        The fraction of missing values for all features

    iv_stats : DataFrame
        IV of all features

    corr_matrix : DataFrame
        All correlations between all features in the data

    record_correlated : DataFrame
        Records the pairs of collinear variables with a correlation coefficient
        above the threshold

    importance_stats : DataFrame
        All feature importances from the gradient boosting machine
    """

    def __init__(self, data, label=None):
        # DataSet and training label
        self.data = data
        self.label = label
        if label is None:
            print("No label provided. The feature importance and iv based "
                  "methods are not available")

        # DataFrame recording information about features
        self.missing_stats = None
        self.iv_stats = None
        self.corr_matrix = None
        self.record_correlated = None  # record cols dropped by correlated
        self.importance_stats = None

        # Hold all columns dropped
        self.drop_cols = []

    def drop_missing(self, missing_threshold=0.9, missing_value=None):
        """
        Drop the features with a fraction of missing values above
         `missing_threshold`

         If `missing_value` specified, then values equal to `missing_value` will
          also be treated as missing
        """
        # Calculate the fraction of missing in each column
        total = self.data.shape[0]
        if missing_value is not None:
            missing_series = (self.data.isnull() | (self.data == missing_value)
                              ).sum() / total
        else:
            missing_series = self.data.isnull().sum() / total
        missing_df = pd.DataFrame(missing_series).rename(
            columns={'index': 'feature', 0: 'missing_rate'})

        # Sort with highest number of missing values on top
        missing_df = missing_df.sort_values('missing_rate', ascending=False)
        self.missing_stats = missing_df

        # Find the columns with a missing percentage above the threshold
        to_drop_df = missing_df[missing_df.missing_rate > missing_threshold]
        to_drop = list(to_drop_df.index.values)

        self.drop_cols.extend(to_drop)
        self.data = self.data.drop(columns=to_drop)

        print('%d features with greater than %0.2f missing values.\n' %
              (len(to_drop), missing_threshold))
        return self

    def drop_low_iv(self, iv_threshold=0.02, **kwargs):
        """
        Drop the features with a IV value below `iv_threshold`
        """
        if self.label is None:
            raise ValueError("No training labels provided.")
        self.iv_stats = iv_all(self.data, self.label, **kwargs)

        to_drop_df = self.iv_stats[self.iv_stats.iv < iv_threshold]
        to_drop = list(to_drop_df.index.values)
        self.drop_cols.extend(to_drop)
        self.data = self.data.drop(columns=to_drop)

        print('%d features with iv less than %0.2f.\n' %
              (len(to_drop), iv_threshold))
        return self

    def drop_correlated(self, corr_threshold=0.9):
        """
        Drop collinear features based on the correlation coefficient between
        features. For each pair of features with a correlation coefficient
        greater than `corr_threshold`, the one with lower iv value is identified
        for removal.

        Parameters
        --------
        corr_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying
            correlation features
        """
        # Calculate IV of features if IVs have not been calculated
        if self.iv_stats is None:
            self.drop_low_iv(0)

        # Calculate the correlations between every column
        corr_matrix = self.data.corr()
        self.corr_matrix = corr_matrix

        # Extract the upper&down triangle of the correlation matrix except the
        # diagonal line
        corr = corr_matrix.where(~np.eye(corr_matrix.shape[0]).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        corr_cols = [column for column in corr.columns if
                     any(corr[column].abs() > corr_threshold)]

        # Dataframe to hold correlated pairs
        record_correlated = pd.DataFrame(
            columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of
        # correlated features
        corr_cols = sorted(corr_cols, key=lambda x: self.iv_stats['iv'][x],
                           reverse=True)
        for col in corr_cols:
            if col in record_correlated["drop_feature"].values:
                continue
            # Find the correlated features
            corr_features = list(
                corr.index[corr[col].abs() > corr_threshold])

            for feature in corr_features:
                if feature in record_correlated["drop_feature"].values:
                    continue
                # Record the information (need a temp df for now)
                temp_df = pd.DataFrame(
                    [[feature, col, corr_matrix.loc[col, feature]]],
                    columns=record_correlated.columns
                )
                # Add to dataframe
                record_correlated = pd.concat([record_correlated, temp_df],
                                              ignore_index=True)

        self.record_correlated = record_correlated
        to_drop = list(record_correlated["drop_feature"].values)
        self.drop_cols.extend(to_drop)
        self.data = self.data.drop(columns=to_drop)

        print('%d features dropped with a correlation magnitude greater'
              ' than %0.2f.\n' % (len(to_drop), corr_threshold))
        return self

    def drop_low_importance(self, cumulative_importance=0.95, run_times=10):
        """
        Drop the lowest importance features not needed to account for
        `cumulative_importance` fraction of the total feature importance
        from the gradient boosting machine.
        As an example, if cumulative
        importance is set to 0.95, this will retain only the most important
        features needed to reach 95% of the total feature importance.
        The identified features are those not needed.
        The importance of features are averaged over `run_times` to reduce
        variance.

        Uses the LightGBM implementation
        (http://lightgbm.readthedocs.io/en/latest/index.html)

        Parameters
        --------
        cumulative_importance : float between 0 and 1
            The fraction of cumulative importance to account for
        run_times : int, default = 10
            Number of times to train the gradient boosting machine

        Notes
        --------
        - The gbm is not optimized for any particular task and might need some
          hyperparameter tuning
        - Feature importance can change across runs
        """
        if self.label is None:
            raise ValueError("No training labels provided.")

        feature_names = list(self.data.columns.values)

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))
        print('Training Gradient Boosting Model...')

        # Iterate through each fold
        for _ in range(run_times):
            model = lgb.LGBMClassifier(n_estimators=1000,
                                       learning_rate=0.05, verbose=-1)

            train_features, valid_features, train_labels, valid_labels = \
                train_test_split(self.data, self.label,
                                 test_size=0.2, stratify=self.label)

            # Train the model with early stopping
            model.fit(train_features, train_labels, eval_metric='logloss',
                      eval_set=[(valid_features, valid_labels)],
                      early_stopping_rounds=20, verbose=-1)

            # Clean up memory
            gc.enable()
            del train_features, train_labels, valid_features, valid_labels
            gc.collect()

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / run_times

        importance_df = pd.DataFrame(
            {'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        importance_df = importance_df. \
            sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        importance_df['normalized'] = \
            importance_df['importance'] / importance_df['importance'].sum()
        importance_df['cumulative'] = np.cumsum(
            importance_df['normalized'])

        # Identify the features not needed to reach the cumulative_importance
        to_drop_df = importance_df[
            importance_df['cumulative'] > cumulative_importance]
        to_drop = list(to_drop_df['feature'])

        self.importance_stats = importance_df
        self.drop_cols.extend(to_drop)
        self.data = self.data.drop(columns=to_drop)

        print('\n%d features do not contribute to cumulative importance of'
              ' %0.2f.' % (len(to_drop), cumulative_importance))
        return self

    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.missing_stats is None:
            raise Exception("Missing values have not been calculated."
                            " Run `drop_missing`")

        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(self.missing_stats['missing_rate'],
                 bins=np.linspace(0, 1, 11), edgecolor='k', color='red',
                 linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel('Missing Fraction', size=14)
        plt.ylabel('Count of Features', size=14)
        plt.title("Fraction of Missing Values Histogram", size=16)

    def plot_iv(self, top_n=20):
        """Plots `top_n` features of highest iv"""
        if self.iv_stats is None:
            raise Exception('IV of features have not been calculated.'
                            ' Run `drop_low_iv`')

        # Need to adjust number of features if `top_n` is greater than the
        # features in the data
        if top_n > self.iv_stats.shape[0]:
            top_n = self.iv_stats.shape[0] - 1

        # Make a horizontal bar chart of feature iv
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot highest iv on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.iv_stats.index[:top_n]))),
                list(reversed(list(self.iv_stats['iv'][:top_n]))),
                align='center', edgecolor='k')

        # Plot labeling
        plt.xlabel('IV', size=16)
        plt.title('IV Of Features', size=18)
        plt.show()

    def plot_correlated(self, plot_all=False):
        """
        Heatmap of the correlation values.
        If plot_all = True plots all the correlations otherwise plots only
        those features that have a correlation above the threshold

        Notes
        --------
        - Not all of the plotted correlations are above the threshold because
          this plots all the variables that have been idenfitied as having even
          one correlation above the threshold
        - The features on the x-axis are those that will be removed.
          The features on the y-axis are the correlated features with those
          on the x-axis
        """

        if self.corr_matrix is None:
            raise Exception('Collinear features have not been idenfitied.'
                            ' Run `drop_collinear`.')

        if plot_all:
            corr_matrix_plot = self.corr_matrix
            title = 'All Correlations'

        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis)
            # are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[
                list(set(self.record_correlated['corr_feature'])),
                list(set(self.record_correlated['drop_feature']))]

            title = "Correlations Above Threshold"

        f, ax = plt.subplots(figsize=(10, 8))

        # Diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})

        # Set the ylabels
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index),
                           size=int(160 / corr_matrix_plot.shape[0]))

        # Set the xlabels
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns),
                           size=int(160 / corr_matrix_plot.shape[1]))
        plt.title(title, size=14)
        plt.show()

    def plot_importance(self, top_n=20):
        """
        Plots `top_n` most important features and the cumulative importance
        of features.
        If `threshold` is provided, prints the number of features needed
        to reach `threshold` cumulative importance.

        Parameters
        --------
        top_n : int, default = 20
            Number of most important features to plot.
        """

        if self.importance_stats is None:
            raise Exception('Feature importances have not been determined.'
                            ' Run `drop_low_importance`')

        # Need to adjust number of features if `top_n` is greater than the
        # features in the data
        if top_n > self.importance_stats.shape[0]:
            top_n = self.importance_stats.shape[0] - 1

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.importance_stats.feature[:top_n]))),
                list(reversed(list(self.importance_stats.normalized[:top_n]))),
                align='center', edgecolor='k')

        # Plot labeling
        plt.xlabel('Normalized Importance', size=16)
        plt.title('Feature Importances', size=18)
        plt.show()
