import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class UnknownToNan(BaseEstimator, TransformerMixin):
    """
    Transformer that changes a differently marked missing value to np.nan.
    Parameters:
        missing_dict(dict): A dictonary containing {feature:value} pairs to transform into np.nan values.
    """

    def __init__(self, missing_dict):
        self.missing_dict = missing_dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.missing_dict:
            if self.missing_dict[key] == "nan":
                continue
            else:
                X1.loc[X1[key] == self.missing_dict[key], key] = np.nan

        return X1


class FillMissingDataCurrentOnly(BaseEstimator, TransformerMixin):
    """
    Transformer that fills the missing data for columns used in the second task and also drops columns with too high multicollinarity -
    Columns in columns_to_drop are dropped
    Columns in months_columns either get binarized between missing and present values or fill the missing values with max_value*2
    Columns in max_corr_pairs.index fill the missing values with mean values for binned highest correlated features for each column
    The rest of the missing values get filled with median column value.
    Parameters:
        best_corrs_means_dict({str:pd.Series}): A dictionary containing series with feature_1 mean values for each feature_2 interval for each feature pair in max_corr_pairs.
        max_corr_pairs(pd.Series): A series containing missing_feature and corr_feature pairs.
        months_columns(list): A list containing features that contain "number_of_months_since_X" data
        columns_to_drop(list): A list containing features to drop to avoid multicollinearity
        binarize(str): Whether to binarize features in months_columns list. Values are ["all", "some", "none"] with "all" binarizing all of the features in the list that have more than 50% data missing. Default is "some".
    """

    def __init__(
        self,
        best_corrs_means_dict,
        max_corr_pair_dict,
        columns_to_drop,
        fillna
    ):
        self.best_corrs_means_dict = best_corrs_means_dict
        self.max_corr_pair_dict = max_corr_pair_dict
        self.columns_to_drop = columns_to_drop
        self.fillna = fillna

    def fit(self, X: pd.DataFrame, y=None):
        X1 = X.copy()
        self.median_impute = X1.loc[
            :,
            X1.columns[
                ((~X1.columns.isin(self.columns_to_drop)) & (X1.dtypes == "float"))
            ],
        ].median()
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for _ in range(2):
            for i in range(20):
                max_corr_pairs = self.max_corr_pair_dict[i]
                for column in max_corr_pairs.index:
                    if (((X1[column].isna().sum()*100/X1.shape[0]) == 0) | (X1.loc[X1[column].isna(), max_corr_pairs[column]].isna().sum()*100/X1[X1[column].isna()].shape[0] == 100)):
                      continue

                    column_index = (
                        self.best_corrs_means_dict[f"{column}_{i}"].loc[:, "bin_column"].dtype.categories
                    )
                    X1.loc[:, "bin_column"] = pd.cut(
                        X1[max_corr_pairs[column]], column_index
                    )
                    X1.loc[
                        X1[column] > column_index.right.max(), "bin_column"
                    ] = column_index[-1]
                    X1.loc[
                        X1[column] < column_index.left.min(), "bin_column"
                    ] = column_index[0]
                    mean_column_values = (
                        X1.reset_index()
                        .loc[:, ["index", "bin_column"]]
                        .merge(self.best_corrs_means_dict[f"{column}_{i}"], on="bin_column", how="left")
                        .set_index("index")
                        .drop("bin_column", axis=1)
                    )
                    X1 = X1.merge(
                        mean_column_values, how="left", left_index=True, right_on="index"
                    )
                    X1[column] = X1[column].fillna(X1["medians"])
                    X1 = X1.drop(["medians", "bin_column"], axis=1)

        X1 = X1.loc[:, X1.columns[(~X1.columns.isin(self.columns_to_drop))]]
        if self.fillna == True:
            X1.loc[:, X1.columns[~(X1.dtypes == "object")]] = X1.loc[
                :, X1.columns[~(X1.dtypes == "object")]
            ].fillna(self.median_impute)



        self.column_names = X1.columns
        return X1


class TransformToLog(BaseEstimator, TransformerMixin):
    """
    Transformer that transforms heavily skewed columns into log scale.

    Parameters:
        log_list(list): A list containing features to transform.
        drop_original(bool): A toggle whther to drop the original columns. Default value is "True".
    """

    def __init__(self, log_list, drop_original=True):
        self.log_list = log_list
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for column in self.log_list:
            if column in X1.columns:
                X1.loc[X1[column] == 0, column] = (
                    X1.loc[X1[column] > 0, column].min() / 2
                )
                X1[f"{column}_log"] = np.log(X1[column])
                if self.drop_original:
                    X1 = X1.drop(column, axis=1)
        return X1


class SmallValuesToOther(BaseEstimator, TransformerMixin):
    """
    Transformer that transforms values in categorical features that encompass less than cutoff_num percent of data to "other".
    Parameters:
        column_names(list): List containing column names to transform
        cutoff_num(int): Percent cutoff number for filtering categories
    """

    def __init__(self, column_names: list, cutoff_num=0.5):
        self.cutoff_num = cutoff_num
        self.column_names = column_names

    def fit(self, X: pd.DataFrame, y=None):
        X1 = X.copy()
        filter_values_dict = {}
        for column in self.column_names:
            filter_values_dict[column] = (
                X1[column].value_counts()[
                    X1[column].value_counts() * 100 / X1.shape[0] < self.cutoff_num
                ]
            ).index.tolist()
        self.filter_values_dict = filter_values_dict
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.filter_values_dict:
            X1.loc[X1[key].isin(self.filter_values_dict[key]), key] = "Other"
        return X1
