import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
import xgboost as xg
from sklearn.pipeline import Pipeline


sns.set()
colors = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "grey": "#8C8C8C",
}

###


def draw_simple_barplot(
    x_column: pd.Series,
    y_column: pd.Series,
    metadata: list,
    rotate_x=False,
    largefig=True,
):
    """
    Draws a simple barplot with title and labels set

      Parameters:
        x_column(pd.Series): Column containing data for x axis
        y_column(pd.Series): Column containing data for y axis
        metadata(list): Contains a list of 3 elements : title, label for x axis, label for y axis
        rotate_x(Bool): Sets whether rotate x labels for better readability. Default is "False".
        largefig(Bool): Sets whether to create large or small plot. Default is "True".

      Returns:
        Nothing
    """
    if largefig:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10, 5))

    ax = sns.barplot(x=x_column, y=y_column)

    if rotate_x:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, horizontalalignment="right"
        )
    ax.set_title(metadata[0])
    ax.set(xlabel=metadata[1], ylabel=metadata[2])

    plt.show()


###


def draw_color_barplot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    sort_column: str,
    metadata: list,
    category_list_1: list,
    category_list_2=[],
    rotate_x=False,
    largefig=True,
):
    """
    Draws a barplot from the input features with possibility to color specific features orange, blue and grey

      Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        y_column(str): Name of the column containing data for y axis
        sort_column(str): Name of the column used for sorting the x column
        metadata(list): Contains a list of 3 elements : title, label for x axis, label for y axis
        category_list(list): Contains the categories from x column to be colored blue
        category_list(list): Contains the categories from x column to be colored orange. Default is empty list.
        rotate_x(Bool): Sets whether rotate x labels for better readability. Default is "False".
        largefig(Bool): Sets whether to create large or small plot. Default is "True".

      Returns:
        Nothing
    """
    if not sort_column:
        sort_column = df.columns[0]
    temp_df = (
        df.sort_values(sort_column, ascending=True)
        .reset_index(drop=True)
        .loc[:10, [x_column, y_column]]
    )
    temp_df[x_column] = pd.Categorical(
        temp_df[x_column], categories=list(temp_df.loc[:10, x_column])
    )
    if not category_list_2:
        colors_graph = [
            colors["blue"] if (x in category_list_1) else colors["grey"]
            for x in temp_df[x_column]
        ]
    else:
        colors_graph = [
            colors["blue"]
            if (x in category_list_1)
            else colors["orange"]
            if (x in category_list_2)
            else colors["grey"]
            for x in temp_df[x_column]
        ]

    if largefig:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10, 5))

    ax = sns.barplot(data=temp_df, x=x_column, y=y_column, palette=colors_graph)

    if rotate_x:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, horizontalalignment="right"
        )
    ax.set_title(metadata[0])
    ax.set(xlabel=metadata[1], ylabel=metadata[2])

    plt.show()


###


def draw_boxplot(df: pd.DataFrame, x_column: str, metadata: list):
    """
    Draws a simple boxplot with title and labels set

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        metadata(list): Contains a list of 3 elements : title, label for x axis, label for y axis

      Returns:
        Nothing
    """
    plt.figure(figsize=(20, 3))
    ax = sns.boxplot(data=df, x=x_column)
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###


def draw_histplot(df: pd.DataFrame, x_column: str, metadata: list, hue_param=""):
    """
    Draws either a uniform histplot or a comparison histplot with the x_column values split by a binary column.

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        metadata(list): Contains a list of 4 elements : title, label for x axis, label for y axis, and two labels for legend if the x_column is split (if histplot is drawn uniformly, no need to include last two).
        hue_param(str): Name of the binary column that is used to split the x_column. Default is empty string and the histplot is drawn uniformly

      Returns:
        Nothing
    """
    plt.figure(figsize=(20, 10))
    if hue_param:
        ax = sns.histplot(
            data=df,
            x=x_column,
            hue=hue_param,
            palette=[colors["grey"], colors["orange"]],
        )
        plt.legend(labels=[metadata[3], metadata[4]], title="")
    else:
        ax = sns.histplot(data=df, x=x_column)
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###


def draw_kdeplot(df: pd.DataFrame, x_column: str, hue_param: str, metadata: list):
    """
    Draws a comparison kdeplot with the x_column values split by a binary column. The kde lines each sum up to 1, not together.

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        hue_param(str): Name of the binary column that is used to split the x_column.
        metadata(list): Contains a list of 4 elements : title, label for x axis, label for y axis, and two labels for legend.


      Returns:
        Nothing
    """
    plt.figure(figsize=(20, 10))
    ax = sns.kdeplot(
        data=df,
        x=x_column,
        hue=hue_param,
        palette=[colors["grey"], colors["orange"]],
        common_norm=False,
    )
    plt.legend(labels=[metadata[3], metadata[4]], title="")
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###


def draw_proportion_barplot(data: pd.Series, metadata: list):
    """
    Draws a horizontal barplot that is split between orange and blue colors to show proportions of a binary value count. The input values are transformed into percentages.

      Parameters:
        data(pd.Series): Contains the value counts and names as indexes of the binary data.
        metadata(list): Contains a list of 4 elements : title, label for x axis, two labels for y axis - for the binary class names. If the labels for y axis are given as empty strings, the labels are taken from the index of data parameter.

      Returns:
        Nothing
    """
    percentages = [
        float(data.iloc[0] * 100 / (data.iloc[0] + data.iloc[1])),
        100 - float(data.iloc[0] * 100 / (data.iloc[0] + data.iloc[1])),
    ]

    plt.figure(figsize=(20, 3))
    ax = sns.barplot(x=[100], color=colors["orange"])
    ax = sns.barplot(x=[percentages[1]], color=colors["blue"])
    if metadata[2]:
        ax.set(xlabel=metadata[1], ylabel=metadata[2])
    else:
        ax.set(xlabel=metadata[1], ylabel=data.index[1])
    ax.set_title(metadata[0])
    plt.xlim(0, 100)

    patches = ax.patches
    for i in range(len(patches)):
        if i == 0:
            x = (
                patches[i].get_x()
                + patches[i].get_width()
                - (patches[i].get_width() - patches[1].get_width()) / 2
                - 3
            )
        else:
            x = patches[i].get_x() + patches[i].get_width() / 2 - 3
        y = patches[i].get_y() + 0.5
        ax.annotate(
            "{:.2f}%".format(percentages[i]),
            (x, y),
            size=20,
            xytext=(5, 10),
            textcoords="offset points",
            color="white",
        )

    ax2 = ax.twinx()
    if metadata[3]:
        ax2.set(yticklabels=[], ylabel=metadata[3], yticks=[0.5])
    else:
        ax2.set(yticklabels=[], ylabel=data.index[0], yticks=[0.5])
    ax2.grid(False)

    plt.show()


###


def draw_comparison_barplot(
    df: pd.DataFrame,
    binary_column: str,
    operating_column: str,
    category_list_1: list,
    metadata: list,
    rotate_x=False,
    largefig=True,
    mode="count",
    y_labels=True,
):
    """
    Draws a double barplot for comparing a multiple category column's value counts split by a different binary column. If mode_count is False, then just takes the values of "count" column in the provided dataframe. It's possible to color some columns in orange/blue or grey.

    Parameters:
      df(pd.DataFrame): A dataframe containing both the binary and the operating column
      binary_column(str): Name of the column containing the binary data
      operating_column(str): Name of the column containing multiple category data - plotted on the x axis
      category_list_1(list): A list of categories from the operating column to color orange/blue. If empty, all categories will be orange/blue
      metadata(list): A list containing 5 values - Title, label for x axis, label for y axis, and two labels for the categories in binary column. If the last two are given as empty string, automatically sets the labels as the values from the binary column.
      rotate_x(Bool): Sets whether rotate x labels for better readability. Default is "False".
      largefig(Bool): Sets whether to create large or small plot. Default is "True".
      mode(str): Sets whether to count the amount of values ("count") or the proportions of the values ("proportion") in the binary column or use the "count" column in the provided dataframe. Default is "count".
    """
    temp_df = df
    if (mode == "count") or (mode == "proportion"):
        id_column = temp_df.index.name
        temp_df = (
            temp_df.reset_index()
            .loc[:, [operating_column, binary_column, id_column]]
            .groupby([operating_column, binary_column])
            .count()
            .reset_index()
            .rename(columns={id_column: "count"})
        )
        if mode == "proportion":
            if len(df[operating_column].unique()) < 3:
                temp_df = temp_df.merge(
                    temp_df.groupby(operating_column)["count"]
                    .sum()
                    .rename("count_total"),
                    left_on=operating_column,
                    right_index=True,
                )
            else:
                temp_df = temp_df.merge(
                    temp_df.groupby(binary_column)["count"].sum().rename("count_total"),
                    left_on=binary_column,
                    right_index=True,
                )
            temp_df["count"] = temp_df["count"] / temp_df["count_total"]
            temp_df.drop("count_total", inplace=True, axis=1)
    temp_df = temp_df.set_index([operating_column, binary_column])["count"].unstack()
    temp_df = temp_df.sort_values(operating_column, ascending=True).reset_index()
    temp_df[operating_column] = pd.Categorical(
        temp_df[operating_column], categories=list(temp_df.loc[:, operating_column])
    )

    if category_list_1:
        category_list_2 = [
            x for x in df[operating_column] if (x not in category_list_1)
        ]
    else:
        category_list_1 = [x for x in df[operating_column]]
        category_list_2 = []
    binary_values = [temp_df.columns[1], temp_df.columns[2]]
    colors_list_1 = {
        binary_values[1]: colors["blue"],
        binary_values[0]: colors["orange"],
    }
    colors_list_2 = [colors["grey"] for x in df[operating_column]]

    temp_df = temp_df.melt(id_vars=[operating_column])

    if largefig:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=temp_df[(temp_df[operating_column].isin(category_list_1))],
        x=operating_column,
        y="value",
        hue=binary_column,
        palette=colors_list_1,
    )
    if category_list_2:
        sns.barplot(
            data=temp_df[(temp_df[operating_column].isin(category_list_2))],
            x=operating_column,
            y="value",
            hue=binary_column,
            palette=colors_list_2,
            ax=ax,
        )
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    if rotate_x:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, horizontalalignment="right"
        )
    ax.set_title(metadata[0])

    if metadata[4] and metadata[3]:
        blue_patch = mpatches.Patch(color=colors["blue"], label=metadata[4])
        orange_patch = mpatches.Patch(color=colors["orange"], label=metadata[3])
    else:
        blue_patch = mpatches.Patch(color=colors["blue"], label=binary_values[1])
        orange_patch = mpatches.Patch(color=colors["orange"], label=binary_values[0])
    plt.legend(handles=[orange_patch, blue_patch])

    if not y_labels:
        ax.set_yticks([])

    plt.show()


def draw_confusion_heatmap(matrix: np.ndarray, metadata: list):
    """
    Draws a simple heatmap from the given confusion matrix

    Parameters:
      matrix(np.ndarray): An array containing the confusion matrix
      metadata(list): Contains 3 strings and a list - The title, labels for X and Y axis and a list containing the labels for categories in the confusion matrix

    Returns:
      Nothing
    """
    plt.figure(figsize=(20, 8))
    ax = sns.heatmap(
        matrix, xticklabels=metadata[3], yticklabels=metadata[3], annot=True, fmt="g"
    )
    ax.set_xlabel(metadata[1])
    ax.set_ylabel(metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###

###


def columns_to_drop_due_multicoll(df):
    """
    Function that calculates which between two features with correlation value over 0.7 is higher correlated with the target column. Returns a list with all the columns that should be dropped to avoid multicollinearity.
    """
    columns_to_drop = []
    corr_df = pd.DataFrame(columns=["col1", "col2", "corr"])
    numeric_cols = df.columns[
        ((df.dtypes == "float") | (df.dtypes == "int"))
        & (~df.columns.isin(["TARGET"]))
    ]
    for column in numeric_cols:
        if column in df.columns:
            col_corr = df.corrwith(df[column])
            multicoll_list = col_corr[
                (col_corr > 0.7) & (col_corr.index != column)
            ].index.tolist()
            if multicoll_list:
                for corr in multicoll_list:
                    if np.abs(df[corr].corr(df["TARGET"])) < np.abs(
                        df[column].corr(df["TARGET"])
                    ):
                        columns_to_drop.append(corr)
                    else:
                        columns_to_drop.append(column)
                        break
    return set(columns_to_drop)

def get_best_corrs_missing_cols(df_input, columns_to_drop):
    """
    A function that calculates the most correlated features for columns with missing data. Also calculates mean missing column values for each binned correlated column interval for imputation.
    Parameters:
        df_input(pd.DataFrame): All data
        months_columns(list): A list containing features that contain "number_of_months_since_X" data
        columns_to_drop(list): A list containing features to drop to avoid multicollinearity
    Returns:
        best_corrs_means_dict({str:pd.Series}): A dictionary containing series with missing_col mean values for each corr_col interval for each feature pair in max_corr_pairs.
        max_corr_pairs(pd.Series): A series containing missing_col and corr_col pairs.
    """
    df = df_input.copy()
    missing_values = df.columns[
        ((df.isna().sum() * 100 / df.shape[0]) > 1)
        & (~df.columns.isin(drop_cols))
        & (~(df.dtypes == "object"))
    ]


    corr_df = pd.DataFrame()
    for column in missing_values:
        corr_df[column] = df.loc[:, df.columns[~(df.dtypes == "object")]].corrwith(
            df[column].astype("float")
        )
    corr_df[corr_df > 0.99] = np.nan
    corr_df[np.abs(corr_df) < 0.1] = np.nan

    best_corrs_means_dict = {}
    max_corr_pair_dict = {}
    for i in range(20):

      max_corr_pairs = np.abs(corr_df).idxmax()
      max_corr_pairs = max_corr_pairs.dropna(axis=0)

      max_corr_pair_dict[i] = max_corr_pairs

      for row in max_corr_pairs.index:
        corr_df.loc[max_corr_pairs[row], row] = np.nan

      for column in max_corr_pairs.index:
          df.loc[:, "temp_column"] = pd.cut(df[max_corr_pairs[column]], 10)
          best_corrs_means_dict[f"{column}_{i}"] = (
              df.groupby("temp_column")[column]
              .median()
              .to_frame()
              .reset_index()
              .rename(columns={"temp_column": "bin_column", column: "medians"})
          )
    return best_corrs_means_dict, max_corr_pair_dict

###

###


###

###

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
