import joblib
import numpy as np
from AUS_Functions_transformers_only import *
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import json
import xgboost

app = Flask(__name__, template_folder="templateFiles", static_folder="staticFiles")

prev_cols_to_drop = [
    "FLAG_LAST_APPL_PER_CONTRACT",
    "NFLAG_LAST_APPL_IN_DAY",
    "NAME_CLIENT_TYPE",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    "RATE_INTEREST_PRIVILEGED",
    "RATE_INTEREST_PRIMARY",
]
prev_categorical_cols = [
    "NAME_CONTRACT_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "NAME_CASH_LOAN_PURPOSE",
    "NAME_CONTRACT_STATUS",
    "NAME_PAYMENT_TYPE",
    "CODE_REJECT_REASON",
    "NAME_TYPE_SUITE",
    "NAME_GOODS_CATEGORY",
    "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE",
    "CHANNEL_TYPE",
    "NAME_SELLER_INDUSTRY",
    "NAME_YIELD_GROUP",
    "PRODUCT_COMBINATION",
    "completed_cash",
    "completed_credit",
    "NFLAG_INSURED_ON_APPROVAL",
]
bureau_categorical_cols = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]
current_data_drop_cols = [
    "FLAG_MOBIL",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_21",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_15",
    "FLAG_CONT_MOBILE",
    "DAYS_BIRTH",
    "APARTMENTS_AVG",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_AVG",
    "BASEMENTAREA_MEDI",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_AVG",
    "YEARS_BUILD_MEDI",
    "COMMONAREA_AVG",
    "COMMONAREA_MEDI",
    "ELEVATORS_AVG",
    "ELEVATORS_MEDI",
    "ENTRANCES_AVG",
    "ENTRANCES_MEDI",
    "FLOORSMAX_AVG",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_AVG",
    "FLOORSMIN_MEDI",
    "LANDAREA_AVG",
    "LANDAREA_MEDI",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_AVG",
    "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_AVG",
    "NONLIVINGAREA_MEDI",
]
combined_data_drop_cols = [
    "SK_ID_CURR",
    "prev_SK_ID_CURR",
    "bureau_SK_ID_CURR",
    "prev_NAME_CONTRACT_STATUS_Unused_offer_sum",
    "prev_NAME_CONTRACT_TYPE_nan_sum",
]
credit_na_columns = [
    "prev_atm_drawings_vs_all_drawings_credit_mean",
    "prev_balance_over_limit_credit_ratio_mean",
    "prev_amount_per_drawing_credit_mean_mean",
    "prev_all_drawings_credit_sum_mean",
]


models_dict = {}
models_list = ["model", "one_hot_bureau", "one_hot_previous", "Preprocessor"]


def load_models():
    global models_dict
    for model in models_list:
        models_dict[model] = joblib.load(f"models\\{model}.pkl")




def group_bureau_balance(bureau_balance_df):
    bureau_balance_df = bureau_balance_df.drop(
        bureau_balance_df.index[
            (
                (bureau_balance_df["STATUS"] == "C")
                | (bureau_balance_df["STATUS"] == "X")
            )
        ]
    )
    bureau_balance_df["STATUS"] = bureau_balance_df["STATUS"].astype("int")

    bureau_balance_grouped = (
        bureau_balance_df.groupby("SK_ID_BUREAU")["STATUS"].max().to_frame()
    )
    bureau_balance_grouped = bureau_balance_grouped.rename(
        columns={"STATUS": "DPD_months"}
    )
    bureau_balance_grouped["last_DPD_months"] = (
        bureau_balance_df[bureau_balance_df["STATUS"] > 0]
        .groupby("SK_ID_BUREAU")["MONTHS_BALANCE"]
        .max()
    )
    bureau_balance_grouped = bureau_balance_grouped.reset_index()
    return bureau_balance_grouped




def group_bureau(bureau_df, bureau_balance_grouped):
    bureau_df = bureau_df.merge(bureau_balance_grouped, on="SK_ID_BUREAU", how="left")

    one_hot_categorical_df = models_dict["one_hot_bureau"].transform(
        bureau_df.loc[:, bureau_categorical_cols]
    )
    one_hot_categorical_df = pd.DataFrame(
        one_hot_categorical_df,
        columns=models_dict["one_hot_bureau"].steps[1][1].get_feature_names_out(),
        index=bureau_df.index,
    )

    one_hot_categorical_df.columns = one_hot_categorical_df.columns.str.replace(
        "\s+", "_", regex=True
    )
    one_hot_categorical_columns = one_hot_categorical_df.columns

    num_df = bureau_df.loc[:, bureau_df.columns[~(bureau_df.dtypes == "object")]]
    bureau_df = num_df.join(one_hot_categorical_df)
    del one_hot_categorical_df
    del num_df

    bureau_df_grouped = bureau_df.groupby("SK_ID_CURR")[
        np.append(
            one_hot_categorical_columns,
            [
                "DPD_months",
                "AMT_CREDIT_SUM",
                "CNT_CREDIT_PROLONG",
                "CREDIT_DAY_OVERDUE",
            ],
        )
    ].sum()
    bureau_df_grouped.columns = bureau_df_grouped.columns + "_sum"

    bureau_df_grouped["last_DPD_months_max"] = bureau_df.groupby("SK_ID_CURR")[
        "last_DPD_months"
    ].max()
    bureau_df_grouped = bureau_df_grouped.reset_index()
    return bureau_df_grouped




def group_credit_data(credit_df):
    df_credit_grouped = (
        credit_df.groupby(["SK_ID_PREV", "CNT_INSTALMENT_MATURE_CUM"])["SK_DPD"]
        .max()
        .reset_index()
        .groupby("SK_ID_PREV")["SK_DPD"]
        .sum()
        .to_frame()
    )
    df_credit_grouped["last_DPD_credit"] = (
        credit_df[credit_df["SK_DPD"] > 0].groupby("SK_ID_PREV")["MONTHS_BALANCE"].max()
    )
    df_credit_grouped["atm_drawings_vs_all_drawings_credit"] = (
        credit_df.groupby("SK_ID_PREV")["AMT_DRAWINGS_ATM_CURRENT"].sum()
        / credit_df.groupby("SK_ID_PREV")["AMT_DRAWINGS_CURRENT"].sum()
    )
    credit_df["balance_over_limit_credit"] = (
        credit_df["AMT_CREDIT_LIMIT_ACTUAL"] < credit_df["AMT_BALANCE"]
    )
    df_credit_grouped["balance_over_limit_credit_count"] = credit_df.groupby(
        "SK_ID_PREV"
    )["balance_over_limit_credit"].sum()
    df_credit_grouped["balance_over_limit_credit_ratio"] = (
        credit_df.groupby("SK_ID_PREV")["AMT_BALANCE"].sum()
        / credit_df.groupby("SK_ID_PREV")["AMT_CREDIT_LIMIT_ACTUAL"].sum()
    )
    df_credit_grouped["amount_per_drawing_credit_mean"] = (
        credit_df.groupby("SK_ID_PREV")["AMT_DRAWINGS_CURRENT"].sum()
        / credit_df.groupby("SK_ID_PREV")["CNT_DRAWINGS_CURRENT"].sum()
    )
    credit_df["active_completed_credit"] = (
        credit_df["NAME_CONTRACT_STATUS"] == "Completed"
    )
    df_credit_grouped["completed_credit"] = credit_df.groupby("SK_ID_PREV")[
        "active_completed_credit"
    ].max()
    df_credit_grouped["all_drawings_credit_sum"] = credit_df.groupby("SK_ID_PREV")[
        "AMT_DRAWINGS_CURRENT"
    ].sum()
    df_credit_grouped = df_credit_grouped.rename(columns={"SK_DPD": "DPD_credit_sum"})
    df_credit_grouped = df_credit_grouped.reset_index()
    return df_credit_grouped




def group_cash_data(df_pos_cash):
    df_cash_grouped = (
        df_pos_cash.groupby(["SK_ID_PREV", "CNT_INSTALMENT_FUTURE"])["SK_DPD"]
        .max()
        .reset_index()
        .groupby("SK_ID_PREV")["SK_DPD"]
        .sum()
        .to_frame()
    )
    df_cash_grouped["last_DPD_cash"] = (
        df_pos_cash[df_pos_cash["SK_DPD"] > 0]
        .groupby("SK_ID_PREV")["MONTHS_BALANCE"]
        .max()
    )
    df_pos_cash["active_completed_cash"] = (
        df_pos_cash["NAME_CONTRACT_STATUS"] == "Completed"
    )
    df_cash_grouped["completed_cash"] = df_pos_cash.groupby("SK_ID_PREV")[
        "active_completed_cash"
    ].max()
    df_cash_grouped = df_cash_grouped.rename(columns={"SK_DPD": "DPD_cash_sum"})
    df_cash_grouped = df_cash_grouped.reset_index()
    return df_cash_grouped




def group_previous_application(prev_df, grouped_cash, grouped_credit):
    grouped_cash = grouped_cash[grouped_cash["SK_ID_PREV"].isin(prev_df["SK_ID_PREV"])]
    prev_df = prev_df.merge(
        grouped_cash, how="left", right_on="SK_ID_PREV", left_on="SK_ID_PREV"
    )

    grouped_credit = grouped_credit[
        grouped_credit["SK_ID_PREV"].isin(prev_df["SK_ID_PREV"])
    ]
    prev_df = prev_df.merge(
        grouped_credit, how="left", right_on="SK_ID_PREV", left_on="SK_ID_PREV"
    )

    prev_df[prev_df == "XNA"] = np.nan
    prev_df[prev_df == "XAP"] = np.nan
    prev_df.loc[prev_df["SELLERPLACE_AREA"] == -1, "SELLERPLACE_AREA"] = np.nan
    prev_df.loc[:, prev_df.columns.drop("SK_ID_CURR")] = prev_df.loc[
        :, prev_df.columns.drop("SK_ID_CURR")
    ].replace(365243.0, np.nan)

    prev_df = prev_df.drop(prev_cols_to_drop, axis=1)

    one_hot_categorical_df = models_dict["one_hot_previous"].transform(
        prev_df.loc[:, prev_categorical_cols]
    )
    one_hot_categorical_df = pd.DataFrame(
        one_hot_categorical_df,
        columns=models_dict["one_hot_previous"].steps[1][1].get_feature_names_out(),
        index=prev_df.index,
    )

    one_hot_categorical_df.columns = one_hot_categorical_df.columns.str.replace(
        "[^a-zA-Z\d\s]", " ", regex=True
    )
    one_hot_categorical_df.columns = one_hot_categorical_df.columns.str.replace(
        "\s+", "_", regex=True
    )

    one_hot_categorical_df.loc[
        ~(one_hot_categorical_df["NAME_CONTRACT_STATUS_Approved"] == 1),
        ["completed_cash_nan", "completed_credit_nan", "NFLAG_INSURED_ON_APPROVAL_nan"],
    ] = 0
    one_hot_categorical_df.loc[
        one_hot_categorical_df["NAME_CONTRACT_STATUS_Approved"] == 1,
        "CODE_REJECT_REASON_nan",
    ] = 0
    one_hot_categorical_df.loc[
        one_hot_categorical_df["NAME_CONTRACT_STATUS_Approved"] == 1,
        "CODE_REJECT_REASON_nan",
    ] = 0

    one_hot_categorical_df.loc[
        one_hot_categorical_df["NAME_CONTRACT_TYPE_Revolving_loans"] == 1,
        [
            "NAME_PAYMENT_TYPE_nan",
            "NAME_YIELD_GROUP_nan",
            "NAME_CASH_LOAN_PURPOSE_nan",
            "completed_cash_nan",
        ],
    ] = 0
    one_hot_categorical_df.loc[
        one_hot_categorical_df["NAME_CONTRACT_TYPE_Cash_loans"] == 1,
        ["NAME_GOODS_CATEGORY_nan", "completed_credit_nan"],
    ] = 0
    one_hot_categorical_df.loc[
        one_hot_categorical_df["NAME_CONTRACT_TYPE_Consumer_loans"] == 1,
        ["NAME_CASH_LOAN_PURPOSE_nan", "NAME_PRODUCT_TYPE_nan", "completed_credit_nan"],
    ] = 0

    num_columns = prev_df.columns[
        (
            ~(prev_df.dtypes == "object")
            & ~prev_df.columns.isin(["NFLAG_INSURED_ON_APPROVAL"])
        )
    ]
    num_df = prev_df.loc[:, num_columns]

    prev_df = num_df.join(one_hot_categorical_df)
    del one_hot_categorical_df
    del num_df

    prev_df_grouped = prev_df.groupby("SK_ID_CURR")[
        np.append(
            prev_df.columns.drop(num_columns),
            ["DPD_cash_sum", "DPD_credit_sum", "balance_over_limit_credit_count"],
        )
    ].sum()
    prev_df_grouped.columns = prev_df_grouped.columns + "_sum"

    mean_cols = num_columns.drop(
        [
            "SK_ID_PREV",
            "SK_ID_CURR",
            "DAYS_DECISION",
            "last_DPD_cash",
            "last_DPD_credit",
            "DPD_cash_sum",
            "DPD_credit_sum",
            "balance_over_limit_credit_count",
        ]
    )
    temp_grouped = prev_df.groupby("SK_ID_CURR")[mean_cols].mean()
    temp_grouped.columns = temp_grouped.columns + "_mean"
    prev_df_grouped = prev_df_grouped.join(temp_grouped)

    temp_grouped = prev_df.groupby("SK_ID_CURR")[
        "DAYS_DECISION", "last_DPD_cash", "last_DPD_credit"
    ].max()
    temp_grouped.columns = temp_grouped.columns + "_max"
    prev_df_grouped = prev_df_grouped.join(temp_grouped)
    prev_df_grouped = prev_df_grouped.reset_index()
    return prev_df_grouped


def clean_current_data(df):
    df["age"] = (df["DAYS_BIRTH"] / -365).round()
    df = df.drop(current_data_drop_cols, axis=1)
    return df


def clean_combined_data(df, grouped_previous_data, bureau_df_grouped):
    grouped_previous_data.columns = "prev_" + grouped_previous_data.columns
    bureau_df_grouped.columns = "bureau_" + bureau_df_grouped.columns

    df = df.merge(
        grouped_previous_data,
        left_on="SK_ID_CURR",
        right_on="prev_SK_ID_CURR",
        how="left",
    )
    df = df.merge(
        bureau_df_grouped,
        left_on="SK_ID_CURR",
        right_on="bureau_SK_ID_CURR",
        how="left",
    )

    df = df.drop(combined_data_drop_cols, axis=1)

    for column in df.columns[~(df.dtypes == "object")][
        np.isinf(df.loc[:, df.columns[~(df.dtypes == "object")]]).any()
    ]:
        df.loc[np.isinf(df[column]), column] = np.nan

    for column in [
        "prev_last_DPD_credit_max",
        "bureau_last_DPD_months_max",
        "prev_last_DPD_cash_max",
        "prev_DAYS_DECISION_max",
    ]:
        df[column] = df[column].fillna(df[column].min() * 2)

    df.loc[:, credit_na_columns] = df.loc[:, credit_na_columns].fillna(0)
    df.loc[
        df["prev_NAME_CONTRACT_TYPE_Cash_loans_sum"].isna(),
        df.columns[df.columns.str.contains("prev_")],
    ] = df.loc[
        df["prev_NAME_CONTRACT_TYPE_Cash_loans_sum"].isna(),
        df.columns[df.columns.str.contains("prev_")],
    ].fillna(
        0
    )
    df.loc[
        df["bureau_CREDIT_ACTIVE_Closed_sum"].isna(),
        df.columns[df.columns.str.contains("bureau_")],
    ] = df.loc[
        df["bureau_CREDIT_ACTIVE_Closed_sum"].isna(),
        df.columns[df.columns.str.contains("bureau_")],
    ].fillna(
        0
    )

    df["annual_income_vs_credit"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["previous_credit_vs_current"] = df["AMT_CREDIT"] / df["prev_AMT_CREDIT_mean"]

    for column in df.columns[~(df.dtypes == "object")][
        np.isinf(df.loc[:, df.columns[~(df.dtypes == "object")]]).any()
    ]:
        df.loc[np.isinf(df[column]), column] = np.nan

    return df


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result")
def result():
    return render_template("result.html")


THRESHOLD = 0.3


@app.route("/predict", methods=["GET", "POST"])
def get_prediction():
    global attributes_accepted_df
    if request.method == "POST":
        file = request.files["file_upload"]
        myfile = file.read()

        all_data_dict = json.loads(myfile)
        grouped_cash = group_cash_data(
            pd.DataFrame.from_records(all_data_dict["POS_CASH_balance"])
        )
        grouped_credit = group_credit_data(
            pd.DataFrame.from_records(all_data_dict["credit_card_balance"])
        )
        grouped_previous_data = group_previous_application(
            pd.DataFrame.from_records(all_data_dict["previous_application"]),
            grouped_cash,
            grouped_credit,
        )

        bureau_balance_grouped = group_bureau_balance(
            pd.DataFrame.from_records(all_data_dict["bureau_balance"])
        )
        bureau_df_grouped = group_bureau(
            pd.DataFrame.from_records(all_data_dict["bureau"]), bureau_balance_grouped
        )

        current_data = clean_current_data(
            pd.DataFrame.from_records(all_data_dict["current_application"])
        )
        combined_data = clean_combined_data(
            current_data, grouped_previous_data, bureau_df_grouped
        )

        preprocessed_data = models_dict["Preprocessor"].transform(combined_data)
        test_xg_DMatrix = xgboost.DMatrix(data=preprocessed_data)
        result_proba = models_dict["model"].predict(test_xg_DMatrix)
        result = (result_proba > THRESHOLD).astype(int)
        return redirect(
            url_for(
                "result",
                _anchor="anchor",
                prediction_result=result[0],
                prediction_result_proba=np.round(result_proba[0], 2),
                prediction_threshold=THRESHOLD,
            )
        )

    return render_template("index.html")


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
