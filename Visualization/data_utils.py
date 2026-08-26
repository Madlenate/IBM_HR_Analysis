from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "ibm-hr-analytics-attrition-dataset"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

DROP_COLUMNS = ["Over18", "EmployeeCount", "EmployeeNumber", "StandardHours"]


@st.cache_data(show_spinner="Loading HR attrition dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=DROP_COLUMNS, inplace=True)
    return df


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include="object").columns if col != "Attrition"]


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="int").columns)


def split_by_attrition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_copy = df.copy()
    df_copy["Attrition"] = LabelEncoder().fit_transform(df_copy["Attrition"])
    attrition_yes_df = df_copy[df_copy["Attrition"] == 1]
    attrition_no_df = df_copy[df_copy["Attrition"] == 0]
    return attrition_yes_df, attrition_no_df
