import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Encoding features for modeling...")
def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    dummy_features = [
        col for col in df.drop("Attrition", axis=1).columns if df[col].nunique() < 20
    ]
    data_dummy = pd.get_dummies(df, columns=dummy_features, drop_first=True, dtype="uint8")
    data_dummy = data_dummy.T.drop_duplicates().T
    data_dummy.drop_duplicates(inplace=True)
    data_dummy["Attrition"] = data_dummy["Attrition"].map({"Yes": 1, "No": 0})
    return data_dummy


@st.cache_resource(show_spinner="Splitting and scaling data...")
def split_and_scale(data_dummy: pd.DataFrame):
    X = data_dummy.drop("Attrition", axis=1)
    y = data_dummy["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_std, X_test_std, scaler


@st.cache_resource(show_spinner="Training Logistic Regression...")
def train_logistic_regression(X_train_std, y_train):
    lr_clf = LogisticRegression(solver="liblinear", penalty="l1")
    lr_clf.fit(X_train_std, y_train)
    return lr_clf


@st.cache_resource(show_spinner="Training Random Forest...")
def train_random_forest(X_train, y_train, n_estimators=100, bootstrap=False):
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators, bootstrap=bootstrap, random_state=42
    )
    rf_clf.fit(X_train, y_train)
    return rf_clf


@st.cache_resource(show_spinner="Running hyperparameter grid search (this can take a while)...")
def tune_random_forest(X_train, y_train):
    param_grid = dict(
        n_estimators=[100, 500, 900],
        max_features=["sqrt", "log2"],
        max_depth=[2, 3, 5, 10, 15, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        bootstrap=[True, False],
    )
    base_rf = RandomForestClassifier(random_state=42)
    search = GridSearchCV(
        base_rf, param_grid=param_grid, scoring="roc_auc", cv=5, verbose=0, n_jobs=-1
    )
    search.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(**search.best_params_, random_state=42)
    rf_clf.fit(X_train, y_train)
    return rf_clf, search.best_params_


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "train_confusion_matrix": confusion_matrix(y_train, y_train_pred),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred),
        "train_report": pd.DataFrame(
            classification_report(y_train, y_train_pred, output_dict=True)
        ),
        "test_report": pd.DataFrame(
            classification_report(y_test, y_test_pred, output_dict=True)
        ),
        "train_roc_auc": roc_auc_score(y_train, y_train_pred),
        "test_roc_auc": roc_auc_score(y_test, y_test_pred),
    }


def plot_precision_recall_and_roc(model, X_test, y_test) -> plt.Figure:
    y_pred = model.predict(X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(thresholds, precisions[:-1], "b-", label="Precision")
    axes[0].plot(thresholds, recalls[:-1], "r--", label="Recall")
    axes[0].set_xlabel("Threshold")
    axes[0].legend(loc="best")
    axes[0].set_title("Precision/Recall Tradeoff")

    axes[1].plot(recalls, precisions)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")

    axes[2].plot(fpr, tpr, linewidth=2)
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve")

    fig.tight_layout()
    return fig


def get_dummy_feature_list(df: pd.DataFrame) -> list[str]:
    return [col for col in df.drop("Attrition", axis=1).columns if df[col].nunique() < 20]


def encode_new_sample(sample: dict, df: pd.DataFrame, X_columns: pd.Index) -> pd.DataFrame:
    """Encode a single raw-feature employee record the same way training data was
    one-hot encoded (matching drop_first + column-dedup behavior) so it can be fed
    straight into the trained Random Forest model."""
    dummy_features = get_dummy_feature_list(df)
    row = {col: 0 for col in X_columns}

    for col, val in sample.items():
        if col in dummy_features:
            dummy_col_name = f"{col}_{val}"
            if dummy_col_name in row:
                row[dummy_col_name] = 1
        elif col in row:
            row[col] = val

    return pd.DataFrame([row], columns=X_columns)


def feature_importance(X: pd.DataFrame, model) -> pd.DataFrame:
    fi = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    )
    return fi.sort_values(by="importance", ascending=False)


def plot_feature_importance(fi: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    top = fi.head(top_n).set_index("feature")
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    top.plot(kind="barh", ax=ax, legend=False, color="teal")
    ax.invert_yaxis()
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig
