import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_attrition_count(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    df["Attrition"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "indianred"])
    ax.set_title("Attrition Count")
    ax.set_xlabel("Attrition (yes or no)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: list[str], title: str) -> plt.Figure:
    corr_matrix = df[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_categorical_comparison(df_yes: pd.DataFrame, df_no: pd.DataFrame, col: str) -> plt.Figure:
    counts_yes = df_yes[col].value_counts()
    counts_no = df_no[col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(counts_yes.index.astype(str), counts_yes.values, color="indianred")
    axes[0].set_title(f"{col} - Attrition: Yes")
    axes[0].invert_yaxis()

    axes[1].barh(counts_no.index.astype(str), counts_no.values, color="steelblue")
    axes[1].set_title(f"{col} - Attrition: No")
    axes[1].invert_yaxis()

    fig.tight_layout()
    return fig


def plot_corr_with_attrition(data_dummy: pd.DataFrame) -> plt.Figure:
    corr = data_dummy.drop("Attrition", axis=1).corrwith(data_dummy["Attrition"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(6, len(corr) * 0.22)))
    corr.plot(kind="barh", ax=ax, color=["indianred" if v > 0 else "steelblue" for v in corr])
    ax.set_title("Feature Correlation with Attrition")
    fig.tight_layout()
    return fig
