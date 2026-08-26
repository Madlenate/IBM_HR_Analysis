import streamlit as st

from data_utils import (
    get_categorical_columns,
    get_numerical_columns,
    load_data,
    split_by_attrition,
)
from eda_utils import (
    plot_attrition_count,
    plot_categorical_comparison,
    plot_corr_with_attrition,
    plot_correlation_heatmap,
)
from model_utils import (
    encode_new_sample,
    evaluate_model,
    feature_importance,
    get_dummy_feature_list,
    plot_feature_importance,
    plot_precision_recall_and_roc,
    prepare_model_data,
    split_and_scale,
    train_logistic_regression,
    train_random_forest,
    tune_random_forest,
)

st.set_page_config(page_title="IBM HR Attrition Analysis", layout="wide")

df = load_data()
attrition_yes_df, attrition_no_df = split_by_attrition(df)
categorical_cols = get_categorical_columns(df)
numerical_cols = get_numerical_columns(df)

st.sidebar.title("IBM HR Attrition")
section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Attrition Breakdown",
        "Correlation Heatmaps",
        "Categorical Comparisons",
        "Attrition Risk Modeling",
        "Feature Importance & Predictor",
    ],
)

# ----------------------------------------------------------------------------
if section == "Overview":
    st.title("IBM HR Analytics – Employee Attrition")
    st.markdown(
        "Exploring which employee attributes are associated with attrition, "
        "then using classification models to quantify and predict attrition risk."
    )

    col1, col2 = st.columns(2)
    col1.metric("Employees", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    with st.expander("Categorical Variables"):
        for col in categorical_cols:
            st.write(f"**{col}** ({df[col].nunique()} unique)")
            st.write(df[col].value_counts())

    with st.expander("Numerical Variables"):
        st.write(numerical_cols)

# ----------------------------------------------------------------------------
elif section == "Attrition Breakdown":
    st.title("Attrition Breakdown")
    st.markdown("The dataset is **unbalanced** — far fewer employees left than stayed.")
    st.pyplot(plot_attrition_count(df))

    stayed = (df["Attrition"] == "No").mean()
    left = (df["Attrition"] == "Yes").mean()
    col1, col2 = st.columns(2)
    col1.metric("Stayed", f"{stayed * 100:.1f}%")
    col2.metric("Left", f"{left * 100:.1f}%")

# ----------------------------------------------------------------------------
elif section == "Correlation Heatmaps":
    st.title("Correlation Heatmaps")
    st.markdown(
        "Numerical feature correlations, split by whether the employee left. "
        "Comparing the two heatmaps highlights relationships that shift with attrition."
    )
    tab_yes, tab_no = st.tabs(["Attrition: Yes", "Attrition: No"])
    with tab_yes:
        st.pyplot(
            plot_correlation_heatmap(
                attrition_yes_df, numerical_cols, "Correlation Heatmap - Attrition (Yes)"
            )
        )
    with tab_no:
        st.pyplot(
            plot_correlation_heatmap(
                attrition_no_df, numerical_cols, "Correlation Heatmap - Attrition (No)"
            )
        )

# ----------------------------------------------------------------------------
elif section == "Categorical Comparisons":
    st.title("Categorical Comparisons")
    st.markdown("Compare how a categorical feature's distribution differs between leavers and stayers.")
    col = st.selectbox("Categorical feature", categorical_cols)
    st.pyplot(plot_categorical_comparison(attrition_yes_df, attrition_no_df, col))

    st.divider()
    st.markdown(
        "From the analysis of both numerical and categorical features, a few attributes "
        "stand out as strong indicators of attrition: **Overtime, Job Role, pay-related "
        "fields, and years with the company.** These point to either unmet pay expectations "
        "or employees seeking new/higher positions elsewhere."
    )

# ----------------------------------------------------------------------------
elif section == "Attrition Risk Modeling":
    st.title("Attrition Risk Modeling")
    st.markdown(
        """
        Beyond descriptive exploration, we can train classification models to **learn the
        patterns behind attrition** and use them to flag employees who share the profile of
        past leavers. This has direct business value: HR can proactively intervene — with
        retention conversations, compensation review, or workload changes — before a valued
        employee resigns, rather than reacting after the fact.

        Two models are compared here:
        - **Logistic Regression** — a simple, interpretable baseline.
        - **Random Forest** — captures non-linear interactions between features and typically
          performs better on this kind of tabular HR data.
        """
    )

    data_dummy = prepare_model_data(df)
    X, y, X_train, X_test, y_train, y_test, X_train_std, X_test_std, scaler = split_and_scale(
        data_dummy
    )

    stay_rate = (y_train == 0).mean()
    leave_rate = (y_train == 1).mean()
    col1, col2 = st.columns(2)
    col1.metric("Training set - stay rate", f"{stay_rate * 100:.1f}%")
    col2.metric("Training set - leave rate", f"{leave_rate * 100:.1f}%")
    st.caption(
        "Because far more employees stay than leave, accuracy alone is misleading — "
        "watch precision/recall and ROC AUC for the 'leave' class instead."
    )

    model_choice = st.radio("Model", ["Logistic Regression", "Random Forest"], horizontal=True)

    if model_choice == "Logistic Regression":
        model = train_logistic_regression(X_train_std, y_train)
        results = evaluate_model(model, X_train_std, X_test_std, y_train, y_test)
        roc_fig = plot_precision_recall_and_roc(model, X_test_std, y_test)
    else:
        tuned = st.checkbox(
            "Run full hyperparameter grid search (slow — several minutes)", value=False
        )
        if tuned:
            model, best_params = tune_random_forest(X_train, y_train)
            st.caption(f"Best params: {best_params}")
        else:
            model = train_random_forest(X_train, y_train)
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        roc_fig = plot_precision_recall_and_roc(model, X_test, y_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Test ROC AUC", f"{results['test_roc_auc']:.3f}")
    col2.metric("Train ROC AUC", f"{results['train_roc_auc']:.3f}")
    col3.metric("Test Accuracy", f"{results['test_report'].loc['precision', 'accuracy']:.3f}" if 'accuracy' in results['test_report'].columns else "-")

    tab_test, tab_train = st.tabs(["Test Results", "Train Results"])
    with tab_test:
        st.write("Confusion Matrix (rows=actual, cols=predicted)")
        st.write(results["test_confusion_matrix"])
        st.write("Classification Report")
        st.dataframe(results["test_report"])
    with tab_train:
        st.write("Confusion Matrix (rows=actual, cols=predicted)")
        st.write(results["train_confusion_matrix"])
        st.write("Classification Report")
        st.dataframe(results["train_report"])

    st.subheader("Precision/Recall & ROC")
    st.pyplot(roc_fig)

    st.session_state["X"] = X
    st.session_state["rf_ready"] = model_choice == "Random Forest"
    if model_choice == "Random Forest":
        st.session_state["rf_model"] = model

# ----------------------------------------------------------------------------
elif section == "Feature Importance & Predictor":
    st.title("Feature Importance & Attrition Risk Predictor")
    st.markdown(
        "The Random Forest model's feature importances tell us **which factors matter "
        "most** when predicting attrition — useful for HR to prioritize what to address."
    )

    data_dummy = prepare_model_data(df)
    X, y, X_train, X_test, y_train, y_test, X_train_std, X_test_std, scaler = split_and_scale(
        data_dummy
    )
    rf_model = train_random_forest(X_train, y_train)

    fi = feature_importance(X, rf_model)
    top_n = st.slider("Top N features", min_value=5, max_value=40, value=20)
    st.pyplot(plot_feature_importance(fi, top_n))

    st.divider()
    st.subheader("Predict Attrition Risk for a Hypothetical Employee")
    st.markdown(
        "Fill in an employee profile below to see the model's estimated attrition risk. "
        "This is exactly how the model could be used in practice: run it against current "
        "employee records to flag high-risk individuals for retention outreach."
    )

    dummy_features = get_dummy_feature_list(df)
    raw_cols = df.drop(columns="Attrition").columns

    with st.form("predict_form"):
        cols = st.columns(3)
        sample = {}
        for i, col in enumerate(raw_cols):
            target_col = cols[i % 3]
            if col in dummy_features:
                options = sorted(df[col].unique(), key=str)
                sample[col] = target_col.selectbox(col, options)
            else:
                sample[col] = target_col.number_input(
                    col,
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=float(df[col].median()),
                )
        submitted = st.form_submit_button("Predict")

    if submitted:
        encoded = encode_new_sample(sample, df, X.columns)
        proba = rf_model.predict_proba(encoded)[0][1]
        pred = rf_model.predict(encoded)[0]

        st.metric("Predicted attrition risk", f"{proba * 100:.1f}%")
        if pred == 1:
            st.warning("Model flags this employee as **likely to leave**.")
        else:
            st.success("Model flags this employee as **likely to stay**.")
