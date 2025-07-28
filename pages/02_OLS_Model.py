# 📄 02_OLS_Model.py — Enhanced OLS Streamlit Page

import streamlit as st
import pandas as pd
from streamlit_app.utils.st_helpers import (
    load_default_data,
    get_numeric_columns,
    run_ols_model,
    show_model_summary,
    plot_residuals,
    plot_actual_vs_predicted,
    display_model_coefficients,
    plot_qq,
    plot_influence,
    export_model_summary,
)

# -----------------------------------------------
# 🎯 Page Configuration
# -----------------------------------------------
st.set_page_config(page_title="OLS Modeling", layout="wide")
st.title("📈 OLS Regression Analysis")
st.markdown("""
Build and analyze a linear regression model using the **statsmodels** library.
Use the sidebar to configure the model, view diagnostics, and interpret results.
""")

# -----------------------------------------------
# 📥 Load & Preview Data
# -----------------------------------------------
df = load_default_data()
num_cols = get_numeric_columns(df)

with st.expander("📂 Preview Dataset"):
    st.dataframe(df.head(), use_container_width=True)

# -----------------------------------------------
# ⚙️ Model Setup
# -----------------------------------------------
st.sidebar.header("Model Configuration")
response = st.sidebar.selectbox("🎯 Response Variable (Y)", num_cols)
predictors = st.sidebar.multiselect("📐 Predictor Variables (X)", [col for col in num_cols if col != response])

# -----------------------------------------------
# 🧠 Model Execution
# -----------------------------------------------
if response and predictors:
    model = run_ols_model(df, response, predictors)

    st.success("✅ Model successfully fitted!")

    # 📊 Summary
    st.subheader("📋 Model Summary")
    show_model_summary(model)

    # 🧾 Coefficients
    st.subheader("🧮 Coefficients Table")
    display_model_coefficients(model)

    # 📉 Residual Plots
    st.subheader("📉 Residual Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        plot_residuals(model)
    with col2:
        plot_qq(model)

    # 🧪 Influence Plot
    st.subheader("📌 Influence Diagnostics")
    plot_influence(model)

    # 🔁 Predictions
    st.subheader("🔁 Actual vs Predicted")
    plot_actual_vs_predicted(model, df, response)

    # 📤 Export
    with st.expander("📤 Export Model Output"):
        if st.button("Download Summary as Text"):
            export_model_summary(model)

    # 📘 Interpretation Aids
    st.subheader("📘 Interpretation Guide")
    st.markdown("""
    - **Intercept**: Expected value of Y when all X are 0.
    - **Coef**: Impact of each X on Y, assuming others are fixed.
    - **p-value < 0.05**: Statistically significant predictor.
    - **R-squared**: Proportion of variance in Y explained by Xs.
    - **QQ Plot**: Should be linear if residuals are normal.
    - **Influence Plot**: Look for high-leverage points or outliers.
    """)
else:
    st.warning("Please select a response and at least one predictor to proceed.")
