# 08_Model_Selection.py

import streamlit as st
from streamlit_app.utils import st_helpers as sth
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Model Selection & Comparison", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "ols_diagnostics.csv"
df = pd.read_csv(DATA_PATH)

st.title("🧮 Model Selection — AIC, BIC, R², Log-Likelihood")


# 📊 Dataset Preview
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Summary
st.markdown("### 🔍 Summary of Numeric Features")
st.dataframe(df.describe().T.style.format(precision=2))

# === Model Definitions ===
target = st.selectbox("Select Response Variable", sth.get_numeric_columns(df))
all_feats = [col for col in sth.get_numeric_columns(df) if col != target]
st.sidebar.header("🔧 Model Config")
n1 = st.sidebar.slider("Model 1: Top N Predictors", 1, min(5, len(all_feats)), 1)
n2 = st.sidebar.slider("Model 2: Top N Predictors", 2, min(8, len(all_feats)), 3)

candidate_sets = {
    f"Model 1 – Simple ({n1} var)": all_feats[:n1],
    f"Model 2 – Basic ({n2} vars)": all_feats[:n2],
    "Model 3 – Full": all_feats
}


# === Build and Compare Models ===
models = {}
for label, predictors in candidate_sets.items():
    if predictors:
        models[label] = sth.run_ols_model(df, target, predictors)

with st.expander("ℹ️ Metric Definitions", expanded=False):
    st.markdown("""
    - **AIC (Akaike Info Criterion)**: Lower is better. Penalizes complexity.
    - **BIC (Bayesian Info Criterion)**: Lower is better. Stronger penalty on complexity.
    - **R² (R-squared)**: Higher is better. Variance explained by model.
    - **Log-Likelihood**: Higher is better. Model fit to data.
    """)


if models:
    st.subheader("📊 Model Metrics")
    comp_df = sth.compare_models(models)
    st.dataframe(comp_df.style.format(precision=3))



    st.download_button(
    label="📥 Download Metrics Table",
    data=comp_df.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv"
)
    
    
    col1, col2 = st.columns(2)

    col1.subheader("📊 AIC Scores")
    col1.bar_chart(comp_df["AIC"])

    col2.subheader("📉 BIC Scores")
    col2.bar_chart(comp_df["BIC"])


else:
    st.warning("Please select a valid target and ensure predictors are available.")

# === Footer ===
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Model Selection Module")
