# 09_Inference_Interpretation.py

import streamlit as st
from streamlit_app.utils import st_helpers as sth
from pathlib import Path
import pandas as pd

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="Inference & Confidence Intervals", layout="wide")

# Load custom dataset
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "ols_diagnostics.csv"
df = pd.read_csv(DATA_PATH)

st.title("🧠 Inference Interpretation — Confidence Intervals & Coefficients")

# ---------------------------------------
# 🗂 Dataset Preview
# ---------------------------------------
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("### 📊 Feature Summary")
st.dataframe(df.describe().T.style.format(precision=2))

# ---------------------------------------
# ⚙️ Model Configuration
# ---------------------------------------
st.sidebar.header("🔧 Model Configuration")
target = st.selectbox("🎯 Select Response Variable", sth.get_numeric_columns(df))
predictors = st.multiselect("📌 Select Predictors", [col for col in sth.get_numeric_columns(df) if col != target])

# ---------------------------------------
# 📈 Inference Results
# ---------------------------------------
if target and predictors:
    model = sth.run_ols_model(df, target, predictors)
    ci_df = sth.extract_confidence_intervals(model)

    # Standardized Coefficients (Beta Weights)
    std_coefs = sth.get_standardized_coefficients(df, target, predictors)

    st.subheader("📐 Standardized Coefficients (Beta Weights)")
    st.dataframe(std_coefs.style.format(precision=3))


    st.subheader("📋 Coefficients with 95% Confidence Intervals")
    st.dataframe(ci_df.style.format(precision=3))

    # Plot CIs
    sth.plot_ci_intervals(ci_df, title="95% Confidence Intervals for Coefficients")

    # 🆕 Section Heading
    st.subheader("🚨 Statistical Significance Check")
    st.markdown("Rows highlighted in red indicate coefficients whose 95% confidence intervals include zero, suggesting *statistical insignificance*.")

# Highlight insignificant rows

    # Flag if CI crosses zero
    ci_df["Insignificant (CI crosses 0)"] = ((ci_df["Lower CI"] < 0) & (ci_df["Upper CI"] > 0))
    def highlight_insignificant(val):
        return 'background-color: #fdd' if val else ''

    styled_ci_df = ci_df.style.format(precision=3)
    styled_ci_df = styled_ci_df.applymap(highlight_insignificant, subset=["Insignificant (CI crosses 0)"])

    st.dataframe(styled_ci_df)


    # Download Option
    st.download_button(
        label="📥 Download Coefficients with CI",
        data=ci_df.to_csv(index=False).encode("utf-8"),
        file_name="coefficients_with_ci.csv",
        mime="text/csv"
    )

    # Full Summary
    with st.expander("📜 Full Model Summary"):
        sth.show_model_summary(model)

        # Predictions and Residuals
        pred_df = pd.DataFrame({
            "Actual": df[target],
            "Predicted": model.fittedvalues,
            "Residual": model.resid
        })

        st.subheader("📄 Prediction and Residuals")
        st.dataframe(pred_df.head())

        st.download_button(
            label="📥 Download Predictions & Residuals",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="model_predictions_residuals.csv",
            mime="text/csv"
        )


else:
    st.warning("⚠️ Please select both a response variable and at least one predictor.")

# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Inference & Interpretation Module")
