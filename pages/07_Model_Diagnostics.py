# 07_Model_Diagnostics.py

import streamlit as st
from streamlit_app.utils import st_helpers as sth
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Model Diagnostics", layout="wide")
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "ols_diagnostics.csv"
df = pd.read_csv(DATA_PATH)

st.title("🧪 OLS Model Diagnostics – Residual Checks & Influence")

# Preview Dataset
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())
    st.markdown("**📊 Summary Statistics**")
    st.dataframe(df.describe().T)


# === Model Config ===
target = st.selectbox("Select Response Variable", sth.get_numeric_columns(df))
features = st.multiselect("Select Predictor(s)", [col for col in sth.get_numeric_columns(df) if col != target])

if target and features:
    model = sth.run_ols_model(df, target, features)
    st.subheader("📄 Model Summary")
    sth.show_model_summary(model)

    st.subheader("📊 Multicollinearity – VIF")
    vif_df = sth.calculate_vif(df[features])
    st.dataframe(vif_df)

    if vif_df["VIF"].gt(10).any():
        st.warning("⚠️ High multicollinearity detected (VIF > 10). Consider removing/reducing predictors.")


    st.subheader("📉 Residual Plot")
    sth.plot_residuals(model)

    st.subheader("🟣 Q–Q Plot")
    sth.plot_qq(model)

    st.subheader("🟠 Leverage vs Residual²")
    sth.plot_leverage(model)

    st.subheader("🔴 Cook’s Distance")
    sth.show_cooks_distance(model)

    st.subheader("🚨 Influential Observations")
    influence_df = sth.get_influential_points_df(model, df)
    st.dataframe(influence_df)

    st.download_button(
        label="📥 Download Influential Points CSV",
        data=influence_df.to_csv(index=False).encode("utf-8"),
        file_name="influential_points.csv",
        mime="text/csv"
    )


    st.subheader("📤 Export Predictions & Residuals")
    export_df = df.copy()
    export_df["Predicted"] = model.fittedvalues
    export_df["Residuals"] = model.resid


    st.download_button(
        label="📥 Download Predictions CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions_with_residuals.csv",
        mime="text/csv"
    )

else:
    st.warning("Please select both response and at least one predictor variable.")

# === Footer ===
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Model Diagnostics Module")
