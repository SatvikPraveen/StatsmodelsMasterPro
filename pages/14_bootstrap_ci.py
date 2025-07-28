# 14_bootstrap_ci.py

import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_app.utils import st_helpers as sth

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="Bootstrap CI Comparison", layout="wide")
st.title("🎯 Bootstrap Confidence Intervals — Statsmodels vs Manual")

# ---------------------------------------
# 📂 Load Dataset
# ---------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "ols_diagnostics.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------------------
# 🗂 Dataset Preview
# ---------------------------------------
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("### 📊 Feature Summary")
st.dataframe(df.describe().T.style.format(precision=2))

# ---------------------------------------
# ⚙️ User Inputs
# ---------------------------------------
st.sidebar.header("🔧 Bootstrap CI Configuration")
num_cols = sth.get_numeric_columns(df)
selected_col = st.selectbox("📈 Select Numeric Column", num_cols)

alpha = st.sidebar.slider("🎯 Confidence Level (1 - α)", 0.80, 0.99, 0.95, step=0.01)
reps = st.sidebar.slider("🔁 Number of Bootstrap Repetitions", 500, 5000, 1000, step=500)

# ---------------------------------------
# 📐 Bootstrap CI Computation
# ---------------------------------------
if selected_col:
    st.success(f"Generating {int(alpha*100)}% CI for **{selected_col}** using {reps} bootstrap samples")
    data = df[selected_col].dropna().values

    # Explanation
    with st.expander("📘 What is Bootstrapping?", expanded=False):
        st.markdown("""
        - **Bootstrapping** resamples with replacement to estimate the distribution of a statistic.
        - **No distributional assumptions** (unlike t-based CIs).
        - Ideal for small or non-normal datasets.
        - Statsmodels uses the `bootstrap` function with percentiles. NumPy manually does the same.
        """)

    # CI using Statsmodels
    ci_statsmodels = sth.get_bootstrap_ci_statsmodels(data, alpha=1 - alpha, reps=reps)

    # CI using manual NumPy implementation
    ci_manual = sth.get_bootstrap_ci_numpy(data, alpha=1 - alpha, reps=reps)

    # === Results ===
    st.subheader("📘 Statsmodels Bootstrap CI")
    st.markdown(f"- Lower Bound: `{ci_statsmodels[0]:.4f}`")
    st.markdown(f"- Upper Bound: `{ci_statsmodels[1]:.4f}`")

    st.subheader("📘 Manual NumPy Bootstrap CI")
    st.markdown(f"- Lower Bound: `{ci_manual[0]:.4f}`")
    st.markdown(f"- Upper Bound: `{ci_manual[1]:.4f}`")

    st.caption("ℹ️ Bootstrap methods do not assume normality — ideal for small or skewed samples.")

    # Export as DataFrame
    with st.expander("📥 Export CI Results", expanded=False):
        df_out = pd.DataFrame({
            "Method": ["Statsmodels", "Manual NumPy"],
            "Lower Bound": [ci_statsmodels[0], ci_manual[0]],
            "Upper Bound": [ci_statsmodels[1], ci_manual[1]],
            "Repetitions": [reps, reps],
            "Confidence Level": [alpha, alpha]
        })
        st.dataframe(df_out.style.format(precision=4))

        st.download_button(
            label="📥 Download CI Results",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="bootstrap_ci_comparison.csv",
            mime="text/csv"
        )

else:
    st.warning("⚠️ Please select a numeric column to begin.")

# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Bootstrap Confidence Interval Module")
