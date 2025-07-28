# 13_ci_comparison.py

import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_app.utils import st_helpers as sth

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="CI Comparison", layout="wide")
st.title("📏 Confidence Interval Comparison — Statsmodels vs SciPy")

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
st.sidebar.header("🔧 CI Configuration")
num_cols = sth.get_numeric_columns(df)
selected_col = st.selectbox("📌 Select Numeric Column", num_cols)

alpha = st.sidebar.slider("🔬 Confidence Level (1 - α)", 0.80, 0.99, 0.95, step=0.01)

# ---------------------------------------
# 📐 Confidence Interval Comparison
# ---------------------------------------
if selected_col:
    st.success(f"Computing Confidence Intervals for **{selected_col}** at **{int(alpha*100)}%** level")

    data = df[selected_col].dropna()

    try:
        ci_statsmodels = sth.get_confidence_interval_statsmodels(data, alpha=1 - alpha)
        ci_scipy = sth.get_confidence_interval_scipy(data, alpha=1 - alpha)

        st.subheader("📘 Statsmodels CI")
        st.markdown(f"- **Lower Bound**: `{ci_statsmodels[0]:.4f}`")
        st.markdown(f"- **Upper Bound**: `{ci_statsmodels[1]:.4f}`")

        st.subheader("📘 SciPy CI")
        st.markdown(f"- **Lower Bound**: `{ci_scipy[0]:.4f}`")
        st.markdown(f"- **Upper Bound**: `{ci_scipy[1]:.4f}`")

        st.caption("ℹ️ Both methods use t-distribution assumptions and should agree on large samples.")

        # Optional: Side-by-side comparison
        with st.expander("📊 Side-by-Side CI Comparison Table", expanded=False):
            ci_df = pd.DataFrame({
                "Library": ["Statsmodels", "SciPy"],
                "Lower Bound": [ci_statsmodels[0], ci_scipy[0]],
                "Upper Bound": [ci_statsmodels[1], ci_scipy[1]]
            })
            st.dataframe(ci_df.style.format(precision=4))

            st.download_button(
                label="📥 Download CI Comparison",
                data=ci_df.to_csv(index=False).encode("utf-8"),
                file_name="ci_comparison.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.warning("⚠️ Please select a numeric column.")

# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Confidence Interval Comparison Module")
