# 11_ttest_comparison.py

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from streamlit_app.utils import st_helpers as sth

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="T-Test Comparison", layout="wide")
st.title("📊 T-Test Comparison — Statsmodels vs SciPy")

# ---------------------------------------
# 📂 Load Dataset
# ---------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "posthoc_dataset.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------------------
# 🗂 Dataset Preview
# ---------------------------------------
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("### 📊 Feature Summary")
st.dataframe(df.describe(include="all").T.style.format(precision=2))

# ---------------------------------------
# ⚙️ User Inputs
# ---------------------------------------
st.sidebar.header("🔧 T-Test Configuration")
numeric_col = st.selectbox("🎯 Select Numeric Variable", sth.get_numeric_columns(df))
group_col = st.selectbox("📌 Select Binary Grouping Variable", sth.get_categorical_columns(df))

# ---------------------------------------
# 🧪 T-Test Results
# ---------------------------------------
if numeric_col and group_col:
    st.success(f"Running independent t-tests for **{numeric_col}** grouped by **{group_col}**")

    try:
        result = sth.compare_ttests(df, numeric_col, group_col)

        # 📘 Statsmodels
        st.subheader("📘 Statsmodels Result")
        st.markdown(f"- **t-statistic**: `{result['Statsmodels']['t-stat']:.4f}`")
        st.markdown(f"- **p-value**: `{result['Statsmodels']['p-value']:.4g}`")

        # 📘 SciPy
        st.subheader("📘 SciPy Result")
        st.markdown(f"- **t-statistic**: `{result['SciPy']['t-stat']:.4f}`")
        st.markdown(f"- **p-value**: `{result['SciPy']['p-value']:.4g}`")

        st.caption("ℹ️ Assumes equal variance (`equal_var=True`). Use Welch’s test if variances are unequal.")

        # 📈 Optional Visualization
        with st.expander("📈 Show Boxplot", expanded=False):
            fig = px.box(df, x=group_col, y=numeric_col, points="all", title="Group-wise Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # 📥 Optional Download
        with st.expander("📥 Export Test Results", expanded=False):
            df_out = pd.DataFrame(result).T.reset_index().rename(columns={"index": "Library"})
            st.dataframe(df_out.style.format(precision=4))
            st.download_button(
                label="📥 Download T-Test Comparison",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="ttest_comparison.csv",
                mime="text/csv"
            )

    except ValueError as e:
        st.error(f"❌ {str(e)}")
else:
    st.warning("⚠️ Please select both a numeric variable and a **binary** grouping variable.")

# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • T-Test Comparison Module")
