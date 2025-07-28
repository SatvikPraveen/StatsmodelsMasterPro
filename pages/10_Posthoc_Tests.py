# 10_Posthoc_Analysis.py

import streamlit as st
from streamlit_app.utils import st_helpers as sth
from pathlib import Path
import pandas as pd

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="Posthoc Analysis", layout="wide")
st.title("🔁 Posthoc Tests — Tukey HSD & Bonferroni")

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
st.sidebar.header("🔧 Posthoc Test Configuration")
response = st.selectbox("🎯 Select Numeric Response Variable", sth.get_numeric_columns(df))
group_col = st.selectbox("📌 Select Categorical Grouping Variable", sth.get_categorical_columns(df))

# ---------------------------------------
# 🧪 Posthoc Results
# ---------------------------------------
if response and group_col:
    st.success(f"Running posthoc tests for **{response}** grouped by **{group_col}**")

    # === Tukey HSD ===
    st.subheader("📊 Tukey HSD Results")
    with st.expander("📘 Tukey HSD Explained"):
        st.markdown("""
        - **Tukey's Honest Significant Difference** test controls for Type I error across multiple pairwise comparisons.
        - Assumes homogeneity of variances and normal distribution.
        - Results show which group pairs differ significantly in their means.
        """)
    tukey_df = sth.run_tukey_hsd(df, response, group_col)
    st.dataframe(tukey_df.style.format(precision=3))

    st.download_button(
        label="📥 Download Tukey HSD Results",
        data=tukey_df.to_csv(index=False).encode("utf-8"),
        file_name="tukey_hsd_results.csv",
        mime="text/csv"
    )


    # === Bonferroni Pairwise Tests ===
    st.subheader("📊 Pairwise T-tests (Bonferroni-adjusted)")
    with st.expander("📘 Bonferroni Correction Explained"):
        st.markdown("""
        - Conducts **independent t-tests** between all pairs of groups.
        - Adjusts p-values using the **Bonferroni correction**, making it more conservative.
        - Useful when assumptions for ANOVA or Tukey aren’t perfectly met.
        """)
    bonf_df = sth.run_pairwise_tests(df, response, group_col)
    st.dataframe(bonf_df.style.format(precision=3))

    st.download_button(
        label="📥 Download Bonferroni Results",
        data=bonf_df.to_csv(index=False).encode("utf-8"),
        file_name="bonferroni_pairwise_tests.csv",
        mime="text/csv"
    )


else:
    st.warning("⚠️ Please select both a numeric response variable and a categorical grouping variable.")

# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Posthoc Analysis Module")
