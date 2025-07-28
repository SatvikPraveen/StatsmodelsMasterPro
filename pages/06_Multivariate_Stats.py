# 06_Multivariate_Stats.py

import streamlit as st
import pandas as pd
from streamlit_app.utils import st_helpers as sth
from pathlib import Path

st.set_page_config(page_title="Multivariate Stats", layout="wide")
st.title("📈 Multivariate Statistics – Hotelling's T² Test")

# ---------------------------------------
# 📥 Load Dataset
# ---------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "multivariate_group_data.csv"
df = pd.read_csv(DATA_PATH)
numeric_cols = sth.get_numeric_columns(df)
categorical_cols = sth.get_categorical_columns(df)

# Dataset Preview
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())

col1, col2 = st.columns(2)
col1.metric("🔢 Numeric Columns", len(numeric_cols))
col2.metric("🔠 Categorical Columns", len(categorical_cols))

# ---------------------------------------
# ⚙️ User Selections
# ---------------------------------------
st.subheader("🔧 Configuration")

group_col = st.selectbox("👥 Select Group Column (Categorical, Exactly 2 Groups)", categorical_cols)
selected_cols = st.multiselect("📌 Select Variables (2+)", numeric_cols, default=numeric_cols[:2])

group_values = df[group_col].dropna().unique() if group_col else []

if len(group_values) == 2 and len(selected_cols) >= 2:
    g1_df = df[df[group_col] == group_values[0]]
    g2_df = df[df[group_col] == group_values[1]]

    st.markdown(f"**Groups Detected**: `{group_values[0]}` vs `{group_values[1]}`")

    # ---------------------------------------
    # 📊 Pairplot
    # ---------------------------------------
    with st.expander("📊 Pairplot of Selected Features", expanded=False):
        st.write("Visualizing selected variables by group:")
        sth.show_multivariate_distributions(df[[*selected_cols, group_col]], selected_cols)


    # ---------------------------------------
    # 🧪 Hotelling’s T² Test
    # ---------------------------------------
    st.subheader("🧮 Hotelling’s T² Test Result")

    result = sth.compute_hotelling_t2(g1_df, g2_df, selected_cols)
    st.markdown(f"- **T² Statistic**: `{result['T2']:.4f}`")
    st.markdown(f"- **F Equivalent**: `{result['F']:.4f}`")
    st.markdown(f"- **p-value**: `{result['p_value']:.4f}`")

    if result["p_value"] < 0.05:
        st.success("✅ Reject Null Hypothesis — Group Means Differ Significantly")
    else:
        st.info("ℹ️ Fail to Reject Null — No Significant Group Difference")

    # ---------------------------------------
    # 📥 Export
    # ---------------------------------------
    result_df = sth.get_hotelling_summary_df(result)
    st.download_button(
        label="📥 Download Result as CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="hotelling_t2_result.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Please select a categorical group column with exactly 2 unique values and at least 2 numeric variables.")

# ---------------------------------------
# 🧾 Footer
# ---------------------------------------
st.markdown("---")
st.caption("📘 StatsmodelsMasterPro • Multivariate Statistics Module")
