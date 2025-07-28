# 12_correlation_comparison.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from streamlit_app.utils import st_helpers as sth

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="Correlation Comparison", layout="wide")
st.title("🔗 Correlation Comparison — Statsmodels vs SciPy")

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
st.sidebar.header("🔧 Correlation Configuration")
num_cols = sth.get_numeric_columns(df)

col1 = st.selectbox("📈 Select First Variable", num_cols, key="col1")
col2 = st.selectbox("📉 Select Second Variable", [c for c in num_cols if c != col1], key="col2")

# ---------------------------------------
# 📊 Correlation Comparison
# ---------------------------------------
if col1 and col2:
    st.success(f"Comparing correlation between **{col1}** and **{col2}**")

    try:
        result = sth.compare_correlations(df, col1, col2)

        # === Statsmodels ===
        st.subheader("📘 Statsmodels Pearson Correlation")
        r_val = result['Statsmodels']['Pearson r']
        p_val = result['Statsmodels']['p-value']
        p_val_fmt = f"{p_val:.4g}" if p_val is not None else "N/A"

        st.markdown(f"- **Pearson r**: `{r_val:.4f}`")
        st.markdown(f"- **p-value**: `{p_val_fmt}`")

        # === SciPy ===
        st.subheader("📘 SciPy Correlations")
        st.markdown(f"**Pearson Correlation**\n- r: `{result['SciPy']['Pearson r']:.4f}`\n- p: `{result['SciPy']['Pearson p']:.4g}`")
        st.markdown(f"**Spearman Correlation**\n- r: `{result['SciPy']['Spearman r']:.4f}`\n- p: `{result['SciPy']['Spearman p']:.4g}`")

        st.caption("ℹ️ Spearman correlation is non-parametric and only available via SciPy.")

        # === Scatterplot ===
        with st.expander("📈 Show Scatter Plot", expanded=False):
            fig = px.scatter(df, x=col1, y=col2, trendline="ols", title="Scatter Plot with OLS Trendline")
            st.plotly_chart(fig, use_container_width=True)

        # === Download Results ===
        with st.expander("📥 Export Correlation Results", expanded=False):
            df_out = pd.DataFrame(result).T.reset_index().rename(columns={"index": "Library"})
            st.dataframe(df_out.style.format(precision=4))

            st.download_button(
                label="📥 Download Correlation Comparison",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="correlation_comparison.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.warning("⚠️ Please select two numeric variables for correlation.")

# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Correlation Comparison Module")
