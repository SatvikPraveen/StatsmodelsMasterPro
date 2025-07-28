# 01_Descriptive_Stats.py

import streamlit as st
import pandas as pd
from streamlit_app.utils.st_helpers import (
    load_default_data,
    get_numeric_columns,
    plot_correlation_matrix,
    compute_skew_kurtosis,
    plot_histograms,
    plot_boxplots
)

# -----------------------------------------------
# 🎯 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Descriptive Analysis",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Descriptive Statistics & EDA")
st.markdown("Explore summary statistics, distribution shape, and correlation across variables.")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
st.sidebar.header("📁 Data Options")
use_default = st.sidebar.checkbox("Use default synthetic dataset", value=True)

if use_default:
    df = load_default_data()
    dataset_name = "ols_data.csv (default)"
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            st.stop()
    else:
        st.warning("Upload a CSV file or use the default synthetic dataset.")
        st.stop()

# -----------------------------------------------
# 📊 Dataset Context
# -----------------------------------------------
st.markdown(f"### 🗂️ Dataset Preview: `{dataset_name}`")
st.markdown(f"📏 Shape: `{df.shape[0]} rows × {df.shape[1]} columns`")
numeric_cols = get_numeric_columns(df)
st.markdown(f"🔢 Numeric Columns Detected: `{len(numeric_cols)}`")

# -----------------------------------------------
# 🧪 Column Selection
# -----------------------------------------------
selected_cols = st.multiselect(
    "Select numeric columns to analyze:",
    numeric_cols,
    default=numeric_cols
)

if not selected_cols:
    st.warning("Please select at least one numeric column.")
    st.stop()

df_selected = df[selected_cols]

# -----------------------------------------------
# 📋 Summary Statistics
# -----------------------------------------------
st.subheader("📋 Summary Statistics")
st.dataframe(df_selected.describe().T)

# Optional: download filtered data
st.download_button(
    label="📥 Download Filtered Data",
    data=df_selected.to_csv(index=False),
    file_name="filtered_descriptive_stats.csv",
    mime="text/csv"
)

# -----------------------------------------------
# 🔄 Skewness and Kurtosis
# -----------------------------------------------
st.subheader("📐 Skewness & Kurtosis")
shape_stats = compute_skew_kurtosis(df_selected)
st.dataframe(pd.DataFrame(shape_stats).T)

# -----------------------------------------------
# 📊 Distribution Plots
# -----------------------------------------------
with st.expander("📈 Show Histograms"):
    plot_histograms(df_selected)

with st.expander("📦 Show Boxplots"):
    plot_boxplots(df_selected)

# -----------------------------------------------
# 🔗 Correlation Matrix
# -----------------------------------------------
st.subheader("🔗 Correlation Matrix")
plot_correlation_matrix(df_selected)

# -----------------------------------------------
# ✅ Footer
# -----------------------------------------------
st.success("✅ Descriptive analysis complete. Use the sidebar to switch datasets or explore other pages.")
