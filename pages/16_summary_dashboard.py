# 16_summary_dashboard.py

import streamlit as st
import pandas as pd
from streamlit_app.utils import st_helpers as sth
import io
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


# ---------------------------------------
# 🔧 Page Config
# ---------------------------------------
st.set_page_config(page_title="📊 Project Summary Dashboard", layout="wide")
st.title("📈 StatsmodelsMasterPro — Summary Dashboard")

# ---------------------------------------
# 📤 Upload or Load Dataset
# ---------------------------------------
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Custom dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()
else:
    DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "multivariate_group_data.csv"
    df = pd.read_csv(DATA_PATH)

    st.sidebar.success("✅ Loaded multivariate_group_data.csv successfully!")

# ---------------------------------------
# 📦 Dataset Overview
# ---------------------------------------
st.markdown("### 📦 Dataset Overview")
sth.display_project_metrics(df)
sth.display_random_sample(df)

# ---------------------------------------
# 📊 Summary Statistics
# ---------------------------------------
st.markdown("### 📊 Summary Statistics")

try:
    summary_df = df.describe(include="all").T
    st.dataframe(summary_df.style.format(precision=3))

    with st.expander("💾 Download Summary Stats"):
        csv = summary_df.to_csv().encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name="summary_statistics.csv", mime="text/csv")
except Exception as e:
    st.warning(f"⚠️ Could not compute summary statistics: {e}")

# ---------------------------------------
# 🔍 Histogram Visuals
# ---------------------------------------
st.markdown("### 🔍 Histogram Visuals")

numeric_cols = sth.get_numeric_columns(df)
if not numeric_cols:
    st.warning("No numeric columns found.")
else:
    selected_cols = st.multiselect("Select numeric columns for histogram:", numeric_cols, default=numeric_cols[:2])

    # Plot and capture
    export_bufs = {}
    for col in selected_cols:
        st.markdown(f"**📉 Histogram — `{col}`**")
        fig = plt.figure()
        try:
            sth.show_histogram(df, col)
            export_bufs[f"histogram_{col}.png"] = fig
        except Exception as e:
            st.error(f"❌ Could not generate histogram for `{col}`: {e}")
        finally:
            plt.close(fig)

    if export_bufs:
        with st.expander("📥 Export Visuals as PNG"):
            for filename, fig in export_bufs.items():
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=buf.getvalue(),
                    file_name=filename,
                    mime="image/png"
                )

# ---------------------------------------
# 📦 Boxplot Section (Expandable)
# ---------------------------------------
with st.expander("📦 Show Boxplots (Optional Grouping)"):
    group_by = st.selectbox("Group by column (optional)", [None] + sth.get_categorical_columns(df))

    for col in selected_cols:
        st.markdown(f"**📦 Boxplot — `{col}`**")
        try:
            sth.show_boxplot(df, col, group=group_by)
        except Exception as e:
            st.error(f"❌ Failed to render boxplot for `{col}`: {e}")

# ---------------------------------------
# 🙏 Project Footer — Thank You Message
# ---------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        🙌 Thank you for exploring <strong>StatsmodelsMasterPro</strong>.<br>
        This project was built with ❤️ to help you master statistical modeling.<br>
        Happy analyzing and may your models always converge!
    </div>
    """,
    unsafe_allow_html=True
)
