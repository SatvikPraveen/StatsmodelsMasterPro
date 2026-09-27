# Home.py

import streamlit as st

# ✅ Updated relative import — works regardless of module name or Streamlit folder structure
from streamlit_app.utils import st_helpers as sth

st.set_page_config(page_title="StatsmodelsMasterPro", layout="wide")

st.title("📊 StatsmodelsMasterPro Dashboard")
st.markdown("#### 🧠 Master Statistical Modeling with Clarity, Confidence, and Control")

# --- Intro Section
st.markdown("""
Welcome to **StatsmodelsMasterPro** — an interactive, portfolio-ready project designed to help you **learn, revise, and apply statistical modeling using Python's `statsmodels` library.  

This project emphasizes:
- ✅ Clean, **synthetic data** to remove real-world noise.
- ✅ Step-by-step breakdowns of **OLS, GLM, ANOVA, Time Series**, and more.
- ✅ Modular notebooks and **Streamlit dashboards** for each concept.
- ✅ Comparisons with `scipy.stats` for deeper understanding.
""")

# --- Highlights Grid
col1, col2, col3 = st.columns(3)
with col1:
    st.success("📘 10+ Conceptual Notebooks")
    st.info("📈 Clean CI Plots, Residuals & AIC/BIC")
with col2:
    st.warning("🛠️ Utilities & Reusable Functions")
    st.success("📊 Bootstrap + Parametric Inference")
with col3:
    st.info("📦 Synthetic Dataset")
    st.warning("🧪 Tested Comparisons with SciPy")

# --- Dataset Peek
st.markdown("---")
st.markdown("### 🗂️ Preview: OLS Dataset (`ols_data.csv`)")
st.caption("This is one of multiple synthetic datasets used across the app.")
df = sth.load_default_data()
sth.display_project_metrics(df)
sth.display_random_sample(df, n=5)

# --- CTA
st.markdown("---")
st.markdown("""
🎯 **Get Started:**  
Use the sidebar on the left to choose a module — whether you're exploring GLMs, comparing distributions, or running posthoc tests, everything is modular and interactive.
""")
