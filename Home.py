# Home.py

import streamlit as st

from streamlit_app.utils import st_helpers as sth

st.set_page_config(page_title="StatsmodelsMasterPro", layout="wide")

st.title("📊 StatsmodelsMasterPro Dashboard")
st.markdown("#### 🧠 Research-grade statistical modeling, inference, and validation with `statsmodels`")

# --- Intro Section
st.markdown("""
Welcome to **StatsmodelsMasterPro v2.0** — an interactive, reproducible toolkit for **learning, validating, and
reporting** statistical models with Python's `statsmodels` library.

Beyond the foundational modules (OLS, GLM, ANOVA, time series, multivariate, mixed models), v2.0 adds
research-grade capabilities:

- 🛡️ **Robust & resampling inference** — HC/HAC/cluster SEs, BCa bootstrap, wild bootstrap, permutation tests, multiple-testing corrections
- 🎯 **Causal inference** — IPW, propensity matching, difference-in-differences, 2SLS, regression discontinuity, E-values
- 🎲 **Monte Carlo validation** — ADEMP simulation studies with Monte Carlo standard errors for bias, coverage, and power
- 📝 **Publication reporting** — stargazer-style tables (LaTeX/Markdown/HTML), tidy/glance, APA 7 strings, forest plots, diagnostic battery
- 🔬 **Reproducibility** — every synthetic dataset has a documented data-generating process, a SHA-256 manifest, and a parameter-recovery study
""")

# --- Highlights Grid
col1, col2, col3 = st.columns(3)
with col1:
    st.success("📘 29 interactive dashboard pages")
    st.info("📈 13 concept notebooks + 6 SciPy comparisons")
with col2:
    st.warning("🛠️ 15 utility modules (inference, causal, simulation, reporting, …)")
    st.success("🧪 110+ pytest tests validated against known DGPs")
with col3:
    st.info("📦 17 synthetic datasets with hash manifest")
    st.warning("🔁 CI on Python 3.10–3.12 with parameter-recovery gate")

# --- Dataset Peek
st.markdown("---")
st.markdown("### 🗂️ Preview: OLS Dataset (`ols_data.csv`)")
st.caption("DGP: y = 2 + 1.5·X1 − 0.7·X2 + e, e ~ N(0, 1.5²). One of 17 synthetic datasets documented in `docs/DATA_DICTIONARY.md`.")
df = sth.load_default_data()
sth.display_project_metrics(df)
sth.display_random_sample(df, n=5)

# --- CTA
st.markdown("---")
st.markdown("""
🎯 **Get Started:**
Use the sidebar to choose a module. New in v2.0: **26 Robust Inference**, **27 Causal Inference**,
**28 Monte Carlo Validation**, and **29 Publication Reporting**.
""")
