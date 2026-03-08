# 22_Panel_Data.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Panel Data Analysis – StatsmodelsMasterPro",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Panel Data Analysis")
st.markdown("""
Analyze **longitudinal** or **panel** data with repeated observations:
- **Fixed Effects Model** - Control for time-invariant individual heterogeneity
- **Random Effects Model** - Efficient when individual effects are uncorrelated with predictors
- **Pooled OLS** - Baseline ignoring panel structure

**Panel Data:** Observations on multiple individuals over multiple time periods.
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "panel_data.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Individuals", df['individual'].nunique())
col2.metric("⏱️ Time Points", df['time'].nunique())
col3.metric("📏 Total Observations", len(df))
col4.metric("📊 Balanced Panel", "✅ Yes" if len(df) == df['individual'].nunique() * df['time'].nunique() else "❌ No")

# -----------------------------------------------
# 📊 Data Exploration
# -----------------------------------------------
with st.expander("📈 Data Summary"):
    st.dataframe(df.describe().T.style.format(precision=2))

# Visualize panel structure
st.subheader("📉 Panel Data Structure Visualization")

# Sample a few individuals to visualize
sample_ids = df['individual'].unique()[:10]
sample_df = df[df['individual'].isin(sample_ids)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Line plot for sampled individuals
for ind in sample_ids:
    ind_data = sample_df[sample_df['individual'] == ind]
    axes[0].plot(ind_data['time'], ind_data['y'], marker='o', label=ind, alpha=0.7)

axes[0].set_xlabel('Time')
axes[0].set_ylabel('y')
axes[0].set_title('Individual Trajectories (Sample of 10)', fontweight='bold')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0].grid(alpha=0.3)

# Distribution by time
df.boxplot(column='y', by='time', ax=axes[1])
axes[1].set_xlabel('Time')
axes[1].set_ylabel('y')
axes[1].set_title('Distribution of y by Time Point')
axes[1].get_figure().suptitle('')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ===============================================
# 🔹 POOLED OLS (BASELINE)
# ===============================================
st.markdown("---")
st.header("📈 1. Pooled OLS - Baseline Model")

st.markdown("""
**Pooled OLS** ignores the panel structure and treats all observations as independent.
- ❌ Ignores individual-specific effects
- ⚠️ Can lead to biased estimates if individual effects exist
""")

if st.button("🔍 Run Pooled OLS"):
    # Fit pooled OLS
    X = sm.add_constant(df[['X1', 'X2']])
    y = df['y']
    
    pooled_model = sm.OLS(y, X).fit()
    
    st.success("✅ Pooled OLS fitted!")
    
    with st.expander("📄 Model Summary"):
        st.text(pooled_model.summary())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{pooled_model.rsquared:.4f}")
    col2.metric("AIC", f"{pooled_model.aic:.2f}")
    col3.metric("BIC", f"{pooled_model.bic:.2f}")
    
    # Coefficients
    st.subheader("📊 Coefficients")
    coef_df = pd.DataFrame({
        'Variable': pooled_model.params.index,
        'Coefficient': pooled_model.params.values,
        'Std Error': pooled_model.bse.values,
        'p-value': pooled_model.pvalues.values
    })
    st.dataframe(coef_df.style.format(precision=4))

# ===============================================
# 🔹 FIXED EFFECTS MODEL
# ===============================================
st.markdown("---")
st.header("🔒 2. Fixed Effects Model")

st.markdown("""
**Fixed Effects Model** controls for **time-invariant** individual characteristics.
- ✅ Removes individual-specific bias
- Uses **within-individual variation** only
- Equivalent to adding a dummy variable for each individual
""")

if st.button("🔍 Run Fixed Effects Model"):
    # Create individual dummies
    df_fe = df.copy()
    df_fe = pd.get_dummies(df_fe, columns=['individual'], drop_first=False)
    
    # Get dummy column names
    dummy_cols = [col for col in df_fe.columns if col.startswith('individual_')]
    
    # Fit FE model
    fe_formula = 'y ~ X1 + X2 + ' + ' + '.join(dummy_cols[:-1])  # Drop one dummy to avoid collinearity
    
    fe_model = ols(fe_formula, data=df_fe).fit()
    
    st.success("✅ Fixed Effects model fitted!")
    
    with st.expander("📄 Full Model Summary"):
        st.text(fe_model.summary())
    
    # Extract main coefficients (exclude individual dummies)
    main_vars = ['Intercept', 'X1', 'X2']
    main_coefs = fe_model.params[main_vars]
    main_se = fe_model.bse[main_vars]
    main_pvals = fe_model.pvalues[main_vars]
    
    st.subheader("📊 Main Coefficients")
    fe_coef_df = pd.DataFrame({
        'Variable': main_coefs.index,
        'Coefficient': main_coefs.values,
        'Std Error': main_se.values,
        'p-value': main_pvals.values
    })
    st.dataframe(fe_coef_df.style.format(precision=4))
    
    col1, col2 = st.columns(2)
    col1.metric("R² (within)", f"{fe_model.rsquared:.4f}")
    col2.metric("F-statistic", f"{fe_model.fvalue:.2f}")

# ===============================================
# 🔹 RANDOM EFFECTS MODEL
# ===============================================
st.markdown("---")
st.header("🎲 3. Random Effects Model (Mixed Model)")

st.markdown("""
**Random Effects Model** treats individual effects as **random** draws from a distribution.
- ✅ More efficient than FE when assumptions hold
- Uses both **within** and **between** individual variation
- Assumes individual effects uncorrelated with predictors
""")

if st.button("🔍 Run Random Effects Model"):
    from statsmodels.regression.mixed_linear_model import MixedLM
    
    # Fit random effects model
    re_model = MixedLM.from_formula('y ~ X1 + X2', data=df, groups=df['individual'])
    re_result = re_model.fit()
    
    st.success("✅ Random Effects model fitted!")
    
    with st.expander("📄 Model Summary"):
        st.text(re_result.summary())
    
    st.subheader("📊 Coefficients")
    re_coef_df = pd.DataFrame({
        'Variable': re_result.params.index,
        'Coefficient': re_result.params.values,
        'Std Error': re_result.bse.values,
        'z-value': re_result.tvalues.values,
        'p-value': re_result.pvalues.values
    })
    st.dataframe(re_coef_df.style.format(precision=4))
    
    col1, col2 = st.columns(2)
    col1.metric("Log-Likelihood", f"{re_result.llf:.2f}")
    col2.metric("Group Variance", f"{re_result.cov_re.iloc[0, 0]:.4f}")

# ===============================================
# 🔹 MODEL COMPARISON
# ===============================================
st.markdown("---")
st.header("📊 Model Comparison")

st.markdown("""
### Choosing Between Models:

**📌 Pooled OLS vs Fixed/Random Effects:**
- If individual effects exist, pooled OLS is biased
- Use F-test or Breusch-Pagan LM test to check

**📌 Fixed Effects vs Random Effects:**
- **Hausman Test** compares FE and RE
    - If p < 0.05: FE is preferred (RE is inconsistent)
    - If p ≥ 0.05: RE is preferred (more efficient)

**🔑 Rule of Thumb:**
- **Fixed Effects:** When individual effects correlate with predictors
- **Random Effects:** When individual effects are random and uncorrelated
- **Pooled OLS:** When no individual effects exist (rare in panel data)
""")

# Comparison table
comparison_df = pd.DataFrame({
    'Model': ['Pooled OLS', 'Fixed Effects', 'Random Effects'],
    'Individual Effects': ['Ignored', 'Controlled (fixed)', 'Random draws'],
    'Variation Used': ['All', 'Within only', 'Within + Between'],
    'Efficiency': ['Biased if effects exist', 'Unbiased but less efficient', 'Most efficient if assumptions hold'],
    'When to Use': ['No individual effects', 'Effects correlate with X', 'Effects uncorrelated with X']
})

st.dataframe(comparison_df, use_container_width=True)

# -----------------------------------------------
# 📚 Panel Data Notation
# -----------------------------------------------
with st.expander("📚 Panel Data Notation & Formulas"):
    st.markdown("""
    ### Panel Data Structure
    - **i:** Individual index (i = 1, ..., N)
    - **t:** Time index (t = 1, ..., T)
    - **yᵢₜ:** Outcome for individual i at time t
    
    ### Models:
    
    **Pooled OLS:**
    ```
    yᵢₜ = β₀ + β₁X₁ᵢₜ + β₂X₂ᵢₜ + εᵢₜ
    ```
    
    **Fixed Effects:**
    ```
    yᵢₜ = αᵢ + β₁X₁ᵢₜ + β₂X₂ᵢₜ + εᵢₜ
    ```
    where αᵢ is individual-specific fixed effect
    
    **Random Effects:**
    ```
    yᵢₜ = β₀ + β₁X₁ᵢₜ + β₂X₂ᵢₜ + (αᵢ + εᵢₜ)
    ```
    where αᵢ ~ N(0, σ²ₐ) is random individual effect
    """)

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Panel Data Analysis Module")
