# 17_Robust_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import seaborn as sns

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Robust Regression – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🛡️"
)

st.title("🛡️ Robust Regression Methods")
st.markdown("""
Handle **outliers** and **heteroskedasticity** with robust regression techniques:
- **WLS (Weighted Least Squares)** - Account for non-constant variance
- **RLM (Robust Linear Models)** - M-estimators resistant to outliers  
- **Quantile Regression** - Model different percentiles of the response
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "robust_regression_data.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.markdown("This dataset contains **outliers** and **heteroskedastic noise** to demonstrate robust methods.")
st.dataframe(df.head(10), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("📏 Total Observations", df.shape[0])
with col2:
    st.metric("📊 Features", df.shape[1])

# -----------------------------------------------
# 📊 Exploratory Visualization
# -----------------------------------------------
st.subheader("🔍 Data Visualization")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scatter plot with outliers
axes[0].scatter(df['X'], df['y'], alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('Scatter Plot (Notice Outliers)')
axes[0].grid(alpha=0.3)

# Boxplot to show outliers
axes[1].boxplot(df['y'], vert=True)
axes[1].set_ylabel('y')
axes[1].set_title('Boxplot of Response Variable')
axes[1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# -----------------------------------------------
# 🧮 Model Configuration
# -----------------------------------------------
st.markdown("---")
st.subheader("⚙️ Model Configuration")

predictor = st.selectbox("📌 Select Predictor Variable", [col for col in df.columns if col != 'y'])
response = 'y'

# -----------------------------------------------
# 📈 1. Ordinary Least Squares (OLS) - Baseline
# -----------------------------------------------
st.markdown("---")
st.subheader("📈 1. Ordinary Least Squares (OLS) - Baseline")

X = sm.add_constant(df[predictor])
y = df[response]

ols_model = sm.OLS(y, X).fit()

with st.expander("📄 OLS Model Summary", expanded=False):
    st.text(ols_model.summary())

col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{ols_model.rsquared:.4f}")
col2.metric("AIC", f"{ols_model.aic:.2f}")
col3.metric("BIC", f"{ols_model.bic:.2f}")

# -----------------------------------------------
# 📊 2. Weighted Least Squares (WLS)
# -----------------------------------------------
st.markdown("---")
st.subheader("📊 2. Weighted Least Squares (WLS)")
st.markdown("""
**WLS** accounts for **heteroskedasticity** by assigning weights inversely proportional to variance.  
Observations with higher variance get lower weight.
""")

weights = df['weights']
wls_model = sm.WLS(y, X, weights=weights).fit()

with st.expander("📄 WLS Model Summary", expanded=False):
    st.text(wls_model.summary())

col1, col2, col3 = st.columns(3)
col1.metric("R² (WLS)", f"{wls_model.rsquared:.4f}")
col2.metric("AIC", f"{wls_model.aic:.2f}")
col3.metric("BIC", f"{wls_model.bic:.2f}")

# -----------------------------------------------
# 🛡️ 3. Robust Linear Model (RLM)
# -----------------------------------------------
st.markdown("---")
st.subheader("🛡️ 3. Robust Linear Model (RLM)")
st.markdown("""
**RLM** uses **M-estimators** (Huber, Ramsay, etc.) to reduce the influence of outliers.  
These models are more resistant to extreme values compared to OLS.
""")

rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

with st.expander("📄 RLM Model Summary", expanded=False):
    st.text(rlm_model.summary())

st.markdown(f"**Converged:** {rlm_model.converged}")
st.markdown(f"**Scale (residual scale):** {rlm_model.scale:.4f}")

# -----------------------------------------------
# 📐 4. Quantile Regression
# -----------------------------------------------
st.markdown("---")
st.subheader("📐 4. Quantile Regression")
st.markdown("""
**Quantile Regression** models different **percentiles** of the response distribution.  
Unlike OLS (which models the mean), quantile regression can reveal heterogeneous effects.
""")

quantiles = st.multiselect(
    "Select Quantiles to Model",
    [0.1, 0.25, 0.5, 0.75, 0.9],
    default=[0.25, 0.5, 0.75]
)

quantile_models = {}
for q in quantiles:
    qr_model = QuantReg(y, X).fit(q=q)
    quantile_models[q] = qr_model

# Display coefficients
if quantile_models:
    st.markdown("### 📊 Quantile Regression Coefficients")
    
    coef_data = []
    for q, model in quantile_models.items():
        coef_data.append({
            'Quantile': q,
            'Intercept': model.params[0],
            f'Coef_{predictor}': model.params[1]
        })
    
    coef_df = pd.DataFrame(coef_data)
    st.dataframe(coef_df.style.format(precision=4))

# -----------------------------------------------
# 📊 Comparison: Fitted Values Plot
# -----------------------------------------------
st.markdown("---")
st.subheader("📊 Model Comparison: Fitted Lines")

fig, ax = plt.subplots(figsize=(12, 6))

# Scatter original data
ax.scatter(df[predictor], df[response], alpha=0.5, label='Data', s=30, edgecolors='k', linewidth=0.3)

# Sort X for plotting
X_sorted = df[predictor].sort_values()
X_sorted_with_const = sm.add_constant(X_sorted)

# OLS line
ols_pred = ols_model.predict(X_sorted_with_const)
ax.plot(X_sorted, ols_pred, 'r-', linewidth=2, label='OLS', alpha=0.8)

# WLS line
wls_pred = wls_model.predict(X_sorted_with_const)
ax.plot(X_sorted, wls_pred, 'g-', linewidth=2, label='WLS', alpha=0.8)

# RLM line
rlm_pred = rlm_model.predict(X_sorted_with_const)
ax.plot(X_sorted, rlm_pred, 'purple', linewidth=2, label='RLM (Huber)', alpha=0.8)

# Quantile regression lines
colors = ['orange', 'blue', 'brown', 'pink', 'cyan']
for i, (q, model) in enumerate(quantile_models.items()):
    qr_pred = model.predict(X_sorted_with_const)
    ax.plot(X_sorted, qr_pred, '--', linewidth=1.5, 
            label=f'Quantile {q}', color=colors[i % len(colors)])

ax.set_xlabel(predictor, fontsize=12)
ax.set_ylabel(response, fontsize=12)
ax.set_title('Comparison of Regression Methods', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close()

# -----------------------------------------------
# 📥 Export Results
# -----------------------------------------------
st.markdown("---")
st.subheader("📥 Export Results")

results_df = pd.DataFrame({
    'Model': ['OLS', 'WLS', 'RLM'],
    'R²': [ols_model.rsquared, wls_model.rsquared, np.nan],  # RLM doesn't have R²
    'AIC': [ols_model.aic, wls_model.aic, np.nan],
    'BIC': [ols_model.bic, wls_model.bic, np.nan],
    'Intercept': [ols_model.params[0], wls_model.params[0], rlm_model.params[0]],
    f'Coef_{predictor}': [ols_model.params[1], wls_model.params[1], rlm_model.params[1]]
})

st.dataframe(results_df.style.format(precision=4, na_rep='N/A'))

st.download_button(
    label="📥 Download Comparison Table",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="robust_regression_comparison.csv",
    mime="text/csv"
)

# -----------------------------------------------
# 📚 Model Interpretation
# -----------------------------------------------
st.markdown("---")
st.subheader("📚 Key Takeaways")

st.markdown("""
### When to Use Each Method:

**🔹 OLS (Ordinary Least Squares)**
- ✅ Use when: Data meets assumptions (normality, homoskedasticity, no outliers)
- ❌ Avoid when: Outliers or heteroskedasticity present

**🔹 WLS (Weighted Least Squares)**
- ✅ Use when: Heteroskedasticity is present and you know the variance structure
- 📌 Requires: Known or estimated weights (e.g., inverse of variance)

**🔹 RLM (Robust Linear Model)**
- ✅ Use when: Outliers are present and you want outlier-resistant estimates
- 📌 Best for: Data with contamination or extreme values

**🔹 Quantile Regression**
- ✅ Use when: You want to model different parts of the distribution (not just the mean)
- 📌 Best for: Understanding heterogeneous effects across quantiles
""")

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Robust Regression Module")
