# 23_GEE_Models.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable, Independence, 
                                             Autoregressive, Nested)
from statsmodels.genmod.families import Binomial
import seaborn as sns

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="GEE Models – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🔗"
)

st.title("🔗 Generalized Estimating Equations (GEE)")
st.markdown("""
**GEE** models handle **correlated** data (clustered/repeated measures) with flexible correlation structures.

**Use GEE when:**
- Data has clustering or repeated measures
- Want **population-averaged** (marginal) effects
- Focus is on mean response, not individual variability

**vs Mixed Models (Random Effects):**
- Mixed models: Subject-specific interpretations
- GEE: Population-averaged interpretations
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "gee_data.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏘️ Clusters", df['cluster'].nunique())
col2.metric("📏 Observations per Cluster", f"{len(df) / df['cluster'].nunique():.1f}")
col3.metric("📊 Total Observations", len(df))
col4.metric("🎯 Binary Outcome", "Yes" if df['y'].nunique() == 2 else "No")

# Data summary
with st.expander("📈 Data Summary"):
    st.dataframe(df.describe().T.style.format(precision=2))

# -----------------------------------------------
# 🔍 Visualize Clustered Structure
# -----------------------------------------------
st.subheader("📉 Clustered Data Visualization")

# Sample clusters
sample_clusters = df['cluster'].unique()[:8]
sample_df = df[df['cluster'].isin(sample_clusters)]

fig, ax = plt.subplots(figsize=(12, 5))

for cluster in sample_clusters:
    cluster_data = sample_df[sample_df['cluster'] == cluster]
    ax.scatter(cluster_data['X'], cluster_data['y'], label=cluster, alpha=0.7, s=60)

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y (Binary Outcome)', fontsize=12)
ax.set_title('Clustered Binary Outcomes (Sample)', fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ===============================================
# 🔹 GEE MODEL CONFIGURATION
# ===============================================
st.markdown("---")
st.header("⚙️ GEE Model Configuration")

st.markdown("""
**Correlation Structures:**
- **Independence:** No correlation (like GLM)
- **Exchangeable:** Same correlation between any two obs in same cluster
- **Autoregressive (AR-1):** Correlation decreases with time/distance
- **Unstructured:** Estimate all pairwise correlations (data-intensive)
""")

# Model configuration
col1, col2 = st.columns(2)

with col1:
    corr_structure = st.selectbox(
        "Select Correlation Structure",
        ["Exchangeable", "Independence", "Autoregressive"],
        index=0
    )

with col2:
    response_var = 'y'
    predictors = st.multiselect(
        "Select Predictors",
        ['X', 'treatment'],
        default=['X', 'treatment']
    )

# ===============================================
# 🔹 FIT GEE MODEL
# ===============================================
if predictors and st.button("🚀 Fit GEE Model"):
    with st.spinner("Fitting GEE model..."):
        try:
            # Map correlation structure
            if corr_structure == "Exchangeable":
                cov_struct = Exchangeable()
            elif corr_structure == "Independence":
                cov_struct = Independence()
            else:  # Autoregressive
                cov_struct = Autoregressive()
            
            # Prepare data
            formula = f"{response_var} ~ {' + '.join(predictors)}"
            
            # Fit GEE model
            gee_model = GEE.from_formula(
                formula,
                groups=df['cluster'],
                data=df,
                cov_struct=cov_struct,
                family=Binomial()
            )
            
            gee_result = gee_model.fit()
            
            st.success(f"✅ GEE model fitted with {corr_structure} correlation structure!")
            
            # Model summary
            with st.expander("📄 Full Model Summary"):
                st.text(gee_result.summary())
            
            # Coefficients
            st.subheader("📊 Model Coefficients")
            
            # Odds ratios for binary outcome
            coef_df = pd.DataFrame({
                'Variable': gee_result.params.index,
                'Coefficient': gee_result.params.values,
                'Std Error': gee_result.bse.values,
                'z-value': gee_result.tvalues.values,
                'p-value': gee_result.pvalues.values,
                'Odds Ratio': np.exp(gee_result.params.values),
                'OR 95% CI Lower': np.exp(gee_result.conf_int()[0]),
                'OR 95% CI Upper': np.exp(gee_result.conf_int()[1])
            })
            
            st.dataframe(coef_df.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                'z-value': '{:.4f}',
                'p-value': '{:.4f}',
                'Odds Ratio': '{:.4f}',
                'OR 95% CI Lower': '{:.4f}',
                'OR 95% CI Upper': '{:.4f}'
            }))
            
            # Visualization: Odds Ratios
            st.subheader("📈 Odds Ratios with 95% CI")
            
            fig, ax = plt.subplots(figsize=(10, len(predictors) + 2))
            
            y_pos = np.arange(len(coef_df))
            ors = coef_df['Odds Ratio'].values
            lower = coef_df['OR 95% CI Lower'].values
            upper = coef_df['OR 95% CI Upper'].values
            
            ax.errorbar(ors, y_pos, xerr=[ors - lower, upper - ors],
                       fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2)
            ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='OR = 1 (no effect)')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_df['Variable'])
            ax.set_xlabel('Odds Ratio', fontsize=12)
            ax.set_title(f'Odds Ratios (GEE with {corr_structure} Structure)', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Correlation matrix (if available)
            if hasattr(gee_result.cov_struct, 'summary'):
                st.subheader("🔗 Estimated Correlation Structure")
                st.text(gee_result.cov_struct.summary())
            
            # Download results
            st.download_button(
                label="📥 Download GEE Results",
                data=coef_df.to_csv(index=False).encode("utf-8"),
                file_name="gee_model_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error fitting model: {e}")
            st.info("Make sure your data has proper cluster structure and sufficient observations per cluster.")

# ===============================================
# 🔹 COMPARISON WITH GLM
# ===============================================
st.markdown("---")
st.header("📊 GEE vs GLM Comparison")

st.markdown("""
### Why use GEE instead of GLM?

**GLM (Independence Assumption):**
- Assumes all observations are independent
- ❌ Underestimates standard errors when correlation exists
- ❌ Leads to inflated Type I error rates

**GEE:**
- ✅ Accounts for correlation within clusters
- ✅ Correct standard errors
- ✅ Valid inference even if correlation structure is misspecified
""")

if predictors and st.button("📊 Compare GEE vs GLM"):
    try:
        # Fit GLM (ignoring clustering)
        formula = f"{response_var} ~ {' + '.join(predictors)}"
        glm_model = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial())
        glm_result = glm_model.fit()
        
        # Fit GEE
        gee_model = GEE.from_formula(
            formula,
            groups=df['cluster'],
            data=df,
            cov_struct=Exchangeable(),
            family=Binomial()
        )
        gee_result = gee_model.fit()
        
        st.success("✅ Both models fitted!")
        
        # Comparison table
        comparison_data = []
        
        for var in glm_result.params.index:
            comparison_data.append({
                'Variable': var,
                'GLM Coef': glm_result.params[var],
                'GLM SE': glm_result.bse[var],
                'GEE Coef': gee_result.params[var],
                'GEE SE': gee_result.bse[var],
                'SE Ratio (GEE/GLM)': gee_result.bse[var] / glm_result.bse[var]
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        st.subheader("📊 Model Comparison")
        st.dataframe(comp_df.style.format(precision=4))
        
        st.markdown("""
        **Interpretation:**
        - **SE Ratio > 1:** GEE standard errors are larger (accounting for correlation)
        - This typically happens when positive correlation exists within clusters
        - GLM would  underestimate uncertainty in this case
        """)
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

# -----------------------------------------------
# 📚 Key Concepts
# -----------------------------------------------
st.markdown("---")
st.subheader("📚 GEE Key Concepts")

st.markdown("""
### When to Use GEE:

✅ **Perfect for:**
- Clustered data (hospitals, schools, families)
- Longitudinal/repeated measures
- Population-averaged interpretations
- Non-normal outcomes with correlation

❌ **Not ideal for:**
- Predicting individual outcomes
- Small number of clusters (< 30)
- Interest in cluster-specific effects

### GEE vs Mixed Models:

| Feature | GEE | Mixed Models |
|---------|-----|--------------|
| Interpretation | Population-averaged | Subject-specific |
| Correlation | Working correlation | Random effects |
| Robustness | Robust to mis specification | Requires correct distribution |
| Predictors | Marginal effects | Conditional effects |
| Software | Simpler | More complex |

### Correlation Structures:

- **Exchangeable:** Best for clusters with equal correlation
- **AR-1:** Best for time series with decay
- **Independence:** When no correlation (baseline)
- **Unstructured:** Flexible but requires more data
""")

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • GEE Models Module")
