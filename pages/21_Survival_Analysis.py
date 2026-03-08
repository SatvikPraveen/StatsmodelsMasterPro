# 21_Survival_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes
import seaborn as sns

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Survival Analysis – StatsmodelsMasterPro",
    layout="wide",
    page_icon="⏱️"
)

st.title("⏱️ Survival Analysis")
st.markdown("""
Analyze **time-to-event** data with censoring:
- **Kaplan-Meier Estimator** - Non-parametric survival curves
- **Log-Rank Test** - Compare survival between groups
- **Cox Proportional Hazards Model** - Semi-parametric regression for survival
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "survival_data.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📏 Total Subjects", df.shape[0])
col2.metric("⏱️ Events Observed", df['event'].sum())
col3.metric("🚫 Censored", (1 - df['event']).sum())
col4.metric("📊 Event Rate", f"{df['event'].mean()*100:.1f}%")

# Data summary
with st.expander("📈 Variable Summary"):
    st.dataframe(df.describe().T.style.format(precision=2))

# ===============================================
# 🔹 KAPLAN-MEIER SURVIVAL CURVES
# ===============================================
st.markdown("---")
st.header("📈 Kaplan-Meier Survival Curves")

st.markdown("""
**Kaplan-Meier estimator** is a non-parametric method to estimate survival probability over time.
It accounts for censored observations (subjects who don't experience the event during the study).
""")

# Overall survival
st.subheader("🔹 Overall Survival Curve")

kmf = KaplanMeierFitter()
kmf.fit(df['time'], df['event'], label='Overall Population')

fig, ax = plt.subplots(figsize=(10, 6))
kmf.plot_survival_function(ax=ax, ci_show=True)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Survival Curve', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close()

# Median survival time
median_survival = kmf.median_survival_time_
st.metric("Median Survival Time", f"{median_survival:.2f}" if not np.isnan(median_survival) else "Not reached")

# ===============================================
# 🔹 SURVIVAL BY GROUP (Treatment)
# ===============================================
st.markdown("---")
st.subheader("🔹 Survival by Treatment Group")

# Plot KM curves by treatment
fig, ax = plt.subplots(figsize=(12, 6))

for treatment in df['treatment'].unique():
    mask = (df['treatment'] == treatment)
    kmf_group = KaplanMeierFitter()
    kmf_group.fit(df.loc[mask, 'time'], df.loc[mask, 'event'], 
                  label=f'Treatment {int(treatment)}')
    kmf_group.plot_survival_function(ax=ax, ci_show=True)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Curves by Treatment Group', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close()

# Median survival by group
st.markdown("### Median Survival by Treatment")
median_data = []
for treatment in sorted(df['treatment'].unique()):
    mask = (df['treatment'] == treatment)
    kmf_temp = KaplanMeierFitter()
    kmf_temp.fit(df.loc[mask, 'time'], df.loc[mask, 'event'])
    median_data.append({
        'Treatment': int(treatment),
        'N': mask.sum(),
        'Events': df.loc[mask, 'event'].sum(),
        'Median Survival': kmf_temp.median_survival_time_
    })

median_df = pd.DataFrame(median_data)
st.dataframe(median_df.style.format({'Median Survival': '{:.2f}'}))

# ===============================================
# 🔹 LOG-RANK TEST
# ===============================================
st.markdown("---")
st.header("🧪 Log-Rank Test")

st.markdown("""
**Log-Rank Test** compares survival distributions between groups.
- **Null Hypothesis:** No difference in survival between groups
- Tests whether survival curves are significantly different
""")

if st.button("🔍 Run Log-Rank Test"):
    # Split by treatment
    T0 = df.loc[df['treatment'] == 0, 'time']
    E0 = df.loc[df['treatment'] == 0, 'event']
    T1 = df.loc[df['treatment'] == 1, 'time']
    E1 = df.loc[df['treatment'] == 1, 'event']
    
    # Perform log-rank test
    results = logrank_test(T0, T1, E0, E1)
    
    st.success("✅ Log-Rank Test Complete!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Statistic", f"{results.test_statistic:.4f}")
    col2.metric("p-value", f"{results.p_value:.4f}")
    col3.metric("Significant", "✅ Yes" if results.p_value < 0.05 else "❌ No")
    
    st.markdown("### Interpretation")
    if results.p_value < 0.05:
        st.success("✅ **Significant difference** in survival between treatment groups (p < 0.05)")
    else:
        st.info("ℹ️ **No significant difference** in survival between treatment groups (p ≥ 0.05)")

# ===============================================
# 🔹 COX PROPORTIONAL HAZARDS MODEL
# ===============================================
st.markdown("---")
st.header("📊 Cox Proportional Hazards Model")

st.markdown("""
**Cox PH Model** is a semi-parametric regression model for survival data.
- Models the **hazard rate** as a function of covariates
- **Hazard Ratio (HR):** Effect of predictors on the hazard of experiencing the event
    - HR > 1: Increased hazard (worse survival)
    - HR < 1: Decreased hazard (better survival)
""")

# Select covariates
st.subheader("⚙️ Model Configuration")

available_vars = ['age', 'treatment', 'biomarker']
selected_vars = st.multiselect(
    "Select Covariates for Cox Model",
    available_vars,
    default=available_vars
)

if selected_vars and st.button("🚀 Fit Cox Proportional Hazards Model"):
    with st.spinner("Fitting Cox PH model..."):
        try:
            # Prepare data
            cox_data = df[['time', 'event'] + selected_vars].copy()
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='time', event_col='event')
            
            st.success("✅ Model fitted successfully!")
            
            # Model summary
            with st.expander("📄 Full Model Summary", expanded=False):
                st.text(cph.summary)
            
            # Coefficients and Hazard Ratios
            st.subheader("📊 Coefficients and Hazard Ratios")
            
            coef_df = pd.DataFrame({
                'Variable': cph.summary.index,
                'Coefficient': cph.summary['coef'],
                'Hazard Ratio': cph.summary['exp(coef)'],
                'HR 95% CI Lower': cph.summary['exp(coef) lower 95%'],
                'HR 95% CI Upper': cph.summary['exp(coef) upper 95%'],
                'p-value': cph.summary['p'],
                'Significant': ['✅' if p < 0.05 else '❌' for p in cph.summary['p']]
            })
            
            st.dataframe(coef_df.style.format({
                'Coefficient': '{:.4f}',
                'Hazard Ratio': '{:.4f}',
                'HR 95% CI Lower': '{:.4f}',
                'HR 95% CI Upper': '{:.4f}',
                'p-value': '{:.4f}'
            }))
            
            # Model metrics
            col1, col2 = st.columns(2)
            col1.metric("Concordance Index (C-index)", f"{cph.concordance_index_:.4f}")
            col2.metric("Log-Likelihood", f"{cph.log_likelihood_:.2f}")
            
            # Hazard ratio visualization
            st.subheader("📈 Hazard Ratios with 95% CI")
            
            fig, ax = plt.subplots(figsize=(10, len(selected_vars) * 0.8 + 2))
            
            y_pos = np.arange(len(selected_vars))
            hrs = coef_df['Hazard Ratio'].values
            lower = coef_df['HR 95% CI Lower'].values
            upper = coef_df['HR 95% CI Upper'].values
            
            ax.errorbar(hrs, y_pos, xerr=[hrs - lower, upper - hrs], 
                       fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2)
            ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='HR = 1 (no effect)')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_df['Variable'])
            ax.set_xlabel('Hazard Ratio', fontsize=12)
            ax.set_title('Hazard Ratios with 95% Confidence Intervals', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Survival curves for different covariate values
            st.subheader("🔮 Predicted Survival Curves")
            
            st.markdown("Compare survival curves for different covariate profiles:")
            
            # Create example profiles
            if 'treatment' in selected_vars:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for treat in [0, 1]:
                    # Create a profile with median values, varying treatment
                    profile = pd.DataFrame({
                        var: [df[var].median()] for var in selected_vars
                    })
                    profile['treatment'] = treat
                    
                    cph.plot_partial_effects_on_outcome(
                        covariates='treatment',
                        values=[treat],
                        cmap='coolwarm',
                        ax=ax
                    )
                
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Survival Probability', fontsize=12)
                ax.set_title('Predicted Survival by Treatment (other vars at median)', 
                            fontsize=14, fontweight='bold')
                ax.legend([f'Treatment {i}' for i in [0, 1]])
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            
            # Model diagnostics: Check proportional hazards assumption
            st.subheader("🔍 Proportional Hazards Assumption Check")
            
            with st.expander("📊 Schoenfeld Residuals Test"):
                from lifelines.statistics import proportional_hazard_test
                
                ph_test = proportional_hazard_test(cph, cox_data, time_transform='rank')
                
                st.dataframe(ph_test.summary.style.format(precision=4))
                
                st.markdown("""
                **Interpretation:**
                - If p-value > 0.05: Proportional hazards assumption holds
                - If p-value < 0.05: Assumption may be violated for that covariate
                """)
            
            # Download results
            st.download_button(
                label="📥 Download Cox Model Coefficients",
                data=coef_df.to_csv(index=False).encode("utf-8"),
                file_name="cox_model_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error fitting model: {e}")

# -----------------------------------------------
# 📚 Key Concepts
# -----------------------------------------------
st.markdown("---")
st.subheader("📚 Key Concepts in Survival Analysis")

st.markdown("""
### Censoring
- **Right Censoring:** Most common - event hasn't occurred by end of study
- Properly accounting for censoring is critical for valid inference

### Kaplan-Meier
- Non-parametric estimator of survival function
- No assumptions about survival distribution
- Handles censoring correctly

### Cox Proportional Hazards
- Semi-parametric: no assumption about baseline hazard
- **Key Assumption:** Proportional hazards (hazard ratios constant over time)
- Interpretable coefficients (log hazard ratios)

### Hazard Ratio Interpretation
- **HR = 1:** No effect
- **HR > 1:** Increased hazard (higher risk, worse survival)
- **HR < 1:** Decreased hazard (lower risk, better survival)
- Example: HR =  2 means twice the hazard at any time point
""")

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Survival Analysis Module")
