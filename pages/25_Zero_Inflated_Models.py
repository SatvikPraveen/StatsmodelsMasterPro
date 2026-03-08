# 25_Zero_Inflated_Models.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Zero-Inflated Models – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🎯"
)

st.title("🎯 Zero-Inflated Count Models")
st.markdown("""
Handle **excess zeros** in count data with specialized models:
- **ZIP** - Zero-Inflated Poisson
- **ZINB** - Zero-Inflated Negative Binomial
- Compare with standard Poisson/Negative Binomial models
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "zero_inflated_count.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("📏 Observations", len(df))
col2.metric("🎯 Mean Count", f"{df['y'].mean():.2f}")
col3.metric("⭕ Zero Proportion", f"{(df['y'] == 0).mean():.2%}")

# -----------------------------------------------
# 📊 Exploratory Analysis
# -----------------------------------------------
st.subheader("📊 Count Distribution")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(df['y'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Counts', fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Proportion by count
count_props = df['y'].value_counts(normalize=True).sort_index()
axes[1].bar(count_props.index, count_props.values, color='coral', edgecolor='black')
axes[1].set_xlabel('Count Value')
axes[1].set_ylabel('Proportion')
axes[1].set_title('Proportion by Count Value', fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Zero inflation indicator
zero_count = (df['y'] == 0).sum()
total_count = len(df)
zero_prop = zero_count / total_count

st.info(f"**Zero Observations:** {zero_count}/{total_count} ({zero_prop:.1%}) - High zero proportion suggests zero-inflation")

# -----------------------------------------------
# 🔧 Model Selection
# -----------------------------------------------
st.subheader("⚙️ Model Configuration")

model_type = st.selectbox(
    "Select Model Type",
    ["Zero-Inflated Poisson (ZIP)", "Zero-Inflated Negative Binomial (ZINB)", 
     "Compare All Models"]
)

# Select predictors
predictors = st.multiselect(
    "Select Predictors for Count Model",
    ['x1', 'x2', 'x3'],
    default=['x1', 'x2']
)

inflate_predictors = st.multiselect(
    "Select Predictors for Inflation Model (Pr(excess zero))",
    ['x1', 'x2', 'x3'],
    default=['x3']
)

# -----------------------------------------------
# 🎯 ZERO-INFLATED POISSON
# -----------------------------------------------
if model_type == "Zero-Inflated Poisson (ZIP)" and predictors and inflate_predictors:
    st.header("🎯 Zero-Inflated Poisson Model")
    
    if st.button("🚀 Fit ZIP Model"):
        with st.spinner("Fitting Zero-Inflated Poisson model..."):
            try:
                # Prepare data
                X = sm.add_constant(df[predictors])
                X_inflate = sm.add_constant(df[inflate_predictors])
                y = df['y']
                
                # Fit ZIP model
                zip_model = ZeroInflatedPoisson(y, X, exog_infl=X_inflate)
                zip_result = zip_model.fit(disp=False)
                
                st.success("✅ ZIP model fitted successfully!")
                
                # Model Summary
                st.subheader("📄 Model Summary")
                with st.expander("Full Model Summary", expanded=False):
                    st.text(zip_result.summary())
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("AIC", f"{zip_result.aic:.2f}")
                col2.metric("BIC", f"{zip_result.bic:.2f}")
                col3.metric("Log-Likelihood", f"{zip_result.llf:.2f}")
                
                # Coefficients
                st.subheader("📊 Model Coefficients")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Count Model (Poisson)**")
                    count_params = zip_result.params[:len(predictors)+1]
                    count_pvals = zip_result.pvalues[:len(predictors)+1]
                    
                    coef_df = pd.DataFrame({
                        'Variable': ['const'] + predictors,
                        'Coefficient': count_params.values,
                        'p-value': count_pvals.values,
                        'Significant': ['✅' if p < 0.05 else '❌' for p in count_pvals.values]
                    })
                    st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}', 'p-value': '{:.4f}'}))
                
                with col2:
                    st.markdown("**Inflation Model (Logistic)**")
                    inflate_params = zip_result.params[len(predictors)+1:]
                    inflate_pvals = zip_result.pvalues[len(predictors)+1:]
                    
                    infl_df = pd.DataFrame({
                        'Variable': ['const'] + inflate_predictors,
                        'Coefficient': inflate_params.values,
                        'p-value': inflate_pvals.values,
                        'Significant': ['✅' if p < 0.05 else '❌' for p in inflate_pvals.values]
                    })
                    st.dataframe(infl_df.style.format({'Coefficient': '{:.4f}', 'p-value': '{:.4f}'}))
                
                # Predictions
                st.subheader("🔮 Predicted vs Actual")
                
                predictions = zip_result.predict()
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Scatter plot
                axes[0].scatter(df['y'], predictions, alpha=0.5, edgecolor='black')
                axes[0].plot([df['y'].min(), df['y'].max()], 
                           [df['y'].min(), df['y'].max()], 
                           'r--', linewidth=2, label='Perfect Fit')
                axes[0].set_xlabel('Actual Count')
                axes[0].set_ylabel('Predicted Count')
                axes[0].set_title('Predicted vs Actual Counts', fontweight='bold')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                
                # Residuals
                residuals = df['y'] - predictions
                axes[1].scatter(predictions, residuals, alpha=0.5, edgecolor='black')
                axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
                axes[1].set_xlabel('Predicted Count')
                axes[1].set_ylabel('Residuals')
                axes[1].set_title('Residual Plot', fontweight='bold')
                axes[1].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Interpretation
                st.subheader("📖 Interpretation")
                st.markdown("""
                - **Count Model**: Predicts the count when observation is NOT excess zero
                - **Inflation Model**: Predicts probability of being excess zero
                - Positive inflation coefficient → higher probability of excess zero
                - Negative inflation coefficient → lower probability of excess zero
                """)
                
            except Exception as e:
                st.error(f"❌ Error fitting model: {e}")

# -----------------------------------------------
# 🎯 ZERO-INFLATED NEGATIVE BINOMIAL
# -----------------------------------------------
elif model_type == "Zero-Inflated Negative Binomial (ZINB)" and predictors and inflate_predictors:
    st.header("🎯 Zero-Inflated Negative Binomial Model")
    
    if st.button("🚀 Fit ZINB Model"):
        with st.spinner("Fitting Zero-Inflated Negative Binomial model..."):
            try:
                # Prepare data
                X = sm.add_constant(df[predictors])
                X_inflate = sm.add_constant(df[inflate_predictors])
                y = df['y']
                
                # Fit ZINB model
                zinb_model = ZeroInflatedNegativeBinomialP(y, X, exog_infl=X_inflate)
                zinb_result = zinb_model.fit(disp=False, maxiter=1000)
                
                st.success("✅ ZINB model fitted successfully!")
                
                # Model Summary
                st.subheader("📄 Model Summary")
                with st.expander("Full Model Summary", expanded=False):
                    st.text(zinb_result.summary())
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AIC", f"{zinb_result.aic:.2f}")
                col2.metric("BIC", f"{zinb_result.bic:.2f}")
                col3.metric("Log-Likelihood", f"{zinb_result.llf:.2f}")
                
                # Extract alpha (dispersion parameter)
                try:
                    alpha = zinb_result.params['alpha']
                    col4.metric("Alpha (Dispersion)", f"{alpha:.4f}")
                except:
                    col4.metric("Alpha", "N/A")
                
                # Coefficients
                st.subheader("📊 Model Coefficients")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Count Model (Negative Binomial)**")
                    count_params = zinb_result.params[:len(predictors)+1]
                    count_pvals = zinb_result.pvalues[:len(predictors)+1]
                    
                    coef_df = pd.DataFrame({
                        'Variable': ['const'] + predictors,
                        'Coefficient': count_params.values,
                        'p-value': count_pvals.values,
                        'Significant': ['✅' if p < 0.05 else '❌' for p in count_pvals.values]
                    })
                    st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}', 'p-value': '{:.4f}'}))
                
                with col2:
                    st.markdown("**Inflation Model (Logistic)**")
                    # Inflation params come after count params but before alpha
                    n_count = len(predictors) + 1
                    n_inflate = len(inflate_predictors) + 1
                    
                    inflate_params = zinb_result.params[n_count:n_count+n_inflate]
                    inflate_pvals = zinb_result.pvalues[n_count:n_count+n_inflate]
                    
                    infl_df = pd.DataFrame({
                        'Variable': ['const'] + inflate_predictors,
                        'Coefficient': inflate_params.values,
                        'p-value': inflate_pvals.values,
                        'Significant': ['✅' if p < 0.05 else '❌' for p in inflate_pvals.values]
                    })
                    st.dataframe(infl_df.style.format({'Coefficient': '{:.4f}', 'p-value': '{:.4f}'}))
                
                # Predictions
                st.subheader("🔮 Predicted vs Actual")
                
                predictions = zinb_result.predict()
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Scatter plot
                axes[0].scatter(df['y'], predictions, alpha=0.5, edgecolor='black', color='purple')
                axes[0].plot([df['y'].min(), df['y'].max()], 
                           [df['y'].min(), df['y'].max()], 
                           'r--', linewidth=2, label='Perfect Fit')
                axes[0].set_xlabel('Actual Count')
                axes[0].set_ylabel('Predicted Count')
                axes[0].set_title('Predicted vs Actual Counts', fontweight='bold')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                
                # Residuals
                residuals = df['y'] - predictions
                axes[1].scatter(predictions, residuals, alpha=0.5, edgecolor='black', color='purple')
                axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
                axes[1].set_xlabel('Predicted Count')
                axes[1].set_ylabel('Residuals')
                axes[1].set_title('Residual Plot', fontweight='bold')
                axes[1].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Interpretation
                st.subheader("📖 Interpretation")
                st.markdown("""
                - **ZINB** allows for overdispersion (variance > mean) via alpha parameter
                - **Alpha > 0**: Overdispersion present - ZINB preferred over ZIP
                - **Use ZINB when**: Data has both excess zeros AND overdispersion
                """)
                
            except Exception as e:
                st.error(f"❌ Error fitting model: {e}")

# -----------------------------------------------
# 🔍 COMPARE ALL MODELS
# -----------------------------------------------
elif model_type == "Compare All Models" and predictors and inflate_predictors:
    st.header("🔍 Model Comparison")
    
    if st.button("🚀 Fit and Compare All Models"):
        with st.spinner("Fitting all models..."):
            try:
                # Prepare data
                X = sm.add_constant(df[predictors])
                X_inflate = sm.add_constant(df[inflate_predictors])
                y = df['y']
                
                # Fit standard Poisson
                poisson_model = Poisson(y, X)
                poisson_result = poisson_model.fit(disp=False)
                
                # Fit Zero-Inflated Poisson
                zip_model = ZeroInflatedPoisson(y, X, exog_infl=X_inflate)
                zip_result = zip_model.fit(disp=False)
                
                # Fit standard Negative Binomial
                nb_model = NegativeBinomial(y, X)
                nb_result = nb_model.fit(disp=False)
                
                # Fit Zero-Inflated Negative Binomial
                zinb_model = ZeroInflatedNegativeBinomialP(y, X, exog_infl=X_inflate)
                zinb_result = zinb_model.fit(disp=False, maxiter=1000)
                
                st.success("✅ All models fitted successfully!")
                
                # Model Comparison Table
                st.subheader("📊 Model Comparison")
                
                comparison_df = pd.DataFrame({
                    'Model': ['Poisson', 'ZIP', 'Neg. Binomial', 'ZINB'],
                    'AIC': [poisson_result.aic, zip_result.aic, nb_result.aic, zinb_result.aic],
                    'BIC': [poisson_result.bic, zip_result.bic, nb_result.bic, zinb_result.bic],
                    'Log-Likelihood': [poisson_result.llf, zip_result.llf, nb_result.llf, zinb_result.llf],
                })
                
                # Highlight best AIC and BIC
                def highlight_min(s):
                    is_min = s == s.min()
                    return ['background-color: lightgreen' if v else '' for v in is_min]
                
                styled_df = comparison_df.style.format({
                    'AIC': '{:.2f}',
                    'BIC': '{:.2f}',
                    'Log-Likelihood': '{:.2f}'
                }).apply(highlight_min, subset=['AIC', 'BIC'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Best model
                best_aic_idx = comparison_df['AIC'].idxmin()
                best_bic_idx = comparison_df['BIC'].idxmin()
                
                col1, col2 = st.columns(2)
                col1.success(f"**Best by AIC:** {comparison_df.loc[best_aic_idx, 'Model']}")
                col2.success(f"**Best by BIC:** {comparison_df.loc[best_bic_idx, 'Model']}")
                
                # Prediction comparison
                st.subheader("🔮 Prediction Comparison")
                
                pred_poisson = poisson_result.predict()
                pred_zip = zip_result.predict()
                pred_nb = nb_result.predict()
                pred_zinb = zinb_result.predict()
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                models_pred = [
                    (pred_poisson, 'Poisson', 'blue'),
                    (pred_zip, 'ZIP', 'green'),
                    (pred_nb, 'Neg. Binomial', 'orange'),
                    (pred_zinb, 'ZINB', 'purple')
                ]
                
                for idx, (pred, name, color) in enumerate(models_pred):
                    ax = axes[idx // 2, idx % 2]
                    ax.scatter(df['y'], pred, alpha=0.5, edgecolor='black', color=color)
                    ax.plot([df['y'].min(), df['y'].max()], 
                           [df['y'].min(), df['y'].max()], 
                           'r--', linewidth=2, label='Perfect Fit')
                    ax.set_xlabel('Actual Count')
                    ax.set_ylabel('Predicted Count')
                    ax.set_title(f'{name} - Predicted vs Actual', fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Model Selection Guide
                st.subheader("📖 Model Selection Guide")
                st.markdown("""
                **When to use each model:**
                
                1. **Poisson**: Baseline model for count data (mean = variance)
                2. **ZIP**: When you have excess zeros but no overdispersion
                3. **Negative Binomial**: When you have overdispersion but no excess zeros
                4. **ZINB**: When you have BOTH excess zeros AND overdispersion
                
                **Decision criteria:**
                - Compare AIC/BIC (lower is better)
                - Check for overdispersion (variance > mean)
                - Check zero proportion vs Poisson expectation
                - Use likelihood ratio tests when models are nested
                """)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

else:
    st.info("👆 Please select predictors for both count and inflation models to begin.")

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Zero-Inflated Models Module")
