# 24_Mediation_Moderation.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import seaborn as sns

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Mediation & Moderation – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🔀"
)

st.title("🔀 Mediation & Moderation Analysis")
st.markdown("""
Understand **how** and **when** variables influence each other:

**🔹 Mediation:** How does X affect Y? Through what mechanism?
- X → M → Y

**🔹 Moderation:** When does X affect Y? Under what conditions?
- X × W → Y
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "mediation_data.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.metric("📏 Observations", len(df))

with st.expander("📈 Data Summary"):
    st.dataframe(df.describe().T.style.format(precision=2))

# -----------------------------------------------
# 📊 Select Analysis Type
# -----------------------------------------------
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Mediation Analysis", "Moderation Analysis"]
)

# ===============================================
# 🔹 MEDIATION ANALYSIS
# ===============================================
if analysis_type == "Mediation Analysis":
    st.header("🔹 Mediation Analysis")
    
    st.markdown("""
    **Mediation** tests whether the effect of X on Y operates **through** a mediator M.
    
    **Conceptual Model:**
    ```
    X → M → Y
    ```
    
    **Steps (Baron & Kenny, 1986):**
    1. **c:** Total effect of X on Y (without M)
    2. **a:** Effect of X on M
    3. **b:** Effect of M on Y (controlling for X)
    4. **c':** Direct effect of X on Y (controlling for M)
    
    **Mediation occurs when:**
    - **Full mediation:** c' = 0 (all effect through M)
    - **Partial mediation:** c' ≠ 0 but c' < c (some effect through M)
    
    **Indirect Effect:** a × b
    """)
    
    # Variable selection
    st.subheader("⚙️ Variable Assignment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        X_var = st.selectbox("Independent Variable (X)", ['X'], index=0)
    with col2:
        M_var = st.selectbox("Mediator (M)", ['M'], index=0)
    with col3:
        Y_var = st.selectbox("Dependent Variable (Y)", ['Y'], index=0)
    
    if st.button("🔍 Run Mediation Analysis"):
        with st.spinner("Running mediation analysis..."):
            try:
                # Step 1: Total effect (c path): X → Y
                model_total = smf.ols(f'{Y_var} ~ {X_var}', data=df).fit()
                c_coef = model_total.params[X_var]
                c_pval = model_total.pvalues[X_var]
                
                # Step 2: Effect on mediator (a path): X → M
                model_mediator = smf.ols(f'{M_var} ~ {X_var}', data=df).fit()
                a_coef = model_mediator.params[X_var]
                a_pval = model_mediator.pvalues[X_var]
                
                # Step 3: Direct effect (c' path) and mediator effect (b path): X + M → Y
                model_direct = smf.ols(f'{Y_var} ~ {X_var} + {M_var}', data=df).fit()
                c_prime_coef = model_direct.params[X_var]
                c_prime_pval = model_direct.pvalues[X_var]
                b_coef = model_direct.params[M_var]
                b_pval = model_direct.pvalues[M_var]
                
                # Indirect effect
                indirect_effect = a_coef * b_coef
                
                # Proportion mediated
                prop_mediated = indirect_effect / c_coef if c_coef != 0 else np.nan
                
                st.success("✅ Mediation analysis complete!")
                
                # Results summary
                st.subheader("📊 Mediation Results")
                
                results_df = pd.DataFrame({
                    'Path': ['Total Effect (c)', 'X → M (a)', 'M → Y | X (b)', 'Direct Effect (c\')', 'Indirect Effect (a×b)'],
                    'Coefficient': [c_coef, a_coef, b_coef, c_prime_coef, indirect_effect],
                    'p-value': [c_pval, a_pval, b_pval, c_prime_pval, np.nan],
                    'Significant': [
                        '✅' if c_pval < 0.05 else '❌',
                        '✅' if a_pval < 0.05 else '❌',
                        '✅' if b_pval < 0.05 else '❌',
                        '✅' if c_prime_pval < 0.05 else '❌',
                        ''
                    ]
                })
                
                st.dataframe(results_df.style.format({'Coefficient': '{:.4f}', 'p-value': '{:.4f}'}))
                
                # Mediation metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Effect (c)", f"{c_coef:.4f}")
                col2.metric("Direct Effect (c')", f"{c_prime_coef:.4f}")
                col3.metric("Indirect Effect (a×b)", f"{indirect_effect:.4f}")
                
                if not np.isnan(prop_mediated):
                    st.metric("Proportion Mediated", f"{prop_mediated*100:.1f}%")
                
                # Interpretation
                st.subheader("📖 Interpretation")
                
                if a_pval < 0.05 and b_pval < 0.05:
                    if c_prime_pval >= 0.05 and c_pval < 0.05:
                        st.success("✅ **Full Mediation** detected!")
                        st.markdown(f"The effect of **{X_var}** on **{Y_var}** is fully mediated by **{M_var}**.")
                    elif c_prime_pval < 0.05:
                        st.success("✅ **Partial Mediation** detected!")
                        st.markdown(f"**{M_var}** partially mediates the relationship between **{X_var}** and **{Y_var}**.")
                        st.markdown(f"- Direct effect: {c_prime_coef:.4f}")
                        st.markdown(f"- Indirect effect (through M): {indirect_effect:.4f}")
                else:
                    st.info("ℹ️ **No significant mediation** detected.")
                
                # Visualization
                st.subheader("📊 Mediation Model Diagram")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.axis('off')
                
                # Positions
                x_pos = (2, 5)
                m_pos = (5, 8)
                y_pos = (8, 5)
                
                # Draw boxes
                box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2)
                ax.text(x_pos[0], x_pos[1], X_var, fontsize=14, ha='center', va='center', bbox=box_props)
                ax.text(m_pos[0], m_pos[1], M_var, fontsize=14, ha='center', va='center', bbox=box_props)
                ax.text(y_pos[0], y_pos[1], Y_var, fontsize=14, ha='center', va='center', bbox=box_props)
                
                # Draw arrows
                # X → M (a path)
                ax.annotate('', xy=(m_pos[0]-0.5, m_pos[1]-0.3), xytext=(x_pos[0]+0.5, x_pos[1]+0.3),
                           arrowprops=dict(arrowstyle='->', lw=2, color='green'))
                ax.text(3.5, 7, f'a = {a_coef:.3f}***' if a_pval < 0.001 else f'a = {a_coef:.3f}', 
                       fontsize=11, color='green', fontweight='bold')
                
                # M → Y (b path)
                ax.annotate('', xy=(y_pos[0]-0.5, y_pos[1]+0.3), xytext=(m_pos[0]+0.5, m_pos[1]-0.3),
                           arrowprops=dict(arrowstyle='->', lw=2, color='green'))
                ax.text(6.5, 7, f'b = {b_coef:.3f}***' if b_pval < 0.001 else f'b = {b_coef:.3f}', 
                       fontsize=11, color='green', fontweight='bold')
                
                # X → Y (c' path, direct)
                ax.annotate('', xy=(y_pos[0]-0.5, y_pos[1]-0.1), xytext=(x_pos[0]+0.5, x_pos[1]-0.1),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
                ax.text(5, 4, f"c' = {c_prime_coef:.3f}" + ('***' if c_prime_pval < 0.001 else ''), 
                       fontsize=11, color='blue', fontweight='bold')
                
                # Total effect annotation
                ax.text(5, 1.5, f'Total Effect (c) = {c_coef:.3f}', fontsize=12, ha='center', 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                ax.text(5, 0.8, f'Indirect Effect (a×b) = {indirect_effect:.3f}', fontsize=12, ha='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                
                st.pyplot(fig)
                plt.close()
                
                # Model details
                with st.expander("📄 Detailed Model Outputs"):
                    st.markdown("**Model 1: Total Effect (X → Y)**")
                    st.text(model_total.summary())
                    
                    st.markdown("**Model 2: X → M**")
                    st.text(model_mediator.summary())
                    
                    st.markdown("**Model 3: X + M → Y**")
                    st.text(model_direct.summary())
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 MODERATION ANALYSIS
# ===============================================
else:  # Moderation Analysis
    st.header("🔹 Moderation Analysis")
    
    st.markdown("""
    **Moderation** tests whether the effect of X on Y depends on a **moderator** W.
    
    **Conceptual Model:**
    ```
    Y = β₀ + β₁X + β₂W + β₃(X×W) + ε
    ```
    
    **Key:**
    - **β₁:** Effect of X when W = 0
    - **β₂:** Effect of W when X = 0
    - **β₃:** Interaction effect (moderation)
    
    **Moderation exists when:**
    - β₃ is statistically significant
    - The effect of X on Y changes at different levels of W
    """)
    
    # Variable selection
    st.subheader("⚙️ Variable Assignment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        X_var_mod = st.selectbox("Independent Variable (X)", ['X'], index=0)
    with col2:
        W_var = st.selectbox("Moderator (W)", ['W'], index=0)
    with col3:
        Y_var_mod = st.selectbox("Dependent Variable (Y)", ['Y_moderated'], index=0)
    
    if st.button("🔍 Run Moderation Analysis"):
        with st.spinner("Running moderation analysis..."):
            try:
                # Model without interaction (main effects only)
                model_main = smf.ols(f'{Y_var_mod} ~ {X_var_mod} + {W_var}', data=df).fit()
                
                # Model with interaction
                model_interaction = smf.ols(f'{Y_var_mod} ~ {X_var_mod} * {W_var}', data=df).fit()
                
                st.success("✅ Moderation analysis complete!")
                
                # Coefficients
                st.subheader("📊 Moderation Results")
                
                coef_df = pd.DataFrame({
                    'Variable': model_interaction.params.index,
                    'Coefficient': model_interaction.params.values,
                    'Std Error': model_interaction.bse.values,
                    't-value': model_interaction.tvalues.values,
                    'p-value': model_interaction.pvalues.values,
                    'Significant': ['✅' if p < 0.05 else '❌' for p in model_interaction.pvalues.values]
                })
                
                st.dataframe(coef_df.style.format({
                    'Coefficient': '{:.4f}',
                    'Std Error': '{:.4f}',
                    't-value': '{:.4f}',
                    'p-value': '{:.4f}'
                }))
                
                # Interaction coefficient
                interaction_term = f'{X_var_mod}:{W_var}'
                interaction_coef = model_interaction.params[interaction_term]
                interaction_pval = model_interaction.pvalues[interaction_term]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (Main Effects)", f"{model_main.rsquared:.4f}")
                col2.metric("R² (With Interaction)", f"{model_interaction.rsquared:.4f}")
                col3.metric("ΔR²", f"{model_interaction.rsquared - model_main.rsquared:.4f}")
                
                # Interpretation
                st.subheader("📖 Interpretation")
                
                if interaction_pval < 0.05:
                    st.success(f"✅ **Significant Moderation** detected (p = {interaction_pval:.4f})")
                    st.markdown(f"The effect of **{X_var_mod}** on **{Y_var_mod}** depends on the level of **{W_var}**.")
                    st.markdown(f"Interaction coefficient: **{interaction_coef:.4f}**")
                else:
                    st.info(f"ℹ️ **No significant moderation** detected (p = {interaction_pval:.4f})")
                
                # Visualization: Simple slopes
                st.subheader("📊 Simple Slopes Plot")
                
                # Calculate simple slopes at different levels of W
                W_low = df[W_var].mean() - df[W_var].std()
                W_mean = df[W_var].mean()
                W_high = df[W_var].mean() + df[W_var].std()
                
                # Range of X values
                X_range = np.linspace(df[X_var_mod].min(), df[X_var_mod].max(), 100)
                
                # Predicted Y for each level of W
                intercept = model_interaction.params['Intercept']
                beta_X = model_interaction.params[X_var_mod]
                beta_W = model_interaction.params[W_var]
                beta_interaction = model_interaction.params[interaction_term]
                
                Y_low = intercept + beta_X * X_range + beta_W * W_low + beta_interaction * X_range * W_low
                Y_mean = intercept + beta_X * X_range + beta_W * W_mean + beta_interaction * X_range * W_mean
                Y_high = intercept + beta_X * X_range + beta_W * W_high + beta_interaction * X_range * W_high
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(X_range, Y_low, label=f'W = Low (-1 SD)', linewidth=2, color='blue')
                ax.plot(X_range, Y_mean, label=f'W = Mean', linewidth=2, color='green')
                ax.plot(X_range, Y_high, label=f'W = High (+1 SD)', linewidth=2, color='red')
                
                ax.set_xlabel(X_var_mod, fontsize=12)
                ax.set_ylabel(Y_var_mod, fontsize=12)
                ax.set_title('Simple Slopes (Effect of X at Different Levels of W)', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Simple slopes values
                st.subheader("📈 Simple Slopes Coefficients")
                
                slope_low = beta_X + beta_interaction * W_low
                slope_mean = beta_X + beta_interaction * W_mean
                slope_high = beta_X + beta_interaction * W_high
                
                slopes_df = pd.DataFrame({
                    'W Level': ['Low (-1 SD)', 'Mean', 'High (+1 SD)'],
                    'W Value': [W_low, W_mean, W_high],
                    'Simple Slope': [slope_low, slope_mean, slope_high]
                })
                
                st.dataframe(slopes_df.style.format(precision=4))
                
                # Model details
                with st.expander("📄 Detailed Model Output"):
                    st.text(model_interaction.summary())
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

# -----------------------------------------------
# 📚 Key Concepts
# -----------------------------------------------
st.markdown("---")
st.subheader("📚 Mediation vs Moderation")

comparison_df = pd.DataFrame({
    'Aspect': ['Question', 'Focus', 'Statistical Test', 'Interpretation', 'Example'],
    'Mediation': [
        'HOW does X affect Y?',
        'Mechanism/Process',
        'Indirect effect (a×b)',
        'X affects Y through M',
        'Stress → Sleep Quality → Performance'
    ],
    'Moderation': [
        'WHEN does X affect Y?',
        'Boundary conditions',
        'Interaction term (X×W)',
        'Effect of X depends on W',
        'Exercise → Health (stronger for young people)'
    ]
})

st.dataframe(comparison_df, use_container_width=True)

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Mediation & Moderation Module")
