# 20_Power_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import (
    TTestIndPower, TTestPower,
    FTestAnovaPower, FTestPower,
    NormalIndPower
)
import seaborn as sns

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Statistical Power Analysis – StatsmodelsMasterPro",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Statistical Power Analysis")
st.markdown("""
**Power Analysis** helps you determine:
- **Sample size** needed for a study
- **Statistical power** given sample size and effect size
- **Detectable effect size** given power and sample size

**Key Concepts:**
- **Power (1-β):** Probability of detecting an effect when it exists (typically 0.80 or 80%)
- **Alpha (α):** Significance level (typically 0.05)
- **Effect Size:** Magnitude of the difference/effect (Cohen's d, f, etc.)
""")

# -----------------------------------------------
# 📊 Select Analysis Type
# -----------------------------------------------
analysis_type = st.sidebar.selectbox(
    "Select Power Analysis",
    [
        "Independent t-test Power",
        "Paired t-test Power",
        "One-way ANOVA Power",
        "Correlation Power"
    ]
)

# ===============================================
# 🔹 INDEPENDENT T-TEST POWER
# ===============================================
if analysis_type == "Independent t-test Power":
    st.header("🔹 Independent t-test Power Analysis")
    
    st.markdown("""
    Calculate power for comparing **two independent groups**.
    
    **Effect Size (Cohen's d):**
    - Small: 0.2
    - Medium: 0.5
    - Large: 0.8
    """)
    
    # Input parameters
    st.subheader("⚙️ Configuration")
    
    calculation_type = st.radio(
        "What do you want to calculate?",
        ["Sample Size", "Power", "Effect Size"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    with col2:
        if calculation_type != "Sample Size":
            nobs1 = st.number_input("Sample Size Group 1", min_value=5, max_value=1000, value=30)
    
    with col3:
        ratio = st.number_input("Ratio (n2/n1)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    
    if calculation_type == "Sample Size":
        effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.01)
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    elif calculation_type == "Power":
        effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.01)
    else:  # Effect Size
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    
    alternative = st.selectbox("Alternative Hypothesis", ["two-sided", "larger", "smaller"])
    
    if st.button("📊 Calculate"):
        analysis = TTestIndPower()
        
        try:
            if calculation_type == "Sample Size":
                result = analysis.solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha,
                    ratio=ratio,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.subheader("📊 Results")
                
                col1, col2 = st.columns(2)
                col1.metric("Required Sample Size (Group 1)", f"{int(np.ceil(result))}")
                col2.metric("Required Sample Size (Group 2)", f"{int(np.ceil(result * ratio))}")
                
                st.metric("Total Sample Size", f"{int(np.ceil(result * (1 + ratio)))}")
                
            elif calculation_type == "Power":
                result = analysis.solve_power(
                    effect_size=effect_size,
                    nobs1=nobs1,
                    alpha=alpha,
                    ratio=ratio,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.subheader("📊 Results")
                
                col1, col2 = st.columns(2)
                col1.metric("Statistical Power", f"{result:.4f} ({result*100:.2f}%)")
                
                if result < 0.80:
                    col2.markdown("⚠️ **Power is below 0.80** - Consider increasing sample size")
                else:
                    col2.markdown("✅ **Adequate power** for this analysis")
            
            else:  # Effect Size
                result = analysis.solve_power(
                    power=power,
                    nobs1=nobs1,
                    alpha=alpha,
                    ratio=ratio,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.subheader("📊 Results")
                
                st.metric("Detectable Effect Size (Cohen's d)", f"{result:.4f}")
                
                if result > 0.8:
                    st.info("ℹ️ Can only detect large effects with this sample size")
                elif result > 0.5:
                    st.info("ℹ️ Can detect medium to large effects")
                else:
                    st.success("✅ Can detect small to medium effects")
            
            # Power curve
            st.subheader("📈 Power Curve Analysis")
            
            if calculation_type != "Effect Size":
                # Plot power vs sample size
                sample_sizes = np.arange(10, 200, 5)
                powers = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, 
                                                ratio=ratio, alternative=alternative) 
                          for n in sample_sizes]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(sample_sizes, powers, linewidth=2, color='blue')
                ax.axhline(y=0.80, color='r', linestyle='--', label='Power = 0.80')
                ax.axhline(y=power if calculation_type == "Sample Size" else result, 
                          color='g', linestyle=':', label=f'Current Power = {power if calculation_type == "Sample Size" else result:.3f}')
                ax.set_xlabel('Sample Size (n1)', fontsize=12)
                ax.set_ylabel('Power', fontsize=12)
                ax.set_title(f'Power vs Sample Size (d={effect_size}, α={alpha})', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            else:
                # Plot power vs effect size
                effect_sizes = np.arange(0.1, 2.0, 0.05)
                powers = [analysis.solve_power(effect_size=d, nobs1=nobs1, alpha=alpha,
                                               ratio=ratio, alternative=alternative)
                         for d in effect_sizes]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(effect_sizes, powers, linewidth=2, color='blue')
                ax.axhline(y=power, color='r', linestyle='--', label=f'Power = {power}')
                ax.axvline(x=result, color='g', linestyle=':', label=f'Detectable d = {result:.3f}')
                ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12)
                ax.set_ylabel('Power', fontsize=12)
                ax.set_title(f'Power vs Effect Size (n={nobs1}, α={alpha})', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        except Exception as e:
            st.error(f"❌ Error in calculation: {e}")

# ===============================================
# 🔹 PAIRED T-TEST POWER
# ===============================================
elif analysis_type == "Paired t-test Power":
    st.header("🔹 Paired t-test Power Analysis")
    
    st.markdown("""
    Calculate power for **paired samples** (e.g., before/after measurements).
    """)
    
    st.subheader("⚙️ Configuration")
    
    calculation_type = st.radio(
        "What do you want to calculate?",
        ["Sample Size", "Power", "Effect Size"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    with col2:
        if calculation_type != "Sample Size":
            nobs = st.number_input("Number of Pairs", min_value=5, max_value=1000, value=30)
    
    if calculation_type == "Sample Size":
        effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.01)
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    elif calculation_type == "Power":
        effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.01)
    else:
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    
    alternative = st.selectbox("Alternative Hypothesis", ["two-sided", "larger", "smaller"])
    
    if st.button("📊 Calculate"):
        analysis = TTestPower()
        
        try:
            if calculation_type == "Sample Size":
                result = analysis.solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Required Number of Pairs", f"{int(np.ceil(result))}")
            
            elif calculation_type == "Power":
                result = analysis.solve_power(
                    effect_size=effect_size,
                    nobs=nobs,
                    alpha=alpha,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Statistical Power", f"{result:.4f} ({result*100:.2f}%)")
            
            else:
                result = analysis.solve_power(
                    power=power,
                    nobs=nobs,
                    alpha=alpha,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Detectable Effect Size", f"{result:.4f}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 ONE-WAY ANOVA POWER
# ===============================================
elif analysis_type == "One-way ANOVA Power":
    st.header("🔹 One-way ANOVA Power Analysis")
    
    st.markdown("""
    Calculate power for comparing **multiple groups**.
    
    **Effect Size (Cohen's f):**
    - Small: 0.10
    - Medium: 0.25
    - Large: 0.40
    """)
    
    st.subheader("⚙️ Configuration")
    
    calculation_type = st.radio(
        "What do you want to calculate?",
        ["Sample Size", "Power", "Effect Size"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        k_groups = st.number_input("Number of Groups", min_value=2, max_value=10, value=3)
    
    with col2:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    with col3:
        if calculation_type != "Sample Size":
            nobs = st.number_input("Sample Size per Group", min_value=5, max_value=500, value=30)
    
    if calculation_type == "Sample Size":
        effect_size = st.slider("Effect Size (Cohen's f)", 0.05, 1.0, 0.25, 0.01)
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    elif calculation_type == "Power":
        effect_size = st.slider("Effect Size (Cohen's f)", 0.05, 1.0, 0.25, 0.01)
    else:
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    
    if st.button("📊 Calculate"):
        analysis = FTestAnovaPower()
        
        try:
            if calculation_type == "Sample Size":
                result = analysis.solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha,
                    k_groups=k_groups
                )
                
                st.success("✅ Calculation complete!")
                col1, col2 = st.columns(2)
                col1.metric("Required Sample Size per Group", f"{int(np.ceil(result))}")
                col2.metric("Total Sample Size", f"{int(np.ceil(result * k_groups))}")
            
            elif calculation_type == "Power":
                result = analysis.solve_power(
                    effect_size=effect_size,
                    nobs=nobs,
                    alpha=alpha,
                    k_groups=k_groups
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Statistical Power", f"{result:.4f} ({result*100:.2f}%)")
            
            else:
                result = analysis.solve_power(
                    power=power,
                    nobs=nobs,
                    alpha=alpha,
                    k_groups=k_groups
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Detectable Effect Size (Cohen's f)", f"{result:.4f}")
            
            # Visualization: Power vs Sample Size
            if calculation_type != "Effect Size":
                st.subheader("📈 Power Curve")
                
                sample_sizes = np.arange(5, 150, 3)
                powers = [analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha, k_groups=k_groups)
                         for n in sample_sizes]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(sample_sizes, powers, linewidth=2)
                ax.axhline(y=0.80, color='r', linestyle='--', label='Power = 0.80')
                ax.set_xlabel('Sample Size per Group', fontsize=12)
                ax.set_ylabel('Power', fontsize=12)
                ax.set_title(f'Power vs Sample Size (f={effect_size}, k={k_groups}, α={alpha})', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 CORRELATION POWER
# ===============================================
elif analysis_type == "Correlation Power":
    st.header("🔹 Correlation Power Analysis")
    
    st.markdown("""
    Calculate power for **correlation tests** (Pearson's r).
    """)
    
    st.subheader("⚙️ Configuration")
    
    calculation_type = st.radio(
        "What do you want to calculate?",
        ["Sample Size", "Power", "Effect Size (r)"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    with col2:
        if calculation_type != "Sample Size":
            nobs = st.number_input("Sample Size", min_value=10, max_value=1000, value=50)
    
    if calculation_type == "Sample Size":
        effect_size = st.slider("Expected Correlation (r)", 0.05, 0.95, 0.30, 0.01)
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    elif calculation_type == "Power":
        effect_size = st.slider("Expected Correlation (r)", 0.05, 0.95, 0.30, 0.01)
    else:
        power = st.slider("Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01)
    
    alternative = st.selectbox("Alternative", ["two-sided", "larger", "smaller"])
    
    if st.button("📊 Calculate"):
        analysis = NormalIndPower()
        
        # Convert correlation to effect size (Cohen's d approximation)
        # For correlation: d ≈ 2r / sqrt(1 - r²)
        
        try:
            if calculation_type != "Effect Size (r)":
                d = 2 * effect_size / np.sqrt(1 - effect_size**2)
            
            if calculation_type == "Sample Size":
                result = analysis.solve_power(
                    effect_size=d,
                    power=power,
                    alpha=alpha,
                    ratio=1,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Required Sample Size", f"{int(np.ceil(result))}")
            
            elif calculation_type == "Power":
                result = analysis.solve_power(
                    effect_size=d,
                    nobs1=nobs,
                    alpha=alpha,
                    ratio=1,
                    alternative=alternative
                )
                
                st.success("✅ Calculation complete!")
                st.metric("Statistical Power", f"{result:.4f} ({result*100:.2f}%)")
            
            else:
                result_d = analysis.solve_power(
                    power=power,
                    nobs1=nobs,
                    alpha=alpha,
                    ratio=1,
                    alternative=alternative
                )
                
                # Convert back to correlation
                result = result_d / np.sqrt(result_d**2 + 4)
                
                st.success("✅ Calculation complete!")
                st.metric("Detectable Correlation (r)", f"{result:.4f}")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# -----------------------------------------------
# 📚 Effect Size Guidelines
# -----------------------------------------------
st.markdown("---")
st.subheader("📚 Effect Size Guidelines (Cohen, 1988)")

guidelines_df = pd.DataFrame({
    'Test Type': ['t-test (d)', 'ANOVA (f)', 'Correlation (r)'],
    'Small': ['0.20', '0.10', '0.10'],
    'Medium': ['0.50', '0.25', '0.30'],
    'Large': ['0.80', '0.40', '0.50']
})

st.dataframe(guidelines_df, use_container_width=True)

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Power Analysis Module")
