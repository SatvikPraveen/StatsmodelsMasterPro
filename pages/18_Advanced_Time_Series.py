# 18_Advanced_Time_Series.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Advanced Time Series – StatsmodelsMasterPro",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Advanced Time Series Analysis")
st.markdown("""
Master advanced time series techniques:
- **SARIMAX** - Seasonal ARIMA with eXogenous variables
- **VAR** - Vector Autoregression for multivariate time series
- **VECM** - Vector Error Correction Model for cointegrated series
- **Granger Causality** - Test predictive relationships between series
""")

# -----------------------------------------------
# 📥 Select Analysis Type
# -----------------------------------------------
analysis_type = st.sidebar.selectbox(
    "🔧 Select Analysis Type",
    ["SARIMAX - Seasonal ARIMA", "VAR - Multivariate Time Series", 
     "VECM - Cointegration", "Granger Causality Test"]
)

# ===============================================
# 🔹 SARIMAX - SEASONAL ARIMA
# ===============================================
if analysis_type == "SARIMAX - Seasonal ARIMA":
    st.header("🌊 SARIMAX - Seasonal ARIMA with Exogenous Variables")
    
    DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "seasonal_ts_data.csv"
    df = pd.read_csv(DATA_PATH)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📏 Observations", len(df))
    with col2:
        st.metric("📊 Variables", df.shape[1])
    
    # Plot time series
    st.subheader("📉 Time Series Plot")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['y'], linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series with Seasonality', fontweight='bold')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # ACF and PACF
    st.subheader("📊 ACF & PACF Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(df['y'], lags=40, ax=ax)
        ax.set_title('Autocorrelation Function')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_pacf(df['y'], lags=40, ax=ax)
        ax.set_title('Partial Autocorrelation Function')
        st.pyplot(fig)
        plt.close()
    
    # SARIMAX Configuration
    st.subheader("⚙️ SARIMAX Model Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        p = st.number_input("AR (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("I (d)", min_value=0, max_value=2, value=0)
        q = st.number_input("MA (q)", min_value=0, max_value=5, value=1)
    
    with col2:
        st.markdown("**Seasonal Parameters:**")
        P = st.number_input("Seasonal AR (P)", min_value=0, max_value=3, value=1)
        D = st.number_input("Seasonal I (D)", min_value=0, max_value=2, value=0)
        Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=3, value=1)
    
    with col3:
        s = st.number_input("Seasonal Period (s)", min_value=1, max_value=365, value=12)
    
    with col4:
        use_exog = st.checkbox("Use Exogenous Variable", value=True)
        forecast_steps = st.slider("Forecast Steps", 5, 60, 30)
    
    if st.button("🚀 Fit SARIMAX Model"):
        with st.spinner("Fitting SARIMAX model..."):
            try:
                if use_exog:
                    model = SARIMAX(df['y'], exog=df[['exog']], 
                                    order=(p, d, q), 
                                    seasonal_order=(P, D, Q, s))
                else:
                    model = SARIMAX(df['y'], 
                                    order=(p, d, q), 
                                    seasonal_order=(P, D, Q, s))
                
                fitted_model = model.fit(disp=False)
                
                st.success("✅ Model fitted successfully!")
                
                # Model Summary
                with st.expander("📄 Model Summary", expanded=False):
                    st.text(fitted_model.summary())
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("AIC", f"{fitted_model.aic:.2f}")
                col2.metric("BIC", f"{fitted_model.bic:.2f}")
                col3.metric("Log-Likelihood", f"{fitted_model.llf:.2f}")
                
                # Forecast
                st.subheader("🔮 Forecast")
                
                if use_exog:
                    # Generate future exog values (simple: use mean)
                    future_exog = np.full((forecast_steps, 1), df['exog'].mean())
                    forecast = fitted_model.forecast(steps=forecast_steps, exog=future_exog)
                else:
                    forecast = fitted_model.forecast(steps=forecast_steps)
                
                # Plot forecast
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Historical data
                ax.plot(df.index, df['y'], label='Historical', linewidth=1.5)
                
                # Forecast
                forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:]
                ax.plot(forecast_index, forecast, 'r--', label='Forecast', linewidth=2)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title('SARIMAX Forecast', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Diagnostics
                st.subheader("🔍 Residual Diagnostics")
                fig = fitted_model.plot_diagnostics(figsize=(12, 8))
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ Error fitting model: {e}")

# ===============================================
# 🔹 VAR - VECTOR AUTOREGRESSION
# ===============================================
elif analysis_type == "VAR - Multivariate Time Series":
    st.header("📊 VAR - Vector Autoregression")
    
    DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "var_data.csv"
    df = pd.read_csv(DATA_PATH)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.metric("📏 Observations", len(df))
    
    # Plot both series
    st.subheader("📉 Multivariate Time Series Plot")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(df.index, df['y1'], linewidth=1.5, color='blue')
    axes[0].set_ylabel('y1')
    axes[0].set_title('Series y1', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(df.index, df['y2'], linewidth=1.5, color='green')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('y2')
    axes[1].set_title('Series y2', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # VAR Configuration
    st.subheader("⚙️ VAR Model Configuration")
    
    max_lags = st.slider("Maximum Lags to Consider", 1, 10, 5)
    
    if st.button("🔍 Find Optimal Lag Order"):
        with st.spinner("Analyzing lag orders..."):
            try:
                model = VAR(df[['y1', 'y2']])
                lag_order_results = model.select_order(maxlags=max_lags)
                
                st.success("✅ Lag order analysis complete!")
                
                st.markdown("### 📊 Information Criteria by Lag Order")
                ic_df = pd.DataFrame({
                    'Lag': range(max_lags + 1),
                    'AIC': lag_order_results.aic,
                    'BIC': lag_order_results.bic,
                    'FPE': lag_order_results.fpe,
                    'HQIC': lag_order_results.hqic
                })
                st.dataframe(ic_df.style.format(precision=2))
                
                st.markdown(f"**📌 Selected Order (AIC):** {lag_order_results.aic_min_order}")
                st.markdown(f"**📌 Selected Order (BIC):** {lag_order_results.bic_min_order}")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    var_lags = st.number_input("Select Lag Order for VAR Model", min_value=1, max_value=10, value=2)
    forecast_steps_var = st.slider("Forecast Steps", 5, 50, 20)
    
    if st.button("🚀 Fit VAR Model"):
        with st.spinner("Fitting VAR model..."):
            try:
                model = VAR(df[['y1', 'y2']])
                fitted_model = model.fit(var_lags)
                
                st.success("✅ VAR model fitted successfully!")
                
                # Model Summary
                with st.expander("📄 Model Summary", expanded=False):
                    st.text(fitted_model.summary())
                
                # Forecast
                st.subheader("🔮 VAR Forecast")
                
                forecast = fitted_model.forecast(df[['y1', 'y2']].values[-var_lags:], steps=forecast_steps_var)
                forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps_var+1, freq='D')[1:]
                
                fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                
                # y1 forecast
                axes[0].plot(df.index, df['y1'], label='Historical y1', linewidth=1.5)
                axes[0].plot(forecast_index, forecast[:, 0], 'r--', label='Forecast y1', linewidth=2)
                axes[0].set_ylabel('y1')
                axes[0].set_title('y1 Forecast', fontweight='bold')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                
                # y2 forecast
                axes[1].plot(df.index, df['y2'], label='Historical y2', linewidth=1.5, color='green')
                axes[1].plot(forecast_index, forecast[:, 1], 'orange', linestyle='--', label='Forecast y2', linewidth=2)
                axes[1].set_xlabel('Time')
                axes[1].set_ylabel('y2')
                axes[1].set_title('y2 Forecast', fontweight='bold')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Impulse Response Analysis
                st.subheader("📊 Impulse Response Functions")
                irf = fitted_model.irf(10)
                fig = irf.plot(figsize=(12, 8))
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ Error fitting model: {e}")

# ===============================================
# 🔹 VECM - VECTOR ERROR CORRECTION MODEL
# ===============================================
elif analysis_type == "VECM - Cointegration":
    st.header("🔗 VECM - Vector Error Correction Model")
    
    st.markdown("""
    **VECM** is used when time series are **cointegrated** - they share a long-run equilibrium relationship.
    - First, test for cointegration using Johansen test
    - If cointegrated, VECM captures both short-run dynamics and long-run equilibrium
    - VECM is a restricted VAR that includes error correction terms
    """)
    
    DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "var_data.csv"
    df = pd.read_csv(DATA_PATH)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.metric("📏 Observations", len(df))
    
    # Plot both series
    st.subheader("📉 Time Series Plot")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(df.index, df['y1'], linewidth=1.5, color='blue')
    axes[0].set_ylabel('y1')
    axes[0].set_title('Series y1', fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(df.index, df['y2'], linewidth=1.5, color='green')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('y2')
    axes[1].set_title('Series y2', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Step 1: Test for Cointegration
    st.subheader("📊 Step 1: Johansen Cointegration Test")
    
    st.markdown("""
    The **Johansen test** determines:
    1. Whether series are cointegrated
    2. How many cointegration relationships exist
    """)
    
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    
    det_order = st.selectbox("Deterministic Trend Order", 
                            [0, 1, -1], 
                            format_func=lambda x: {0: "No deterministic trend", 
                                                   1: "Constant term", 
                                                   -1: "Constant + linear trend"}[x],
                            index=1)
    
    k_ar_diff = st.slider("Lag order for VECM", 1, 5, 2)
    
    if st.button("🔍 Run Johansen Cointegration Test"):
        with st.spinner("Running Johansen test..."):
            try:
                # Run Johansen test
                johansen_result = coint_johansen(df[['y1', 'y2']], det_order=det_order, k_ar_diff=k_ar_diff)
                
                st.success("✅ Johansen test complete!")
                
                # Display test results
                st.markdown("### 📊 Trace Statistic Test")
                trace_df = pd.DataFrame({
                    'Rank': ['r = 0', 'r ≤ 1'],
                    'Trace Statistic': johansen_result.lr1,
                    '90% Critical': johansen_result.cvt[:, 0],
                    '95% Critical': johansen_result.cvt[:, 1],
                    '99% Critical': johansen_result.cvt[:, 2]
                })
                st.dataframe(trace_df.style.format(precision=4))
                
                st.markdown("### 📊 Maximum Eigenvalue Test")
                max_eig_df = pd.DataFrame({
                    'Rank': ['r = 0', 'r ≤ 1'],
                    'Max-Eigen Statistic': johansen_result.lr2,
                    '90% Critical': johansen_result.cvm[:, 0],
                    '95% Critical': johansen_result.cvm[:, 1],
                    '99% Critical': johansen_result.cvm[:, 2]
                })
                st.dataframe(max_eig_df.style.format(precision=4))
                
                # Interpretation
                st.subheader("📖 Interpretation")
                
                # Check trace statistic at 95% level
                if johansen_result.lr1[0] > johansen_result.cvt[0, 1]:
                    if johansen_result.lr1[1] > johansen_result.cvt[1, 1]:
                        st.success("✅ Evidence of 2 cointegration relationships (both series cointegrated)")
                        coint_rank = 2
                    else:
                        st.success("✅ Evidence of 1 cointegration relationship")
                        coint_rank = 1
                else:
                    st.warning("⚠️ No evidence of cointegration (consider using VAR instead of VECM)")
                    coint_rank = 0
                
                st.markdown("""
                - **Trace Statistic > Critical Value**: Reject null of r cointegration relationships
                - **r = 0**: No cointegration
                - **r = 1**: One cointegration relationship
                - **r = 2**: Series are cointegrated (for 2 variables)
                """)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Step 2: Fit VECM
    st.markdown("---")
    st.subheader("📊 Step 2: Fit VECM Model")
    
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    st.markdown("**VECM Parameters:**")
    col1, col2 = st.columns(2)
    
    with col1:
        coint_rank_input = st.number_input("Cointegration Rank (from Johansen test)", 
                                          min_value=0, max_value=2, value=1)
    with col2:
        vecm_lags = st.number_input("Number of Lags", min_value=1, max_value=10, value=2)
    
    forecast_steps_vecm = st.slider("Forecast Steps", 5, 50, 20, key='vecm_forecast')
    
    if st.button("🚀 Fit VECM Model"):
        if coint_rank_input == 0:
            st.warning("⚠️ Cointegration rank is 0. VECM is not appropriate - use VAR instead.")
        else:
            with st.spinner("Fitting VECM model..."):
                try:
                    # Fit VECM
                    vecm_model = VECM(df[['y1', 'y2']], 
                                     k_ar_diff=vecm_lags, 
                                     coint_rank=coint_rank_input,
                                     deterministic='ci')
                    vecm_result = vecm_model.fit()
                    
                    st.success("✅ VECM model fitted successfully!")
                    
                    # Model Summary
                    with st.expander("📄 Model Summary", expanded=False):
                        st.text(vecm_result.summary())
                    
                    # Error Correction Terms
                    st.subheader("🔄 Error Correction Terms")
                    st.markdown("""
                    **Alpha coefficients** show how quickly each variable adjusts to disequilibrium:
                    - Negative α: Variable adjusts back to equilibrium
                    - Significant α: Variable participates in error correction
                    """)
                    
                    alpha_df = pd.DataFrame(
                        vecm_result.alpha,
                        columns=[f'Coint Eq {i+1}' for i in range(coint_rank_input)],
                        index=['y1', 'y2']
                    )
                    st.dataframe(alpha_df.style.format(precision=4))
                    
                    # Beta coefficients (cointegration vectors)
                    st.subheader("📊 Cointegration Vectors (Beta)")
                    beta_df = pd.DataFrame(
                        vecm_result.beta,
                        columns=[f'Coint Eq {i+1}' for i in range(coint_rank_input)],
                        index=['y1', 'y2']
                    )
                    st.dataframe(beta_df.style.format(precision=4))
                    
                    # Forecast
                    st.subheader("🔮 VECM Forecast")
                    
                    forecast = vecm_result.predict(steps=forecast_steps_vecm)
                    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps_vecm+1, freq='D')[1:]
                    
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                    
                    # y1 forecast
                    axes[0].plot(df.index, df['y1'], label='Historical y1', linewidth=1.5)
                    axes[0].plot(forecast_index, forecast[:, 0], 'r--', label='Forecast y1', linewidth=2)
                    axes[0].set_ylabel('y1')
                    axes[0].set_title('y1 Forecast with VECM', fontweight='bold')
                    axes[0].legend()
                    axes[0].grid(alpha=0.3)
                    
                    # y2 forecast
                    axes[1].plot(df.index, df['y2'], label='Historical y2', linewidth=1.5, color='green')
                    axes[1].plot(forecast_index, forecast[:, 1], 'orange', linestyle='--', 
                               label='Forecast y2', linewidth=2)
                    axes[1].set_xlabel('Time')
                    axes[1].set_ylabel('y2')
                    axes[1].set_title('y2 Forecast with VECM', fontweight='bold')
                    axes[1].legend()
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Model Comparison
                    st.subheader("📊 VECM vs VAR Comparison")
                    st.markdown("""
                    **When to use VECM vs VAR:**
                    
                    | Aspect | VAR | VECM |
                    |--------|-----|------|
                    | **When to use** | Variables NOT cointegrated | Variables ARE cointegrated |
                    | **Focus** | Short-run dynamics only | Short-run + long-run equilibrium |
                    | **Efficiency** | Less efficient if cointegrated | More efficient with cointegration |
                    | **Interpretation** | Simpler | Includes error correction |
                    
                    **Key VECM Components:**
                    - **α (alpha)**: Speed of adjustment to equilibrium
                    - **β (beta)**: Long-run cointegration relationships
                    - **Γ (gamma)**: Short-run dynamics (in full summary)
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error fitting VECM: {e}")

# ===============================================
# 🔹 GRANGER CAUSALITY TEST
# ===============================================
elif analysis_type == "Granger Causality Test":
    st.header("🔗 Granger Causality Test")
    
    st.markdown("""
    **Granger Causality** tests whether one time series can predict another.  
    If X "Granger-causes" Y, past values of X help predict Y beyond what Y's own past can predict.
    """)
    
    DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "var_data.csv"
    df = pd.read_csv(DATA_PATH)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Plot both series
    st.subheader("📉 Time Series Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['y1'], label='y1', linewidth=1.5)
    ax.plot(df.index, df['y2'], label='y2', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Multivariate Time Series', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Test configuration
    st.subheader("⚙️ Test Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        cause_var = st.selectbox("Potential Cause (X)", ['y1', 'y2'])
    with col2:
        effect_var = st.selectbox("Effect Variable (Y)", ['y2', 'y1'])
    
    max_lag = st.slider("Maximum Lag to Test", 1, 10, 4)
    
    if cause_var == effect_var:
        st.warning("⚠️ Please select different variables for cause and effect.")
    else:
        if st.button("🔍 Run Granger Causality Test"):
            with st.spinner("Running Granger causality test..."):
                try:
                    # Prepare data
                    test_data = df[[effect_var, cause_var]]
                    
                    # Run test
                    test_result = grangercausalitytests(test_data, max_lag, verbose=False)
                    
                    st.success("✅ Granger causality test complete!")
                    
                    # Extract results
                    st.subheader("📊 Test Results")
                    st.markdown(f"**Hypothesis:** Does **{cause_var}** Granger-cause **{effect_var}**?")
                    
                    results_list = []
                    for lag in range(1, max_lag + 1):
                        ssr_ftest = test_result[lag][0]['ssr_ftest']
                        results_list.append({
                            'Lag': lag,
                            'F-statistic': ssr_ftest[0],
                            'p-value': ssr_ftest[1],
                            'Significant (α=0.05)': '✅ Yes' if ssr_ftest[1] < 0.05 else '❌ No'
                        })
                    
                    results_df = pd.DataFrame(results_list)
                    st.dataframe(results_df.style.format({'F-statistic': '{:.4f}', 'p-value': '{:.4f}'}))
                    
                    # Interpretation
                    st.subheader("📖 Interpretation")
                    significant_lags = results_df[results_df['p-value'] < 0.05]
                    
                    if len(significant_lags) > 0:
                        st.success(f"✅ **{cause_var}** Granger-causes **{effect_var}** at {len(significant_lags)} lag(s).")
                        st.markdown(f"Lags with significant causality: {significant_lags['Lag'].tolist()}")
                    else:
                        st.info(f"ℹ️ No evidence that **{cause_var}** Granger-causes **{effect_var}** at α=0.05.")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Advanced Time Series Module")
