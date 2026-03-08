# 07_Model_Diagnostics.py

import streamlit as st
from streamlit_app.utils import st_helpers as sth
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Model Diagnostics", layout="wide")
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "ols_diagnostics.csv"
df = pd.read_csv(DATA_PATH)

st.title("🧪 OLS Model Diagnostics – Residual Checks & Influence")

# Preview Dataset
with st.expander("🗂 Dataset Preview", expanded=False):
    st.dataframe(df.head())
    st.markdown("**📊 Summary Statistics**")
    st.dataframe(df.describe().T)


# === Model Config ===
target = st.selectbox("Select Response Variable", sth.get_numeric_columns(df))
features = st.multiselect("Select Predictor(s)", [col for col in sth.get_numeric_columns(df) if col != target])

if target and features:
    model = sth.run_ols_model(df, target, features)
    st.subheader("📄 Model Summary")
    sth.show_model_summary(model)

    st.subheader("📊 Multicollinearity – VIF")
    vif_df = sth.calculate_vif(df[features])
    st.dataframe(vif_df)

    if vif_df["VIF"].gt(10).any():
        st.warning("⚠️ High multicollinearity detected (VIF > 10). Consider removing/reducing predictors.")


    st.subheader("📉 Residual Plot")
    sth.plot_residuals(model)

    st.subheader("🟣 Q–Q Plot")
    sth.plot_qq(model)

    st.subheader("🟠 Leverage vs Residual²")
    sth.plot_leverage(model)

    st.subheader("🔴 Cook’s Distance")
    sth.show_cooks_distance(model)
    st.subheader("🧪 Advanced Diagnostic Tests")
    
    # Import additional tests
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.diagnostic import linear_rainbow, linear_reset
    import scipy.stats as stats
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Durbin-Watson Test
        st.markdown("**🔵 Durbin-Watson Test (Autocorrelation)**")
        dw_stat = durbin_watson(model.resid)
        st.metric("DW Statistic", f"{dw_stat:.4f}")
        
        if 1.5 < dw_stat < 2.5:
            st.success("✅ No significant autocorrelation (DW ≈ 2)")
        elif dw_stat < 1.5:
            st.warning("⚠️ Positive autocorrelation detected (DW < 1.5)")
        else:
            st.warning("⚠️ Negative autocorrelation detected (DW > 2.5)")
        
        st.caption("DW ≈ 2: No autocorrelation | DW < 2: Positive | DW > 2: Negative")
        
        # Jarque-Bera Test
        st.markdown("**🟢 Jarque-Bera Test (Normality)**")
        jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(model.resid)
        
        col_jb1, col_jb2 = st.columns(2)
        col_jb1.metric("JB Statistic", f"{jb_stat:.4f}")
        col_jb2.metric("p-value", f"{jb_pval:.4f}")
        
        if jb_pval > 0.05:
            st.success("✅ Residuals appear normally distributed (p > 0.05)")
        else:
            st.warning("⚠️ Residuals deviate from normality (p < 0.05)")
        
        st.caption(f"Skewness: {jb_skew:.4f} | Kurtosis: {jb_kurt:.4f}")
    
    with col2:
        # Rainbow Test
        st.markdown("**🟣 Rainbow Test (Linearity)**")
        try:
            rainbow_stat, rainbow_pval = linear_rainbow(model)
            
            col_rb1, col_rb2 = st.columns(2)
            col_rb1.metric("F-statistic", f"{rainbow_stat:.4f}")
            col_rb2.metric("p-value", f"{rainbow_pval:.4f}")
            
            if rainbow_pval > 0.05:
                st.success("✅ Linear specification appears adequate (p > 0.05)")
            else:
                st.warning("⚠️ Possible nonlinearity detected (p < 0.05)")
            
            st.caption("Tests whether relationship is truly linear")
        except Exception as e:
            st.info(f"Rainbow test unavailable: {str(e)[:50]}")
        
        # RESET Test
        st.markdown("**🟠 RESET Test (Specification)**")
        try:
            reset_result = linear_reset(model, power=2)
            reset_stat = reset_result.fvalue
            reset_pval = reset_result.pvalue
            
            col_rs1, col_rs2 = st.columns(2)
            col_rs1.metric("F-statistic", f"{reset_stat:.4f}")
            col_rs2.metric("p-value", f"{reset_pval:.4f}")
            
            if reset_pval > 0.05:
                st.success("✅ Model specification appears correct (p > 0.05)")
            else:
                st.warning("⚠️ Misspecification detected - consider polynomials/interactions (p < 0.05)")
            
            st.caption("Ramsey RESET test for functional form")
        except Exception as e:
            st.info(f"RESET test unavailable: {str(e)[:50]}")
    st.subheader("🚨 Influential Observations")
    influence_df = sth.get_influential_points_df(model, df)
    st.dataframe(influence_df)

    st.download_button(
        label="📥 Download Influential Points CSV",
        data=influence_df.to_csv(index=False).encode("utf-8"),
        file_name="influential_points.csv",
        mime="text/csv"
    )


    st.subheader("📤 Export Predictions & Residuals")
    export_df = df.copy()
    export_df["Predicted"] = model.fittedvalues
    export_df["Residuals"] = model.resid


    st.download_button(
        label="📥 Download Predictions CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions_with_residuals.csv",
        mime="text/csv"
    )

else:
    st.warning("Please select both response and at least one predictor variable.")

# === Footer ===
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Model Diagnostics Module")
