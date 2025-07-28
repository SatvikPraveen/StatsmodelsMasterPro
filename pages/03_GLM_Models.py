# streamlit_app/03_GLM_Models.py

import streamlit as st
from streamlit_app.utils import st_helpers as sth

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="GLM Models – StatsmodelsMasterPro",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Generalized Linear Models (GLM)")
st.markdown("""
Use **Generalized Linear Models** with different family types (Poisson, Binomial, Gaussian, Gamma)  
to model and visualize relationships in your data.
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
df = sth.load_default_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

numeric_cols = sth.get_numeric_columns(df)
cat_cols = sth.get_categorical_columns(df)

# -----------------------------------------------
# 🧮 Model Configuration
# -----------------------------------------------
st.sidebar.header("🔧 GLM Configuration")

response = st.sidebar.selectbox("🎯 Select Response Variable", numeric_cols)
predictors = st.sidebar.multiselect(
    "📌 Select Predictor(s)",
    [col for col in numeric_cols if col != response]
)

family_name = st.sidebar.selectbox("⚙️ Select GLM Family", ["Poisson", "Binomial", "Gaussian", "Gamma"])

# -----------------------------------------------
# ✅ Run Model if Config is Ready
# -----------------------------------------------
if response and predictors:

    with st.expander("📜 Model Formula", expanded=True):
        formula_str = f"{response} ~ {' + '.join(predictors)}"
        st.code(formula_str, language='markdown')

    # Get family object and fit model
    family = sth.get_glm_family(family_name)
    model = sth.run_glm(df, response, predictors, family)

    # -----------------------------------------------
    # 📄 Summary and Coefficients
    # -----------------------------------------------
    st.subheader("📑 Model Summary")
    sth.show_model_summary(model)

    st.subheader("📊 Coefficients Table")
    sth.display_glm_coefficients(model)
    sth.export_coefficients(model)  # Optional CSV download

    # -----------------------------------------------
    # 📏 AIC/BIC Information
    # -----------------------------------------------
    st.subheader("🧮 Model Fit Metrics")
    aic, bic = model.aic, model.bic
    st.markdown(f"- **AIC**: `{aic:.2f}`")
    st.markdown(f"- **BIC**: `{bic:.2f}`")

    # -----------------------------------------------
    # 🔎 Residual Diagnostics
    # -----------------------------------------------
    st.subheader("🔍 Residual Diagnostics")
    sth.plot_residuals(model)

    # -----------------------------------------------
    # 📉 Actual vs Predicted Plot
    # -----------------------------------------------
    st.subheader("🎯 Actual vs Predicted")
    sth.plot_glm_predictions(model, df, response)

    # ================================
    # 🧪 Prediction Debugging
    # ================================
    with st.expander("🔍 Prediction Debugging"):
        pred_values = model.fittedvalues
        st.write("Prediction Value Counts:")
        st.write(pred_values.value_counts().head(10))

        if pred_values.nunique() <= 2:
            st.error("🚨 Your model is likely underfitting or misconfigured. Try a different GLM family or check your feature selection.")


    # -----------------------------------------------
    # 💾 Export Predictions
    # -----------------------------------------------
    st.subheader("💾 Export Predictions")
    pred_df = sth.get_predictions_df(model, df, response)
    st.dataframe(pred_df.head(), use_container_width=True)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="glm_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please select a response variable and at least one predictor to run the GLM model.")
