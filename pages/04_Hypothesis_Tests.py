# streamlit_app/04_Hypothesis_Tests.py

import streamlit as st
import pandas as pd
from streamlit_app.utils import st_helpers as sth

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Hypothesis Testing – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🧪"
)

st.title("🧪 Hypothesis Testing Module")
st.markdown("""
Perform **Independent T-Tests** and **One-Way ANOVA**  
with effect sizes and exportable results to validate group differences in your data.
""")

# -----------------------------------------------
# 📥 Load Data with Dynamic Switching
# -----------------------------------------------
from pathlib import Path

# Allow user to select dataset from sidebar
st.sidebar.header("📂 Dataset Selection")
dataset_options = {
    "MANOVA": "manova_data.csv",
    "OLS": "ols_data.csv",
    "GLM Logistic": "glm_logistic.csv"
}
selected = st.sidebar.selectbox("Select Dataset", list(dataset_options.keys()))

# Load the selected dataset
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / dataset_options[selected]
df = pd.read_csv(DATA_PATH)

# Detect column types
numeric_cols = sth.get_numeric_columns(df)
categorical_cols = sth.get_categorical_columns(df)

# ⚙️ Additional Sidebar Configuration
st.sidebar.header("🔧 Data Configuration")
st.sidebar.markdown("Choose variables for hypothesis testing.")


# -----------------------------------------------
# 📊 Dataset Preview & Summary
# -----------------------------------------------
st.markdown("### 📋 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

with st.expander("📈 Summary Statistics (.describe())"):
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)


# -----------------------------------------------
# 🔹 Independent T-Test
# -----------------------------------------------
st.subheader("📊 Independent T-Test")

with st.expander("➕ Run T-Test", expanded=True):
    ttest_col = st.selectbox("📈 Select Numeric Column", numeric_cols)
    group_col_ttest = st.selectbox("👥 Select Grouping Column (exactly 2 groups)", categorical_cols)
    use_welch = st.checkbox("Use Welch's T-Test (for unequal variances)", value=False)

    if ttest_col and group_col_ttest:
        if df[group_col_ttest].nunique() == 2:
            t_stat, p_value = sth.run_ttest_ind(df, ttest_col, group_col_ttest, equal_var=not use_welch)
            cohen_d = sth.compute_cohens_d(df, ttest_col, group_col_ttest)

            sth.display_ttest_result(t_stat, p_value, use_welch)
            st.markdown(f"**Cohen's d (effect size):** `{cohen_d:.3f}`")

            sth.plot_group_distributions(df, ttest_col, group_col_ttest)

            # 📥 Export T-Test stats
            ttest_out = pd.DataFrame({
                "t_statistic": [t_stat],
                "p_value": [p_value],
                "cohens_d": [cohen_d],
                "welch_test_used": [use_welch]
            })
            st.download_button(
                label="📥 Download T-Test Result",
                data=ttest_out.to_csv(index=False).encode("utf-8"),
                file_name="ttest_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Please select a grouping column with exactly 2 unique groups.")

# -----------------------------------------------
# 🔸 One-Way ANOVA
# -----------------------------------------------
st.subheader("📈 One-Way ANOVA")

with st.expander("➕ Run ANOVA", expanded=True):
    response_col = st.selectbox("📈 Select Response Variable", numeric_cols, key="anova_response")
    group_col_anova = st.selectbox("👥 Select Grouping Column", categorical_cols, key="anova_group")

    if response_col and group_col_anova:
        model, anova_result = sth.run_anova(df, response_col, group_col_anova)
        eta_squared = sth.compute_eta_squared(model)

        sth.display_anova_table(anova_result)
        st.markdown(f"**Eta-squared (effect size):** `{eta_squared:.3f}`")

        sth.plot_anova_boxplot(df, response_col, group_col_anova)

        # 📥 Export ANOVA results
        anova_export = anova_result.copy()
        anova_export["eta_squared"] = eta_squared
        st.download_button(
            label="📥 Download ANOVA Results",
            data=anova_export.to_csv(index=False).encode("utf-8"),
            file_name="anova_results.csv",
            mime="text/csv"
        )


# -----------------------------------------------
# 🔹 Non-Parametric Tests
# -----------------------------------------------
st.subheader("📊 Non-Parametric Tests")

with st.expander("📉 Mann–Whitney U Test", expanded=False):
    u_col = st.selectbox("📈 Select Numeric Column", numeric_cols, key="u_col")
    group_col_u = st.selectbox("👥 Select Grouping Column (2 groups)", categorical_cols, key="u_group")

    if group_col_u and u_col:
        if df[group_col_u].nunique() == 2:
            u_stat, p_val, cliffs_d = sth.run_mannwhitney_u(df, u_col, group_col_u)
            sth.display_u_test_result(u_stat, p_val)
            st.markdown(f"**Cliff’s Delta (effect size):** `{cliffs_d:.3f}`")
        else:
            st.warning("Please select a column with exactly 2 unique groups.")

with st.expander("📊 Kruskal–Wallis H Test", expanded=False):
    kruskal_col = st.selectbox("📈 Select Numeric Column", numeric_cols, key="kruskal_col")
    group_col_k = st.selectbox("👥 Select Grouping Column", categorical_cols, key="kruskal_group")

    if kruskal_col and group_col_k:
        if df[group_col_k].nunique() >= 2:
            h_stat, p_val, eta2 = sth.run_kruskal(df, kruskal_col, group_col_k)
            sth.display_kruskal_result(h_stat, p_val)
            st.markdown(f"**Eta-squared (effect size):** `{eta2:.3f}`")

            # ✅ Safe export logic inside the same conditional block
            kruskal_export = sth.get_kruskal_result_df(h_stat, p_val, eta2)
            st.download_button(
                label="📥 Download Kruskal–Wallis Result",
                data=kruskal_export.to_csv(index=False).encode("utf-8"),
                file_name="kruskal_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Please select a grouping column with at least 2 unique groups.")


# -----------------------------------------------
# 🔸 Posthoc Analysis: Tukey’s HSD
# -----------------------------------------------
st.subheader("🔎 Posthoc Analysis (Tukey's HSD)")

with st.expander("🧪 Run Tukey HSD Test", expanded=False):
    tukey_col = st.selectbox("📈 Select Response Variable", numeric_cols, key="tukey_response")
    tukey_group = st.selectbox("👥 Select Grouping Column", categorical_cols, key="tukey_group")

    if tukey_col and tukey_group:
        tukey_df, tukey_fig = sth.run_tukey_test(df, tukey_col, tukey_group)
        st.dataframe(tukey_df)

        st.download_button(
            label="📥 Download Tukey HSD Results",
            data=tukey_df.to_csv(index=False).encode("utf-8"),
            file_name="tukey_results.csv",
            mime="text/csv"
        )

        sth.plot_tukey_summary(tukey_fig)


# -----------------------------------------------
# ✅ Footer
# -----------------------------------------------
st.markdown("---")
st.caption("📘 *StatsmodelsMasterPro • Hypothesis Testing Dashboard*")
