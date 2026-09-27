# 29_Publication_Reporting.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import diagnostics, model_selection, reporting  # noqa: E402

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Publication Reporting – StatsmodelsMasterPro",
    layout="wide",
    page_icon="📝",
)

st.title("📝 Publication-Ready Reporting & Model Selection")
st.markdown("""
Turn fitted models into outputs you can drop into a manuscript:

- **Regression comparison tables** in the style of `stargazer` / `modelsummary` (DataFrame, LaTeX, Markdown, HTML)
- **Tidy** coefficient tables and **glance** model summaries (after R's `broom`)
- **Information criteria** with Δ values and **Akaike weights** (Burnham & Anderson, 2002)
- **k-fold cross-validation** for honest out-of-sample error
- **Coefficient forest plots** and **APA 7** reporting strings
- A full **diagnostic battery** for the selected model
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = PROJECT_ROOT / "synthetic_data" / "ols_diagnostics.csv"
df = pd.read_csv(DATA_PATH)
PREDICTORS = [f"X{i}" for i in range(1, 11)]

st.subheader("📊 Dataset Preview: `ols_diagnostics.csv`")
st.caption("DGP: y = 5 + 1.5·X1 − 2·X2 + 0.3·X3 + e; X6 ≈ 0.5·X1 and X7 ≈ −0.4·X2 (multicollinearity); X4, X5, X8–X10 are noise; 10 outliers injected.")
st.dataframe(df.head(8), use_container_width=True)

# -----------------------------------------------
# ⚙️ Sidebar
# -----------------------------------------------
st.sidebar.header("⚙️ Model specification")
m1 = st.sidebar.multiselect("Model 1 predictors", PREDICTORS, default=["X1"])
m2 = st.sidebar.multiselect("Model 2 predictors", PREDICTORS, default=["X1", "X2"])
m3 = st.sidebar.multiselect("Model 3 predictors", PREDICTORS, default=["X1", "X2", "X3"])
cov_type = st.sidebar.selectbox("Covariance estimator", ["nonrobust", "HC1", "HC3"])
k_folds = int(st.sidebar.slider("Cross-validation folds", 3, 10, 5))
seed = int(st.sidebar.number_input("CV seed", min_value=0, max_value=10_000, value=42))
stars = st.sidebar.checkbox("Significance stars in table", value=True)
detail_choice = st.sidebar.selectbox("Model for detailed output", ["Model 1", "Model 2", "Model 3"], index=2)

specs = [("Model 1", m1), ("Model 2", m2), ("Model 3", m3)]
specs = [(name, preds) for name, preds in specs if preds]


def _formula(preds):
    return "y ~ " + " + ".join(preds)


if not specs:
    st.warning("Select at least one predictor for at least one model.")
elif st.button("🔍 Fit models and build report"):
    try:
        models, names = [], []
        for name, preds in specs:
            models.append(smf.ols(_formula(preds), data=df).fit(cov_type=cov_type))
            names.append(name)
        st.success(f"✅ Fitted {len(models)} model(s) with {cov_type} covariance")

        # ---------------- Regression table ----------------
        st.header("🔹 Regression comparison table")
        table_df = reporting.regression_table(models, names, stars=stars)
        st.dataframe(table_df, use_container_width=True)
        latex = reporting.regression_table(models, names, stars=stars, to="latex")
        md = reporting.regression_table(models, names, stars=stars, to="markdown")
        html = reporting.regression_table(models, names, stars=stars, to="html")
        t_latex, t_md, t_html = st.tabs(["LaTeX", "Markdown", "HTML"])
        with t_latex:
            st.code(latex, language="latex")
        with t_md:
            st.code(md, language="markdown")
        with t_html:
            st.code(html, language="html")
        c1, c2, c3 = st.columns(3)
        c1.download_button("📥 Table (.csv)", table_df.to_csv().encode("utf-8"), "regression_table.csv", "text/csv")
        c2.download_button("📥 Table (.tex)", latex.encode("utf-8"), "regression_table.tex", "text/plain")
        c3.download_button("📥 Table (.md)", md.encode("utf-8"), "regression_table.md", "text/markdown")

        # ---------------- Tidy / glance ----------------
        st.header("🔹 Tidy coefficients and model glance")
        sel = detail_choice if detail_choice in names else names[-1]
        idx = names.index(sel)
        model = models[idx]
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**`tidy()` — {sel}**")
            st.dataframe(reporting.tidy(model).style.format(precision=4), use_container_width=True)
        with col2:
            st.markdown(f"**`glance()` — {sel}**")
            st.dataframe(reporting.glance(model).T.rename(columns={0: "value"}).astype(str), use_container_width=True)

        st.markdown("**APA 7 reporting strings**")
        for term in model.params.index:
            st.markdown(f"- `{term}`: {reporting.apa_coefficient(model, term)}")

        # ---------------- Information criteria ----------------
        st.header("🔹 Information criteria and Akaike weights")
        ic = model_selection.information_criteria_table(dict(zip(names, models)))
        st.dataframe(ic.style.format(precision=3), use_container_width=True)
        best = ic.index[0]
        st.markdown(f"Model **{best}** has the lowest AIC with Akaike weight `{ic.loc[best, 'weight_AIC']:.3f}`; "
                    f"an evidence ratio of `{ic['evidence_ratio_AIC'].max():.1f}` separates it from the weakest model.")

        # ---------------- Cross-validation ----------------
        st.header("🔹 k-fold cross-validation")
        cv_rows = []
        for name, preds in specs:
            cv = model_selection.cross_validate(_formula(preds), df, k=k_folds, seed=seed, fit_kwargs={"cov_type": cov_type})
            cv_rows.append({"model": name, **{f"{k}_mean": v for k, v in cv.attrs["mean"].items()}, **{f"{k}_sd": v for k, v in cv.attrs["sd"].items()}})
        cv_table = pd.DataFrame(cv_rows).set_index("model")
        st.dataframe(cv_table.style.format(precision=4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(cv_table.index, cv_table["rmse_mean"], yerr=cv_table["rmse_sd"], capsize=4, color="steelblue")
        ax.set_ylabel(f"out-of-sample RMSE ({k_folds}-fold)")
        st.pyplot(fig)
        plt.close(fig)

        # ---------------- Coefficient plot ----------------
        st.header("🔹 Coefficient forest plot")
        ax = reporting.coefficient_plot(models, names=names)
        st.pyplot(ax.figure)
        plt.close(ax.figure)

        # ---------------- Diagnostic battery ----------------
        st.header(f"🔹 Diagnostic battery — {sel}")
        battery = diagnostics.diagnostic_battery(model)

        def _highlight(row):
            return ["background-color: #ffd6d6" if row["flag"] else "" for _ in row]

        st.dataframe(battery.style.apply(_highlight, axis=1).format({"statistic": "{:.4f}", "p_value": "{:.4f}"}), use_container_width=True)
        n_flags = int(battery["flag"].sum())
        if n_flags:
            st.warning(f"⚠️ {n_flags} assumption check(s) flagged: " + ", ".join(battery.loc[battery["flag"], "test"]))
        else:
            st.info("No assumption checks flagged.")
        with st.expander("📄 Influence summary (flagged observations)"):
            infl = diagnostics.influence_summary(model)
            st.dataframe(infl[infl["any_flag"]].style.format(precision=4), use_container_width=True)
            st.caption(f"Thresholds: {infl.attrs['thresholds']}")
        st.download_button("📥 Diagnostic battery (.csv)", battery.to_csv(index=False).encode("utf-8"), "diagnostic_battery.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
