# 26_Robust_Inference.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import effect_sizes as es  # noqa: E402
from utils import inference  # noqa: E402

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Robust & Resampling Inference – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🛡️",
)

st.title("🛡️ Robust & Resampling Inference")
st.markdown("""
Default `statsmodels` standard errors assume **homoskedastic, independent** errors. This page shows what
to do when that assumption is doubtful:

- **Heteroskedasticity-consistent (HC) SEs** — White (1980), MacKinnon & White (1985)
- **HAC / Newey–West** and **cluster-robust** SEs for dependent errors
- **Bootstrap confidence intervals** — percentile, basic, normal, and BCa (Efron & Tibshirani, 1993)
- **Bootstrap for regression** — pairs, residual, and wild (Freedman, 1981; Wu, 1986)
- **Permutation tests** — exact-in-the-limit randomisation inference (Fisher, 1935)
- **Multiple-testing corrections** — Holm (1979), Benjamini–Hochberg (1995)

All resampling procedures are seeded, so results are exactly reproducible.
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_DIR = PROJECT_ROOT / "synthetic_data"
hetero = pd.read_csv(DATA_DIR / "heteroskedastic_data.csv")
posthoc = pd.read_csv(DATA_DIR / "posthoc_dataset.csv")

st.subheader("📊 Dataset Preview: `heteroskedastic_data.csv`")
st.caption("DGP: y = 3 + 2·X + e,  e ~ N(0, (1 + 2|X|)²) — variance grows with |X|.")
st.dataframe(hetero.head(8), use_container_width=True)

# -----------------------------------------------
# ⚙️ Sidebar
# -----------------------------------------------
st.sidebar.header("⚙️ Settings")
seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1))
n_boot = int(st.sidebar.slider("Bootstrap replicates", 200, 5000, 1000, step=100))
alpha = float(st.sidebar.selectbox("Significance level α", [0.10, 0.05, 0.01], index=1))

section = st.sidebar.radio(
    "Section",
    [
        "Robust standard errors",
        "Bootstrap CI for a statistic",
        "Bootstrap regression",
        "Permutation test & effect sizes",
        "Multiple testing",
    ],
)


def _download(df: pd.DataFrame, label: str, name: str):
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=name, mime="text/csv")


# ===============================================
# 🔹 ROBUST STANDARD ERRORS
# ===============================================
if section == "Robust standard errors":
    st.header("🔹 Robust Standard Errors")
    st.markdown("""
    The same OLS coefficients are reported under several covariance estimators. Under heteroskedasticity
    the **nonrobust** SE is biased; HC0–HC3 correct it (HC3 is recommended for small samples).
    HAC (Newey–West) additionally handles autocorrelation; cluster-robust SEs handle within-group correlation.
    Here, synthetic cluster ids are assigned in blocks of 10 rows for illustration.
    """)
    cov_types = st.multiselect(
        "Covariance estimators",
        ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster"],
        default=["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster"],
    )
    maxlags = int(st.number_input("HAC max lags", min_value=1, max_value=20, value=4))

    if st.button("🔍 Compare standard errors"):
        try:
            model = smf.ols("y ~ X", data=hetero).fit()
            groups = np.repeat(np.arange(len(hetero) // 10 + 1), 10)[: len(hetero)]
            table = inference.robust_se_table(model, cov_types=cov_types, groups=groups, maxlags=maxlags, alpha=alpha)
            st.success("✅ Comparison complete")

            wide = table.pivot(index="term", columns="cov_type", values=["se", "p_value"])
            st.subheader("📋 SE and p-value by estimator")
            st.dataframe(wide.style.format("{:.4f}"), use_container_width=True)

            se_x = table[table["term"] == "X"].set_index("cov_type").loc[cov_types, "se"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(se_x.index, se_x.values, color=["grey" if c == "nonrobust" else "steelblue" for c in se_x.index])
            ax.set_ylabel("SE of slope on X")
            ax.set_title("Standard error of the slope under each covariance estimator")
            st.pyplot(fig)
            plt.close(fig)

            ratio = se_x / se_x.get("nonrobust", se_x.iloc[0])
            st.markdown(f"**HC3 / nonrobust SE ratio:** `{ratio.get('HC3', np.nan):.2f}` — values far from 1 indicate the "
                        "nonrobust SE is unreliable.")
            with st.expander("📄 Long-format table"):
                st.dataframe(table, use_container_width=True)
            _download(table, "📥 Download robust SE table (.csv)", "robust_se_table.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 BOOTSTRAP CI
# ===============================================
elif section == "Bootstrap CI for a statistic":
    st.header("🔹 Bootstrap Confidence Interval")
    st.markdown("""
    Resample the data with replacement `n_boot` times, recompute the statistic, and form an interval from the
    bootstrap distribution. **BCa** corrects for bias and skewness (DiCiccio & Efron, 1996) and is the
    recommended default for non-symmetric statistics.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset = st.selectbox("Dataset", ["heteroskedastic_data", "posthoc_dataset"])
    df_sel = hetero if dataset == "heteroskedastic_data" else posthoc
    numeric = df_sel.select_dtypes(include=np.number).columns.tolist()
    with col2:
        column = st.selectbox("Column", numeric)
    with col3:
        stat_name = st.selectbox("Statistic", ["mean", "median", "trimmed mean (10%)", "std"])
    method = st.radio("Method", ["percentile", "basic", "normal", "bca"], horizontal=True, index=3)

    stat_fns = {
        "mean": np.mean,
        "median": np.median,
        "trimmed mean (10%)": lambda a: float(np.mean(np.sort(a)[int(0.1 * len(a)): len(a) - int(0.1 * len(a))])),
        "std": lambda a: float(np.std(a, ddof=1)),
    }

    if st.button("🔍 Run bootstrap"):
        try:
            data = df_sel[column].to_numpy(dtype=float)
            res = inference.bootstrap_ci(data, stat_fns[stat_name], n_boot=n_boot, alpha=alpha, method=method, seed=seed)
            st.success("✅ Bootstrap complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estimate", f"{res.estimate:.4f}")
            c2.metric(f"{100 * (1 - alpha):.0f}% CI lower", f"{res.ci_low:.4f}")
            c3.metric(f"{100 * (1 - alpha):.0f}% CI upper", f"{res.ci_high:.4f}")
            c4.metric("Bootstrap SE", f"{res.se:.4f}")
            st.caption(f"Bootstrap bias estimate: {res.bias:+.4f}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(res.distribution, bins=40, color="lightsteelblue", edgecolor="white")
            ax.axvline(res.estimate, color="black", label="estimate")
            ax.axvline(res.ci_low, color="red", linestyle="--", label="CI bounds")
            ax.axvline(res.ci_high, color="red", linestyle="--")
            ax.set_title(f"Bootstrap distribution of the {stat_name} of {column} ({method})")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            all_methods = pd.DataFrame(
                [inference.bootstrap_ci(data, stat_fns[stat_name], n_boot=n_boot, alpha=alpha, method=m, seed=seed).to_dict()
                 for m in ["percentile", "basic", "normal", "bca"]]
            ).set_index("method")
            st.subheader("📋 All four methods side by side")
            st.dataframe(all_methods.style.format(precision=4), use_container_width=True)
            _download(all_methods, "📥 Download bootstrap comparison (.csv)", "bootstrap_ci_methods.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 BOOTSTRAP REGRESSION
# ===============================================
elif section == "Bootstrap regression":
    st.header("🔹 Bootstrap for Regression Coefficients")
    st.markdown("""
    - **Pairs** bootstrap resamples rows: robust to heteroskedasticity and misspecified errors.
    - **Residual** bootstrap resamples residuals: assumes i.i.d. errors and a fixed design.
    - **Wild** bootstrap multiplies leverage-adjusted residuals by Rademacher signs: keeps the design fixed
      while remaining robust to heteroskedasticity (Davidson & Flachaire, 2008).
    """)
    method = st.radio("Bootstrap scheme", ["pairs", "residual", "wild"], horizontal=True)
    if st.button("🔍 Run bootstrap regression"):
        try:
            table, draws = inference.bootstrap_regression(
                "y ~ X", hetero, n_boot=n_boot, method=method, alpha=alpha, seed=seed, return_draws=True
            )
            hc3 = smf.ols("y ~ X", data=hetero).fit(cov_type="HC3").bse
            table["HC3_se"] = hc3.reindex(table.index).to_numpy()
            st.success(f"✅ {method} bootstrap complete ({n_boot} replicates)")
            st.dataframe(table.style.format(precision=4), use_container_width=True)
            st.caption("True DGP values: Intercept = 3, X = 2. Compare the bootstrap SE with the asymptotic (nonrobust) and HC3 SEs.")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, (i, name) in zip(axes, enumerate(table.index)):
                ax.hist(draws[:, i], bins=40, color="lightsteelblue", edgecolor="white")
                ax.axvline(table.loc[name, "ci_low"], color="red", linestyle="--")
                ax.axvline(table.loc[name, "ci_high"], color="red", linestyle="--")
                ax.axvline({"Intercept": 3.0, "X": 2.0}.get(name, np.nan), color="green", label="true value")
                ax.set_title(name)
                ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            _download(table, "📥 Download bootstrap regression table (.csv)", f"bootstrap_regression_{method}.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 PERMUTATION TEST
# ===============================================
elif section == "Permutation test & effect sizes":
    st.header("🔹 Permutation Test & Effect Sizes")
    st.markdown("""
    Under the sharp null of exchangeability, group labels are shuffled and the statistic recomputed; the p-value is
    the fraction of shuffles at least as extreme as the observed statistic (with the add-one correction of
    Phipson & Smyth, 2010). Effect sizes report **practical** significance (Cohen, 1988; Lakens, 2013).
    """)
    st.caption("Dataset: `posthoc_dataset.csv` — Group A ~ N(60, 10), B ~ N(70, 12), C ~ N(65, 8).")
    groups = sorted(posthoc["Group"].unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        g1 = st.selectbox("Group 1", groups, index=0)
    with col2:
        g2 = st.selectbox("Group 2", groups, index=1)
    with col3:
        statistic = st.selectbox("Statistic", ["mean_diff", "median_diff", "t"])
    n_perm = int(st.slider("Permutations", 500, 10000, 2000, step=500))
    alternative = st.radio("Alternative", ["two-sided", "greater", "less"], horizontal=True)

    if st.button("🔍 Run permutation test"):
        if g1 == g2:
            st.warning("Choose two different groups.")
        else:
            try:
                x = posthoc.loc[posthoc["Group"] == g1, "Score"].to_numpy()
                y = posthoc.loc[posthoc["Group"] == g2, "Score"].to_numpy()
                res = inference.permutation_test(x, y, statistic=statistic, n_perm=n_perm, alternative=alternative, seed=seed)
                st.success("✅ Permutation test complete")
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Observed {statistic}", f"{res.observed:.4f}")
                c2.metric("Permutation p-value", f"{res.p_value:.4f}")
                c3.metric("Permutations", f"{res.n_perm}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(res.null_distribution, bins=50, color="lightsteelblue", edgecolor="white")
                ax.axvline(res.observed, color="red", label="observed")
                ax.set_title(f"Permutation null distribution ({g1} vs {g2})")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📏 Effect sizes")
                d, g, cd = es.cohens_d(x, y, alpha=alpha), es.hedges_g(x, y, alpha=alpha), es.cliffs_delta(x, y)
                eff = pd.DataFrame(
                    [
                        {"measure": "Cohen's d", "value": d["d"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "interpretation": d["interpretation"]},
                        {"measure": "Hedges' g", "value": g["g"], "ci_low": g["ci_low"], "ci_high": g["ci_high"], "interpretation": g["interpretation"]},
                        {"measure": "Cliff's δ", "value": cd["delta"], "ci_low": np.nan, "ci_high": np.nan, "interpretation": cd["interpretation"]},
                        {"measure": "Common-language ES", "value": es.common_language_effect_size(x, y), "ci_low": np.nan, "ci_high": np.nan, "interpretation": "P(x > y)"},
                    ]
                ).set_index("measure")
                st.dataframe(eff.style.format(precision=3), use_container_width=True)
                _download(eff, "📥 Download effect sizes (.csv)", "effect_sizes.csv")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 MULTIPLE TESTING
# ===============================================
else:
    st.header("🔹 Multiple-Testing Corrections")
    st.markdown("""
    All pairwise comparisons between the three groups are tested with permutation tests; the raw p-values are then
    adjusted. **Holm** controls the family-wise error rate; **Benjamini–Hochberg** controls the false discovery rate
    and is less conservative.
    """)
    method = st.selectbox("Adjustment method", ["holm", "bonferroni", "sidak", "holm-sidak", "fdr_bh", "fdr_by"])
    n_perm = int(st.slider("Permutations per comparison", 500, 5000, 1000, step=500))
    if st.button("🔍 Run pairwise comparisons"):
        try:
            groups = sorted(posthoc["Group"].unique())
            pairs, pvals, obs = [], [], []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    x = posthoc.loc[posthoc["Group"] == groups[i], "Score"].to_numpy()
                    y = posthoc.loc[posthoc["Group"] == groups[j], "Score"].to_numpy()
                    r = inference.permutation_test(x, y, n_perm=n_perm, seed=seed)
                    pairs.append(f"{groups[i]} vs {groups[j]}")
                    pvals.append(r.p_value)
                    obs.append(r.observed)
            adj = inference.multiple_testing(pvals, method=method, alpha=alpha, labels=pairs)
            adj.insert(0, "mean_diff", obs)
            st.success("✅ Adjustment complete")
            st.dataframe(adj.style.format({"mean_diff": "{:.3f}", "p_raw": "{:.4f}", "p_adj": "{:.4f}"}), use_container_width=True)
            st.markdown(f"**Rejected at α = {alpha}:** {int(adj['reject'].sum())} of {len(adj)} comparisons.")
            _download(adj, "📥 Download adjusted p-values (.csv)", "multiple_testing.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")
