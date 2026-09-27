# 28_Monte_Carlo_Validation.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import power as pw  # noqa: E402
from utils import simulation as sim  # noqa: E402

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Monte Carlo Validation – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🎲",
)

st.title("🎲 Monte Carlo Validation of Statistical Procedures")
st.markdown("""
A simulation study answers: *does this estimator actually behave the way theory says?* Following the
**ADEMP** structure (Morris, White & Crowther, 2019) — Aims, Data-generating mechanism, Estimands, Methods,
Performance measures — this page repeatedly draws data from a known DGP, fits a model, and reports:

- **Bias** and **relative bias** of the point estimate
- **Empirical SE** (SD of estimates) versus the **mean model-based SE** (their ratio should be ≈ 1)
- **RMSE**, **coverage** of the 95% CI (should be ≈ 0.95), and **rejection rate**
- A **Monte Carlo standard error (MCSE)** for every performance measure, so simulation noise is quantified

Each replicate uses an independent child seed spawned from the master seed, so the study is exactly reproducible.
""")

# -----------------------------------------------
# ⚙️ Sidebar
# -----------------------------------------------
st.sidebar.header("⚙️ Simulation design")
n = int(st.sidebar.slider("Sample size per replicate (n)", 20, 500, 60, step=10))
n_reps = int(st.sidebar.slider("Replicates", 100, 2000, 500, step=100))
seed = int(st.sidebar.number_input("Master seed", min_value=0, max_value=10_000, value=2024, step=1))
error = st.sidebar.selectbox("Error distribution", ["normal", "t3", "skewed"])
hetero = st.sidebar.checkbox("Heteroskedastic errors (SD ∝ 1 + |X1|)", value=False)
cov_type = st.sidebar.selectbox("Covariance estimator", ["nonrobust", "HC1", "HC3"])
alpha = 0.05

TRUE = {"beta_Intercept": 2.0, "beta_X1": 1.5, "beta_X2": -0.7}
SIGMA = 1.5


def _download(df: pd.DataFrame, label: str, name: str):
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=name, mime="text/csv")


@st.cache_data(show_spinner=False)
def run_ols_study(n: int, n_reps: int, seed: int, error: str, hetero: bool, cov_type: str) -> pd.DataFrame:
    dgp = sim.dgp_linear(n=n, beta=(2.0, 1.5, -0.7), sigma=SIGMA, heteroskedastic=hetero, error=error)
    est = sim.ols_estimator("y ~ X1 + X2", cov_type=cov_type, alpha=alpha)
    return sim.monte_carlo(dgp, est, n_reps=n_reps, seed=seed)


@st.cache_data(show_spinner=False)
def run_power_study(n1: int, n2: int, effects: tuple, n_reps: int, seed: int, dist: str, equal_var: bool) -> pd.DataFrame:
    return sim.simulate_power(
        lambda d: sim.dgp_two_sample(n1, n2, d, sd=1.0, dist=dist),
        lambda data: float(stats.ttest_ind(*data, equal_var=equal_var).pvalue),
        effects=list(effects),
        n_reps=n_reps,
        alpha=alpha,
        seed=seed,
    )


# ===============================================
# 🔹 SECTION 1: OLS ESTIMATOR PERFORMANCE
# ===============================================
st.header("🔹 1. Performance of OLS under the chosen DGP")
st.markdown(f"""
**DGP:** `y = 2 + 1.5·X1 − 0.7·X2 + e`, `X ~ N(0, 1)`, error scale σ = {SIGMA}
{"with SD scaled by (1 + |X1|)" if hetero else "(homoskedastic)"}, error family **{error}**.
**Method:** OLS with `{cov_type}` covariance. Try *heteroskedastic + nonrobust* to see coverage fall below 0.95,
then switch to HC3 to see it restored.
""")

if st.button("🔍 Run simulation study", key="mc_btn"):
    try:
        with st.spinner(f"Running {n_reps} replicates..."):
            results = run_ols_study(n, n_reps, seed, error, hetero, cov_type)
        perf = sim.performance_summary(results, TRUE, alpha=alpha)
        st.success(f"✅ {n_reps} replicates complete ({perf.attrs['n_failed']} failed fits)")

        st.subheader("📋 Performance measures with Monte Carlo SEs")
        show_cols = ["true", "mean_estimate", "bias", "bias_mcse", "relative_bias", "empirical_se", "mean_model_se", "se_ratio", "rmse", "coverage", "coverage_mcse", "rejection_rate"]
        st.dataframe(perf[show_cols].style.format(precision=4), use_container_width=True)

        cov_x1, mcse_x1 = perf.loc["beta_X1", "coverage"], perf.loc["beta_X1", "coverage_mcse"]
        if abs(cov_x1 - 0.95) > 2 * mcse_x1 + 0.005:
            st.warning(f"⚠️ Coverage for β(X1) = {cov_x1:.3f} ± {mcse_x1:.3f} differs from the nominal 0.95 by more than 2 MCSE — the interval is miscalibrated under this DGP.")
        else:
            st.info(f"Coverage for β(X1) = {cov_x1:.3f} ± {mcse_x1:.3f} is consistent with the nominal 0.95.")

        st.subheader("📊 Sampling distributions")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, name in zip(axes, ["beta_X1", "beta_X2"]):
            vals = results[name].dropna()
            ax.hist(vals, bins=40, color="lightsteelblue", edgecolor="white", density=True)
            ax.axvline(TRUE[name], color="green", linewidth=2, label="true value")
            ax.axvline(vals.mean(), color="red", linestyle="--", label="mean estimate")
            ax.set_title(f"{name}: empirical SE = {vals.std(ddof=1):.3f}")
            ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("🎯 Coverage vs nominal")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names = ["beta_Intercept", "beta_X1", "beta_X2"]
        ax.bar(names, perf.loc[names, "coverage"], yerr=1.96 * perf.loc[names, "coverage_mcse"], capsize=4, color="steelblue")
        ax.axhline(0.95, color="red", linestyle="--", label="nominal 0.95")
        ax.set_ylim(min(0.8, perf.loc[names, "coverage"].min() - 0.05), 1.0)
        ax.set_ylabel("95% CI coverage")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📄 Per-replicate results"):
            st.dataframe(results.head(50), use_container_width=True)
        _download(perf, "📥 Download performance summary (.csv)", "mc_performance_summary.csv")
        _download(results, "📥 Download per-replicate results (.csv)", "mc_replicates.csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 SECTION 2: EMPIRICAL VS ANALYTIC POWER
# ===============================================
st.markdown("---")
st.header("🔹 2. Empirical power curve vs analytic power")
st.markdown("""
The analytic power of the two-sample t-test (`utils.power.power_ttest`) assumes normal errors. The simulated curve
(`utils.simulation.simulate_power`) makes no such assumption — switch the distribution to `lognormal` or `t3` to
see the two diverge. The effect of 0 gives the **empirical type-I error rate**.
""")
col1, col2, col3 = st.columns(3)
with col1:
    n_per_group = int(st.slider("n per group", 10, 200, 30, step=5, key="pw_n"))
with col2:
    dist = st.selectbox("Sampling distribution", ["normal", "lognormal", "t3"], key="pw_dist")
with col3:
    equal_var = st.checkbox("Assume equal variances (Student t)", value=True, key="pw_eqv")
effects = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
pw_reps = int(st.slider("Replicates per effect size", 100, 1000, 300, step=100, key="pw_reps"))

if st.button("🔍 Run power study", key="pw_btn"):
    try:
        with st.spinner(f"Simulating {len(effects)} × {pw_reps} tests..."):
            emp = run_power_study(n_per_group, n_per_group, effects, pw_reps, seed, dist, equal_var)
        grid = np.linspace(0, 1.0, 41)
        analytic = [pw.power_ttest(effect_size=d, nobs=n_per_group, alpha=alpha)["power"] if d > 0 else alpha for d in grid]
        st.success("✅ Power study complete")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(grid, analytic, color="black", label="analytic (normal theory)")
        ax.errorbar(emp["effect"], emp["power"], yerr=1.96 * emp["mcse"], fmt="o", capsize=4, color="firebrick", label=f"simulated ({dist}) ± 1.96 MCSE")
        ax.axhline(alpha, color="grey", linestyle=":", label=f"α = {alpha}")
        ax.set_xlabel("standardised effect size d")
        ax.set_ylabel("power (rejection rate)")
        ax.set_ylim(0, 1.02)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        emp["analytic_power"] = [pw.power_ttest(effect_size=d, nobs=n_per_group, alpha=alpha)["power"] if d > 0 else alpha for d in emp["effect"]]
        st.dataframe(emp.set_index("effect").style.format(precision=4), use_container_width=True)
        t1 = emp.loc[emp["effect"] == 0.0].iloc[0]
        st.markdown(f"**Empirical type-I error:** `{t1['power']:.3f} ± {t1['mcse']:.3f}` (nominal {alpha}).")
        _download(emp, "📥 Download power curve (.csv)", "power_curve.csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
