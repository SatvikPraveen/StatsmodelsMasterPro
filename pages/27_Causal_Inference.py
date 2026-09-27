# 27_Causal_Inference.py

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import causal  # noqa: E402

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Causal Inference – StatsmodelsMasterPro",
    layout="wide",
    page_icon="🎯",
)

st.title("🎯 Causal Inference from Observational Data")
st.markdown("""
Regression coefficients are only causal under strong assumptions. This page demonstrates **design-based
estimators** that make those assumptions explicit, each on simulated data with a **known true effect** so you can
judge how well each method recovers it:

| Method | Identifying assumption | Reference |
|---|---|---|
| Inverse-probability weighting (IPW) | Conditional ignorability given X | Rosenbaum & Rubin (1983); Hirano, Imbens & Ridder (2003) |
| Propensity-score matching | Conditional ignorability, common support | Austin (2011) |
| Difference-in-differences | Parallel trends | Card & Krueger (1994) |
| Instrumental variables (2SLS) | Relevance + exclusion restriction | Angrist & Pischke (2009); Staiger & Stock (1997) |
| Regression discontinuity | Continuity of potential outcomes at the cutoff | Imbens & Lemieux (2008) |
| E-value | Sensitivity to unmeasured confounding | VanderWeele & Ding (2017) |
""")

# -----------------------------------------------
# ⚙️ Sidebar
# -----------------------------------------------
st.sidebar.header("⚙️ Simulation settings")
seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1))
n = int(st.sidebar.slider("Sample size", 200, 5000, 1500, step=100))
alpha = float(st.sidebar.selectbox("Significance level α", [0.10, 0.05, 0.01], index=1))
rng = np.random.default_rng(seed)


def _download(df: pd.DataFrame, label: str, name: str):
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=name, mime="text/csv")


def _summary_row(res: dict, keys=("estimate", "se", "ci_low", "ci_high", "p_value")) -> pd.DataFrame:
    return pd.DataFrame([{k: res[k] for k in keys if k in res}])


tab_ipw, tab_match, tab_did, tab_iv, tab_rd, tab_ev = st.tabs(
    ["IPW", "Matching", "Difference-in-differences", "Instrumental variables", "Regression discontinuity", "E-value"]
)

# ===============================================
# 🔹 IPW
# ===============================================
with tab_ipw:
    st.header("🔹 Inverse-Probability Weighting")
    st.markdown("""
    **DGP:** two confounders `x1`, `x2` raise both the probability of treatment and the outcome, so the naive
    treated-minus-control difference is biased upward. **True ATE = 2.0.**

    Weights are stabilised and propensity scores trimmed to [0.01, 0.99]. Balance is assessed with standardised
    mean differences (|SMD| < 0.1 is the conventional target).
    """)
    true_ate = 2.0
    col1, col2 = st.columns(2)
    with col1:
        estimand = st.radio("Estimand", ["ATE", "ATT"], horizontal=True, key="ipw_estimand")
    with col2:
        n_boot = int(st.slider("Bootstrap replicates for SE (0 = analytic only)", 0, 200, 0, step=50, key="ipw_boot"))

    if st.button("🔍 Estimate with IPW", key="ipw_btn"):
        try:
            x1, x2 = rng.normal(size=n), rng.normal(size=n)
            p = 1 / (1 + np.exp(-(0.8 * x1 + 0.5 * x2)))
            t = rng.binomial(1, p)
            y = 1 + true_ate * t + 1.5 * x1 + 1.0 * x2 + rng.normal(size=n)
            df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

            with st.spinner("Fitting propensity model and weighting..."):
                res = causal.ipw(df, "y", "t", ["x1", "x2"], estimand=estimand, alpha=alpha, n_boot=n_boot, seed=seed)
            st.success("✅ IPW complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True effect", f"{true_ate:.2f}")
            c2.metric("Naive difference", f"{res['naive_difference']:.3f}", delta=f"{res['naive_difference'] - true_ate:+.3f} bias")
            c3.metric(f"IPW {estimand}", f"{res['estimate']:.3f}", delta=f"{res['estimate'] - true_ate:+.3f} bias")
            c4.metric("Effective sample size", f"{res['effective_sample_size']:.0f} / {n}")

            keys = ["estimate", "se", "ci_low", "ci_high", "p_value"] + (["boot_se", "boot_ci_low", "boot_ci_high"] if n_boot else [])
            st.dataframe(_summary_row(res, keys).style.format(precision=4), use_container_width=True)

            st.subheader("⚖️ Covariate balance (love plot)")
            bal = pd.DataFrame({"before": res["balance_before"]["smd"], "after": res["balance_after"]["smd"]})
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ypos = np.arange(len(bal))
            ax.scatter(bal["before"].abs(), ypos, label="before weighting", color="firebrick")
            ax.scatter(bal["after"].abs(), ypos, label="after weighting", color="seagreen")
            ax.axvline(0.1, linestyle="--", color="grey", label="|SMD| = 0.1")
            ax.set_yticks(ypos)
            ax.set_yticklabels(bal.index)
            ax.set_xlabel("|standardised mean difference|")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(bal.style.format(precision=3), use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist(res["pscore"][df["t"] == 1], bins=30, alpha=0.6, label="treated")
            ax.hist(res["pscore"][df["t"] == 0], bins=30, alpha=0.6, label="control")
            ax.set_xlabel("propensity score")
            ax.set_title("Common support")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            _download(bal, "📥 Download balance table (.csv)", "ipw_balance.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 MATCHING
# ===============================================
with tab_match:
    st.header("🔹 Propensity-Score Matching")
    st.markdown("""
    Same DGP as the IPW tab (**true effect = 2.0**). Each treated unit is matched to the nearest control on the
    logit propensity score within a caliper (in SDs of the logit score). The ATT is the mean within-pair difference.
    """)
    caliper = float(st.slider("Caliper (SD of logit propensity)", 0.05, 1.0, 0.2, step=0.05, key="caliper"))
    replace = st.checkbox("Match with replacement", value=False, key="replace")

    if st.button("🔍 Match and estimate ATT", key="match_btn"):
        try:
            x1, x2 = rng.normal(size=n), rng.normal(size=n)
            p = 1 / (1 + np.exp(-(0.8 * x1 + 0.5 * x2)))
            t = rng.binomial(1, p)
            y = 1 + 2.0 * t + 1.5 * x1 + 1.0 * x2 + rng.normal(size=n)
            df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

            ps = causal.estimate_propensity(df, "t", ["x1", "x2"])
            matched = causal.nearest_neighbor_match(df, "t", ps, caliper=caliper, replace=replace, seed=seed)
            att = causal.matching_att(matched, "y", "t", alpha=alpha)
            st.success("✅ Matching complete")
            c1, c2, c3 = st.columns(3)
            c1.metric("Matched pairs", f"{matched.attrs['n_matched']} / {matched.attrs['n_treated']} treated")
            c2.metric("Matched ATT", f"{att['estimate']:.3f}", delta=f"{att['estimate'] - 2.0:+.3f} bias")
            c3.metric("Naive difference", f"{df.loc[df.t == 1, 'y'].mean() - df.loc[df.t == 0, 'y'].mean():.3f}")
            st.dataframe(pd.DataFrame([att]).style.format(precision=4), use_container_width=True)

            before = causal.covariate_balance(df, "t", ["x1", "x2"])["smd"]
            after = causal.covariate_balance(matched, "t", ["x1", "x2"])["smd"]
            bal = pd.DataFrame({"before": before, "after matching": after})
            st.subheader("⚖️ Balance before vs after matching")
            st.dataframe(bal.style.format(precision=3), use_container_width=True)
            with st.expander("📄 Matched sample"):
                st.dataframe(matched.head(20), use_container_width=True)
            _download(matched, "📥 Download matched sample (.csv)", "matched_sample.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 DIFFERENCE-IN-DIFFERENCES
# ===============================================
with tab_did:
    st.header("🔹 Difference-in-Differences")
    st.markdown("""
    **DGP:** two groups observed in two periods. Both share a common time trend (+0.5), the treated group has a
    permanently higher level (+0.7), and treatment adds **3.0** in the post period. Under parallel trends the
    interaction `treated:post` recovers the effect; SEs are clustered by unit.
    """)
    n_units = int(st.slider("Units per group", 20, 500, 100, step=10, key="did_units"))
    if st.button("🔍 Estimate DiD", key="did_btn"):
        try:
            rows = []
            for u in range(2 * n_units):
                treated = int(u < n_units)
                fe = rng.normal()
                for post in (0, 1):
                    yv = 1 + fe + 0.5 * post + 0.7 * treated + 3.0 * treated * post + rng.normal(scale=0.5)
                    rows.append({"unit": u, "treated": treated, "post": post, "y": yv})
            df = pd.DataFrame(rows)
            res = causal.difference_in_differences(df, "y", "treated", "post", cluster="unit", alpha=alpha)
            st.success("✅ DiD complete")
            c1, c2, c3 = st.columns(3)
            c1.metric("True effect", "3.00")
            c2.metric("DiD estimate (regression)", f"{res['estimate']:.3f}")
            c3.metric("Manual (2×2 table)", f"{res['manual_did']:.3f}")
            st.dataframe(_summary_row(res).style.format(precision=4), use_container_width=True)

            means = res["group_period_means"]
            st.subheader("📊 Group means by period")
            st.dataframe(means.rename(index={0: "control", 1: "treated"}, columns={0: "pre", 1: "post"}).style.format(precision=3))
            fig, ax = plt.subplots(figsize=(7, 4))
            for g, label in [(0, "control"), (1, "treated")]:
                ax.plot([0, 1], means.loc[g].values, marker="o", label=label)
            counter = [means.loc[1, 0], means.loc[1, 0] + (means.loc[0, 1] - means.loc[0, 0])]
            ax.plot([0, 1], counter, linestyle="--", color="grey", label="treated counterfactual")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["pre", "post"])
            ax.set_ylabel("mean outcome")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("📄 Regression output"):
                st.text(res["result"].summary())
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 INSTRUMENTAL VARIABLES
# ===============================================
with tab_iv:
    st.header("🔹 Instrumental Variables (2SLS)")
    st.markdown("""
    **DGP:** an unobserved confounder `u` raises both `x` and `y`, so OLS is biased. The instrument `z` shifts `x`
    but affects `y` only through `x`. **True effect of x on y = 1.0.** Weaken the instrument to see the
    first-stage F fall below 10 and the 2SLS interval widen.
    """)
    strength = float(st.slider("Instrument strength (coefficient of z in first stage)", 0.05, 1.5, 0.8, step=0.05, key="iv_str"))
    if st.button("🔍 Estimate 2SLS", key="iv_btn"):
        try:
            u, z = rng.normal(size=n), rng.normal(size=n)
            x = strength * z + u + rng.normal(scale=0.5, size=n)
            y = 1 + 1.0 * x + 2 * u + rng.normal(size=n)
            df = pd.DataFrame({"x": x, "y": y, "z": z})
            res = causal.two_stage_least_squares(df, "y", "x", ["z"], alpha=alpha)
            st.success("✅ 2SLS complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True effect", "1.00")
            c2.metric("OLS (biased)", f"{res['ols_estimate']:.3f}")
            c3.metric("2SLS", f"{res['estimate']:.3f}")
            c4.metric("First-stage F", f"{res['first_stage_f']:.1f}")
            if res["weak_instruments"]:
                st.warning("⚠️ First-stage F < 10: weak instrument. 2SLS is biased toward OLS and its SE is unreliable.")
            else:
                st.info("First-stage F ≥ 10: instrument relevance is adequate (Staiger & Stock rule of thumb).")
            st.dataframe(_summary_row(res, ("estimate", "se", "ci_low", "ci_high", "p_value", "ols_estimate", "first_stage_f", "first_stage_r2")).style.format(precision=4), use_container_width=True)
            with st.expander("📄 First-stage regression"):
                st.text(res["first_stage"].summary())
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 REGRESSION DISCONTINUITY
# ===============================================
with tab_rd:
    st.header("🔹 Sharp Regression Discontinuity")
    st.markdown("""
    **DGP:** `y = 1 + 0.5·r + 2.0·1[r ≥ 0] + noise` with running variable `r ~ U(−1, 1)`. **True jump = 2.0.**
    A local linear regression with triangular kernel weights is fitted on each side of the cutoff. Always check
    sensitivity to the bandwidth.
    """)
    kernel = st.selectbox("Kernel", ["triangular", "uniform", "epanechnikov"], key="rd_kernel")
    poly = int(st.radio("Polynomial order", [1, 2], horizontal=True, key="rd_poly"))
    bw_choice = st.slider("Bandwidth (0 = rule-of-thumb default)", 0.0, 1.0, 0.0, step=0.05, key="rd_bw")
    if st.button("🔍 Estimate RD", key="rd_btn"):
        try:
            r = rng.uniform(-1, 1, n)
            y = 1 + 0.5 * r + 2.0 * (r >= 0) + rng.normal(scale=0.3, size=n)
            df = pd.DataFrame({"r": r, "y": y})
            res = causal.regression_discontinuity(df, "y", "r", cutoff=0.0, bandwidth=bw_choice or None, kernel=kernel, polynomial=poly, alpha=alpha)
            st.success("✅ RD complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True jump", "2.00")
            c2.metric("RD estimate", f"{res['estimate']:.3f}")
            c3.metric("Bandwidth", f"{res['bandwidth']:.3f}")
            c4.metric("n left / right", f"{res['n_left']} / {res['n_right']}")
            st.dataframe(_summary_row(res).style.format(precision=4), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(df["r"], df["y"], s=6, alpha=0.3, color="grey")
            h = res["bandwidth"]
            params = res["result"].params
            for side, (lo, hi) in [(0, (-h, 0)), (1, (0, h))]:
                grid = np.linspace(lo, hi, 50)
                fitted = params["const"] + params["D"] * side
                for p_ in range(1, poly + 1):
                    fitted = fitted + params[f"x{p_}"] * grid**p_ + params[f"D_x{p_}"] * side * grid**p_
                ax.plot(grid, fitted, color="red", linewidth=2)
            ax.axvline(0, linestyle="--", color="black")
            ax.set_xlabel("running variable r")
            ax.set_ylabel("y")
            ax.set_title("Local polynomial fit within the bandwidth")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("📉 Bandwidth sensitivity")
            grid_bw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
            sens = causal.rd_bandwidth_sensitivity(df, "y", "r", bandwidths=grid_bw, kernel=kernel, polynomial=poly, alpha=alpha)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.errorbar(sens["bandwidth"], sens["estimate"], yerr=[sens["estimate"] - sens["ci_low"], sens["ci_high"] - sens["estimate"]], fmt="o-", capsize=3)
            ax.axhline(2.0, color="green", linestyle="--", label="true jump")
            ax.set_xlabel("bandwidth")
            ax.set_ylabel("RD estimate")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(sens.style.format(precision=4), use_container_width=True)
            _download(sens, "📥 Download bandwidth sensitivity (.csv)", "rd_sensitivity.csv")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# 🔹 E-VALUE
# ===============================================
with tab_ev:
    st.header("🔹 E-value Sensitivity Analysis")
    st.markdown("""
    The **E-value** is the minimum strength of association (on the risk-ratio scale) that an unmeasured confounder
    would need to have with *both* treatment and outcome to explain away the observed effect. Odds ratios, hazard
    ratios, and standardised mean differences are converted to an approximate risk ratio first
    (for an SMD d, RR ≈ exp(0.91·d)).
    """)
    col1, col2 = st.columns(2)
    with col1:
        scale = st.selectbox("Effect scale", ["rr", "or", "hr", "d"], key="ev_scale")
        rare = st.checkbox("Outcome is rare (< 15%)", value=False, key="ev_rare")
    with col2:
        est = float(st.number_input("Point estimate", value=2.0 if scale != "d" else 0.5, key="ev_est"))
        lo = float(st.number_input("CI lower", value=1.5 if scale != "d" else 0.2, key="ev_lo"))
        hi = float(st.number_input("CI upper", value=2.7 if scale != "d" else 0.8, key="ev_hi"))
    if st.button("🔍 Compute E-value", key="ev_btn"):
        try:
            res = causal.e_value(est, lo, hi, scale=scale, rare=rare)
            c1, c2, c3 = st.columns(3)
            c1.metric("RR-equivalent", f"{res['rr_equivalent']:.3f}")
            c2.metric("E-value (estimate)", f"{res['e_value']:.3f}")
            c3.metric("E-value (CI bound)", f"{res.get('e_value_ci', float('nan')):.3f}")
            st.markdown(
                f"An unmeasured confounder associated with both treatment and outcome by a risk ratio of at least "
                f"**{res['e_value']:.2f}** each (above and beyond the measured covariates) could explain away the point "
                f"estimate; **{res.get('e_value_ci', float('nan')):.2f}** would be needed to move the confidence interval to include the null."
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
