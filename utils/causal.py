"""Causal inference from observational data.

Design-based estimators implemented with ``statsmodels`` primitives:

* Propensity scores, covariate balance (standardised mean differences),
  inverse-probability weighting (IPW) for the ATE/ATT with robust standard
  errors, and nearest-neighbour matching (Rosenbaum & Rubin, 1983;
  Hirano, Imbens & Ridder, 2003; Austin, 2011).
* Difference-in-differences with cluster-robust inference
  (Card & Krueger, 1994; Bertrand, Duflo & Mullainathan, 2004).
* Two-stage least squares with a first-stage weak-instrument F statistic
  (Staiger & Stock, 1997).
* Sharp regression discontinuity with local polynomial regression and kernel
  weights (Imbens & Lemieux, 2008; Lee & Lemieux, 2010).
* E-values for sensitivity to unmeasured confounding
  (VanderWeele & Ding, 2017).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.sandbox.regression.gmm import IV2SLS

__all__ = [
    "estimate_propensity",
    "covariate_balance",
    "ipw",
    "nearest_neighbor_match",
    "matching_att",
    "difference_in_differences",
    "two_stage_least_squares",
    "regression_discontinuity",
    "rd_bandwidth_sensitivity",
    "e_value",
]


# --------------------------------------------------------------------------- #
# Propensity scores and balance
# --------------------------------------------------------------------------- #
def estimate_propensity(df: pd.DataFrame, treatment: str, covariates: Sequence[str], method: str = "logit") -> pd.Series:
    """Estimate P(T = 1 | X) with a logit or probit model. Returns a Series aligned to ``df``."""
    formula = f"{treatment} ~ " + " + ".join(covariates)
    fit = smf.logit if method == "logit" else smf.probit
    res = fit(formula, data=df).fit(disp=0)
    ps = res.predict(df)
    ps.name = "pscore"
    return ps


def covariate_balance(df: pd.DataFrame, treatment: str, covariates: Sequence[str], weights: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Standardised mean differences (SMD) between treated and control units.

    ``|SMD| < 0.1`` is the conventional threshold for adequate balance
    (Austin, 2009). Weighted means and variances are used when ``weights``
    is supplied, which lets you compare balance before and after weighting.
    """
    t = df[treatment].astype(int).to_numpy()
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    rows = []
    for c in covariates:
        x = df[c].astype(float).to_numpy()
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1])
        v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
        smd = (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0
        rows.append({"covariate": c, "mean_treated": m1, "mean_control": m0, "smd": float(smd), "balanced": abs(smd) < 0.1})
    return pd.DataFrame(rows).set_index("covariate")


# --------------------------------------------------------------------------- #
# Inverse probability weighting
# --------------------------------------------------------------------------- #
def ipw(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: Sequence[str],
    estimand: str = "ATE",
    stabilized: bool = True,
    trim: tuple[float, float] | None = (0.01, 0.99),
    pscore: pd.Series | None = None,
    alpha: float = 0.05,
    n_boot: int = 0,
    seed: int | None = None,
) -> dict:
    """Inverse-probability-weighted estimate of the ATE or ATT.

    The point estimate is the coefficient on ``treatment`` in a weighted
    least-squares regression of ``outcome`` on ``treatment`` using the IPW
    weights, with HC1 robust standard errors (a Horvitz–Thompson estimator with
    a sandwich variance that ignores propensity-model uncertainty). Set
    ``n_boot > 0`` to obtain a bootstrap SE that re-estimates the propensity
    model in every replicate.
    """
    estimand = estimand.upper()
    if estimand not in {"ATE", "ATT"}:
        raise ValueError("estimand must be 'ATE' or 'ATT'.")
    data = df.reset_index(drop=True)
    ps = estimate_propensity(data, treatment, covariates) if pscore is None else pd.Series(np.asarray(pscore, float))
    if trim is not None:
        ps = ps.clip(*trim)
    t = data[treatment].astype(float)
    if estimand == "ATE":
        w = t / ps + (1 - t) / (1 - ps)
        if stabilized:
            p_t = t.mean()
            w = t * p_t / ps + (1 - t) * (1 - p_t) / (1 - ps)
    else:
        w = t + (1 - t) * ps / (1 - ps)

    X = sm.add_constant(t.to_numpy())
    res = sm.WLS(data[outcome].astype(float).to_numpy(), X, weights=w.to_numpy()).fit(cov_type="HC1")
    est, se = float(res.params[1]), float(res.bse[1])
    z = stats.norm.ppf(1 - alpha / 2)
    out = {
        "estimand": estimand,
        "estimate": est,
        "se": se,
        "ci_low": est - z * se,
        "ci_high": est + z * se,
        "p_value": float(res.pvalues[1]),
        "naive_difference": float(data.loc[t == 1, outcome].mean() - data.loc[t == 0, outcome].mean()),
        "effective_sample_size": float(w.sum() ** 2 / (w**2).sum()),
        "weights": w,
        "pscore": ps,
        "balance_before": covariate_balance(data, treatment, covariates),
        "balance_after": covariate_balance(data, treatment, covariates, weights=w),
    }
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            sample = data.iloc[rng.integers(0, len(data), len(data))].reset_index(drop=True)
            boots[b] = ipw(sample, outcome, treatment, covariates, estimand, stabilized, trim, None, alpha, 0)["estimate"]
        out["boot_se"] = float(boots.std(ddof=1))
        out["boot_ci_low"], out["boot_ci_high"] = (float(q) for q in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    return out


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def nearest_neighbor_match(
    df: pd.DataFrame,
    treatment: str,
    pscore: pd.Series | np.ndarray,
    caliper: float | None = 0.2,
    replace: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """1:1 nearest-neighbour matching on the (logit) propensity score.

    ``caliper`` is expressed in standard deviations of the logit propensity
    score (0.2 is the usual recommendation; Austin, 2011). Treated units are
    processed in random order. Returns the matched sample with a ``match_id``
    column shared by each treated/control pair.
    """
    data = df.reset_index(drop=True).copy()
    ps = np.clip(np.asarray(pscore, float), 1e-6, 1 - 1e-6)
    lp = np.log(ps / (1 - ps))
    data["_lp"] = lp
    t_idx = np.flatnonzero(data[treatment].to_numpy() == 1)
    c_idx = np.flatnonzero(data[treatment].to_numpy() == 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(t_idx)
    max_dist = caliper * lp.std(ddof=1) if caliper is not None else np.inf
    available = np.ones(len(c_idx), dtype=bool)
    pairs = []
    for k, i in enumerate(order):
        d = np.abs(lp[c_idx] - lp[i])
        if not replace:
            d = np.where(available, d, np.inf)
        j = int(np.argmin(d))
        if d[j] <= max_dist:
            pairs.append((i, c_idx[j], k))
            if not replace:
                available[j] = False
    rows = []
    for i, j, k in pairs:
        rows.append(data.loc[i].to_dict() | {"match_id": k})
        rows.append(data.loc[j].to_dict() | {"match_id": k})
    matched = pd.DataFrame(rows).drop(columns="_lp")
    matched.attrs["n_treated"] = len(t_idx)
    matched.attrs["n_matched"] = len(pairs)
    return matched


def matching_att(matched: pd.DataFrame, outcome: str, treatment: str, alpha: float = 0.05) -> dict:
    """ATT from a matched sample: mean within-pair difference with a paired-t interval."""
    wide = matched.pivot_table(index="match_id", columns=treatment, values=outcome)
    diff = (wide[1] - wide[0]).dropna()
    est = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    tcrit = stats.t.ppf(1 - alpha / 2, len(diff) - 1)
    return {"estimate": est, "se": se, "ci_low": est - tcrit * se, "ci_high": est + tcrit * se, "p_value": float(stats.ttest_1samp(diff, 0).pvalue), "n_pairs": int(len(diff))}


# --------------------------------------------------------------------------- #
# Difference-in-differences
# --------------------------------------------------------------------------- #
def difference_in_differences(
    df: pd.DataFrame,
    outcome: str,
    treated: str,
    post: str,
    covariates: Sequence[str] | None = None,
    cluster: str | None = None,
    alpha: float = 0.05,
) -> dict:
    """Two-period / two-group DiD: ``y ~ treated * post (+ covariates)``.

    The coefficient on ``treated:post`` is the DiD estimate under parallel
    trends. Standard errors are clustered on ``cluster`` when given, otherwise
    HC1-robust.
    """
    rhs = f"{treated} * {post}"
    if covariates:
        rhs += " + " + " + ".join(covariates)
    formula = f"{outcome} ~ {rhs}"
    if cluster is not None:
        res = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    else:
        res = smf.ols(formula, data=df).fit(cov_type="HC1")
    term = f"{treated}:{post}"
    ci = res.conf_int(alpha=alpha).loc[term]
    means = df.groupby([treated, post])[outcome].mean().unstack()
    return {
        "estimate": float(res.params[term]),
        "se": float(res.bse[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(res.pvalues[term]),
        "group_period_means": means,
        "manual_did": float((means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])),
        "result": res,
    }


# --------------------------------------------------------------------------- #
# Instrumental variables
# --------------------------------------------------------------------------- #
def two_stage_least_squares(
    df: pd.DataFrame,
    outcome: str,
    endog: str,
    instruments: Sequence[str],
    exog: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> dict:
    """2SLS estimate of the effect of ``endog`` on ``outcome`` using ``instruments``.

    Returns the second-stage coefficient with CI, the first-stage F statistic
    for the excluded instruments (``< 10`` indicates weak instruments), and the
    naive OLS estimate for comparison.
    """
    exog = list(exog or [])
    X = sm.add_constant(df[[endog, *exog]].astype(float))
    Z = sm.add_constant(df[[*instruments, *exog]].astype(float))
    y = df[outcome].astype(float)
    res = IV2SLS(y, X, Z).fit()
    first = sm.OLS(df[endog].astype(float), Z).fit()
    hyp = ", ".join(f"{z} = 0" for z in instruments)
    f_first = float(first.f_test(hyp).fvalue)
    ols = sm.OLS(y, X).fit()
    ci = res.conf_int(alpha=alpha)
    return {
        "estimate": float(res.params[endog]),
        "se": float(res.bse[endog]),
        "ci_low": float(ci.loc[endog, 0]),
        "ci_high": float(ci.loc[endog, 1]),
        "p_value": float(res.pvalues[endog]),
        "ols_estimate": float(ols.params[endog]),
        "first_stage_f": f_first,
        "weak_instruments": f_first < 10,
        "first_stage_r2": float(first.rsquared),
        "result": res,
        "first_stage": first,
    }


# --------------------------------------------------------------------------- #
# Regression discontinuity
# --------------------------------------------------------------------------- #
def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    a = np.abs(u)
    if kernel == "triangular":
        return np.clip(1 - a, 0, None)
    if kernel == "uniform":
        return (a <= 1).astype(float)
    if kernel == "epanechnikov":
        return np.clip(0.75 * (1 - u**2), 0, None)
    raise ValueError("kernel must be 'triangular', 'uniform', or 'epanechnikov'.")


def regression_discontinuity(
    df: pd.DataFrame,
    outcome: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
    polynomial: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Sharp RD estimate: the jump in E[Y | running] at ``cutoff``.

    Fits a local polynomial (default linear) with separate slopes on each side
    of the cutoff using kernel weights within ``bandwidth``. The default
    bandwidth is a rule of thumb, ``1.84 · sd(running) · n^(-1/5)``, and should
    be checked with :func:`rd_bandwidth_sensitivity`.
    """
    x = df[running].astype(float).to_numpy() - cutoff
    y = df[outcome].astype(float).to_numpy()
    h = bandwidth if bandwidth is not None else 1.84 * x.std(ddof=1) * len(x) ** (-1 / 5)
    w = _kernel_weights(x / h, kernel)
    keep = w > 0
    x, y, w = x[keep], y[keep], w[keep]
    d = (x >= 0).astype(float)
    cols = {"const": np.ones_like(x), "D": d}
    for p in range(1, polynomial + 1):
        cols[f"x{p}"] = x**p
        cols[f"D_x{p}"] = d * x**p
    X = pd.DataFrame(cols)
    res = sm.WLS(y, X, weights=w).fit(cov_type="HC1")
    ci = res.conf_int(alpha=alpha).loc["D"]
    return {
        "estimate": float(res.params["D"]),
        "se": float(res.bse["D"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(res.pvalues["D"]),
        "bandwidth": float(h),
        "kernel": kernel,
        "polynomial": polynomial,
        "n_left": int((d == 0).sum()),
        "n_right": int((d == 1).sum()),
        "result": res,
    }


def rd_bandwidth_sensitivity(df, outcome, running, bandwidths: Sequence[float], cutoff: float = 0.0, **kwargs) -> pd.DataFrame:
    """Re-estimate the RD effect across a grid of bandwidths."""
    rows = []
    for h in bandwidths:
        r = regression_discontinuity(df, outcome, running, cutoff, bandwidth=h, **kwargs)
        rows.append({k: r[k] for k in ("bandwidth", "estimate", "se", "ci_low", "ci_high", "p_value", "n_left", "n_right")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #
def _to_rr(value: float, scale: str, rare: bool) -> float:
    if scale == "rr" or scale == "hr" and rare:
        return value
    if scale == "or":
        return value if rare else float(np.sqrt(value))
    if scale == "hr":
        return (1 - 0.5 ** np.sqrt(value)) / (1 - 0.5 ** np.sqrt(1 / value))
    if scale in {"d", "smd"}:
        return float(np.exp(0.91 * value))
    raise ValueError("scale must be 'rr', 'or', 'hr', 'd', or 'smd'.")


def e_value(estimate: float, ci_low: float | None = None, ci_high: float | None = None, scale: str = "rr", rare: bool = False) -> dict:
    """E-value: minimum strength of association an unmeasured confounder would
    need with both treatment and outcome to fully explain away the estimate
    (VanderWeele & Ding, 2017).

    ``scale`` converts odds ratios, hazard ratios, or standardised mean
    differences to an approximate risk ratio first. For an SMD ``d`` the
    conversion is ``RR ≈ exp(0.91 d)``.
    """
    rr = _to_rr(estimate, scale, rare)
    rr_star = rr if rr >= 1 else 1 / rr
    ev = rr_star + np.sqrt(rr_star * (rr_star - 1))
    out = {"rr_equivalent": float(rr), "e_value": float(ev)}
    if ci_low is not None and ci_high is not None:
        lo, hi = _to_rr(ci_low, scale, rare), _to_rr(ci_high, scale, rare)
        if lo <= 1 <= hi:
            out["e_value_ci"] = 1.0
        else:
            bound = lo if rr >= 1 else 1 / hi
            out["e_value_ci"] = float(bound + np.sqrt(bound * (bound - 1)))
    return out
