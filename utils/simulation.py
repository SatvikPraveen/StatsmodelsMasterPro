"""Monte Carlo simulation framework for validating statistical procedures.

The framework follows the ADEMP structure of Morris, White & Crowther (2019):
*Aims, Data-generating mechanisms, Estimands, Methods, Performance measures*.

* ``monte_carlo`` – run an estimator over repeated draws from a
  data-generating process (DGP) and collect per-replicate results.
* ``performance_summary`` – bias, relative bias, empirical SE, model SE,
  RMSE, coverage, and rejection rate, each with its Monte Carlo standard
  error (MCSE), so simulation noise is reported alongside the estimate.
* ``simulate_power`` – empirical power / type-I error curves for any test.
* Ready-made DGPs (``dgp_linear``, ``dgp_two_sample``, ``dgp_poisson``) and
  estimators (``ols_estimator``, ``ttest_estimator``) that plug straight in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

__all__ = [
    "monte_carlo",
    "performance_summary",
    "rejection_rate",
    "simulate_power",
    "dgp_linear",
    "dgp_two_sample",
    "dgp_poisson",
    "ols_estimator",
    "glm_estimator",
    "ttest_estimator",
]


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #
def monte_carlo(
    dgp: Callable[[np.random.Generator], object],
    estimator: Callable[[object], dict],
    n_reps: int = 1000,
    seed: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Repeat ``estimator(dgp(rng))`` ``n_reps`` times.

    Each replicate uses an independent child generator spawned from ``seed``,
    so results are reproducible and replicates are statistically independent.
    Estimator exceptions are recorded in an ``error`` column rather than
    aborting the study.
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_reps)
    rows = []
    iterator = range(n_reps)
    if progress:  # pragma: no cover - cosmetic
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Monte Carlo")
        except ImportError:
            pass
    for i in iterator:
        rng = np.random.default_rng(children[i])
        try:
            row = dict(estimator(dgp(rng)))
            row["error"] = None
        except Exception as exc:  # noqa: BLE001 - record and continue
            row = {"error": f"{type(exc).__name__}: {exc}"}
        row["rep"] = i
        rows.append(row)
    table = pd.DataFrame(rows).set_index("rep")
    table.attrs["n_reps"] = n_reps
    table.attrs["seed"] = seed
    return table


def rejection_rate(pvalues, alpha: float = 0.05) -> tuple[float, float]:
    """Proportion of p-values below ``alpha`` and its Monte Carlo SE."""
    p = np.asarray(pvalues, float)
    p = p[~np.isnan(p)]
    rate = float(np.mean(p < alpha))
    return rate, float(np.sqrt(rate * (1 - rate) / len(p)))


def performance_summary(
    results: pd.DataFrame,
    true_values: dict[str, float],
    alpha: float = 0.05,
    ci_suffixes: tuple[str, str] = ("_ci_low", "_ci_high"),
    se_suffix: str = "_se",
    p_prefix: str = "p_",
) -> pd.DataFrame:
    """Performance measures with Monte Carlo standard errors (Morris et al., 2019, Table 6).

    For each key in ``true_values`` the column of the same name holds point
    estimates. Optional companion columns ``<key>_ci_low`` / ``<key>_ci_high``
    give coverage, ``<key>_se`` gives the average model-based SE, and
    ``p_<key>`` gives the rejection rate.
    """
    ok = results[results["error"].isna()] if "error" in results else results
    rows = []
    for name, truth in true_values.items():
        est = ok[name].astype(float).to_numpy()
        est = est[~np.isnan(est)]
        n = len(est)
        bias = est.mean() - truth
        emp_se = est.std(ddof=1)
        row = {
            "estimand": name,
            "true": truth,
            "n_reps": n,
            "mean_estimate": est.mean(),
            "bias": bias,
            "bias_mcse": emp_se / np.sqrt(n),
            "relative_bias": bias / truth if truth != 0 else np.nan,
            "empirical_se": emp_se,
            "empirical_se_mcse": emp_se / np.sqrt(2 * (n - 1)),
            "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
        }
        lo_col, hi_col = f"{name}{ci_suffixes[0]}", f"{name}{ci_suffixes[1]}"
        if lo_col in ok and hi_col in ok:
            cov = np.mean((ok[lo_col] <= truth) & (truth <= ok[hi_col]))
            row["coverage"] = float(cov)
            row["coverage_mcse"] = float(np.sqrt(cov * (1 - cov) / n))
            row["nominal_coverage"] = 1 - alpha
        se_col = f"{name}{se_suffix}"
        if se_col in ok:
            model_se = ok[se_col].astype(float).mean()
            row["mean_model_se"] = float(model_se)
            row["se_ratio"] = float(model_se / emp_se)
        p_col = f"{p_prefix}{name}"
        if p_col in ok:
            rate, mcse = rejection_rate(ok[p_col], alpha)
            row["rejection_rate"] = rate
            row["rejection_rate_mcse"] = mcse
        rows.append(row)
    table = pd.DataFrame(rows).set_index("estimand")
    table.attrs["n_failed"] = int(results["error"].notna().sum()) if "error" in results else 0
    return table


def simulate_power(
    dgp_factory: Callable[[float], Callable[[np.random.Generator], object]],
    test: Callable[[object], float],
    effects: Sequence[float],
    n_reps: int = 500,
    alpha: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """Empirical rejection rate as a function of effect size.

    ``dgp_factory(effect)`` must return a DGP callable; ``test(data)`` returns
    a p-value. An effect of ``0`` yields the empirical type-I error rate.
    """
    rows = []
    for k, eff in enumerate(effects):
        res = monte_carlo(dgp_factory(eff), lambda d: {"p": test(d)}, n_reps=n_reps, seed=None if seed is None else seed + k)
        rate, mcse = rejection_rate(res["p"], alpha)
        rows.append({"effect": eff, "power": rate, "mcse": mcse, "n_reps": n_reps, "alpha": alpha})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Data-generating processes
# --------------------------------------------------------------------------- #
def dgp_linear(n: int = 100, beta: Sequence[float] = (2.0, 1.5, -0.7), sigma: float = 1.5, heteroskedastic: bool = False, error: str = "normal"):
    """Linear DGP ``y = b0 + b1 X1 + b2 X2 + ... + e`` returning a DataFrame with X1..Xk and y."""
    beta = np.asarray(beta, float)
    k = len(beta) - 1

    def draw(rng: np.random.Generator) -> pd.DataFrame:
        X = rng.normal(size=(n, k))
        mu = beta[0] + X @ beta[1:]
        scale = sigma * (1 + np.abs(X[:, 0])) if heteroskedastic else sigma
        if error == "normal":
            e = rng.normal(0, scale, size=n)
        elif error == "t3":
            e = rng.standard_t(3, size=n) * scale / np.sqrt(3)
        elif error == "skewed":
            e = (rng.exponential(1.0, size=n) - 1.0) * scale
        else:
            raise ValueError("error must be 'normal', 't3', or 'skewed'.")
        df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(k)])
        df["y"] = mu + e
        return df

    return draw


def dgp_two_sample(n1: int = 30, n2: int = 30, delta: float = 0.0, sd: float = 1.0, dist: str = "normal"):
    """Two independent samples with mean shift ``delta`` (group 2 = group 1 + delta)."""

    def draw(rng: np.random.Generator):
        if dist == "normal":
            x, y = rng.normal(0, sd, n1), rng.normal(delta, sd, n2)
        elif dist == "lognormal":
            x, y = rng.lognormal(0, sd, n1), rng.lognormal(0, sd, n2) + delta
        elif dist == "t3":
            x, y = rng.standard_t(3, n1) * sd, rng.standard_t(3, n2) * sd + delta
        else:
            raise ValueError("dist must be 'normal', 'lognormal', or 't3'.")
        return x, y

    return draw


def dgp_poisson(n: int = 200, beta: Sequence[float] = (0.5, 0.9), overdispersion: float = 0.0):
    """Poisson (or negative-binomial when ``overdispersion > 0``) counts with log link."""
    beta = np.asarray(beta, float)

    def draw(rng: np.random.Generator) -> pd.DataFrame:
        x = rng.normal(size=n)
        mu = np.exp(beta[0] + beta[1] * x)
        if overdispersion > 0:
            mu = rng.gamma(1 / overdispersion, overdispersion * mu)
        return pd.DataFrame({"X": x, "y": rng.poisson(mu)})

    return draw


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def _model_row(res, alpha: float, prefix: str = "beta_") -> dict:
    ci = res.conf_int(alpha=alpha)
    row = {}
    for term in res.params.index:
        key = f"{prefix}{term}"
        row[key] = float(res.params[term])
        row[f"{key}_se"] = float(res.bse[term])
        row[f"{key}_ci_low"] = float(ci.loc[term, 0])
        row[f"{key}_ci_high"] = float(ci.loc[term, 1])
        row[f"p_{key}"] = float(res.pvalues[term])
    return row


def ols_estimator(formula: str = "y ~ X1 + X2", cov_type: str = "nonrobust", alpha: float = 0.05, **fit_kwargs):
    """Estimator returning OLS coefficients, SEs, CIs, and p-values keyed as ``beta_<term>``."""

    def estimate(data: pd.DataFrame) -> dict:
        res = smf.ols(formula, data=data).fit(cov_type=cov_type, **fit_kwargs)
        return _model_row(res, alpha)

    return estimate


def glm_estimator(formula: str = "y ~ X", family=None, alpha: float = 0.05):
    """Estimator for GLMs (default Poisson)."""
    fam = family or sm.families.Poisson()

    def estimate(data: pd.DataFrame) -> dict:
        res = smf.glm(formula, data=data, family=fam).fit()
        return _model_row(res, alpha)

    return estimate


def ttest_estimator(equal_var: bool = True, alpha: float = 0.05):
    """Estimator for two-sample data returning the mean difference, CI, and p-value."""

    def estimate(data) -> dict:
        x, y = data
        t = stats.ttest_ind(y, x, equal_var=equal_var)
        diff = float(np.mean(y) - np.mean(x))
        ci = t.confidence_interval(1 - alpha)
        return {"diff": diff, "diff_ci_low": float(ci.low), "diff_ci_high": float(ci.high), "p_diff": float(t.pvalue)}

    return estimate
