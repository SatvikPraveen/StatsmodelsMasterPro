"""Resampling and robust inference.

This module provides research-grade inference tools that go beyond the default
asymptotic standard errors reported by ``statsmodels``:

* ``bootstrap_ci`` – percentile, basic, normal, and bias-corrected accelerated
  (BCa) bootstrap confidence intervals for arbitrary statistics
  (Efron & Tibshirani, 1993; DiCiccio & Efron, 1996).
* ``bootstrap_regression`` – pairs, residual, and wild bootstrap for regression
  coefficients (Freedman, 1981; Wu, 1986; Davidson & Flachaire, 2008).
* ``permutation_test`` / ``paired_permutation_test`` – exact-in-the-limit
  randomisation tests for two-sample and paired designs (Fisher, 1935).
* ``robust_se_table`` – heteroskedasticity-consistent (HC0–HC3), HAC
  (Newey & West, 1987), and cluster-robust standard errors side by side
  (White, 1980; MacKinnon & White, 1985).
* ``wald_test`` / ``likelihood_ratio_test`` – tests of linear restrictions and
  nested-model comparisons.
* ``multiple_testing`` – family-wise and false-discovery-rate corrections
  (Holm, 1979; Benjamini & Hochberg, 1995).

All random procedures accept a ``seed`` argument and use
``numpy.random.default_rng`` so results are exactly reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ArrayLike = Sequence[float] | np.ndarray | pd.Series

__all__ = [
    "BootstrapResult",
    "PermutationResult",
    "bootstrap_ci",
    "bootstrap_regression",
    "permutation_test",
    "paired_permutation_test",
    "robust_se_table",
    "wald_test",
    "likelihood_ratio_test",
    "multiple_testing",
    "ci_from_se",
]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapResult:
    """Container for a bootstrap confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    se: float
    method: str
    n_boot: int
    alpha: float
    distribution: np.ndarray = field(repr=False)

    @property
    def bias(self) -> float:
        """Bootstrap estimate of bias: mean of the bootstrap distribution minus the estimate."""
        return float(np.mean(self.distribution) - self.estimate)

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "se": self.se,
            "bias": self.bias,
            "method": self.method,
            "n_boot": self.n_boot,
            "alpha": self.alpha,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        level = 100 * (1 - self.alpha)
        return (
            f"{self.estimate:.4f} [{level:.0f}% {self.method} CI: "
            f"{self.ci_low:.4f}, {self.ci_high:.4f}], bootstrap SE = {self.se:.4f}"
        )


@dataclass
class PermutationResult:
    """Container for a permutation / randomisation test."""

    observed: float
    p_value: float
    n_perm: int
    alternative: str
    null_distribution: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "n_perm": self.n_perm,
            "alternative": self.alternative,
        }


# --------------------------------------------------------------------------- #
# Bootstrap for arbitrary statistics
# --------------------------------------------------------------------------- #
def _as_arrays(data) -> list[np.ndarray]:
    if isinstance(data, (tuple, list)) and len(data) > 0 and np.ndim(data[0]) >= 1:
        arrays = [np.asarray(a, dtype=float) for a in data]
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("All arrays passed to the bootstrap must have the same length.")
        return arrays
    return [np.asarray(data, dtype=float)]


def _jackknife(arrays: list[np.ndarray], statistic: Callable) -> np.ndarray:
    n = len(arrays[0])
    idx = np.arange(n)
    return np.array([statistic(*[a[idx != i] for a in arrays]) for i in range(n)], dtype=float)


def bootstrap_ci(
    data,
    statistic: Callable = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence interval for an arbitrary statistic.

    Parameters
    ----------
    data
        A 1-D array, or a tuple/list of equal-length 1-D arrays that are
        resampled *jointly* (e.g. ``(x, y)`` for a correlation coefficient).
    statistic
        Function mapping the resampled array(s) to a scalar. Called as
        ``statistic(*arrays)``.
    n_boot
        Number of bootstrap replicates.
    alpha
        Significance level; a ``100 * (1 - alpha)`` % interval is returned.
    method
        ``"percentile"``, ``"basic"`` (reverse percentile), ``"normal"``
        (estimate ± z · bootstrap SE), or ``"bca"`` (bias-corrected and
        accelerated; recommended for skewed statistics).
    seed
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    BootstrapResult

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
    DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals.
    *Statistical Science*, 11(3), 189–228.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    method = method.lower()
    if method not in {"percentile", "basic", "normal", "bca"}:
        raise ValueError(f"Unknown method {method!r}.")

    arrays = _as_arrays(data)
    n = len(arrays[0])
    if n < 2:
        raise ValueError("Need at least two observations to bootstrap.")

    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(*arrays))

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = statistic(*[a[idx[b]] for a in arrays])

    se = float(np.std(boot, ddof=1))
    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    if method == "percentile":
        lo, hi = np.quantile(boot, [lo_q, hi_q])
    elif method == "basic":
        q_lo, q_hi = np.quantile(boot, [lo_q, hi_q])
        lo, hi = 2 * theta_hat - q_hi, 2 * theta_hat - q_lo
    elif method == "normal":
        z = stats.norm.ppf(hi_q)
        lo, hi = theta_hat - z * se, theta_hat + z * se
    else:  # bca
        prop_less = np.mean(boot < theta_hat)
        prop_less = min(max(prop_less, 1.0 / (n_boot + 1)), n_boot / (n_boot + 1))
        z0 = stats.norm.ppf(prop_less)
        jack = _jackknife(arrays, statistic)
        jack_mean = jack.mean()
        num = np.sum((jack_mean - jack) ** 3)
        den = 6.0 * np.sum((jack_mean - jack) ** 2) ** 1.5
        a = num / den if den > 0 else 0.0
        z_lo, z_hi = stats.norm.ppf(lo_q), stats.norm.ppf(hi_q)
        adj_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
        adj_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
        lo, hi = np.quantile(boot, [adj_lo, adj_hi])

    return BootstrapResult(
        estimate=theta_hat,
        ci_low=float(lo),
        ci_high=float(hi),
        se=se,
        method=method,
        n_boot=n_boot,
        alpha=alpha,
        distribution=boot,
    )


# --------------------------------------------------------------------------- #
# Bootstrap for regression coefficients
# --------------------------------------------------------------------------- #
def bootstrap_regression(
    formula: str,
    data: pd.DataFrame,
    n_boot: int = 1000,
    method: str = "pairs",
    alpha: float = 0.05,
    seed: int | None = None,
    fit_fn: Callable = smf.ols,
    return_draws: bool = False,
    **fit_kwargs,
):
    """Bootstrap standard errors and percentile CIs for regression coefficients.

    Parameters
    ----------
    formula, data
        A patsy formula and the DataFrame it refers to.
    n_boot
        Number of bootstrap replicates.
    method
        ``"pairs"`` – resample rows (robust to heteroskedasticity and
        misspecified error distributions);
        ``"residual"`` – resample residuals and add them to fitted values
        (assumes i.i.d. errors and fixed design);
        ``"wild"`` – multiply residuals by Rademacher draws (robust to
        heteroskedasticity while keeping the design fixed).
    fit_fn
        A ``statsmodels.formula.api`` model constructor. Residual and wild
        bootstrap require a linear-mean model such as ``smf.ols``.
    return_draws
        If True also return the ``(n_boot, k)`` array of bootstrap coefficients.

    Returns
    -------
    pandas.DataFrame
        Index = parameter names; columns ``coef, boot_se, ci_low, ci_high,
        asymptotic_se``.

    References
    ----------
    Freedman, D. A. (1981). Bootstrapping regression models. *Ann. Statist.*, 9(6).
    Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling methods in
    regression analysis. *Ann. Statist.*, 14(4).
    """
    method = method.lower()
    if method not in {"pairs", "residual", "wild"}:
        raise ValueError("method must be 'pairs', 'residual', or 'wild'.")

    rng = np.random.default_rng(seed)
    base = fit_fn(formula, data=data).fit(**fit_kwargs)
    names = list(base.params.index)
    n = int(base.nobs)
    draws = np.empty((n_boot, len(names)), dtype=float)

    if method == "pairs":
        for b in range(n_boot):
            sample = data.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
            draws[b] = fit_fn(formula, data=sample).fit(**fit_kwargs).params.reindex(names).to_numpy()
    else:
        response = formula.split("~")[0].strip()
        fitted = np.asarray(base.fittedvalues)
        resid = np.asarray(base.resid)
        # Leverage adjustment (HC3-style) for the wild bootstrap
        try:
            h = np.asarray(base.get_influence().hat_matrix_diag)
            resid_adj = resid / (1 - h)
        except Exception:  # pragma: no cover - non-OLS models
            resid_adj = resid
        for b in range(n_boot):
            if method == "residual":
                e = resid[rng.integers(0, n, size=n)]
            else:
                e = resid_adj * rng.choice([-1.0, 1.0], size=n)
            boot_data = data.copy()
            boot_data[response] = fitted + e
            draws[b] = fit_fn(formula, data=boot_data).fit(**fit_kwargs).params.reindex(names).to_numpy()

    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    table = pd.DataFrame(
        {
            "coef": base.params.to_numpy(),
            "boot_se": draws.std(axis=0, ddof=1),
            "ci_low": lo,
            "ci_high": hi,
            "asymptotic_se": base.bse.to_numpy(),
        },
        index=names,
    )
    table.attrs["method"] = method
    table.attrs["n_boot"] = n_boot
    return (table, draws) if return_draws else table


# --------------------------------------------------------------------------- #
# Permutation tests
# --------------------------------------------------------------------------- #
_STATISTICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mean_diff": lambda a, b: float(np.mean(a) - np.mean(b)),
    "median_diff": lambda a, b: float(np.median(a) - np.median(b)),
    "t": lambda a, b: float(stats.ttest_ind(a, b, equal_var=False).statistic),
}


def _perm_p_value(observed: float, null: np.ndarray, alternative: str) -> float:
    n = len(null)
    if alternative == "two-sided":
        count = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    elif alternative == "greater":
        count = np.sum(null >= observed - 1e-12)
    elif alternative == "less":
        count = np.sum(null <= observed + 1e-12)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    # Add-one correction (Phipson & Smyth, 2010) keeps p-values strictly positive.
    return float((count + 1) / (n + 1))


def permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    statistic: str | Callable = "mean_diff",
    n_perm: int = 5000,
    alternative: str = "two-sided",
    seed: int | None = None,
) -> PermutationResult:
    """Two-sample permutation test by random relabelling.

    Under the sharp null of exchangeability, group labels are shuffled
    ``n_perm`` times and the statistic recomputed. The p-value uses the
    add-one correction of Phipson & Smyth (2010).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stat_fn = _STATISTICS[statistic] if isinstance(statistic, str) else statistic
    rng = np.random.default_rng(seed)

    pooled = np.concatenate([x, y])
    n_x = len(x)
    observed = float(stat_fn(x, y))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null[i] = stat_fn(perm[:n_x], perm[n_x:])
    return PermutationResult(observed, _perm_p_value(observed, null, alternative), n_perm, alternative, null)


def paired_permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    n_perm: int = 5000,
    alternative: str = "two-sided",
    seed: int | None = None,
) -> PermutationResult:
    """Paired permutation (sign-flip) test on the mean of within-pair differences."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    observed = float(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    return PermutationResult(observed, _perm_p_value(observed, null, alternative), n_perm, alternative, null)


# --------------------------------------------------------------------------- #
# Robust standard errors
# --------------------------------------------------------------------------- #
def _refit_with_cov(result, cov_type: str, cov_kwds: dict | None):
    """Return a results object whose covariance uses ``cov_type``.

    Refitting through ``result.model.fit`` keeps pandas labels on the
    parameters (``get_robustcov_results`` returns bare arrays).
    """
    if cov_type == "nonrobust":
        return result.model.fit()
    return result.model.fit(cov_type=cov_type, cov_kwds=cov_kwds or {})


def robust_se_table(
    result,
    cov_types: Sequence[str] = ("nonrobust", "HC0", "HC1", "HC2", "HC3"),
    groups: ArrayLike | None = None,
    maxlags: int | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare coefficient standard errors under several covariance estimators.

    Parameters
    ----------
    result
        A fitted ``statsmodels`` results object (OLS, WLS, GLM, ...).
    cov_types
        Any of ``"nonrobust"``, ``"HC0"``–``"HC3"``, ``"HAC"``, ``"cluster"``.
    groups
        Cluster identifiers, required for ``"cluster"``.
    maxlags
        Bandwidth for ``"HAC"`` (Newey–West). Defaults to ``floor(4 (n/100)^(2/9))``.

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns ``term, cov_type, coef, se, statistic,
        p_value, ci_low, ci_high``.
    """
    n = int(result.nobs)
    rows = []
    for cov_type in cov_types:
        cov_kwds: dict = {}
        if cov_type == "cluster":
            if groups is None:
                raise ValueError("groups must be supplied for cluster-robust standard errors.")
            cov_kwds = {"groups": np.asarray(groups)}
        elif cov_type == "HAC":
            lags = maxlags if maxlags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
            cov_kwds = {"maxlags": lags}
        res = _refit_with_cov(result, cov_type, cov_kwds)
        ci = res.conf_int(alpha=alpha)
        for term in res.params.index:
            rows.append(
                {
                    "term": term,
                    "cov_type": cov_type,
                    "coef": float(res.params[term]),
                    "se": float(res.bse[term]),
                    "statistic": float(res.tvalues[term]),
                    "p_value": float(res.pvalues[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Hypothesis tests on fitted models
# --------------------------------------------------------------------------- #
def wald_test(result, hypotheses: str | Sequence[str], use_f: bool | None = None) -> pd.DataFrame:
    """Test one or more linear restrictions such as ``"X1 = 0"`` or ``"X1 - X2 = 0"``.

    Each hypothesis string is tested separately and, when more than one is
    given, jointly as well. Returns a tidy DataFrame.
    """
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    rows = []
    for h in hypotheses:
        wt = result.wald_test(h, use_f=use_f, scalar=True)
        rows.append({"hypothesis": h, "statistic": float(wt.statistic), "p_value": float(wt.pvalue), "df": _df_of(wt)})
    if len(hypotheses) > 1:
        wt = result.wald_test(", ".join(hypotheses), use_f=use_f, scalar=True)
        rows.append({"hypothesis": "joint", "statistic": float(wt.statistic), "p_value": float(wt.pvalue), "df": _df_of(wt)})
    return pd.DataFrame(rows)


def _df_of(wt) -> float | tuple:
    df_denom = getattr(wt, "df_denom", None)
    df_num = getattr(wt, "df_num", None)
    if df_num is None:
        return float("nan")
    return (int(df_num), int(df_denom)) if df_denom is not None else int(df_num)


def likelihood_ratio_test(restricted, full) -> dict:
    """Likelihood-ratio test for nested models fitted by maximum likelihood.

    ``LR = 2 (ll_full - ll_restricted) ~ chi2(df_full - df_restricted)``.
    """
    lr = 2.0 * (full.llf - restricted.llf)
    df = int(full.df_model - restricted.df_model)
    if df <= 0:
        raise ValueError("The 'full' model must have more parameters than the 'restricted' model.")
    p = float(stats.chi2.sf(lr, df))
    return {"lr_statistic": float(lr), "df": df, "p_value": p}


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
def multiple_testing(
    pvalues: ArrayLike,
    method: str = "holm",
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Adjust p-values for multiple comparisons.

    ``method`` is any ``statsmodels`` option: ``"bonferroni"``, ``"holm"``,
    ``"sidak"``, ``"holm-sidak"``, ``"fdr_bh"`` (Benjamini–Hochberg),
    ``"fdr_by"`` (Benjamini–Yekutieli), ...
    """
    p = np.asarray(pvalues, dtype=float)
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=method)
    idx = list(labels) if labels is not None else list(range(len(p)))
    return pd.DataFrame({"p_raw": p, "p_adj": p_adj, "reject": reject, "method": method}, index=idx)


def ci_from_se(estimate: float, se: float, alpha: float = 0.05, df: float | None = None) -> tuple[float, float]:
    """Symmetric confidence interval from an estimate and standard error (z or t based)."""
    crit = stats.t.ppf(1 - alpha / 2, df) if df is not None else stats.norm.ppf(1 - alpha / 2)
    return float(estimate - crit * se), float(estimate + crit * se)
