"""Survival analysis helpers built on ``lifelines`` (optional dependency).

* ``kaplan_meier_table`` – survival estimates with confidence bands per group
  and median survival times.
* ``logrank`` – two-group or multivariate log-rank test.
* ``fit_cox`` – Cox proportional-hazards model with a tidy hazard-ratio table.
* ``check_proportional_hazards`` – Schoenfeld-residual based PH test
  (Grambsch & Therneau, 1994).
* ``parametric_comparison`` – AIC comparison of Weibull, exponential,
  log-normal, and log-logistic fits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

try:
    import lifelines
    from lifelines import (
        CoxPHFitter,
        ExponentialFitter,
        KaplanMeierFitter,
        LogLogisticFitter,
        LogNormalFitter,
        WeibullFitter,
    )
    from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test

    HAS_LIFELINES = True
except ImportError:  # pragma: no cover
    HAS_LIFELINES = False

__all__ = ["HAS_LIFELINES", "kaplan_meier_table", "logrank", "fit_cox", "check_proportional_hazards", "parametric_comparison"]


def _require():
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required: pip install lifelines")


def kaplan_meier_table(df: pd.DataFrame, duration: str, event: str, group: str | None = None, times: Sequence[float] | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Kaplan–Meier survival probabilities (with CIs) at ``times`` for each group."""
    _require()
    groups = [(None, df)] if group is None else list(df.groupby(group))
    rows = []
    for name, sub in groups:
        km = KaplanMeierFitter(alpha=alpha).fit(sub[duration], sub[event], label=str(name))
        eval_times = np.asarray(times if times is not None else np.quantile(sub[duration], [0.25, 0.5, 0.75]), float)
        surv = km.survival_function_at_times(eval_times).to_numpy()
        ci = km.confidence_interval_
        lo = np.interp(eval_times, ci.index, ci.iloc[:, 0])
        hi = np.interp(eval_times, ci.index, ci.iloc[:, 1])
        for t, s, l, h in zip(eval_times, surv, lo, hi):
            rows.append({"group": name, "time": float(t), "survival": float(s), "ci_low": float(l), "ci_high": float(h), "median_survival": float(km.median_survival_time_), "n": len(sub), "events": int(sub[event].sum())})
    return pd.DataFrame(rows)


def logrank(df: pd.DataFrame, duration: str, event: str, group: str) -> dict:
    """Log-rank test across the levels of ``group`` (multivariate when > 2 levels)."""
    _require()
    levels = df[group].unique()
    if len(levels) == 2:
        a, b = (df[df[group] == lvl] for lvl in levels)
        res = logrank_test(a[duration], b[duration], a[event], b[event])
    else:
        res = multivariate_logrank_test(df[duration], df[group], df[event])
    return {"statistic": float(res.test_statistic), "p_value": float(res.p_value), "df": int(len(levels) - 1)}


def fit_cox(df: pd.DataFrame, duration: str, event: str, covariates: Sequence[str], robust: bool = False, alpha: float = 0.05):
    """Fit a Cox PH model and return ``(fitter, tidy_table)`` with hazard ratios."""
    _require()
    cph = CoxPHFitter(alpha=alpha)
    cph.fit(df[[duration, event, *covariates]], duration_col=duration, event_col=event, robust=robust)
    s = cph.summary
    table = pd.DataFrame(
        {
            "term": s.index,
            "coef": s["coef"].to_numpy(),
            "hazard_ratio": s["exp(coef)"].to_numpy(),
            "se": s["se(coef)"].to_numpy(),
            "hr_ci_low": s[f"exp(coef) lower {100 * (1 - alpha):g}%"].to_numpy(),
            "hr_ci_high": s[f"exp(coef) upper {100 * (1 - alpha):g}%"].to_numpy(),
            "p_value": s["p"].to_numpy(),
        }
    ).set_index("term")
    table.attrs["concordance"] = float(cph.concordance_index_)
    table.attrs["log_likelihood"] = float(cph.log_likelihood_)
    table.attrs["aic_partial"] = float(cph.AIC_partial_)
    return cph, table


def check_proportional_hazards(cph, df: pd.DataFrame, duration: str, event: str, covariates: Sequence[str], time_transform: str = "rank") -> pd.DataFrame:
    """Schoenfeld-residual test of the PH assumption for each covariate."""
    _require()
    res = proportional_hazard_test(cph, df[[duration, event, *covariates]], time_transform=time_transform)
    s = res.summary
    out = pd.DataFrame({"term": s.index.get_level_values(0), "statistic": s["test_statistic"].to_numpy(), "p_value": s["p"].to_numpy()}).set_index("term")
    out["ph_violated"] = out["p_value"] < 0.05
    return out


def parametric_comparison(df: pd.DataFrame, duration: str, event: str) -> pd.DataFrame:
    """Compare parametric survival distributions by AIC."""
    _require()
    fitters = {"Weibull": WeibullFitter(), "Exponential": ExponentialFitter(), "LogNormal": LogNormalFitter(), "LogLogistic": LogLogisticFitter()}
    rows = []
    for name, f in fitters.items():
        f.fit(df[duration], df[event])
        rows.append({"distribution": name, "aic": float(f.AIC_), "log_likelihood": float(f.log_likelihood_), "median_survival": float(f.median_survival_time_)})
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    return table
