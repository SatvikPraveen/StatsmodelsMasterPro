"""Power analysis and sample-size planning.

Thin, consistent wrappers around ``statsmodels.stats.power`` that solve for
whichever of effect size, sample size, or power is left as ``None``, plus a
Fisher-z correlation power routine and power-curve tabulation. Pair these
analytic results with :mod:`utils.simulation` to verify them empirically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize

__all__ = ["power_ttest", "power_anova", "power_proportions", "power_correlation", "power_curve", "minimum_detectable_effect"]


def _solve(solver, **kwargs) -> dict:
    missing = [k for k in ("effect_size", "nobs1", "nobs", "power") if k in kwargs and kwargs[k] is None]
    if len(missing) != 1:
        raise ValueError("Exactly one of effect size, sample size, or power must be None.")
    solved = float(solver.solve_power(**kwargs))
    out = {k: v for k, v in kwargs.items() if v is not None}
    out[missing[0]] = solved
    out["solved_for"] = missing[0]
    return out


def power_ttest(effect_size: float | None = None, nobs: float | None = None, alpha: float = 0.05, power: float | None = None, ratio: float = 1.0, alternative: str = "two-sided", paired: bool = False) -> dict:
    """Two-sample (or paired/one-sample) t-test power. ``nobs`` is the size of group 1."""
    if paired:
        out = _solve(TTestPower(), effect_size=effect_size, nobs=nobs, alpha=alpha, power=power, alternative=alternative)
    else:
        out = _solve(TTestIndPower(), effect_size=effect_size, nobs1=nobs, alpha=alpha, power=power, ratio=ratio, alternative=alternative)
        out["nobs2"] = out["nobs1"] * ratio
        out["nobs"] = out.pop("nobs1")
        if out["solved_for"] == "nobs1":
            out["solved_for"] = "nobs"
    out["test"] = "paired t" if paired else "independent t"
    return out


def power_anova(effect_size: float | None = None, k_groups: int = 3, nobs: float | None = None, alpha: float = 0.05, power: float | None = None) -> dict:
    """One-way ANOVA power with Cohen's f. ``nobs`` is the *total* sample size."""
    out = _solve(FTestAnovaPower(), effect_size=effect_size, nobs=nobs, alpha=alpha, power=power, k_groups=k_groups)
    out["test"] = "one-way ANOVA"
    return out


def power_proportions(p1: float, p2: float, nobs: float | None = None, alpha: float = 0.05, power: float | None = None, ratio: float = 1.0, alternative: str = "two-sided") -> dict:
    """Two-proportion z-test power using Cohen's h."""
    h = abs(float(proportion_effectsize(p1, p2)))  # sign is irrelevant for power
    out = _solve(NormalIndPower(), effect_size=h, nobs1=nobs, alpha=alpha, power=power, ratio=ratio, alternative=alternative)
    out["cohens_h"] = h
    out["nobs"] = out.pop("nobs1")
    if out["solved_for"] == "nobs1":
        out["solved_for"] = "nobs"
    out["test"] = "two-proportion z"
    return out


def power_correlation(r: float | None = None, nobs: float | None = None, alpha: float = 0.05, power: float | None = None, alternative: str = "two-sided") -> dict:
    """Power for testing ρ = 0 via the Fisher z transform."""
    z_a = stats.norm.ppf(1 - alpha / 2) if alternative == "two-sided" else stats.norm.ppf(1 - alpha)
    if power is None:
        if r is None or nobs is None:
            raise ValueError("Provide r and nobs to solve for power.")
        zr = np.arctanh(r) * np.sqrt(nobs - 3)
        pw = stats.norm.sf(z_a - abs(zr)) + (stats.norm.cdf(-z_a - abs(zr)) if alternative == "two-sided" else 0)
        return {"r": r, "nobs": nobs, "alpha": alpha, "power": float(pw), "solved_for": "power", "test": "correlation"}
    z_b = stats.norm.ppf(power)
    if nobs is None:
        if r is None:
            raise ValueError("Provide r to solve for nobs.")
        n = ((z_a + z_b) / np.arctanh(abs(r))) ** 2 + 3
        return {"r": r, "nobs": float(np.ceil(n)), "alpha": alpha, "power": power, "solved_for": "nobs", "test": "correlation"}
    r_min = float(np.tanh((z_a + z_b) / np.sqrt(nobs - 3)))
    return {"r": r_min, "nobs": nobs, "alpha": alpha, "power": power, "solved_for": "r", "test": "correlation"}


def power_curve(effect_sizes: Sequence[float], sample_sizes: Sequence[int], alpha: float = 0.05, test: str = "t", **kwargs) -> pd.DataFrame:
    """Long-format table of power over a grid of effect and sample sizes."""
    fn = {"t": lambda d, n: power_ttest(d, n, alpha, None, **kwargs)["power"], "anova": lambda f, n: power_anova(f, kwargs.get("k_groups", 3), n, alpha, None)["power"]}
    if test not in fn:
        raise ValueError("test must be 't' or 'anova'.")
    rows = [{"effect_size": d, "nobs": n, "power": fn[test](d, n)} for d in effect_sizes for n in sample_sizes]
    return pd.DataFrame(rows)


def minimum_detectable_effect(nobs: int, alpha: float = 0.05, power: float = 0.8, test: str = "t", **kwargs) -> float:
    """Smallest standardised effect detectable with the given design."""
    if test == "t":
        return power_ttest(None, nobs, alpha, power, **kwargs)["effect_size"]
    if test == "anova":
        return power_anova(None, kwargs.get("k_groups", 3), nobs, alpha, power)["effect_size"]
    raise ValueError("test must be 't' or 'anova'.")
