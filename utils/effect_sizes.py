"""Effect sizes with confidence intervals and conventional interpretations.

Effect sizes are essential for reporting practical significance alongside
p-values (Cohen, 1988; Lakens, 2013). This module covers:

* Standardised mean differences: Cohen's d, Hedges' g, Glass's Δ
* Non-parametric: Cliff's δ, common-language effect size
* ANOVA: η², partial η², ω² from a ``statsmodels`` ANOVA table
* Categorical: Cramér's V (bias-corrected), odds ratio with Woolf CI
* Conversions between d and r, and Cohen-style interpretation labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "cliffs_delta",
    "common_language_effect_size",
    "anova_effect_sizes",
    "cramers_v",
    "odds_ratio",
    "d_to_r",
    "r_to_d",
    "interpret_effect_size",
]


def _pooled_sd(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    return float(np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)))


def _smd_ci(d: float, n1: int, n2: int, alpha: float) -> tuple[float, float]:
    """Large-sample CI for a standardised mean difference (Hedges & Olkin, 1985)."""
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(d - z * se), float(d + z * se)


def cohens_d(x, y, alpha: float = 0.05) -> dict:
    """Cohen's d for two independent samples using the pooled standard deviation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = float((x.mean() - y.mean()) / _pooled_sd(x, y))
    lo, hi = _smd_ci(d, len(x), len(y), alpha)
    return {"d": d, "ci_low": lo, "ci_high": hi, "n1": len(x), "n2": len(y), "interpretation": interpret_effect_size(d, "d")}


def hedges_g(x, y, alpha: float = 0.05) -> dict:
    """Hedges' g: Cohen's d with the small-sample bias correction J(df)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    df = len(x) + len(y) - 2
    j = 1 - 3 / (4 * df - 1)
    d = cohens_d(x, y, alpha)["d"]
    g = float(d * j)
    lo, hi = _smd_ci(g, len(x), len(y), alpha)
    return {"g": g, "ci_low": lo, "ci_high": hi, "correction_J": float(j), "interpretation": interpret_effect_size(g, "d")}


def glass_delta(treatment, control, alpha: float = 0.05) -> dict:
    """Glass's Δ: mean difference scaled by the *control* group SD."""
    t, c = np.asarray(treatment, float), np.asarray(control, float)
    delta = float((t.mean() - c.mean()) / c.std(ddof=1))
    lo, hi = _smd_ci(delta, len(t), len(c), alpha)
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "interpretation": interpret_effect_size(delta, "d")}


def cliffs_delta(x, y) -> dict:
    """Cliff's δ = P(x > y) − P(x < y); a robust ordinal effect size in [−1, 1]."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    delta = float((greater - less) / (len(x) * len(y)))
    return {"delta": delta, "interpretation": interpret_effect_size(delta, "cliff")}


def common_language_effect_size(x, y) -> float:
    """Probability that a random draw from ``x`` exceeds a random draw from ``y`` (McGraw & Wong, 1992)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.mean(x[:, None] > y[None, :]) + 0.5 * np.mean(x[:, None] == y[None, :]))


def anova_effect_sizes(anova_table: pd.DataFrame) -> pd.DataFrame:
    """η², partial η², and ω² for each term in a ``statsmodels`` ``anova_lm`` table.

    The residual row must be labelled ``"Residual"`` (the ``anova_lm`` default).
    """
    if "Residual" not in anova_table.index:
        raise ValueError("ANOVA table must contain a 'Residual' row.")
    ss = anova_table["sum_sq"]
    df = anova_table["df"]
    ss_error = ss["Residual"]
    df_error = df["Residual"]
    ms_error = ss_error / df_error
    ss_total = ss.sum()
    terms = [t for t in anova_table.index if t != "Residual"]
    rows = []
    for t in terms:
        eta2 = ss[t] / ss_total
        partial_eta2 = ss[t] / (ss[t] + ss_error)
        omega2 = (ss[t] - df[t] * ms_error) / (ss_total + ms_error)
        rows.append(
            {
                "term": t,
                "eta_squared": float(eta2),
                "partial_eta_squared": float(partial_eta2),
                "omega_squared": float(max(omega2, 0.0)),
                "interpretation": interpret_effect_size(float(eta2), "eta2"),
            }
        )
    return pd.DataFrame(rows).set_index("term")


def cramers_v(contingency, bias_corrected: bool = True) -> dict:
    """Cramér's V for an r × c contingency table, with the Bergsma (2013) bias correction."""
    table = np.asarray(contingency, float)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    if bias_corrected:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    v = float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1e-12)))
    return {"cramers_v": v, "chi2": float(chi2), "p_value": float(p), "dof": int(dof), "interpretation": interpret_effect_size(v, "r")}


def odds_ratio(table_2x2, alpha: float = 0.05, continuity: float = 0.5) -> dict:
    """Odds ratio for a 2 × 2 table ``[[a, b], [c, d]]`` with a Woolf (log) confidence interval.

    A Haldane–Anscombe continuity correction of ``continuity`` is added to every
    cell when any cell is zero.
    """
    t = np.asarray(table_2x2, float)
    if t.shape != (2, 2):
        raise ValueError("table_2x2 must have shape (2, 2).")
    if (t == 0).any():
        t = t + continuity
    a, b, c, d = t.ravel()
    or_ = (a * d) / (b * c)
    se_log = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "odds_ratio": float(or_),
        "ci_low": float(np.exp(np.log(or_) - z * se_log)),
        "ci_high": float(np.exp(np.log(or_) + z * se_log)),
        "log_or_se": float(se_log),
    }


def d_to_r(d: float, n1: int | None = None, n2: int | None = None) -> float:
    """Convert Cohen's d to a point-biserial correlation."""
    a = 4.0 if n1 is None or n2 is None else (n1 + n2) ** 2 / (n1 * n2)
    return float(d / np.sqrt(d**2 + a))


def r_to_d(r: float) -> float:
    """Convert a correlation to Cohen's d (equal group sizes)."""
    return float(2 * r / np.sqrt(1 - r**2))


_THRESHOLDS = {
    "d": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (np.inf, "large")],
    "r": [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (np.inf, "large")],
    "eta2": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (np.inf, "large")],
    "cliff": [(0.147, "negligible"), (0.33, "small"), (0.474, "medium"), (np.inf, "large")],
}


def interpret_effect_size(value: float, kind: str = "d") -> str:
    """Label an effect size using Cohen (1988) / Romano et al. (2006) conventions."""
    if kind not in _THRESHOLDS:
        raise ValueError(f"kind must be one of {sorted(_THRESHOLDS)}")
    v = abs(float(value))
    for cutoff, label in _THRESHOLDS[kind]:
        if v < cutoff:
            return label
    return "large"  # pragma: no cover
