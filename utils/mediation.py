"""Mediation and moderation analysis.

* ``mediation_analysis`` – the causal-steps decomposition (Baron & Kenny, 1986)
  with a bootstrap confidence interval for the indirect effect
  (Preacher & Hayes, 2004, 2008), the Sobel (1982) test, and the proportion
  mediated.
* ``mediation_statsmodels`` – cross-check using ``statsmodels``' potential-
  outcomes implementation of Imai, Keele & Tingley (2010).
* ``moderation_analysis`` – interaction model with simple slopes at
  ``mean ± 1 SD`` of the moderator and the Johnson–Neyman (1936) region of
  significance (Bauer & Curran, 2005).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from utils.inference import bootstrap_ci

__all__ = ["mediation_analysis", "mediation_statsmodels", "moderation_analysis"]


def _rhs(*terms: str, covariates: Sequence[str] | None) -> str:
    return " + ".join([*terms, *(covariates or [])])


def mediation_analysis(
    df: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    covariates: Sequence[str] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int | None = None,
) -> dict:
    """Simple mediation X → M → Y with a bootstrap CI for the indirect effect a·b."""
    data = df.reset_index(drop=True)
    total = smf.ols(f"{y} ~ {_rhs(x, covariates=covariates)}", data=data).fit()
    med = smf.ols(f"{m} ~ {_rhs(x, covariates=covariates)}", data=data).fit()
    out = smf.ols(f"{y} ~ {_rhs(x, m, covariates=covariates)}", data=data).fit()

    a, b = med.params[x], out.params[m]
    c, c_prime = total.params[x], out.params[x]
    indirect = a * b
    se_a, se_b = med.bse[x], out.bse[m]
    sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    sobel_z = indirect / sobel_se
    sobel_p = 2 * stats.norm.sf(abs(sobel_z))

    idx = np.arange(len(data))

    def indirect_stat(rows):
        sample = data.iloc[rows.astype(int)]
        a_b = smf.ols(f"{m} ~ {_rhs(x, covariates=covariates)}", data=sample).fit().params[x]
        b_b = smf.ols(f"{y} ~ {_rhs(x, m, covariates=covariates)}", data=sample).fit().params[m]
        return a_b * b_b

    boot = bootstrap_ci(idx.astype(float), indirect_stat, n_boot=n_boot, alpha=alpha, method=method, seed=seed)

    paths = pd.DataFrame(
        {
            "path": ["a (X→M)", "b (M→Y | X)", "c (total X→Y)", "c' (direct X→Y | M)", "a×b (indirect)"],
            "estimate": [a, b, c, c_prime, indirect],
            "se": [se_a, se_b, total.bse[x], out.bse[x], boot.se],
            "p_value": [med.pvalues[x], out.pvalues[m], total.pvalues[x], out.pvalues[x], sobel_p],
            "ci_low": [*med.conf_int(alpha).loc[x].iloc[:1], *out.conf_int(alpha).loc[m].iloc[:1], *total.conf_int(alpha).loc[x].iloc[:1], *out.conf_int(alpha).loc[x].iloc[:1], boot.ci_low],
            "ci_high": [*med.conf_int(alpha).loc[x].iloc[1:], *out.conf_int(alpha).loc[m].iloc[1:], *total.conf_int(alpha).loc[x].iloc[1:], *out.conf_int(alpha).loc[x].iloc[1:], boot.ci_high],
        }
    ).set_index("path")

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "c_prime": float(c_prime),
        "indirect": float(indirect),
        "indirect_ci": (boot.ci_low, boot.ci_high),
        "indirect_boot_se": boot.se,
        "sobel_z": float(sobel_z),
        "sobel_p": float(sobel_p),
        "proportion_mediated": float(indirect / c) if c != 0 else np.nan,
        "significant_indirect": not (boot.ci_low <= 0 <= boot.ci_high),
        "paths": paths,
        "boot_distribution": boot.distribution,
        "models": {"total": total, "mediator": med, "outcome": out},
    }


def mediation_statsmodels(df: pd.DataFrame, x: str, m: str, y: str, covariates: Sequence[str] | None = None, n_rep: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Potential-outcomes mediation (ACME / ADE) via ``statsmodels.stats.mediation``."""
    from statsmodels.stats.mediation import Mediation

    if seed is not None:
        np.random.seed(seed)  # noqa: NPY002 - statsmodels Mediation uses the global RNG
    outcome_model = sm.OLS.from_formula(f"{y} ~ {_rhs(x, m, covariates=covariates)}", data=df)
    mediator_model = sm.OLS.from_formula(f"{m} ~ {_rhs(x, covariates=covariates)}", data=df)
    res = Mediation(outcome_model, mediator_model, x, m).fit(n_rep=n_rep)
    return res.summary()


def moderation_analysis(
    df: pd.DataFrame,
    x: str,
    w: str,
    y: str,
    covariates: Sequence[str] | None = None,
    center: bool = True,
    alpha: float = 0.05,
    probe_values: Sequence[float] | None = None,
) -> dict:
    """Moderation: ``y ~ x * w`` with simple slopes and the Johnson–Neyman region.

    Simple slopes of ``x`` are probed at ``w`` = mean − 1 SD, mean, mean + 1 SD
    (or ``probe_values``) using exact linear-combination tests. The Johnson–
    Neyman bounds are the values of ``w`` at which the simple slope's t
    statistic equals the critical value.
    """
    data = df.reset_index(drop=True).copy()
    if center:
        data[x] = data[x] - data[x].mean()
        data[w] = data[w] - data[w].mean()
    res = smf.ols(f"{y} ~ {_rhs(f'{x} * {w}', covariates=covariates)}", data=data).fit()
    inter = f"{x}:{w}"
    names = list(res.params.index)
    ix, iw = names.index(x), names.index(inter)

    w_sd = data[w].std(ddof=1)
    probes = list(probe_values) if probe_values is not None else [data[w].mean() - w_sd, data[w].mean(), data[w].mean() + w_sd]
    labels = ["mean - 1 SD", "mean", "mean + 1 SD"] if probe_values is None else [f"{v:g}" for v in probes]
    rows = []
    for lab, val in zip(labels, probes):
        L = np.zeros(len(names))
        L[ix], L[iw] = 1.0, val
        tt = res.t_test(L)
        ci = tt.conf_int(alpha=alpha)
        rows.append({"moderator_level": lab, "w": val, "slope": float(np.squeeze(tt.effect)), "se": float(np.squeeze(tt.sd)), "t": float(np.squeeze(tt.tvalue)), "p_value": float(np.squeeze(tt.pvalue)), "ci_low": float(ci[0, 0]), "ci_high": float(ci[0, 1])})
    simple = pd.DataFrame(rows).set_index("moderator_level")

    V = res.cov_params()
    b1, b3 = res.params[x], res.params[inter]
    v11, v13, v33 = V.loc[x, x], V.loc[x, inter], V.loc[inter, inter]
    tcrit = stats.t.ppf(1 - alpha / 2, res.df_resid)
    qa, qb, qc = b3**2 - tcrit**2 * v33, 2 * b1 * b3 - 2 * tcrit**2 * v13, b1**2 - tcrit**2 * v11
    disc = qb**2 - 4 * qa * qc
    jn = {"bounds": None, "note": "no real roots: simple slope is significant for all w or for none"}
    if qa != 0 and disc >= 0:
        r1, r2 = sorted(((-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)))
        inside = b1 + b3 * (r1 + r2) / 2
        t_mid = inside / np.sqrt(v11 + 2 * ((r1 + r2) / 2) * v13 + ((r1 + r2) / 2) ** 2 * v33)
        region = "outside" if abs(t_mid) < tcrit else "inside"
        jn = {"bounds": (float(r1), float(r2)), "note": f"simple slope of {x} is significant {region} [{r1:.3f}, {r2:.3f}] on the {'centred ' if center else ''}{w} scale"}

    return {
        "result": res,
        "interaction": {"coef": float(b3), "se": float(res.bse[inter]), "p_value": float(res.pvalues[inter])},
        "simple_slopes": simple,
        "johnson_neyman": jn,
        "centered": center,
        "delta_r2_interaction": float(res.rsquared - smf.ols(f"{y} ~ {_rhs(x, w, covariates=covariates)}", data=data).fit().rsquared),
    }
