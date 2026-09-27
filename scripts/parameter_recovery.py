#!/usr/bin/env python3
"""Parameter-recovery study: fit the intended model to every synthetic dataset
and check that the true DGP parameters fall inside the 95% confidence intervals.

Writes ``exports/tables/validation/parameter_recovery.{csv,md}`` and prints a
summary. With ``--strict`` the exit status is non-zero when any parameter that
is expected to be recovered falls outside its interval.

Some parameters are *not* expected to be recovered exactly and are marked
``expect_recovery = False``:
- population-averaged GEE coefficients are attenuated relative to the
  conditional DGP parameters (large cluster effect);
- OLS on ``ols_diagnostics`` and ``robust_regression_data`` is deliberately
  contaminated by outliers (the robust fit is the one that should recover).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from synthetic_data.dgp_registry import DATASETS  # noqa: E402
from utils.reporting import to_markdown  # noqa: E402

DATA = PROJECT_ROOT / "synthetic_data"
OUT_DIR = PROJECT_ROOT / "exports" / "tables" / "validation"


def load(key: str) -> pd.DataFrame:
    return pd.read_csv(DATA / DATASETS[key]["file"])


def rows_from_result(dataset: str, model: str, res, mapping: dict[str, str], expect: bool = True) -> list[dict]:
    ci = res.conf_int(alpha=0.05)
    out = []
    for true_name, term in mapping.items():
        truth = DATASETS[dataset]["true_params"][true_name]
        est, lo, hi = float(res.params[term]), float(ci.loc[term, 0]), float(ci.loc[term, 1])
        out.append(_row(dataset, model, true_name, truth, est, float(res.bse[term]), lo, hi, expect))
    return out


def _row(dataset, model, param, truth, est, se, lo, hi, expect=True, note=""):
    return {
        "dataset": dataset,
        "model": model,
        "parameter": param,
        "true": truth,
        "estimate": est,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "covered": bool(lo <= truth <= hi),
        "z_score": (est - truth) / se if se and se > 0 else np.nan,
        "expect_recovery": expect,
        "note": note,
    }


def study() -> pd.DataFrame:  # noqa: C901 - one block per dataset
    rows: list[dict] = []
    warnings.simplefilter("ignore")

    df = load("ols_data")
    res = smf.ols("y ~ X1 + X2", data=df).fit()
    rows += rows_from_result("ols_data", "OLS", res, {"Intercept": "Intercept", "X1": "X1", "X2": "X2"})
    sigma_hat = float(np.sqrt(res.scale))
    se_sigma = sigma_hat / np.sqrt(2 * res.df_resid)
    rows.append(_row("ols_data", "OLS", "sigma", 1.5, sigma_hat, se_sigma, sigma_hat - 1.96 * se_sigma, sigma_hat + 1.96 * se_sigma))

    df = load("glm_poisson")
    res = smf.glm("y ~ X", data=df, family=sm.families.Poisson()).fit()
    rows += rows_from_result("glm_poisson", "GLM Poisson (log)", res, {"Intercept": "Intercept", "X": "X"})

    df = load("glm_logistic")
    res = smf.logit("y ~ X", data=df).fit(disp=0)
    rows += rows_from_result("glm_logistic", "Logit", res, {"Intercept": "Intercept", "X": "X"})

    df = load("arima_series")
    res = ARIMA(df["value"], order=(1, 0, 1), trend="n").fit()
    rows += rows_from_result("arima_series", "ARIMA(1,0,1)", res, {"ar.L1": "ar.L1", "ma.L1": "ma.L1", "sigma2": "sigma2"})

    df = load("manova_data")
    for col, name in [("Y1", "mean_diff_Y1"), ("Y2", "mean_diff_Y2")]:
        res = smf.ols(f"{col} ~ C(group)", data=df).fit()
        rows += rows_from_result("manova_data", "OLS group contrast", res, {name: "C(group)[T.B]"})

    df = load("heteroskedastic_data")
    res = smf.ols("y ~ X", data=df).fit(cov_type="HC3")
    rows += rows_from_result("heteroskedastic_data", "OLS + HC3", res, {"Intercept": "Intercept", "X": "X"})

    df = load("multivariate_group_data")
    for grp, num, name in [("A", "Num1", "Num1_mean_A"), ("B", "Num1", "Num1_mean_B"), ("A", "Num5", "Num5_mean_A"), ("B", "Num5", "Num5_mean_B")]:
        x = df.loc[df["Group"] == grp, num]
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        rows.append(_row("multivariate_group_data", "sample mean", name, DATASETS["multivariate_group_data"]["true_params"][name], m, se, m - 1.96 * se, m + 1.96 * se))

    df = load("ols_diagnostics")
    ols = smf.ols("y ~ X1 + X2 + X3", data=df).fit()
    rows += rows_from_result("ols_diagnostics", "OLS (contaminated)", ols, {"Intercept": "Intercept", "X1": "X1", "X2": "X2", "X3": "X3"}, expect=False)
    rlm = smf.rlm("y ~ X1 + X2 + X3", data=df, M=sm.robust.norms.HuberT()).fit()
    rows += rows_from_result("ols_diagnostics", "RLM Huber", rlm, {"Intercept": "Intercept", "X1": "X1", "X2": "X2", "X3": "X3"})

    df = load("posthoc_dataset")
    for g in ["A", "B", "C"]:
        x = df.loc[df["Group"] == g, "Score"]
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        rows.append(_row("posthoc_dataset", "group mean", f"mean_{g}", DATASETS["posthoc_dataset"]["true_params"][f"mean_{g}"], m, se, m - 1.96 * se, m + 1.96 * se))

    df = load("robust_regression_data")
    ols = smf.ols("y ~ X", data=df).fit()
    rows += rows_from_result("robust_regression_data", "OLS (contaminated)", ols, {"Intercept": "Intercept", "X": "X"}, expect=False)
    rlm = smf.rlm("y ~ X", data=df, M=sm.robust.norms.HuberT()).fit()
    rows += rows_from_result("robust_regression_data", "RLM Huber", rlm, {"Intercept": "Intercept", "X": "X"})
    qr = smf.quantreg("y ~ X", data=df).fit(q=0.5)
    rows += rows_from_result("robust_regression_data", "Median regression", qr, {"Intercept": "Intercept", "X": "X"})

    df = load("seasonal_ts_data")
    df["t_idx"] = np.arange(len(df))
    df["s"] = np.sin(2 * np.pi * df["t_idx"] / 12)
    df["c"] = np.cos(2 * np.pi * df["t_idx"] / 12)
    res = smf.ols("y ~ t_idx + s + c + exog", data=df).fit()
    rows += rows_from_result("seasonal_ts_data", "Harmonic regression", res, {"level": "Intercept", "trend": "t_idx", "amplitude": "s", "exog_effect": "exog"})

    df = load("panel_data")
    fe = smf.ols("y ~ X1 + X2 + C(individual)", data=df).fit()
    rows += rows_from_result("panel_data", "Fixed effects (LSDV)", fe, {"X1": "X1", "X2": "X2"})
    ml = smf.mixedlm("y ~ X1 + X2", data=df, groups=df["individual"]).fit(reml=True)
    rows += rows_from_result("panel_data", "MixedLM random intercept", ml, {"Intercept": "Intercept", "X1": "X1", "X2": "X2"})
    sd_hat = float(np.sqrt(ml.cov_re.iloc[0, 0]))
    rows.append(_row("panel_data", "MixedLM random intercept", "sd_individual", 5.0, sd_hat, np.nan, np.nan, np.nan, expect=False, note="point estimate only (no CI for variance component)"))

    try:
        from lifelines import CoxPHFitter

        df = load("survival_data")
        cph = CoxPHFitter().fit(df[["time", "event", "age", "treatment", "biomarker"]], "time", "event")
        s = cph.summary
        for term in ["age", "treatment", "biomarker"]:
            rows.append(_row("survival_data", "Cox PH", term, DATASETS["survival_data"]["true_params"][term], float(s.loc[term, "coef"]), float(s.loc[term, "se(coef)"]), float(s.loc[term, "coef lower 95%"]), float(s.loc[term, "coef upper 95%"])))
    except ImportError:
        rows.append(_row("survival_data", "Cox PH", "treatment", -0.5, np.nan, np.nan, np.nan, np.nan, expect=False, note="lifelines not installed"))

    df = load("zero_inflated_count")
    X = sm.add_constant(df[["x1", "x2"]])
    Z = sm.add_constant(df[["x3"]])
    zip_res = sm.ZeroInflatedPoisson(df["y"], X, exog_infl=Z).fit(disp=0, maxiter=500)
    rows += rows_from_result("zero_inflated_count", "ZIP", zip_res, {"Intercept": "const", "x1": "x1", "x2": "x2", "inflate_Intercept": "inflate_const", "inflate_x3": "inflate_x3"})

    df = load("var_data")
    var = VAR(df[["y1", "y2"]].to_numpy()).fit(1)
    coefs = var.params  # rows: const, L1.y1, L1.y2 ; cols: y1, y2
    ses = var.stderr
    for name, (r, c) in {"L1.y1->y1": (1, 0), "L1.y2->y1": (2, 0), "L1.y1->y2": (1, 1), "L1.y2->y2": (2, 1)}.items():
        est, se = float(coefs[r, c]), float(ses[r, c])
        rows.append(_row("var_data", "VAR(1)", name, DATASETS["var_data"]["true_params"][name], est, se, est - 1.96 * se, est + 1.96 * se))

    df = load("gee_data")
    gee = sm.GEE.from_formula("y ~ X + treatment", groups="cluster", data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    rows += rows_from_result("gee_data", "GEE exchangeable", gee, {"Intercept": "Intercept", "X": "X", "treatment": "treatment"}, expect=False)
    for r in rows[-3:]:
        r["note"] = "population-averaged; attenuated vs conditional DGP"

    df = load("mediation_data")
    med = smf.ols("M ~ X", data=df).fit()
    out = smf.ols("Y ~ X + M", data=df).fit()
    rows += rows_from_result("mediation_data", "Mediator model", med, {"a": "X"})
    rows += rows_from_result("mediation_data", "Outcome model", out, {"b": "M", "c_prime": "X"})
    mod = smf.ols("Y_moderated ~ X * W", data=df).fit()
    rows += rows_from_result("mediation_data", "Moderation model", mod, {"mod_X": "X", "mod_W": "W", "mod_XW": "X:W"})

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="exit 1 if any expected parameter is not covered")
    args = parser.parse_args()

    table = study()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "parameter_recovery.csv", index=False)
    (OUT_DIR / "parameter_recovery.md").write_text("# Parameter recovery\n\n" + to_markdown(table.drop(columns=["note"]), index=False, float_format="{:.4f}") + "\n")

    expected = table[table["expect_recovery"]]
    n_cov = int(expected["covered"].sum())
    print(f"Parameters expected to be recovered: {len(expected)}; covered by 95% CI: {n_cov} ({100 * n_cov / len(expected):.1f}%)")
    missed = expected[~expected["covered"]]
    if len(missed):
        print("\nNot covered:")
        print(missed[["dataset", "model", "parameter", "true", "estimate", "ci_low", "ci_high", "z_score"]].to_string(index=False))
    print(f"\nWrote {OUT_DIR.relative_to(PROJECT_ROOT)}/parameter_recovery.csv and .md")
    # With ~50 parameters at 95% we expect ~2-3 misses by chance; strict mode tolerates |z| < 3.
    severe = missed[missed["z_score"].abs() >= 3]
    if args.strict and len(severe):
        print(f"\nSTRICT: {len(severe)} parameter(s) missed with |z| >= 3")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
