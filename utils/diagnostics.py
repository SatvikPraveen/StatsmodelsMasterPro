# diagnostics.py

"""
Diagnostic functions for computing distribution shape metrics using pandas,
to align with the StatsmodelsMasterPro philosophy — no scipy used.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as sm


def compute_skewness_kurtosis(df, cols):
    """
    Compute skewness and Pearson-style kurtosis for selected columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cols (list): List of numeric column names

    Returns:
        dict: Dictionary with skewness and kurtosis per column
    """
    result = {}
    for col in cols:
        result[col] = {
            'skewness': df[col].skew(skipna=True),
            'kurtosis': df[col].kurt(skipna=True) + 3  # Convert to Pearson-style
        }
    return result


def plot_fitted_vs_actual(y_true, y_pred, title="Fitted vs Actual"):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=y_true)
    plt.xlabel("Fitted Values")
    plt.ylabel("Actual Values")
    plt.title(title)
    plt.axline((0, 0), slope=1, color='red', linestyle='--')
    plt.tight_layout()


def plot_residuals(model, title="Residuals vs Fitted"):
    plt.figure(figsize=(6, 4))
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()



def plot_residual_histogram(model, title="Residual Histogram"):
    plt.figure(figsize=(6, 4))
    sns.histplot(model.resid, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()


def plot_acf_pacf(series, lags=40, title_prefix=""):
    """
    Plot ACF and PACF side by side
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix} ACF")

    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title(f"{title_prefix} PACF")

    plt.tight_layout()


def plot_qq_residuals(model, title="Q–Q Plot of Residuals"):

    fig = sm.qqplot(model.resid, line='45', fit=True)
    plt.title(title)
    plt.tight_layout()


def plot_leverage_cooks(model, title="Influence Plot"):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    influence_plot(model, ax=ax)
    plt.title(title)
    plt.tight_layout()


def run_heteroskedasticity_tests(model):
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    residuals = model.resid
    exog = model.model.exog
    
    bp_test = het_breuschpagan(residuals, exog)
    white_test = het_white(residuals, exog)
    
    return {
        "Breusch-Pagan": {
            "LM Stat": bp_test[0],
            "p-value": bp_test[1]
        },
        "White": {
            "Stat": white_test[0],
            "p-value": white_test[1]
        }
    }




# =========================================================================== #
# Research-grade diagnostic battery
# =========================================================================== #
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as _stats  # noqa: E402
from statsmodels.stats.diagnostic import (  # noqa: E402
    acorr_ljungbox,
    het_breuschpagan,
    het_white,
    linear_harvey_collier,
    linear_rainbow,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402
from statsmodels.stats.stattools import durbin_watson, jarque_bera  # noqa: E402


def vif_table(exog, drop_constant: bool = True) -> pd.DataFrame:
    """Variance inflation factors and tolerances for a design matrix.

    ``exog`` may be a DataFrame of predictors or a fitted results object, in
    which case the model's design matrix is used.
    """
    if hasattr(exog, "model"):
        X = pd.DataFrame(exog.model.exog, columns=exog.model.exog_names)
    else:
        X = pd.DataFrame(exog).astype(float)
    if drop_constant:
        X = X.loc[:, X.std(ddof=0) > 0]
    if X.shape[1] < 2:
        return pd.DataFrame({"feature": X.columns, "vif": [1.0] * X.shape[1], "tolerance": [1.0] * X.shape[1]})
    vals = [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])]
    out = pd.DataFrame({"feature": X.columns, "vif": vals})
    out["tolerance"] = 1 / out["vif"]
    out["flag"] = out["vif"] > 10
    return out.sort_values("vif", ascending=False).reset_index(drop=True)


def influence_summary(result, cooks_threshold: float | None = None) -> pd.DataFrame:
    """Per-observation influence measures with conventional cut-off flags.

    Flags: Cook's D > 4/n, leverage > 2p/n, |studentized residual| > 3,
    |DFFITS| > 2·sqrt(p/n) (Belsley, Kuh & Welsch, 1980).
    """
    infl = result.get_influence()
    n = int(result.nobs)
    p = int(result.df_model) + 1
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    stud = infl.resid_studentized_external
    dffits = infl.dffits[0]
    thr_cooks = cooks_threshold if cooks_threshold is not None else 4.0 / n
    out = pd.DataFrame(
        {
            "leverage": leverage,
            "cooks_d": cooks,
            "studentized_resid": stud,
            "dffits": dffits,
        }
    )
    out["flag_cooks"] = out["cooks_d"] > thr_cooks
    out["flag_leverage"] = out["leverage"] > 2.0 * p / n
    out["flag_outlier"] = out["studentized_resid"].abs() > 3
    out["flag_dffits"] = out["dffits"].abs() > 2 * np.sqrt(p / n)
    out["any_flag"] = out[["flag_cooks", "flag_leverage", "flag_outlier", "flag_dffits"]].any(axis=1)
    out.attrs["thresholds"] = {
        "cooks_d": thr_cooks,
        "leverage": 2.0 * p / n,
        "studentized_resid": 3.0,
        "dffits": 2 * np.sqrt(p / n),
    }
    return out


def diagnostic_battery(result, lags: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """Run a full battery of OLS assumption tests and return a tidy table.

    Columns: ``test, statistic, p_value, null_hypothesis, flag, note``. ``flag``
    is True when the null (i.e. the assumption) is rejected at ``alpha`` or,
    for statistics without a p-value, when a conventional rule of thumb is
    violated.

    Tests
    -----
    - Heteroskedasticity: Breusch–Pagan (1979), White (1980)
    - Normality: Jarque–Bera (1987), D'Agostino–Pearson omnibus
    - Autocorrelation: Durbin–Watson (1950), Ljung–Box (1978)
    - Functional form: Rainbow (Utts, 1982), Ramsey RESET (1969), Harvey–Collier (1977)
    - Multicollinearity: condition number, maximum VIF
    """
    resid = np.asarray(result.resid)
    exog = np.asarray(result.model.exog)
    rows = []

    def add(test, stat, p, null, flag, note=""):
        rows.append(
            {
                "test": test,
                "statistic": float(stat) if stat is not None else np.nan,
                "p_value": float(p) if p is not None else np.nan,
                "null_hypothesis": null,
                "flag": bool(flag),
                "note": note,
            }
        )

    lm, lm_p, _, _ = het_breuschpagan(resid, exog)
    add("Breusch-Pagan", lm, lm_p, "Homoskedastic errors", lm_p < alpha)
    try:
        w, w_p, _, _ = het_white(resid, exog)
        add("White", w, w_p, "Homoskedastic errors", w_p < alpha)
    except Exception as exc:  # pragma: no cover - singular cross products
        add("White", None, None, "Homoskedastic errors", False, f"not computed: {exc}")

    jb, jb_p, skew, kurt = jarque_bera(resid)
    add("Jarque-Bera", jb, jb_p, "Normal errors", jb_p < alpha, f"skew={skew:.2f}, kurtosis={kurt:.2f}")
    if len(resid) >= 20:
        om, om_p = _stats.normaltest(resid)
        add("Omnibus (D'Agostino-Pearson)", om, om_p, "Normal errors", om_p < alpha)

    dw = durbin_watson(resid)
    add("Durbin-Watson", dw, None, "No first-order autocorrelation", not (1.5 <= dw <= 2.5), "rule of thumb: 1.5-2.5")
    lb = acorr_ljungbox(resid, lags=[min(lags, len(resid) // 2 - 1)], return_df=True)
    add("Ljung-Box", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0], "No autocorrelation up to lag", lb["lb_pvalue"].iloc[0] < alpha, f"lags={lags}")

    try:
        rb, rb_p = linear_rainbow(result)
        add("Rainbow", rb, rb_p, "Linear specification", rb_p < alpha)
    except Exception as exc:  # pragma: no cover
        add("Rainbow", None, None, "Linear specification", False, f"not computed: {exc}")
    try:
        rs = linear_reset(result, power=3, use_f=True)
        add("Ramsey RESET", rs.statistic, rs.pvalue, "No omitted nonlinearity", rs.pvalue < alpha, "powers 2-3")
    except Exception as exc:  # pragma: no cover
        add("Ramsey RESET", None, None, "No omitted nonlinearity", False, f"not computed: {exc}")
    try:
        hc, hc_p = linear_harvey_collier(result)
        add("Harvey-Collier", hc, hc_p, "Linear specification", hc_p < alpha)
    except Exception as exc:
        add("Harvey-Collier", None, None, "Linear specification", False, f"not computed: {type(exc).__name__}")

    cond = float(getattr(result, "condition_number", np.linalg.cond(exog)))
    add("Condition number", cond, None, "Well-conditioned design", cond > 30, "rule of thumb: > 30 indicates collinearity")
    if exog.shape[1] > 2:
        vif = vif_table(result)
        add("Max VIF", vif["vif"].max(), None, "No severe multicollinearity", vif["vif"].max() > 10, f"feature={vif.iloc[0]['feature']}")

    table = pd.DataFrame(rows)
    table.attrs["alpha"] = alpha
    table.attrs["nobs"] = int(result.nobs)
    return table
