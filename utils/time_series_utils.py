"""Time-series testing, order selection, and out-of-sample forecast evaluation.

* ``stationarity_tests`` – ADF (Dickey & Fuller, 1979) and KPSS
  (Kwiatkowski et al., 1992) reported together with a joint decision rule.
* ``auto_arima_order`` – information-criterion grid search over (p, d, q).
* ``residual_diagnostics`` – Ljung–Box, Jarque–Bera, and heteroskedasticity
  tests on fitted ARIMA/SARIMAX residuals.
* ``rolling_origin_evaluation`` – expanding-window out-of-sample forecasts
  (Tashman, 2000).
* ``diebold_mariano`` – test of equal predictive accuracy with the
  Harvey–Leybourne–Newbold (1997) small-sample correction.
* ``granger_causality_matrix`` – pairwise Granger (1969) causality p-values.
* ``forecast_accuracy`` – RMSE, MAE, MAPE, sMAPE, MASE.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

__all__ = [
    "stationarity_tests",
    "auto_arima_order",
    "residual_diagnostics",
    "rolling_origin_evaluation",
    "diebold_mariano",
    "granger_causality_matrix",
    "forecast_accuracy",
]


def stationarity_tests(series, regression: str = "c", alpha: float = 0.05) -> pd.DataFrame:
    """ADF and KPSS tests with a combined conclusion in ``table.attrs['conclusion']``.

    Decision rule (both at level ``alpha``):
    - ADF rejects unit root & KPSS does not reject stationarity → *stationary*
    - ADF does not reject & KPSS rejects → *non-stationary (difference the series)*
    - both reject → *trend-stationary or structural break (inspect)*
    - neither rejects → *inconclusive (low power)*
    """
    x = np.asarray(pd.Series(series).dropna(), float)
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(x, regression=regression, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kp_stat, kp_p, kp_lags, kp_crit = kpss(x, regression=regression, nlags="auto")
    table = pd.DataFrame(
        [
            {"test": "ADF", "statistic": adf_stat, "p_value": adf_p, "lags": adf_lags, "null_hypothesis": "unit root (non-stationary)", "reject": adf_p < alpha, "crit_5%": adf_crit["5%"]},
            {"test": "KPSS", "statistic": kp_stat, "p_value": kp_p, "lags": kp_lags, "null_hypothesis": "stationary", "reject": kp_p < alpha, "crit_5%": kp_crit["5%"]},
        ]
    ).set_index("test")
    adf_rej, kp_rej = adf_p < alpha, kp_p < alpha
    if adf_rej and not kp_rej:
        conclusion = "stationary"
    elif not adf_rej and kp_rej:
        conclusion = "non-stationary: difference the series"
    elif adf_rej and kp_rej:
        conclusion = "conflicting: possibly trend-stationary or a structural break"
    else:
        conclusion = "inconclusive: tests lack power at this sample size"
    table.attrs["conclusion"] = conclusion
    return table


def auto_arima_order(
    series,
    p: Iterable[int] = range(0, 4),
    d: Iterable[int] = range(0, 3),
    q: Iterable[int] = range(0, 4),
    criterion: str = "aic",
    seasonal_order: tuple | None = None,
    exog=None,
    trend: str | None = None,
) -> tuple[tuple[int, int, int], pd.DataFrame]:
    """Grid search over ARIMA orders; returns the best order and the full table."""
    if criterion not in {"aic", "bic", "aicc", "hqic"}:
        raise ValueError("criterion must be 'aic', 'bic', 'aicc', or 'hqic'.")
    rows = []
    for order in itertools.product(p, d, q):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(series, order=order, seasonal_order=seasonal_order or (0, 0, 0, 0), exog=exog, trend=trend).fit()
            rows.append({"order": order, "aic": res.aic, "bic": res.bic, "aicc": res.aicc, "hqic": res.hqic, "llf": res.llf, "converged": bool(res.mle_retvals.get("converged", True))})
        except Exception as exc:  # noqa: BLE001 - keep searching
            rows.append({"order": order, "aic": np.nan, "bic": np.nan, "aicc": np.nan, "hqic": np.nan, "llf": np.nan, "converged": False, "error": str(exc)[:60]})
    table = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)
    best = tuple(table.iloc[0]["order"])
    table.attrs["criterion"] = criterion
    return best, table


def residual_diagnostics(result, lags: int = 10) -> pd.DataFrame:
    """Ljung–Box, Jarque–Bera, and heteroskedasticity tests on a fitted state-space model."""
    lb = result.test_serial_correlation("ljungbox", lags=lags)[0]
    jb = result.test_normality("jarquebera")[0]
    het = result.test_heteroskedasticity("breakvar")[0]
    return pd.DataFrame(
        [
            {"test": "Ljung-Box", "statistic": float(lb[0][-1]), "p_value": float(lb[1][-1]), "null_hypothesis": "no residual autocorrelation"},
            {"test": "Jarque-Bera", "statistic": float(jb[0]), "p_value": float(jb[1]), "null_hypothesis": "normal residuals"},
            {"test": "Heteroskedasticity (break-variance)", "statistic": float(het[0]), "p_value": float(het[1]), "null_hypothesis": "constant residual variance"},
        ]
    ).set_index("test")


def forecast_accuracy(actual, predicted, training=None, seasonality: int = 1) -> dict:
    """Point-forecast accuracy metrics. ``MASE`` needs the training series (Hyndman & Koehler, 2006)."""
    a, f = np.asarray(actual, float), np.asarray(predicted, float)
    e = a - f
    out = {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "mape": float(np.mean(np.abs(e / a)) * 100) if np.all(a != 0) else np.nan,
        "smape": float(np.mean(2 * np.abs(e) / (np.abs(a) + np.abs(f))) * 100),
        "bias": float(np.mean(e)),
    }
    if training is not None:
        tr = np.asarray(training, float)
        scale = np.mean(np.abs(tr[seasonality:] - tr[:-seasonality]))
        out["mase"] = float(np.mean(np.abs(e)) / scale) if scale > 0 else np.nan
    return out


def rolling_origin_evaluation(
    series,
    order: tuple[int, int, int],
    initial: int,
    horizon: int = 1,
    step: int = 1,
    seasonal_order: tuple | None = None,
    refit: bool = False,
    max_origins: int | None = None,
) -> pd.DataFrame:
    """Expanding-window ``horizon``-step-ahead forecasts from successive origins.

    With ``refit=False`` the model is estimated once on the initial window and
    parameters are held fixed while the state is updated (fast); with
    ``refit=True`` it is re-estimated at every origin.
    """
    y = np.asarray(pd.Series(series).dropna(), float)
    if initial + horizon > len(y):
        raise ValueError("initial + horizon must not exceed the series length.")
    origins = list(range(initial, len(y) - horizon + 1, step))
    if max_origins is not None:
        origins = origins[:max_origins]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = ARIMA(y[:initial], order=order, seasonal_order=seasonal_order or (0, 0, 0, 0)).fit()
    rows = []
    for t in origins:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if refit:
                res = ARIMA(y[:t], order=order, seasonal_order=seasonal_order or (0, 0, 0, 0)).fit()
            else:
                res = base.apply(y[:t])
            fc = np.asarray(res.forecast(steps=horizon))
        rows.append({"origin": t, "target": t + horizon - 1, "forecast": float(fc[-1]), "actual": float(y[t + horizon - 1])})
    table = pd.DataFrame(rows)
    table["error"] = table["actual"] - table["forecast"]
    table.attrs.update(forecast_accuracy(table["actual"], table["forecast"], training=y[:initial]))
    table.attrs["horizon"] = horizon
    return table


def diebold_mariano(e1, e2, h: int = 1, loss: str = "squared", alternative: str = "two-sided") -> dict:
    """Diebold–Mariano test for equal forecast accuracy of two forecast error series.

    A negative statistic favours the first forecast. Uses a rectangular HAC
    variance with ``h - 1`` autocovariances and the Harvey–Leybourne–Newbold
    small-sample correction with a t(n − 1) reference distribution.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'.")
    n = len(d)
    d_bar = d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[: n - k] - d_bar)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n
    dm = d_bar / np.sqrt(var_d)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(hln), n - 1)
    elif alternative == "less":
        p = stats.t.cdf(hln, n - 1)
    elif alternative == "greater":
        p = stats.t.sf(hln, n - 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")
    return {"dm_statistic": float(dm), "hln_statistic": float(hln), "p_value": float(p), "mean_loss_differential": float(d_bar), "n": n, "h": h, "loss": loss}


def granger_causality_matrix(df: pd.DataFrame, maxlag: int = 4, test: str = "ssr_chi2test", columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Matrix of minimum Granger-causality p-values over lags 1..maxlag.

    Entry ``[row, col]`` is the p-value for "``col`` Granger-causes ``row``".
    """
    cols = list(columns or df.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grangercausalitytests(df[[caused, causing]], maxlag=maxlag)
            out.loc[caused, causing] = min(res[lag][0][test][1] for lag in res)
    out.index.name = "caused (y)"
    out.columns.name = "causing (x)"
    return out
