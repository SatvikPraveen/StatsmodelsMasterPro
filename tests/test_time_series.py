import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA

from utils import time_series_utils as tsu


@pytest.fixture
def ar1():
    gen = np.random.default_rng(1)
    n = 300
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + gen.normal()
    return y


@pytest.fixture
def random_walk():
    gen = np.random.default_rng(2)
    return np.cumsum(gen.normal(size=300))


def test_stationarity_tests(ar1, random_walk):
    stat = tsu.stationarity_tests(ar1)
    assert list(stat.index) == ["ADF", "KPSS"]
    assert stat.attrs["conclusion"] == "stationary"
    rw = tsu.stationarity_tests(random_walk)
    assert rw.attrs["conclusion"].startswith("non-stationary") or rw.attrs["conclusion"].startswith("conflicting")
    assert rw.loc["KPSS", "reject"]


def test_auto_arima_order_prefers_ar1(ar1):
    best, table = tsu.auto_arima_order(ar1, p=range(0, 3), d=range(0, 2), q=range(0, 2), criterion="bic")
    assert best[0] >= 1 and best[1] == 0
    assert len(table) == 12 and table.attrs["criterion"] == "bic"
    assert table["bic"].is_monotonic_increasing
    with pytest.raises(ValueError):
        tsu.auto_arima_order(ar1, criterion="rmse")


def test_residual_diagnostics(ar1):
    res = ARIMA(ar1, order=(1, 0, 0)).fit()
    diag = tsu.residual_diagnostics(res, lags=8)
    assert list(diag.index)[0] == "Ljung-Box"
    assert diag["p_value"].between(0, 1).all() and (diag["statistic"] >= 0).all()


def test_rolling_origin_evaluation(ar1):
    ev = tsu.rolling_origin_evaluation(ar1, order=(1, 0, 0), initial=200, horizon=1, step=5)
    assert len(ev) == 20
    assert (ev["target"] == ev["origin"]).all()
    assert ev.attrs["rmse"] > 0 and 0 < ev.attrs["mase"] < 2
    refit = tsu.rolling_origin_evaluation(ar1, order=(1, 0, 0), initial=250, horizon=3, step=10, refit=True, max_origins=3)
    assert len(refit) == 3 and (refit["target"] - refit["origin"] == 2).all()
    with pytest.raises(ValueError):
        tsu.rolling_origin_evaluation(ar1, (1, 0, 0), initial=299, horizon=5)


def test_diebold_mariano():
    gen = np.random.default_rng(3)
    e_good = gen.normal(scale=1.0, size=200)
    e_bad = gen.normal(scale=2.0, size=200)
    res = tsu.diebold_mariano(e_good, e_bad)
    assert res["dm_statistic"] < 0 and res["p_value"] < 0.01
    same = tsu.diebold_mariano(e_good, e_good + gen.normal(scale=0.01, size=200), h=2, loss="absolute")
    assert same["p_value"] > 0.05
    assert tsu.diebold_mariano(e_good, e_bad, alternative="less")["p_value"] < 0.01
    with pytest.raises(ValueError):
        tsu.diebold_mariano(e_good, e_bad, loss="huber")


def test_granger_matrix(data_dir):
    df = pd.read_csv(data_dir / "var_data.csv")
    pm = tsu.granger_causality_matrix(df[["y1", "y2"]], maxlag=2)
    assert pm.shape == (2, 2) and np.isnan(pm.loc["y1", "y1"])
    # DGP: y1 depends on lagged y2 (0.2) and y2 on lagged y1 (0.3)
    assert pm.loc["y2", "y1"] < 0.05
    assert pm.loc["y1", "y2"] < 0.05


def test_forecast_accuracy():
    acc = tsu.forecast_accuracy([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8], training=[0, 1, 2, 3, 4])
    assert acc["rmse"] == pytest.approx(np.sqrt(np.mean([0.01, 0.01, 0.04, 0.04])))
    assert acc["mase"] == pytest.approx(0.15)
    assert acc["smape"] > 0
