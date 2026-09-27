import numpy as np
import pandas as pd
import pytest
from scipy import stats

from utils import simulation as sim


def test_monte_carlo_reproducible_and_records_errors():
    dgp = sim.dgp_linear(n=50)
    est = sim.ols_estimator("y ~ X1 + X2")
    a = sim.monte_carlo(dgp, est, n_reps=5, seed=1)
    b = sim.monte_carlo(dgp, est, n_reps=5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["n_reps"] == 5 and a["error"].isna().all()

    def bad(data):
        raise RuntimeError("boom")

    c = sim.monte_carlo(dgp, bad, n_reps=3, seed=0)
    assert c["error"].str.startswith("RuntimeError").all()


def test_performance_summary_ols_unbiased_with_nominal_coverage():
    res = sim.monte_carlo(sim.dgp_linear(n=80, beta=(2, 1.5, -0.7), sigma=1.5), sim.ols_estimator("y ~ X1 + X2"), n_reps=300, seed=2)
    perf = sim.performance_summary(res, {"beta_X1": 1.5, "beta_X2": -0.7, "beta_Intercept": 2.0})
    for name in ["beta_X1", "beta_X2"]:
        row = perf.loc[name]
        assert abs(row["bias"]) < 3 * row["bias_mcse"] + 0.02
        assert 0.9 <= row["coverage"] <= 0.99
        assert 0.85 < row["se_ratio"] < 1.15
        assert row["rejection_rate"] > 0.9
    assert perf.attrs["n_failed"] == 0
    assert {"rmse", "empirical_se_mcse", "coverage_mcse", "nominal_coverage"} <= set(perf.columns)


def test_nonrobust_se_undercovers_under_heteroskedasticity_but_hc3_fixes_it():
    dgp = sim.dgp_linear(n=60, beta=(1, 1), sigma=1.0, heteroskedastic=True)
    naive = sim.performance_summary(sim.monte_carlo(dgp, sim.ols_estimator("y ~ X1"), n_reps=300, seed=3), {"beta_X1": 1.0})
    hc3 = sim.performance_summary(sim.monte_carlo(dgp, sim.ols_estimator("y ~ X1", cov_type="HC3"), n_reps=300, seed=3), {"beta_X1": 1.0})
    assert naive.loc["beta_X1", "coverage"] < hc3.loc["beta_X1", "coverage"]
    assert hc3.loc["beta_X1", "coverage"] > 0.9


def test_simulate_power_type1_and_increasing_power():
    curve = sim.simulate_power(lambda d: sim.dgp_two_sample(30, 30, d), lambda data: stats.ttest_ind(*data).pvalue, effects=[0.0, 0.5, 1.0], n_reps=300, seed=4)
    assert list(curve["effect"]) == [0.0, 0.5, 1.0]
    assert 0.02 <= curve.loc[0, "power"] <= 0.09
    assert curve["power"].is_monotonic_increasing
    assert curve.loc[2, "power"] > 0.9
    assert (curve["mcse"] >= 0).all()


def test_dgps_and_estimators():
    rng = np.random.default_rng(0)
    x, y = sim.dgp_two_sample(20, 25, 0.3, dist="lognormal")(rng)
    assert len(x) == 20 and len(y) == 25
    row = sim.ttest_estimator(equal_var=False)((x, y))
    assert row["diff_ci_low"] <= row["diff"] <= row["diff_ci_high"]
    df = sim.dgp_poisson(100, (0.5, 0.9), overdispersion=0.5)(rng)
    assert (df["y"] >= 0).all() and np.issubdtype(df["y"].dtype, np.integer)
    glm_row = sim.glm_estimator("y ~ X")(df)
    assert "beta_X" in glm_row and "p_beta_X" in glm_row
    skewed = sim.dgp_linear(n=30, beta=(0, 1), error="skewed")(rng)
    assert skewed.shape == (30, 2)
    with pytest.raises(ValueError):
        sim.dgp_linear(error="cauchy")(rng)
    with pytest.raises(ValueError):
        sim.dgp_two_sample(dist="beta")(rng)


def test_rejection_rate_handles_nan():
    rate, mcse = sim.rejection_rate([0.01, 0.2, np.nan, 0.03], alpha=0.05)
    assert rate == pytest.approx(2 / 3) and mcse > 0
