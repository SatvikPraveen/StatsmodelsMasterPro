import numpy as np
import pandas as pd
import pytest

from utils import causal


@pytest.fixture
def confounded():
    """Treatment depends on x; true ATE = 2.0; naive difference is biased upward."""
    gen = np.random.default_rng(10)
    n = 2000
    x = gen.normal(size=n)
    p = 1 / (1 + np.exp(-(0.8 * x)))
    t = gen.binomial(1, p)
    y = 1 + 2.0 * t + 1.5 * x + gen.normal(size=n)
    return pd.DataFrame({"x": x, "t": t, "y": y})


def test_propensity_and_balance(confounded):
    ps = causal.estimate_propensity(confounded, "t", ["x"])
    assert ps.between(0, 1).all() and ps.name == "pscore"
    before = causal.covariate_balance(confounded, "t", ["x"])
    assert abs(before.loc["x", "smd"]) > 0.3 and not before.loc["x", "balanced"]


def test_ipw_removes_confounding(confounded):
    res = causal.ipw(confounded, "y", "t", ["x"], n_boot=30, seed=1)
    assert res["naive_difference"] > 2.4
    assert res["estimate"] == pytest.approx(2.0, abs=0.2)
    assert res["ci_low"] < 2.0 < res["ci_high"]
    assert abs(res["balance_after"].loc["x", "smd"]) < 0.1
    assert res["boot_se"] > 0 and res["boot_ci_low"] < res["boot_ci_high"]
    att = causal.ipw(confounded, "y", "t", ["x"], estimand="ATT", stabilized=False)
    assert att["estimate"] == pytest.approx(2.0, abs=0.25)
    with pytest.raises(ValueError):
        causal.ipw(confounded, "y", "t", ["x"], estimand="LATE")


def test_matching(confounded):
    ps = causal.estimate_propensity(confounded, "t", ["x"])
    matched = causal.nearest_neighbor_match(confounded, "t", ps, caliper=0.2, seed=3)
    assert matched.attrs["n_matched"] > 100
    assert (matched.groupby("match_id").size() == 2).all()
    att = causal.matching_att(matched, "y", "t")
    assert att["estimate"] == pytest.approx(2.0, abs=0.3)
    assert att["n_pairs"] == matched.attrs["n_matched"]
    with_rep = causal.nearest_neighbor_match(confounded, "t", ps, caliper=None, replace=True)
    assert with_rep.attrs["n_matched"] == with_rep.attrs["n_treated"]


def test_difference_in_differences():
    gen = np.random.default_rng(4)
    units, periods = 40, 2
    rows = []
    for u in range(units):
        treated = int(u < units / 2)
        fe = gen.normal()
        for p in range(periods):
            y = 1 + fe + 0.5 * p + 3.0 * treated * p + 0.7 * treated + gen.normal(scale=0.5)
            rows.append({"unit": u, "treated": treated, "post": p, "y": y})
    df = pd.DataFrame(rows)
    res = causal.difference_in_differences(df, "y", "treated", "post", cluster="unit")
    assert res["estimate"] == pytest.approx(3.0, abs=0.4)
    assert res["manual_did"] == pytest.approx(res["estimate"], abs=1e-8)
    assert res["p_value"] < 1e-4
    hc = causal.difference_in_differences(df, "y", "treated", "post")
    assert hc["estimate"] == pytest.approx(res["estimate"])


def test_two_stage_least_squares():
    gen = np.random.default_rng(6)
    n = 3000
    u = gen.normal(size=n)  # unobserved confounder
    z = gen.normal(size=n)
    x = 0.8 * z + u + gen.normal(scale=0.5, size=n)
    y = 1 + 1.0 * x + 2 * u + gen.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    res = causal.two_stage_least_squares(df, "y", "x", ["z"])
    assert res["ols_estimate"] > 1.4  # biased by u
    assert res["estimate"] == pytest.approx(1.0, abs=0.15)
    assert res["first_stage_f"] > 10 and not res["weak_instruments"]
    assert res["ci_low"] < 1.0 < res["ci_high"]


def test_regression_discontinuity():
    gen = np.random.default_rng(7)
    n = 3000
    r = gen.uniform(-1, 1, n)
    y = 1 + 0.5 * r + 2.0 * (r >= 0) + gen.normal(scale=0.3, size=n)
    df = pd.DataFrame({"r": r, "y": y})
    res = causal.regression_discontinuity(df, "y", "r", cutoff=0.0)
    assert res["estimate"] == pytest.approx(2.0, abs=0.2)
    assert res["n_left"] > 0 and res["n_right"] > 0 and res["bandwidth"] > 0
    sens = causal.rd_bandwidth_sensitivity(df, "y", "r", bandwidths=[0.2, 0.4, 0.8], kernel="uniform")
    assert len(sens) == 3 and sens["estimate"].between(1.7, 2.3).all()
    quad = causal.regression_discontinuity(df, "y", "r", polynomial=2, kernel="epanechnikov", bandwidth=0.5)
    assert quad["estimate"] == pytest.approx(2.0, abs=0.25)
    with pytest.raises(ValueError):
        causal.regression_discontinuity(df, "y", "r", kernel="gaussian")


def test_e_value():
    rr = causal.e_value(2.0, 1.5, 2.7)
    assert rr["e_value"] == pytest.approx(2 + np.sqrt(2), rel=1e-6)
    assert rr["e_value_ci"] == pytest.approx(1.5 + np.sqrt(1.5 * 0.5), rel=1e-6)
    protective = causal.e_value(0.5, 0.3, 0.9)
    assert protective["e_value"] == pytest.approx(2 + np.sqrt(2), rel=1e-6)
    null_ci = causal.e_value(1.3, 0.9, 1.8)
    assert null_ci["e_value_ci"] == 1.0
    d = causal.e_value(0.5, scale="d")
    assert d["rr_equivalent"] == pytest.approx(np.exp(0.455))
    assert causal.e_value(4.0, scale="or")["rr_equivalent"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        causal.e_value(1.2, scale="beta")
