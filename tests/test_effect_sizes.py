import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import effect_sizes as es


def test_cohens_d_recovers_known_shift():
    gen = np.random.default_rng(0)
    x, y = gen.normal(0.8, 1, 500), gen.normal(0, 1, 500)
    res = es.cohens_d(x, y)
    assert res["d"] == pytest.approx(0.8, abs=0.15)
    assert res["ci_low"] < res["d"] < res["ci_high"]
    assert res["interpretation"] in {"medium", "large"}


def test_hedges_g_smaller_than_d_for_small_samples():
    gen = np.random.default_rng(1)
    x, y = gen.normal(1, 1, 10), gen.normal(0, 1, 10)
    d = es.cohens_d(x, y)["d"]
    g = es.hedges_g(x, y)
    assert abs(g["g"]) < abs(d)
    assert 0.9 < g["correction_J"] < 1


def test_glass_delta_uses_control_sd():
    t = np.array([2.0, 3.0, 4.0, 5.0])
    c = np.array([0.0, 1.0, 2.0, 3.0])
    res = es.glass_delta(t, c)
    assert res["delta"] == pytest.approx((t.mean() - c.mean()) / c.std(ddof=1))


def test_cliffs_delta_bounds_and_sign():
    assert es.cliffs_delta([5, 6, 7], [1, 2, 3])["delta"] == 1.0
    assert es.cliffs_delta([1, 2, 3], [5, 6, 7])["delta"] == -1.0
    assert es.cliffs_delta([1, 2, 3], [1, 2, 3])["delta"] == 0.0


def test_common_language_effect_size():
    assert es.common_language_effect_size([5, 6], [1, 2]) == 1.0
    assert es.common_language_effect_size([1, 2], [1, 2]) == pytest.approx(0.5)


def test_anova_effect_sizes(data_dir):
    df = pd.read_csv(data_dir / "posthoc_dataset.csv")
    model = smf.ols("Score ~ C(Group)", data=df).fit()
    table = es.anova_effect_sizes(sm.stats.anova_lm(model, typ=2))
    row = table.loc["C(Group)"]
    assert 0 < row["omega_squared"] <= row["eta_squared"] <= 1
    assert row["partial_eta_squared"] == pytest.approx(row["eta_squared"])  # one-way design
    with pytest.raises(ValueError):
        es.anova_effect_sizes(pd.DataFrame({"sum_sq": [1.0], "df": [1.0]}, index=["A"]))


def test_cramers_v_and_odds_ratio():
    table = [[30, 10], [10, 30]]
    v = es.cramers_v(table)
    assert 0.3 < v["cramers_v"] < 0.6 and v["p_value"] < 0.001
    orr = es.odds_ratio(table)
    assert orr["odds_ratio"] == pytest.approx(9.0)
    assert orr["ci_low"] < 9 < orr["ci_high"]
    zero = es.odds_ratio([[5, 0], [3, 4]])
    assert np.isfinite(zero["odds_ratio"])
    with pytest.raises(ValueError):
        es.odds_ratio([[1, 2, 3]])


def test_conversions_roundtrip():
    d = 0.7
    assert es.r_to_d(es.d_to_r(d)) == pytest.approx(d)


@pytest.mark.parametrize("value,kind,label", [(0.1, "d", "negligible"), (0.3, "d", "small"), (0.6, "d", "medium"), (1.2, "d", "large"), (0.03, "eta2", "small"), (0.07, "eta2", "medium")])
def test_interpretation_labels(value, kind, label):
    assert es.interpret_effect_size(value, kind) == label


def test_interpretation_bad_kind():
    with pytest.raises(ValueError):
        es.interpret_effect_size(0.5, "zeta")
