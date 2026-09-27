import pytest

from utils import power


def test_power_ttest_roundtrip():
    n = power.power_ttest(effect_size=0.5, power=0.8)
    assert n["solved_for"] == "nobs" and n["nobs"] == pytest.approx(63.8, abs=1)
    p = power.power_ttest(effect_size=0.5, nobs=64)
    assert p["power"] == pytest.approx(0.8, abs=0.01)
    d = power.power_ttest(nobs=64, power=0.8)
    assert d["effect_size"] == pytest.approx(0.5, abs=0.01)
    paired = power.power_ttest(effect_size=0.5, power=0.8, paired=True)
    assert paired["nobs"] < n["nobs"] and paired["test"] == "paired t"
    with pytest.raises(ValueError):
        power.power_ttest(effect_size=0.5)


def test_power_anova_and_proportions():
    a = power.power_anova(effect_size=0.25, k_groups=3, power=0.8)
    assert a["nobs"] == pytest.approx(158, abs=3)
    pr = power.power_proportions(0.5, 0.6, power=0.8)
    assert pr["cohens_h"] == pytest.approx(0.201, abs=0.01)
    assert 380 < pr["nobs"] < 400


def test_power_correlation():
    n = power.power_correlation(r=0.3, power=0.8)
    assert n["nobs"] == pytest.approx(85, abs=3)
    pw = power.power_correlation(r=0.3, nobs=85)
    assert pw["power"] == pytest.approx(0.8, abs=0.03)
    r = power.power_correlation(nobs=85, power=0.8)
    assert r["r"] == pytest.approx(0.3, abs=0.01)
    with pytest.raises(ValueError):
        power.power_correlation(power=0.8)


def test_power_curve_and_mde():
    curve = power.power_curve([0.2, 0.5], [20, 50, 100])
    assert len(curve) == 6
    assert curve.sort_values(["effect_size", "nobs"])["power"].is_monotonic_increasing
    assert power.minimum_detectable_effect(64) == pytest.approx(0.5, abs=0.01)
    assert power.minimum_detectable_effect(90, test="anova") > 0
    with pytest.raises(ValueError):
        power.power_curve([0.2], [10], test="chi2")
