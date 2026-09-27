import pandas as pd
import pytest

from utils import survival_utils as su

pytestmark = pytest.mark.skipif(not su.HAS_LIFELINES, reason="lifelines not installed")


@pytest.fixture
def surv(data_dir):
    return pd.read_csv(data_dir / "survival_data.csv")


def test_kaplan_meier_table(surv):
    km = su.kaplan_meier_table(surv, "time", "event", group="treatment", times=[2, 5, 10])
    assert len(km) == 6
    assert km["survival"].between(0, 1).all()
    assert (km["ci_low"] <= km["survival"] + 1e-9).all()
    # Treatment reduces hazard (coef -0.5) so survival is higher in the treated group.
    s = km.set_index(["group", "time"])["survival"]
    assert s.loc[(1, 5)] > s.loc[(0, 5)]
    single = su.kaplan_meier_table(surv, "time", "event")
    assert single["group"].isna().all() and len(single) == 3


def test_logrank(surv):
    res = su.logrank(surv, "time", "event", "treatment")
    assert res["df"] == 1 and res["p_value"] < 0.05


def test_cox_and_ph_check(surv):
    cph, table = su.fit_cox(surv, "time", "event", ["age", "treatment", "biomarker"])
    assert list(table.index) == ["age", "treatment", "biomarker"]
    assert table.loc["treatment", "coef"] == pytest.approx(-0.5, abs=0.35)
    assert table.loc["treatment", "hazard_ratio"] < 1
    assert 0.5 < table.attrs["concordance"] < 1
    ph = su.check_proportional_hazards(cph, surv, "time", "event", ["age", "treatment", "biomarker"])
    assert set(ph.index) == {"age", "treatment", "biomarker"}
    assert "ph_violated" in ph.columns


def test_parametric_comparison(surv):
    table = su.parametric_comparison(surv, "time", "event")
    assert len(table) == 4 and table.loc[0, "delta_aic"] == 0
    assert table["aic"].is_monotonic_increasing
