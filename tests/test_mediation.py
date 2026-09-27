import numpy as np
import pandas as pd
import pytest

from utils import mediation


@pytest.fixture
def med_df(data_dir):
    return pd.read_csv(data_dir / "mediation_data.csv")


def test_mediation_recovers_dgp(med_df):
    # DGP: M = 0.7 X + e, Y = 0.4 X + 0.6 M + e  -> a=0.7, b=0.6, c'=0.4, ab=0.42
    res = mediation.mediation_analysis(med_df, "X", "M", "Y", n_boot=300, seed=1, method="bca")
    assert res["a"] == pytest.approx(0.7, abs=0.1)
    assert res["b"] == pytest.approx(0.6, abs=0.15)
    assert res["c_prime"] == pytest.approx(0.4, abs=0.15)
    assert res["indirect"] == pytest.approx(0.42, abs=0.12)
    lo, hi = res["indirect_ci"]
    assert lo < 0.42 < hi and res["significant_indirect"]
    assert res["sobel_p"] < 1e-4
    assert 0.3 < res["proportion_mediated"] < 0.7
    assert list(res["paths"].index)[-1] == "a×b (indirect)"
    assert (res["paths"]["ci_low"] <= res["paths"]["estimate"]).all()


def test_mediation_with_covariates_and_no_effect():
    gen = np.random.default_rng(2)
    n = 300
    x, cov = gen.normal(size=n), gen.normal(size=n)
    m = 0.5 * cov + gen.normal(size=n)  # M unrelated to X
    y = 0.5 * x + 0.5 * m + gen.normal(size=n)
    df = pd.DataFrame({"X": x, "M": m, "Y": y, "Z": cov})
    res = mediation.mediation_analysis(df, "X", "M", "Y", covariates=["Z"], n_boot=200, seed=3)
    lo, hi = res["indirect_ci"]
    assert lo <= 0 <= hi and not res["significant_indirect"]


def test_mediation_statsmodels_crosscheck(med_df):
    summary = mediation.mediation_statsmodels(med_df, "X", "M", "Y", n_rep=100, seed=5)
    assert "ACME (average)" in summary.index
    assert summary.loc["ACME (average)", "Estimate"] == pytest.approx(0.42, abs=0.12)
    assert summary.loc["Total effect", "Estimate"] == pytest.approx(0.82, abs=0.15)


def test_moderation_recovers_interaction(med_df):
    # DGP: Y_moderated = 0.5 X + 0.3 W + 0.4 X*W + e
    res = mediation.moderation_analysis(med_df, "X", "W", "Y_moderated")
    assert res["interaction"]["coef"] == pytest.approx(0.4, abs=0.12)
    assert res["interaction"]["p_value"] < 1e-4
    slopes = res["simple_slopes"]
    assert list(slopes.index) == ["mean - 1 SD", "mean", "mean + 1 SD"]
    assert slopes.loc["mean + 1 SD", "slope"] > slopes.loc["mean", "slope"] > slopes.loc["mean - 1 SD", "slope"]
    assert slopes.loc["mean", "slope"] == pytest.approx(0.5, abs=0.15)
    assert res["delta_r2_interaction"] > 0.05
    jn = res["johnson_neyman"]
    assert jn["bounds"] is not None and jn["bounds"][0] < jn["bounds"][1]
    custom = mediation.moderation_analysis(med_df, "X", "W", "Y_moderated", probe_values=[-2, 0, 2], center=False)
    assert list(custom["simple_slopes"].index) == ["-2", "0", "2"]
    assert not custom["centered"]
