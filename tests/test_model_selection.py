import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from utils import model_selection as ms


@pytest.fixture
def noisy_df(linear_df):
    gen = np.random.default_rng(5)
    df = linear_df.copy()
    df["noise1"] = gen.normal(size=len(df))
    df["noise2"] = gen.normal(size=len(df))
    return df


def test_information_criteria_table(noisy_df):
    full = smf.ols("y ~ X1 + X2", data=noisy_df).fit()
    over = smf.ols("y ~ X1 + X2 + noise1 + noise2", data=noisy_df).fit()
    under = smf.ols("y ~ X1", data=noisy_df).fit()
    table = ms.information_criteria_table({"true": full, "overfit": over, "underfit": under})
    assert table.index[0] == "true"
    assert table.loc["true", "delta_AIC"] == 0
    assert table["weight_AIC"].sum() == pytest.approx(1.0)
    assert table["weight_BIC"].sum() == pytest.approx(1.0)
    assert (table["AICc"] >= table["AIC"]).all()
    assert table.loc["underfit", "evidence_ratio_AIC"] > 100


def test_cross_validate_reports_folds(noisy_df):
    cv = ms.cross_validate("y ~ X1 + X2", noisy_df, k=5, seed=1)
    assert len(cv) == 5
    assert cv["n_test"].sum() == len(noisy_df)
    assert cv.attrs["mean"]["rmse"] == pytest.approx(1.5, abs=0.4)
    assert cv.attrs["mean"]["r2_oos"] > 0.8
    with pytest.raises(ValueError):
        ms.cross_validate("y ~ X1", noisy_df, k=1)


def test_cross_validate_glm(data_dir):
    df = pd.read_csv(data_dir / "glm_poisson.csv")
    cv = ms.cross_validate("y ~ X", df, k=4, seed=2, fit_fn=smf.glm, fit_kwargs=None)
    assert np.isfinite(cv["rmse"]).all()


def test_best_subset_ranks_true_model_first(noisy_df):
    table = ms.best_subset(noisy_df, "y", ["X1", "X2", "noise1", "noise2"], criterion="bic")
    assert len(table) == 16
    assert set(table.iloc[0]["predictors"].split(" + ")) == {"X1", "X2"}
    assert table.attrs["criterion"] == "bic"
    with pytest.raises(ValueError):
        ms.best_subset(noisy_df, "y", ["X1"], criterion="mse")


@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
def test_stepwise_selection(noisy_df, direction):
    result, selected, trace = ms.stepwise_selection(noisy_df, "y", ["X1", "X2", "noise1", "noise2"], direction=direction, criterion="bic")
    assert set(selected) == {"X1", "X2"}
    assert trace.iloc[0]["action"] == "start"
    assert len(trace) >= 2
    assert result.nobs == len(noisy_df)


def test_stepwise_bad_direction(noisy_df):
    with pytest.raises(ValueError):
        ms.stepwise_selection(noisy_df, "y", ["X1"], direction="sideways")


def test_nested_f_test(noisy_df):
    restricted = smf.ols("y ~ X1", data=noisy_df).fit()
    full = smf.ols("y ~ X1 + X2", data=noisy_df).fit()
    res = ms.nested_f_test(restricted, full)
    assert res["df_num"] == 1 and res["p_value"] < 1e-6
    lr = ms.likelihood_ratio_test(restricted, full)
    assert lr["p_value"] < 1e-6
