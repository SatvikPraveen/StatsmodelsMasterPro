"""Tests for the original utility modules (model_utils, diagnostics, compare_models)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from utils import compare_models, diagnostics, model_utils


def test_summarize_stats(linear_df):
    result = model_utils.summarize_stats(linear_df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {"X1", "X2", "y"}
    assert "mean" in result.columns


def test_compute_central_tendency(linear_df):
    result = model_utils.compute_central_tendency(linear_df, ["X1", "X2"])
    assert list(result.columns) == ["mean", "median", "mode"]
    assert np.isclose(result.loc["X1", "mean"], linear_df["X1"].mean())


def test_summarize_model_coefficients(ols_model):
    result = model_utils.summarize_model_coefficients(ols_model)
    assert list(result.columns) == ["coef", "p_value", "ci_lower", "ci_upper"]
    assert (result["ci_lower"] <= result["coef"]).all()
    assert (result["coef"] <= result["ci_upper"]).all()


def test_coefficients_recover_truth(ols_model):
    assert ols_model.params["X1"] == pytest.approx(1.5, abs=0.15)
    assert ols_model.params["X2"] == pytest.approx(-0.7, abs=0.15)


def test_extract_aic_bic(ols_model):
    result = model_utils.extract_aic_bic(ols_model)
    assert set(result) == {"AIC", "BIC", "Log-Likelihood"}
    assert result["BIC"] > result["AIC"]  # BIC penalises 3 params more heavily for n=200


def test_compare_models_by_ic(linear_df, ols_model):
    reduced = smf.ols("y ~ X1", data=linear_df).fit()
    table = model_utils.compare_models_by_ic(ols_model, reduced)
    assert list(table.index) == ["Model_1", "Model_2"]
    assert table.loc["Model_1", "AIC"] < table.loc["Model_2", "AIC"]


def test_export_model_summary_as_text(ols_model, tmp_path):
    out = tmp_path / "summary.txt"
    model_utils.export_model_summary_as_text(ols_model, out)
    assert "OLS Regression Results" in out.read_text()


def test_compute_hotelling_t2_detects_shift():
    gen = np.random.default_rng(3)
    g1 = pd.DataFrame(gen.normal(0, 1, (80, 2)), columns=["a", "b"])
    g2 = pd.DataFrame(gen.normal(1, 1, (80, 2)), columns=["a", "b"])
    res = model_utils.compute_hotelling_t2(g1, g2, ["a", "b"])
    assert res["p_value"] < 1e-6
    assert res["T2"] > 0


def test_compute_hotelling_t2_null_not_rejected():
    gen = np.random.default_rng(4)
    g1 = pd.DataFrame(gen.normal(0, 1, (80, 2)), columns=["a", "b"])
    g2 = pd.DataFrame(gen.normal(0, 1, (80, 2)), columns=["a", "b"])
    res = model_utils.compute_hotelling_t2(g1, g2, ["a", "b"])
    assert res["p_value"] > 0.01


def test_bootstrap_mean_ci_contains_mean(rng):
    data = rng.normal(5, 1, 100)
    lo, hi = model_utils.bootstrap_mean_ci(data, n_bootstrap=500)
    assert lo < data.mean() < hi


def test_compute_skewness_kurtosis(linear_df):
    result = diagnostics.compute_skewness_kurtosis(linear_df, ["X1", "y"])
    assert set(result) == {"X1", "y"}
    assert result["X1"]["kurtosis"] == pytest.approx(3, abs=1.0)


def test_run_heteroskedasticity_tests(hetero_df):
    model = smf.ols("y ~ X", data=hetero_df).fit()
    result = diagnostics.run_heteroskedasticity_tests(model)
    assert result["Breusch-Pagan"]["p-value"] < 0.05 or result["White"]["p-value"] < 0.05


def test_compare_model_metrics_sorted(linear_df, ols_model):
    reduced = smf.ols("y ~ X1", data=linear_df).fit()
    table = compare_models.compare_model_metrics({"full": ols_model, "reduced": reduced})
    assert table.iloc[0]["Model"] == "full"


def test_forward_stepwise_selects_true_predictors(linear_df):
    gen = np.random.default_rng(1)
    df = linear_df.copy()
    df["noise"] = gen.normal(size=len(df))
    model, selected = compare_models.forward_stepwise_selection(df, "y", ["X1", "X2", "noise"], verbose=False)
    assert set(selected) >= {"X1", "X2"}
