import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import reporting


def test_tidy_and_glance_ols(ols_model):
    t = reporting.tidy(ols_model)
    assert list(t.columns) == ["term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]
    assert list(t["term"]) == ["Intercept", "X1", "X2"]
    g = reporting.glance(ols_model)
    assert g.shape[0] == 1
    assert 0.8 < g.loc[0, "r_squared"] <= 1
    assert g.loc[0, "cov_type"] == "nonrobust"


def test_tidy_exponentiate_logit(data_dir):
    df = pd.read_csv(data_dir / "glm_logistic.csv")
    res = smf.logit("y ~ X", data=df).fit(disp=0)
    t = reporting.tidy(res, exponentiate=True)
    assert "std_error" not in t.columns
    assert t.set_index("term").loc["X", "estimate"] == pytest.approx(np.exp(res.params["X"]))
    g = reporting.glance(res)
    assert "pseudo_r_squared" in g.columns and g.loc[0, "converged"]


def test_regression_table_formats(linear_df, ols_model):
    reduced = smf.ols("y ~ X1", data=linear_df).fit()
    df = reporting.regression_table([reduced, ols_model], ["Reduced", "Full"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Reduced", "Full"]
    assert "X2" in df.index and "R²" in df.index
    tex = reporting.regression_table([reduced, ols_model], to="latex")
    assert "\\begin{table}" in tex or "\\begin{tabular}" in tex
    md = reporting.regression_table([reduced, ols_model], to="markdown")
    assert md.startswith("|") and "X1" in md
    assert "<table" in reporting.regression_table([reduced], to="html")
    with pytest.raises(ValueError):
        reporting.regression_table([reduced], to="pdf")


def test_apa_strings(ols_model, data_dir):
    assert reporting.format_p(0.0001) == "p < .001"
    assert reporting.format_p(0.0345) == "p = .035"
    assert reporting.format_p(np.nan) == "p = n/a"
    assert reporting.format_ci(1.2, 2.34) == "95% CI [1.20, 2.34]"
    s = reporting.apa_coefficient(ols_model, "X1")
    assert s.startswith("b = 1.") and "t(197)" in s and "p < .001" in s
    assert reporting.apa_ttest(2.31, 58, 0.024, d=0.6) == "t(58) = 2.31, p = .024, d = 0.60"
    df = pd.read_csv(data_dir / "posthoc_dataset.csv")
    anova = sm.stats.anova_lm(smf.ols("Score ~ C(Group)", data=df).fit(), typ=2)
    s2 = reporting.apa_anova(anova, "C(Group)", eta_squared=0.14)
    assert s2.startswith("F(2, 297)") and "η² = .14" in s2


def test_coefficient_plot(linear_df, ols_model):
    import matplotlib

    matplotlib.use("Agg")
    reduced = smf.ols("y ~ X1", data=linear_df).fit()
    ax = reporting.coefficient_plot([reduced, ols_model], names=["Reduced", "Full"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["X1", "X2"]
    ax2 = reporting.coefficient_plot(ols_model, terms=["X2"], exclude_intercept=False)
    assert [t.get_text() for t in ax2.get_yticklabels()] == ["X2"]


def test_to_markdown_and_save_table(tmp_path):
    df = pd.DataFrame({"a": [1.5, np.nan], "b": ["x", "y"]}, index=["r1", "r2"])
    md = reporting.to_markdown(df, float_format="{:.1f}")
    assert md.splitlines()[0] == "| index | a | b |"
    assert "| r1 | 1.5 | x |" in md
    paths = reporting.save_table(df, tmp_path / "out" / "table")
    assert [p.suffix for p in paths] == [".csv", ".tex", ".md"]
    assert all(p.exists() for p in paths)
    with pytest.raises(ValueError):
        reporting.save_table(df, tmp_path / "t", formats=("xlsx",))


def test_to_latex_escapes_and_formats():
    df = pd.DataFrame({"est_1": [0.5, 1.25]}, index=["a_b", "c"])
    tex = reporting.to_latex(df, float_format="%.2f", caption="Cap", label="tab:x")
    assert "\\begin{tabular}{lr}" in tex and "\\toprule" in tex
    assert "a\\_b & 0.50 \\\\" in tex and "\\caption{Cap}" in tex
