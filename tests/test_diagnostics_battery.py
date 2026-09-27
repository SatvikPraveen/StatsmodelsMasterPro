import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from utils import diagnostics


def test_battery_flags_heteroskedasticity(hetero_df):
    model = smf.ols("y ~ X", data=hetero_df).fit()
    table = diagnostics.diagnostic_battery(model)
    assert {"test", "statistic", "p_value", "null_hypothesis", "flag", "note"} <= set(table.columns)
    idx = table.set_index("test")
    # Variance grows with |X| symmetrically, so the linear Breusch-Pagan auxiliary
    # regression has little power; White's test (which includes X^2) must catch it.
    assert idx.loc["White", "flag"] and idx.loc["White", "p_value"] < 0.05
    assert table.attrs["nobs"] == len(hetero_df)


def test_battery_clean_model_mostly_passes(ols_model):
    table = diagnostics.diagnostic_battery(ols_model).set_index("test")
    for name in ["Breusch-Pagan", "Jarque-Bera", "Ljung-Box", "Rainbow", "Ramsey RESET", "Condition number", "Max VIF"]:
        assert name in table.index
    assert not table.loc["Breusch-Pagan", "flag"]
    assert not table.loc["Jarque-Bera", "flag"]
    assert not table.loc["Max VIF", "flag"]
    assert 1.5 <= table.loc["Durbin-Watson", "statistic"] <= 2.5


def test_battery_detects_nonlinearity():
    gen = np.random.default_rng(8)
    x = gen.uniform(-3, 3, 300)
    df = pd.DataFrame({"x": x, "y": 1 + x**2 + gen.normal(scale=0.5, size=300)})
    table = diagnostics.diagnostic_battery(smf.ols("y ~ x", data=df).fit()).set_index("test")
    assert table.loc["Ramsey RESET", "flag"]


def test_vif_table_detects_collinearity(data_dir):
    df = pd.read_csv(data_dir / "ols_diagnostics.csv")
    vif = diagnostics.vif_table(df[["X1", "X2", "X6", "X8"]])
    assert list(vif.columns) == ["feature", "vif", "tolerance", "flag"]
    assert vif.iloc[0]["feature"] in {"X1", "X6"}
    assert vif.iloc[0]["vif"] > 5
    from_model = diagnostics.vif_table(smf.ols("y ~ X1 + X6", data=df).fit())
    assert "const" not in from_model["feature"].tolist() and "Intercept" not in from_model["feature"].tolist()


def test_vif_single_predictor():
    vif = diagnostics.vif_table(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))
    assert vif["vif"].tolist() == [1.0]


def test_influence_summary_flags_injected_outlier(linear_df):
    df = linear_df.copy()
    df.loc[0, "y"] += 40
    model = smf.ols("y ~ X1 + X2", data=df).fit()
    infl = diagnostics.influence_summary(model)
    assert len(infl) == len(df)
    assert infl.loc[0, "flag_cooks"] and infl.loc[0, "flag_outlier"]
    assert infl.attrs["thresholds"]["cooks_d"] == pytest.approx(4 / len(df))
    assert infl["any_flag"].sum() < 0.15 * len(df)
