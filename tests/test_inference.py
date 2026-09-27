"""Tests for utils.inference: bootstrap, permutation, robust SEs, model tests."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from utils import inference


class TestBootstrapCI:
    def test_reproducible_with_seed(self, rng):
        x = rng.normal(size=50)
        a = inference.bootstrap_ci(x, seed=1, n_boot=300)
        b = inference.bootstrap_ci(x, seed=1, n_boot=300)
        assert a.ci_low == b.ci_low and a.ci_high == b.ci_high

    @pytest.mark.parametrize("method", ["percentile", "basic", "normal", "bca"])
    def test_interval_brackets_estimate(self, rng, method):
        x = rng.exponential(2.0, size=80)
        res = inference.bootstrap_ci(x, np.mean, n_boot=500, method=method, seed=0)
        assert res.ci_low < res.estimate < res.ci_high
        assert res.method == method
        assert res.se > 0

    def test_joint_resampling_for_correlation(self, rng):
        x = rng.normal(size=100)
        y = 0.6 * x + rng.normal(size=100)
        res = inference.bootstrap_ci((x, y), lambda a, b: np.corrcoef(a, b)[0, 1], n_boot=500, method="bca", seed=2)
        assert 0.2 < res.ci_low < res.estimate < res.ci_high < 0.9

    def test_bca_shifts_interval_for_skewed_statistic(self, rng):
        x = rng.lognormal(0, 1, size=60)
        perc = inference.bootstrap_ci(x, np.mean, n_boot=1000, method="percentile", seed=5)
        bca = inference.bootstrap_ci(x, np.mean, n_boot=1000, method="bca", seed=5)
        # For a right-skewed mean, BCa pushes the interval to the right.
        assert bca.ci_high >= perc.ci_high - 1e-9

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            inference.bootstrap_ci([1.0], n_boot=10)
        with pytest.raises(ValueError):
            inference.bootstrap_ci([1.0, 2.0], method="magic")
        with pytest.raises(ValueError):
            inference.bootstrap_ci(([1, 2, 3], [1, 2]), np.mean)

    @pytest.mark.slow
    def test_percentile_coverage_near_nominal(self):
        gen = np.random.default_rng(99)
        hits = 0
        reps = 200
        for _ in range(reps):
            x = gen.normal(1.0, 1.0, 40)
            res = inference.bootstrap_ci(x, np.mean, n_boot=400, seed=int(gen.integers(1e9)))
            hits += res.ci_low <= 1.0 <= res.ci_high
        assert 0.88 <= hits / reps <= 0.99


class TestBootstrapRegression:
    @pytest.mark.parametrize("method", ["pairs", "residual", "wild"])
    def test_coefficients_and_ci(self, linear_df, method):
        table = inference.bootstrap_regression("y ~ X1 + X2", linear_df, n_boot=200, method=method, seed=3)
        assert list(table.index) == ["Intercept", "X1", "X2"]
        assert (table["ci_low"] <= table["coef"]).all() and (table["coef"] <= table["ci_high"]).all()
        assert table.loc["X1", "ci_low"] < 1.5 < table.loc["X1", "ci_high"]
        assert table.attrs["method"] == method

    def test_pairs_bootstrap_se_close_to_hc_under_heteroskedasticity(self, hetero_df):
        table, draws = inference.bootstrap_regression("y ~ X", hetero_df, n_boot=400, method="pairs", seed=4, return_draws=True)
        assert draws.shape == (400, 2)
        hc3 = smf.ols("y ~ X", data=hetero_df).fit(cov_type="HC3").bse["X"]
        assert table.loc["X", "boot_se"] == pytest.approx(hc3, rel=0.25)

    def test_bad_method(self, linear_df):
        with pytest.raises(ValueError):
            inference.bootstrap_regression("y ~ X1", linear_df, method="nope")


class TestPermutation:
    def test_detects_shift(self, two_groups):
        x, y = two_groups
        res = inference.permutation_test(x, y, n_perm=2000, seed=0)
        assert res.p_value < 0.01
        assert res.observed == pytest.approx(x.mean() - y.mean())

    def test_null_uniformish(self, rng):
        x, y = rng.normal(size=40), rng.normal(size=40)
        res = inference.permutation_test(x, y, n_perm=1000, seed=1)
        assert res.p_value > 0.05

    def test_one_sided_and_callable(self, two_groups):
        x, y = two_groups
        less = inference.permutation_test(x, y, statistic=lambda a, b: a.mean() - b.mean(), alternative="less", n_perm=1000, seed=2)
        greater = inference.permutation_test(x, y, alternative="greater", n_perm=1000, seed=2)
        assert less.p_value < 0.05 < greater.p_value

    def test_paired(self, rng):
        x = rng.normal(size=30)
        y = x + 0.5 + rng.normal(scale=0.3, size=30)
        res = inference.paired_permutation_test(y, x, n_perm=2000, seed=3)
        assert res.p_value < 0.01

    def test_bad_alternative(self, two_groups):
        with pytest.raises(ValueError):
            inference.permutation_test(*two_groups, alternative="sideways", n_perm=50)


class TestRobustSE:
    def test_hc_inflates_se_under_heteroskedasticity(self, hetero_df):
        model = smf.ols("y ~ X", data=hetero_df).fit()
        table = inference.robust_se_table(model)
        wide = table.pivot(index="term", columns="cov_type", values="se")
        assert set(wide.columns) == {"nonrobust", "HC0", "HC1", "HC2", "HC3"}
        assert wide.loc["X", "HC3"] > wide.loc["X", "HC0"] > 0
        assert wide.loc["X", "HC3"] > wide.loc["X", "nonrobust"]

    def test_cluster_and_hac(self, linear_df):
        model = smf.ols("y ~ X1 + X2", data=linear_df).fit()
        groups = np.repeat(np.arange(20), 10)
        table = inference.robust_se_table(model, cov_types=("cluster", "HAC"), groups=groups)
        assert set(table["cov_type"]) == {"cluster", "HAC"}
        assert (table["se"] > 0).all()

    def test_cluster_requires_groups(self, ols_model):
        with pytest.raises(ValueError):
            inference.robust_se_table(ols_model, cov_types=("cluster",))

    def test_glm_supported(self, data_dir):
        df = pd.read_csv(data_dir / "glm_poisson.csv")
        import statsmodels.api as sm

        res = smf.glm("y ~ X", data=df, family=sm.families.Poisson()).fit()
        table = inference.robust_se_table(res, cov_types=("nonrobust", "HC0"))
        assert len(table) == 4


class TestModelTests:
    def test_wald_single_and_joint(self, ols_model):
        table = inference.wald_test(ols_model, ["X1 = 0", "X2 = 0"])
        assert list(table["hypothesis"]) == ["X1 = 0", "X2 = 0", "joint"]
        assert (table["p_value"] < 1e-6).all()

    def test_wald_true_restriction_not_rejected(self, ols_model):
        table = inference.wald_test(ols_model, "X1 = 1.5")
        assert table.loc[0, "p_value"] > 0.05

    def test_lr_test(self, linear_df, ols_model):
        restricted = smf.ols("y ~ X1", data=linear_df).fit()
        res = inference.likelihood_ratio_test(restricted, ols_model)
        assert res["df"] == 1 and res["p_value"] < 1e-6
        with pytest.raises(ValueError):
            inference.likelihood_ratio_test(ols_model, restricted)


class TestMultipleTesting:
    def test_holm_and_bh(self):
        p = [0.001, 0.01, 0.03, 0.04, 0.2]
        holm = inference.multiple_testing(p, method="holm", labels=list("abcde"))
        bh = inference.multiple_testing(p, method="fdr_bh")
        assert holm.loc["a", "reject"] and not holm.loc["e", "reject"]
        assert (bh["p_adj"] >= np.asarray(p)).all()
        assert bh["reject"].sum() >= holm["reject"].sum()

    def test_ci_from_se(self):
        lo, hi = inference.ci_from_se(1.0, 0.5, df=10)
        lo_z, hi_z = inference.ci_from_se(1.0, 0.5)
        assert lo < lo_z < 1.0 < hi_z < hi
