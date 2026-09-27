# Data Dictionary

Every dataset in `synthetic_data/` is simulated from a fully specified data-generating process (DGP)
with a fixed seed, so the *true* parameter values are known. This document is generated from
`synthetic_data/dgp_registry.py` by `scripts/generate_data_dictionary.py`; do not edit it by hand.

Integrity hashes live in `synthetic_data/MANIFEST.json` and are checked by `scripts/verify_manifest.py`.
Parameter recovery for every dataset is reported in `exports/tables/validation/parameter_recovery.csv`
(produced by `scripts/parameter_recovery.py`).

## Summary

| Dataset | Rows | Generator | Intended models |
|---|---:|---|---|
| `ols_data.csv` | 200 | `generate_ols_data` | OLS |
| `glm_poisson.csv` | 300 | `generate_glm_data` | GLM Poisson |
| `glm_logistic.csv` | 300 | `generate_glm_data` | Logit, GLM Binomial |
| `arima_series.csv` | 250 | `generate_time_series_data` | ARIMA(1,0,1) |
| `manova_data.csv` | 300 | `generate_manova_data` | MANOVA, Hotelling T2 |
| `heteroskedastic_data.csv` | 200 | `generate_heteroskedastic_data` | OLS + HC3, WLS |
| `multivariate_group_data.csv` | 1000 | `generate_multivariate_group_data` | descriptives, t-tests, Hotelling T2 |
| `ols_diagnostics.csv` | 600 | `generate_ols_diagnostics_data` | OLS diagnostics, RLM |
| `posthoc_dataset.csv` | 300 | `generate_posthoc_data` | one-way ANOVA, Tukey HSD |
| `robust_regression_data.csv` | 200 | `generate_robust_regression_data` | RLM (Huber), WLS, Quantile regression |
| `seasonal_ts_data.csv` | 365 | `generate_seasonal_ts_data` | SARIMAX, harmonic regression |
| `panel_data.csv` | 500 | `generate_panel_data` | fixed effects, random effects / MixedLM |
| `survival_data.csv` | 200 | `generate_survival_data` | Cox PH, Kaplan-Meier, Weibull AFT |
| `zero_inflated_count.csv` | 300 | `generate_zero_inflated_count_data` | ZIP, ZINB |
| `var_data.csv` | 250 | `generate_var_data` | VAR(1), Granger causality |
| `gee_data.csv` | 250 | `generate_gee_data` | GEE (exchangeable), mixed logit |
| `mediation_data.csv` | 300 | `generate_mediation_data` | mediation (bootstrap), moderation |

## Datasets

### `ols_data.csv`

Clean multiple linear regression with two Gaussian predictors and homoskedastic noise.

- **Generator:** `generate_ols_data` (n = 200, seed: module (42, order-dependent))
- **DGP:** `y = 2 + 1.5 X1 - 0.7 X2 + e,  e ~ N(0, 1.5^2)`
- **SHA-256:** `f0df2e1ca104a7ce8502d6b281368b5fdfbb71e1fd52f6bbc3e8216fe18a3385`

| Column | Definition |
|---|---|
| `X1` | N(5, 2) predictor |
| `X2` | N(10, 3) predictor |
| `y` | response |

| True parameter | Value |
|---|---:|
| `Intercept` | 2.0 |
| `X1` | 1.5 |
| `X2` | -0.7 |
| `sigma` | 1.5 |

### `glm_poisson.csv`

Poisson counts with a log link and one predictor.

- **Generator:** `generate_glm_data` (n = 300, seed: module (42, order-dependent))
- **DGP:** `y ~ Poisson(exp(0.5 + 0.9 X))`
- **SHA-256:** `dcd1dd7f61e8f56f1466a1cb6f9abd779a3ec05763c7114bd467e9505947a7c2`

| Column | Definition |
|---|---|
| `X` | N(2, 1) predictor |
| `y` | Poisson count |

| True parameter | Value |
|---|---:|
| `Intercept` | 0.5 |
| `X` | 0.9 |

### `glm_logistic.csv`

Bernoulli outcomes with a logit link and one predictor.

- **Generator:** `generate_glm_data` (n = 300, seed: module (42, order-dependent))
- **DGP:** `P(y = 1) = logistic(-0.5 + 0.8 X)`
- **SHA-256:** `1f8b0836792b8e7d7552e30e7f7e155dc6ebabdce81c5d9218cc399a87113c07`

| Column | Definition |
|---|---|
| `X` | N(0, 1) predictor |
| `y` | binary outcome |

| True parameter | Value |
|---|---:|
| `Intercept` | -0.5 |
| `X` | 0.8 |

### `arima_series.csv`

Daily ARMA(1,1) series.

- **Generator:** `generate_time_series_data` (n = 250, seed: module (42, order-dependent))
- **DGP:** `y_t = 0.8 y_{t-1} + e_t + 0.4 e_{t-1},  e ~ N(0, 1)`
- **SHA-256:** `d43cad500dd60ba7221dcbc63576e15c74d65dd3782d9642dfe28b14b6ef5be9`

| Column | Definition |
|---|---|
| `t` | daily date index from 2020-01-01 |
| `value` | ARMA(1,1) realisation |

| True parameter | Value |
|---|---:|
| `ar.L1` | 0.8 |
| `ma.L1` | 0.4 |
| `sigma2` | 1.0 |

### `manova_data.csv`

Two bivariate-normal groups with a mean shift of 2 on both responses.

- **Generator:** `generate_manova_data` (n = 300, seed: module (42, order-dependent))
- **DGP:** `A ~ MVN([0, 0], [[1, .5], [.5, 1]]);  B ~ MVN([2, 2], same cov)`
- **SHA-256:** `001a4ccd767490d52ce7d51b9a242d7c6a8cd72524f7f399d23f4cc046f5b13b`

| Column | Definition |
|---|---|
| `Y1` | response 1 |
| `Y2` | response 2 |
| `group` | A or B (150 each) |

| True parameter | Value |
|---|---:|
| `mean_diff_Y1` | 2.0 |
| `mean_diff_Y2` | 2.0 |
| `corr` | 0.5 |

### `heteroskedastic_data.csv`

Linear mean with error variance growing in |X|; for WLS and robust SEs.

- **Generator:** `generate_heteroskedastic_data` (n = 200, seed: module (42, order-dependent))
- **DGP:** `y = 3 + 2 X + e,  e ~ N(0, (1 + 2|X|)^2)`
- **SHA-256:** `ad1a4cfedb8dcfeef7b6a348f48b5e52ba03a475e41d333e574b0ecdda95b6f4`

| Column | Definition |
|---|---|
| `X` | N(0, 1) predictor |
| `y` | response |

| True parameter | Value |
|---|---:|
| `Intercept` | 3.0 |
| `X` | 2.0 |

### `multivariate_group_data.csv`

Five numeric and five categorical columns across two groups with different distributions.

- **Generator:** `generate_multivariate_group_data` (n = 1000, seed: 42)
- **DGP:** `group-specific marginal distributions, independent columns`
- **SHA-256:** `46dcbd5615ef0c816f1ddcca8f6c6e752695c65d7ecb17b785af02803211bf5b`

| Column | Definition |
|---|---|
| `Num1` | A: N(50,10); B: N(60,15) |
| `Num2` | A: U(10,100); B: U(20,120) |
| `Num3` | A: LogN(3,.5); B: LogN(4,.8) |
| `Num4` | A: Exp(1.5); B: Exp(1.0) |
| `Num5` | A: N(0,50); B: N(20,60) |
| `Group` | A/B |
| `Cat1-Cat5` | independent categorical noise |

| True parameter | Value |
|---|---:|
| `Num1_mean_A` | 50.0 |
| `Num1_mean_B` | 60.0 |
| `Num5_mean_A` | 0.0 |
| `Num5_mean_B` | 20.0 |

### `ols_diagnostics.csv`

Ten predictors with multicollinearity (X6~X1, X7~X2), irrelevant predictors, and 10 injected outliers.

- **Generator:** `generate_ols_diagnostics_data` (n = 600, seed: 42)
- **DGP:** `y = 5 + 1.5 X1 - 2 X2 + 0.3 X3 + e, e ~ N(0,10^2); 10 rows get +N(100, 20)`
- **SHA-256:** `f42daa4787895a16259640d7d3adf30020f3983a66c36ae1484ba0ef26abdca2`

| Column | Definition |
|---|---|
| `X1` | N(50,10) |
| `X2` | N(30,5) |
| `X3` | U(10,100) |
| `X4` | Exp(1.2) |
| `X5` | N(0,20) |
| `X6` | 0.5 X1 + N(0,2) |
| `X7` | -0.4 X2 + N(0,1.5) |
| `X8` | N(0,1) |
| `X9` | U(0,1) |
| `X10` | Chi2(2) |
| `y` | response |

| True parameter | Value |
|---|---:|
| `Intercept` | 5.0 |
| `X1` | 1.5 |
| `X2` | -2.0 |
| `X3` | 0.3 |

### `posthoc_dataset.csv`

Three groups of 100 with different means and variances for ANOVA and post-hoc tests.

- **Generator:** `generate_posthoc_data` (n = 300, seed: 42)
- **DGP:** `A ~ N(60, 10), B ~ N(70, 12), C ~ N(65, 8)`
- **SHA-256:** `b33d8b8d0057b9fa327c88c47967795485d1c6cab0fd9c971f019f725bb685b8`

| Column | Definition |
|---|---|
| `Group` | A/B/C |
| `Score` | response |

| True parameter | Value |
|---|---:|
| `mean_A` | 60.0 |
| `mean_B` | 70.0 |
| `mean_C` | 65.0 |

### `robust_regression_data.csv`

Linear relationship with heteroskedastic noise and 15 injected +/-30 outliers; includes inverse-variance weights.

- **Generator:** `generate_robust_regression_data` (n = 200, seed: 42)
- **DGP:** `y = 5 + 2.5 X + e, e ~ N(0, (0.5 + 0.3|X|)^2); 15 rows +/- 30`
- **SHA-256:** `f59d3da2588a981eebfafc18e9e144aa1bda01a5005398153ad86b91d31485bc`

| Column | Definition |
|---|---|
| `X` | N(10, 3) |
| `y` | response |
| `weights` | 1 / (0.5 + 0.3|X|) |

| True parameter | Value |
|---|---:|
| `Intercept` | 5.0 |
| `X` | 2.5 |

### `seasonal_ts_data.csv`

Trend + period-12 sinusoid + noise with an independent exogenous regressor.

- **Generator:** `generate_seasonal_ts_data` (n = 365, seed: 42)
- **DGP:** `y_t = 50 + 0.05 t + 10 sin(2 pi t / 12) + e_t, e ~ N(0, 4)`
- **SHA-256:** `9e5b804d63fec0c15c2ff1a88ed10e1ddd22a776d750bec3ba0395006a80b08b`

| Column | Definition |
|---|---|
| `t` | daily date |
| `y` | series |
| `exog` | N(5, 1.5), unrelated to y |

| True parameter | Value |
|---|---:|
| `level` | 50.0 |
| `trend` | 0.05 |
| `amplitude` | 10.0 |
| `period` | 12 |
| `sigma` | 2.0 |
| `exog_effect` | 0.0 |

### `panel_data.csv`

Balanced panel: 50 individuals x 10 periods with individual random intercepts.

- **Generator:** `generate_panel_data` (n = 500, seed: 42)
- **DGP:** `y_it = 10 + u_i + 1.5 X1 - 0.8 X2 + e_it, u_i ~ N(0, 25), e ~ N(0, 1)`
- **SHA-256:** `8be96865a9711655ed25121c653f6cbbf11d696bdb2e89889bf2907dae72d906`

| Column | Definition |
|---|---|
| `individual` | ID_0..ID_49 |
| `time` | 0..9 |
| `X1` | N(10, 2) |
| `X2` | N(5, 1.5) |
| `y` | response |

| True parameter | Value |
|---|---:|
| `Intercept` | 10.0 |
| `X1` | 1.5 |
| `X2` | -0.8 |
| `sd_individual` | 5.0 |
| `sigma` | 1.0 |

### `survival_data.csv`

Exponential survival times with covariate-dependent hazard and uniform censoring.

- **Generator:** `generate_survival_data` (n = 200, seed: 42)
- **DGP:** `T ~ Exp(rate = exp(-3 + 0.02 age - 0.5 treatment + 0.01 biomarker)); C ~ U(0, 15)`
- **SHA-256:** `ddfe8023abd082cce241c3212130582e3b58e00c8bf7e79a47b0df026313b378`

| Column | Definition |
|---|---|
| `age` | N(55, 12) |
| `treatment` | Bernoulli(0.5) |
| `biomarker` | N(100, 20) |
| `time` | min(T, C) |
| `event` | 1 if T <= C |

| True parameter | Value |
|---|---:|
| `age` | 0.02 |
| `treatment` | -0.5 |
| `biomarker` | 0.01 |

### `zero_inflated_count.csv`

Zero-inflated Poisson: structural zeros driven by x3, counts driven by x1 and x2.

- **Generator:** `generate_zero_inflated_count_data` (n = 300, seed: 42)
- **DGP:** `P(structural zero) = logistic(-0.5 + 0.7 x3); y ~ Poisson(exp(0.5 + 0.6 x1 + 0.3 x2)) otherwise`
- **SHA-256:** `b95cb346cfc0dde0e89b52d69144cffd3154a648de967dca61be6c6f9c1e737b`

| Column | Definition |
|---|---|
| `x1` | N(2, 1) |
| `x2` | N(0, 1.5) |
| `x3` | N(-1, 0.8) inflation predictor |
| `y` | count |

| True parameter | Value |
|---|---:|
| `Intercept` | 0.5 |
| `x1` | 0.6 |
| `x2` | 0.3 |
| `inflate_Intercept` | -0.5 |
| `inflate_x3` | 0.7 |

### `var_data.csv`

Bivariate VAR(1) with cross-lag feedback.

- **Generator:** `generate_var_data` (n = 250, seed: 42)
- **DGP:** `y1_t = 0.5 y1_{t-1} + 0.2 y2_{t-1} + e1;  y2_t = 0.3 y1_{t-1} + 0.6 y2_{t-1} + e2`
- **SHA-256:** `b9d6525d9f4ac24e1f2949b90ff30c0ed7416a21be8a72a7445cc77e086427ef`

| Column | Definition |
|---|---|
| `t` | daily date |
| `y1` | series 1 |
| `y2` | series 2 |

| True parameter | Value |
|---|---:|
| `L1.y1->y1` | 0.5 |
| `L1.y2->y1` | 0.2 |
| `L1.y1->y2` | 0.3 |
| `L1.y2->y2` | 0.6 |

### `gee_data.csv`

50 clusters x 5 binary outcomes with a large cluster random effect (sd 3).

- **Generator:** `generate_gee_data` (n = 250, seed: 42)
- **DGP:** `logit P(y=1) = -1 + 0.5 X + 0.8 treatment + u_c, u_c ~ N(0, 9)`
- **Note:** GEE estimates population-averaged effects, which are attenuated relative to these conditional (subject-specific) parameters.
- **SHA-256:** `c81887a7822e1f4104468c405f012b49f2346f65b17fd9467be39b6de5723427`

| Column | Definition |
|---|---|
| `cluster` | C_0..C_49 |
| `observation` | 0..4 |
| `X` | N(5, 2) |
| `treatment` | Bernoulli(0.5) |
| `y` | binary |

| True parameter | Value |
|---|---:|
| `Intercept` | -1.0 |
| `X` | 0.5 |
| `treatment` | 0.8 |
| `sd_cluster` | 3.0 |

### `mediation_data.csv`

Simple mediation X -> M -> Y plus a separate moderated outcome.

- **Generator:** `generate_mediation_data` (n = 300, seed: 42)
- **DGP:** `M = 0.7 X + N(0, .25); Y = 0.4 X + 0.6 M + N(0, .64); Y_mod = 0.5 X + 0.3 W + 0.4 X W + N(0, .64)`
- **SHA-256:** `dcfaa8c5421ce2cb889f956a6323789279443010b17ba507a03971cf559e9922`

| Column | Definition |
|---|---|
| `X` | N(0, 1) |
| `M` | mediator |
| `Y` | outcome |
| `W` | N(0, 1) moderator |
| `Y_moderated` | moderated outcome |

| True parameter | Value |
|---|---:|
| `a` | 0.7 |
| `b` | 0.6 |
| `c_prime` | 0.4 |
| `indirect` | 0.42 |
| `mod_X` | 0.5 |
| `mod_W` | 0.3 |
| `mod_XW` | 0.4 |

## Reproducibility note

The following generators do not reseed and therefore consume the module-level `np.random.seed(42)`
stream in call order: `generate_ols_data`, `generate_glm_data`, `generate_time_series_data`, `generate_manova_data`, `generate_heteroskedastic_data`.
Changing their order in `generate_all_datasets()` changes those files; rebuild the manifest if you do.
