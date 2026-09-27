# Statistical Methodology

This document describes the statistical procedures implemented in `utils/`, the
assumptions behind them, the design choices that were made, and the primary
references. It is intended to let a reviewer audit *what* each function computes
and *why* that estimator was chosen.

All procedures are validated in two ways:

1. **Unit tests against known data-generating processes** (`tests/`): every
   estimator is run on simulated data with known parameters and must recover them
   within tolerance, and every confidence-interval procedure is checked for
   bracket correctness. A slow-marked test checks empirical bootstrap coverage.
2. **Parameter recovery on the shipped datasets** (`scripts/parameter_recovery.py`):
   the intended model is fitted to each of the 17 synthetic datasets and the true
   DGP values are compared with 95% confidence intervals.

---

## 1. Resampling inference (`utils/inference.py`)

### 1.1 Bootstrap confidence intervals

`bootstrap_ci` resamples observations with replacement (jointly across arrays when a
tuple is passed, so paired statistics such as correlations are handled correctly)
and supports four interval constructions:

| Method | Interval | Notes |
|---|---|---|
| percentile | `[q_{α/2}, q_{1-α/2}]` of the bootstrap distribution | Simple, transformation-respecting, but not bias-corrected. |
| basic | `[2θ̂ − q_{1-α/2}, 2θ̂ − q_{α/2}]` | Reverse-percentile; better when the bootstrap distribution is shifted. |
| normal | `θ̂ ± z_{1-α/2} · SE_boot` | Assumes approximate normality. |
| BCa | percentile with bias-correction `z₀` and acceleration `a` | Second-order accurate; `a` from the jackknife. Recommended default for skewed statistics. |

BCa follows DiCiccio & Efron (1996), §2: `z₀ = Φ⁻¹(#{θ*<θ̂}/B)`,
`a = Σ(θ̄₍·₎ − θ₍ᵢ₎)³ / 6[Σ(θ̄₍·₎ − θ₍ᵢ₎)²]^{3/2}`, with the adjusted quantiles
`Φ(z₀ + (z₀+z_α)/(1 − a(z₀+z_α)))`. The proportion used for `z₀` is clipped to
`(1/(B+1), B/(B+1))` so that it is always finite.

### 1.2 Regression bootstraps

`bootstrap_regression` implements three schemes (Freedman, 1981; Wu, 1986):

- **pairs** – resample rows. Valid under heteroskedasticity and random design;
  the default.
- **residual** – resample residuals and add them to fitted values. Assumes i.i.d.
  errors and a fixed design; most efficient when those hold.
- **wild** – multiply leverage-adjusted residuals `e_i/(1−h_ii)` by Rademacher
  weights. Keeps the design fixed while remaining valid under heteroskedasticity
  (Davidson & Flachaire, 2008).

### 1.3 Permutation tests

`permutation_test` relabels the pooled sample; `paired_permutation_test` flips the
signs of within-pair differences. P-values use the add-one estimator
`(b + 1)/(B + 1)` (Phipson & Smyth, 2010), which is exact-in-expectation and never
zero.

### 1.4 Robust covariance estimators

`robust_se_table` refits the model with `cov_type` set to each requested estimator:

- **HC0–HC3** (White, 1980; MacKinnon & White, 1985). HC3 is the recommended
  default in samples below ~250 (Long & Ervin, 2000).
- **HAC** (Newey & West, 1987) with default bandwidth `⌊4(n/100)^{2/9}⌋`.
- **cluster** with one-way clustering; requires the number of clusters to be
  reasonably large (≥ 40–50) for the asymptotics to be reliable.

### 1.5 Model tests and multiplicity

`wald_test` evaluates linear restrictions (individually and jointly),
`likelihood_ratio_test` compares nested ML fits with `2(ℓ_full − ℓ_restricted) ~ χ²`,
and `multiple_testing` wraps `statsmodels.stats.multitest.multipletests`
(Bonferroni, Holm 1979, Šidák, Benjamini–Hochberg 1995, Benjamini–Yekutieli 2001).

---

## 2. Effect sizes (`utils/effect_sizes.py`)

Standardised mean differences (Cohen's *d* with pooled SD, Hedges' *g* with the
small-sample correction `J = 1 − 3/(4df − 1)`, Glass's Δ with the control SD) use
the large-sample variance `(n₁+n₂)/(n₁n₂) + d²/(2(n₁+n₂))` (Hedges & Olkin, 1985).
Cliff's δ and the common-language effect size are distribution-free. ANOVA effect
sizes (η², partial η², ω²) are computed from an `anova_lm` table; ω² is truncated
at zero. Cramér's V uses the Bergsma (2013) bias correction by default. Odds
ratios use the Woolf logit interval with a Haldane–Anscombe 0.5 correction when a
cell is zero. Interpretation labels follow Cohen (1988) and, for Cliff's δ,
Romano et al. (2006), and are heuristics rather than substantive thresholds.

---

## 3. Causal inference (`utils/causal.py`)

All estimators assume the identifying conditions stated below; the module cannot
verify those conditions from data.

| Estimator | Identifying assumption | Implementation |
|---|---|---|
| IPW (`ipw`) | Conditional exchangeability given covariates, positivity | Logit propensity; stabilised ATE weights `P(T)/e(X)` and `(1−P(T))/(1−e(X))`, ATT weights `e/(1−e)` for controls; weights trimmed at the 1st/99th percentile by default; point estimate from weighted least squares with HC1 sandwich SE; optional bootstrap that re-estimates the propensity model. |
| Matching (`nearest_neighbor_match`) | As above | 1:1 nearest neighbour on the logit propensity with a caliper of 0.2 SD (Austin, 2011); ATT from paired differences. |
| Difference-in-differences | Parallel trends | `y ~ treated * post` with cluster-robust or HC1 SEs (Bertrand et al., 2004). |
| 2SLS | Instrument relevance and exclusion | `IV2SLS`; first-stage F on the excluded instruments, flagged when `< 10` (Staiger & Stock, 1997). |
| Sharp RD | Continuity of potential outcomes at the cutoff | Local polynomial (default linear) with separate slopes each side, triangular kernel, rule-of-thumb bandwidth `1.84·sd·n^{−1/5}`; always report `rd_bandwidth_sensitivity`. |

Balance is summarised by standardised mean differences (Austin, 2009); `|SMD| < 0.1`
is the usual adequacy threshold. `e_value` converts the estimate (RR, OR, HR, or
*d* via `RR ≈ exp(0.91d)`) to the minimum confounder association needed to explain
it away (VanderWeele & Ding, 2017).

---

## 4. Mediation and moderation (`utils/mediation.py`)

The indirect effect `a·b` is tested with a bootstrap CI (Preacher & Hayes, 2008)
rather than the Sobel test alone, because the product of two normal coefficients is
not normal; the Sobel statistic is reported for comparison. The percentile or BCa
bootstrap is applied to row indices so that both regressions are refitted in every
replicate. `mediation_statsmodels` cross-checks the result with the
potential-outcomes ACME/ADE decomposition of Imai, Keele & Tingley (2010).

Moderation uses mean-centred predictors by default; simple slopes at
`W = mean ± 1 SD` are exact linear-combination *t*-tests
(`result.t_test`), and the Johnson–Neyman bounds solve
`(b₁ + b₃w)² = t²_crit · Var(b₁ + b₃w)` for `w` (Bauer & Curran, 2005).

---

## 5. Monte Carlo validation (`utils/simulation.py`)

The framework follows the ADEMP structure of Morris, White & Crowther (2019).
Each replicate draws an independent child generator from a `SeedSequence`, so
studies are reproducible and replicates are independent. `performance_summary`
reports each measure with its Monte Carlo standard error:

| Measure | Estimate | MCSE |
|---|---|---|
| Bias | `mean(θ̂) − θ` | `SD(θ̂)/√n` |
| Empirical SE | `SD(θ̂)` | `EmpSE/√(2(n−1))` |
| Coverage | `mean(CI ∋ θ)` | `√(cov(1−cov)/n)` |
| Rejection rate | `mean(p < α)` | `√(rate(1−rate)/n)` |

`se_ratio` (mean model SE / empirical SE) flags SE misspecification: values well
below 1 indicate that the model-based SE is too small, which is exactly what the
heteroskedasticity study in `tests/test_simulation.py` demonstrates for non-robust
OLS SEs relative to HC3.

---

## 6. Model selection (`utils/model_selection.py`)

Information criteria are reported with `Δ` values and Akaike weights
`w_i = exp(−Δ_i/2)/Σ exp(−Δ_j/2)` (Burnham & Anderson, 2002); `AICc` adds the
small-sample correction `2k(k+1)/(n−k−1)`. Cross-validation uses shuffled k-fold
splits with out-of-sample RMSE, MAE, and R². Best-subset search is exhaustive;
stepwise search records every step so the path can be reported. All of these are
*exploratory*: inference after selection is not corrected for the selection
process.

---

## 7. Time series (`utils/time_series_utils.py`)

ADF and KPSS are reported together because they have opposite null hypotheses;
the joint decision rule (stationary / non-stationary / conflicting / inconclusive)
avoids the common mistake of treating an ADF non-rejection as evidence of a unit
root. Out-of-sample forecasts use an expanding window (Tashman, 2000).
`diebold_mariano` uses a rectangular HAC variance with `h−1` autocovariances and the
Harvey–Leybourne–Newbold (1997) correction with a `t(n−1)` reference distribution.
Granger causality p-values are the minimum over lags 1..`maxlag` and are therefore
liberal; treat them as screening.

---

## 8. Diagnostics (`utils/diagnostics.py`)

`diagnostic_battery` reports each test with its null hypothesis and a `flag`
column so the direction of a rejection is never ambiguous. Two cautions:

- The Breusch–Pagan auxiliary regression is linear in the regressors and has
  little power against variance patterns that are symmetric in a regressor (for
  example variance proportional to `|X|`); White's test, which includes squares
  and cross-products, catches these. The test suite includes exactly this case.
- Durbin–Watson and the condition number have no p-value; the flags use the
  conventional 1.5–2.5 and 30 thresholds.

---

## 9. Survival and power

Survival helpers wrap `lifelines`: Kaplan–Meier with Greenwood intervals, the
log-rank test, Cox proportional hazards with the Schoenfeld-residual PH test
(Grambsch & Therneau, 1994), and AIC comparison of parametric families. Power
routines wrap `statsmodels.stats.power` and use the Fisher-*z* approximation for
correlations; analytic power should be checked against `simulate_power` when the
test's assumptions are in doubt.

---

## References

- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. *Statistics in Medicine*, 28(25), 3083–3107.
- Austin, P. C. (2011). Optimal caliper widths for propensity-score matching. *Pharmaceutical Statistics*, 10(2), 150–161.
- Bauer, D. J., & Curran, P. J. (2005). Probing interactions in fixed and multilevel regression. *Multivariate Behavioral Research*, 40(3), 373–400.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289–300.
- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42(3), 323–328.
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *QJE*, 119(1), 249–275.
- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum.
- Davidson, R., & Flachaire, E. (2008). The wild bootstrap, tamed at last. *Journal of Econometrics*, 146(1), 162–169.
- DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3), 189–228.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Freedman, D. A. (1981). Bootstrapping regression models. *Annals of Statistics*, 9(6), 1218–1228.
- Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*, 81(3), 515–526.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *IJF*, 13(2), 281–291.
- Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*. Academic Press.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Imai, K., Keele, L., & Tingley, D. (2010). A general approach to causal mediation analysis. *Psychological Methods*, 15(4), 309–334.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615–635.
- Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*, 54(3), 217–224.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325.
- Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine*, 38(11), 2074–2102.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. *Behavior Research Methods*, 40(3), 879–891.
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41–55.
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak instruments. *Econometrica*, 65(3), 557–586.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *IJF*, 16(4), 437–450.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268–274.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
- Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling methods in regression analysis. *Annals of Statistics*, 14(4), 1261–1295.
