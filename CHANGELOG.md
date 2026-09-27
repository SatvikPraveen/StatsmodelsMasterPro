# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-09-27

Research-grade release. The project moves from a teaching portfolio to a
validated, reproducible toolkit.

### Added
- **Packaging and CI**: `pyproject.toml` (installable with `pip install -e ".[dev]"`),
  GitHub Actions running ruff, pytest on Python 3.10–3.12, manifest verification,
  and a parameter-recovery gate.
- **Test suite** (`tests/`, 130+ tests): every estimator is checked against a known
  data-generating process; bootstrap coverage and Monte Carlo behaviour are
  verified empirically.
- `utils/inference.py`: percentile/basic/normal/BCa bootstrap CIs, pairs/residual/wild
  regression bootstrap, two-sample and paired permutation tests, HC0–HC3/HAC/cluster
  robust SE comparison, Wald and likelihood-ratio tests, multiple-testing corrections.
- `utils/effect_sizes.py`: Cohen's d, Hedges' g, Glass's Δ, Cliff's δ, CLES,
  η²/partial η²/ω², Cramér's V, odds ratios, with CIs and interpretation labels.
- `utils/causal.py`: propensity scores and balance tables, stabilised IPW (ATE/ATT),
  caliper matching, difference-in-differences, 2SLS with weak-instrument F,
  sharp regression discontinuity with bandwidth sensitivity, E-values.
- `utils/mediation.py`: bootstrap indirect-effect CIs, Sobel test, statsmodels
  ACME/ADE cross-check, moderation with simple slopes and Johnson–Neyman regions.
- `utils/simulation.py`: ADEMP Monte Carlo engine with Monte Carlo standard errors
  (bias, empirical/model SE, RMSE, coverage, rejection rate), power curves,
  plug-in DGPs and estimators.
- `utils/model_selection.py`: AIC/AICc/BIC with Akaike weights, k-fold CV, best-subset
  and stepwise search with trace, nested F-test.
- `utils/reporting.py`: tidy/glance tables, stargazer-style regression tables in
  DataFrame/LaTeX/Markdown/HTML, APA 7 strings, coefficient forest plots,
  dependency-free Markdown and LaTeX writers.
- `utils/time_series_utils.py`: joint ADF/KPSS decision, ARIMA order search,
  residual diagnostics, rolling-origin evaluation, Diebold–Mariano, Granger matrix,
  MASE and other accuracy metrics.
- `utils/survival_utils.py`, `utils/power.py`: Kaplan–Meier, log-rank, Cox PH with
  PH checks, parametric comparison; solve-for-any-unknown power routines.
- `utils/diagnostics.py`: `diagnostic_battery`, `vif_table`, `influence_summary`.
- **Reproducibility layer**: `synthetic_data/dgp_registry.py` (true parameters for all
  17 datasets), `MANIFEST.json` with SHA-256 hashes, `scripts/verify_manifest.py`,
  `scripts/generate_data_dictionary.py`, `scripts/parameter_recovery.py`.
- **Docs**: `docs/METHODOLOGY.md`, `docs/REPRODUCIBILITY.md`, `docs/DATA_DICTIONARY.md`,
  `CITATION.cff`.
- Streamlit pages 26–29 (robust inference, causal inference, Monte Carlo validation,
  publication reporting) and notebooks 11–13.

### Fixed
- Duplicate function definitions in `streamlit_app/utils/st_helpers.py` and
  `utils/model_utils.py`; a non-functional `DescrStatsW.bootstrap` helper.
- Deprecated seaborn `ci=` arguments.
- Dataset validation expected the wrong columns for `gee_data.csv` and
  `zero_inflated_count.csv`.
- README leftovers and a broken clone URL.

### Changed
- All commits are now attributed to a single author identity.
- Legacy bootstrap/simulation helpers accept a `seed` and use `numpy.random.Generator`.

## [1.0.0] - 2025-09

Initial release: 25 Streamlit modules, 10 notebooks, 18 synthetic datasets,
automation scripts, Docker support.
