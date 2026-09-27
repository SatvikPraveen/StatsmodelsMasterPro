# Reproducibility

StatsmodelsMasterPro is designed so that every number in the repository can be
regenerated exactly and every claim about an estimator can be re-verified.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,survival]"      # library + tests + lifelines
pip install -e ".[app,notebooks]"     # optional: Streamlit dashboard and JupyterLab
```

`requirements_dev.txt` is a full `pip freeze` of a known-good environment
(numpy 2.3, pandas 2.3, scipy 1.16, statsmodels 0.14.5) for exact pinning.

## Datasets

| Step | Command | What it guarantees |
|---|---|---|
| Regenerate | `python synthetic_data/generate_datasets.py` | Rebuilds all 17 CSVs from fixed seeds. |
| Verify | `python scripts/verify_manifest.py --regenerate` | Every CSV matches the SHA-256 in `synthetic_data/MANIFEST.json`, **and** a fresh regeneration reproduces the same hashes byte-for-byte. |
| Document | `python scripts/generate_data_dictionary.py` | `docs/DATA_DICTIONARY.md` reflects `synthetic_data/dgp_registry.py`. |

Five generators (`generate_ols_data`, `generate_glm_data`, `generate_time_series_data`,
`generate_manova_data`, `generate_heteroskedastic_data`) draw from the module-level
`np.random.seed(42)` stream and are therefore order-dependent. The call order in
`generate_all_datasets()` is part of the specification; if it changes, rebuild the
manifest with `python scripts/build_manifest.py` and commit the new hashes.

## Validation

```bash
pytest                                   # 130+ tests, ~15 s (add -m slow for coverage studies)
python scripts/parameter_recovery.py --strict
```

The parameter-recovery study fits the intended model to every dataset and writes
`exports/tables/validation/parameter_recovery.{csv,md}`. Across 57 parameters that are
expected to be recovered, 55 fall inside their 95% confidence intervals; the two
misses (`manova_data` group contrasts) have |z| ≈ 2.3–2.6, which is consistent with
nominal coverage for a single realisation. Estimates that are *not* expected to
recover the conditional DGP (OLS on outlier-contaminated data, population-averaged
GEE) are reported but excluded from the strict gate.

## Randomness policy

- Library code never uses the global NumPy RNG; every stochastic function takes a
  `seed` and builds `np.random.default_rng(seed)`.
- Monte Carlo replicates use `SeedSequence.spawn`, so replicate `i` of a study is
  the same regardless of how many replicates are run.
- The one exception is `mediation_statsmodels`, which seeds the global RNG because
  `statsmodels.stats.mediation` draws from it; this is documented inline.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. `ruff check` on `utils`, `tests`, `scripts`, `synthetic_data`;
2. `pytest` with coverage on Python 3.10, 3.11, and 3.12;
3. `scripts/verify_manifest.py`;
4. `scripts/parameter_recovery.py --strict`, with the resulting tables uploaded as an artifact.

## Notebooks

Notebooks `11`–`13` are written to be executed top-to-bottom from the `notebooks/`
directory and use fixed seeds throughout:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/1[123]_*.ipynb
```
