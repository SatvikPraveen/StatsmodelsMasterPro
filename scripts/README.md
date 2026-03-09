# 🤖 Automation Scripts Documentation

This directory contains 15 comprehensive Python scripts that add professional automation, testing, and tooling capabilities to StatsmodelsMasterPro.

---

## 📋 Table of Contents

1. [Testing Infrastructure](#testing-infrastructure)
2. [Batch Processing & Automation](#batch-processing--automation)
3. [Model Comparison & Validation](#model-comparison--validation)
4. [CLI Tools](#cli-tools)
5. [Documentation & CI/CD](#documentation--cicd)
6. [Quick Start Guide](#quick-start-guide)

---

## 🧪 Testing Infrastructure

### `test_utils.py`
Comprehensive unit tests for all utility modules.

**Tests:**
- Model utilities (coefficients, AIC/BIC, comparisons)
- Diagnostic functions (skewness, kurtosis, heteroscedasticity)
- Integration tests for complete workflows

**Usage:**
```bash
python3 scripts/test_utils.py
```

**Output:** Pass/fail status for all utility functions with detailed error messages.

---

### `test_data_generation.py`
Validates all synthetic datasets for quality and correctness.

**Checks:**
- File existence and loadability
- Expected column structure
- Missing values
- Data type validation (counts are integers, binary is 0/1, etc.)
- Statistical properties

**Usage:**
```bash
python3 scripts/test_data_generation.py
```

**Output:** Validation report for each dataset with ✅/❌ indicators.

---

### `run_all_tests.py`
Master test runner that executes all test suites.

**Features:**
- Runs utility tests
- Runs dataset validation
- Import verification
- Summary report with pass/fail counts

**Usage:**
```bash
python3 scripts/run_all_tests.py
```

**Output:** Complete test suite results with execution summary.

---

## ⚡ Batch Processing & Automation

### `batch_run_all_models.py`
Executes all 25+ statistical models programmatically.

**Models Run:**
- OLS Regression
- Poisson & Logistic GLM
- ARIMA & SARIMAX Time Series
- MANOVA
- Robust Regression
- Panel Data Models
- Survival Analysis

**Usage:**
```bash
python3 scripts/batch_run_all_models.py
```

**Output:** 
- Console summary of each model
- JSON file with results saved to `exports/batch_results/`

---

### `generate_all_reports.py`
Creates comprehensive markdown and HTML reports.

**Generates:**
- Model summaries with statistics
- Coefficient tables
- Interpretation guides
- Markdown report (convertible to HTML with pypandoc)

**Usage:**
```bash
python3 scripts/generate_all_reports.py
```

**Output:** `exports/reports/analysis_report_[timestamp].md`

---

### `export_all_results.py`
Batch exports all plots and tables.

**Exports:**
- Diagnostic plots (residuals, Q-Q, influence)
- Model comparison tables
- Coefficient tables
- Summary statistics
- Correlation heatmaps
- Time series plots

**Usage:**
```bash
python3 scripts/export_all_results.py
```

**Output:** 
- Plots saved to `exports/plots/`
- Tables saved to `exports/tables/`

---

## 📊 Model Comparison & Validation

### `compare_all_models.py`
Automated model comparison suite.

**Comparisons:**
- Nested OLS models (likelihood ratio tests)
- GLM family comparisons (Poisson vs Gaussian vs Negative Binomial)
- Robust regression methods (OLS vs RLM vs WLS)
- ARIMA order selection

**Usage:**
```bash
python3 scripts/compare_all_models.py
```

**Output:** CSV files with comparison results in `exports/comparisons/`

---

### `validate_datasets.py`
Comprehensive dataset assumption validation.

**Validates:**
- **OLS**: Linearity, multicollinearity (VIF), normality, homoscedasticity, outliers, independence
- **GLM**: Count data types, mean-variance relationship, binary outcomes
- **Time Series**: Stationarity (ADF test), autocorrelation
- **Panel Data**: Balance, structure, missing values

**Usage:**
```bash
python3 scripts/validate_datasets.py
```

**Output:** Detailed validation report with statistical test results.

---

### `benchmark_performance.py`
Performance benchmarking for all model types.

**Benchmarks:**
- Model fitting time
- Memory usage (peak and current)
- Prediction speed
- Scalability with data size

**Usage:**
```bash
python3 scripts/benchmark_performance.py
```

**Output:** 
- Console ranking of fastest to slowest models
- CSV file saved to `exports/benchmarks/`

---

## 💻 CLI Tools

### `cli.py`
Professional command-line interface for the entire project.

**Commands:**
```bash
# Run all tests
python3 scripts/cli.py test all

# Run specific analysis
python3 scripts/cli.py analyze ols

# Export all results
python3 scripts/cli.py export all

# Validate datasets
python3 scripts/cli.py validate

# Generate reports
python3 scripts/cli.py report

# Run benchmarks
python3 scripts/cli.py benchmark

# Compare models
python3 scripts/cli.py compare

# Show project info
python3 scripts/cli.py info
```

**Features:**
- Unified interface for all automation
- Subcommands for different operations
- Help text with examples

---

### `quick_analysis.py`
Run statistical analyses from command line.

**Usage:**
```bash
# Interactive mode
python3 scripts/quick_analysis.py

# Direct analysis
python3 scripts/quick_analysis.py ols
python3 scripts/quick_analysis.py glm poisson
python3 scripts/quick_analysis.py glm logistic
python3 scripts/quick_analysis.py arima

# Custom data
python3 scripts/quick_analysis.py ols --data mydata.csv
```

**Features:**
- Interactive or command-line mode
- Auto-detects columns
- Pretty-printed results
- Coefficient tables with p-values

---

## 📚 Documentation & CI/CD

### `generate_api_docs.py`
Auto-generates API documentation from docstrings.

**Generates:**
- Function signatures
- Parameter descriptions
- Docstrings
- Source code snippets
- Table of contents

**Usage:**
```bash
python3 scripts/generate_api_docs.py
```

**Output:** `docs/API_DOCUMENTATION.md`

---

### `check_project_health.py`
Comprehensive project health checker.

**Checks:**
- Required files exist
- Python syntax errors
- Import validity
- Dataset availability
- Streamlit pages
- Jupyter notebooks
- Git repository status

**Usage:**
```bash
python3 scripts/check_project_health.py
```

**Output:** Health report with errors, warnings, and overall status.

---

### `update_readme_stats.py`
Auto-updates README with current project statistics.

**Updates:**
- File counts (pages, notebooks, datasets)
- Lines of code
- Git commit count
- Last updated timestamp

**Usage:**
```bash
python3 scripts/update_readme_stats.py
```

**Output:** Updated README.md with statistics section.

---

## 🚀 Quick Start Guide

### First Time Setup
```bash
# Make all scripts executable
chmod +x scripts/*.py

# Check project health
python3 scripts/cli.py info

# Run all tests
python3 scripts/cli.py test all

# Validate datasets
python3 scripts/cli.py validate
```

### Daily Workflow
```bash
# Run a quick analysis
python3 scripts/quick_analysis.py

# Generate reports
python3 scripts/cli.py report

# Export results
python3 scripts/cli.py export all
```

### Portfolio Preparation
```bash
# Run everything
python3 scripts/batch_run_all_models.py
python3 scripts/export_all_results.py
python3 scripts/generate_all_reports.py

# Update documentation
python3 scripts/generate_api_docs.py
python3 scripts/update_readme_stats.py

# Final health check
python3 scripts/check_project_health.py
```

---

## 📂 Output Directories

All scripts save their outputs to organized directories:

```
exports/
├── batch_results/     # JSON files from batch model runs
├── benchmarks/        # Performance benchmark results
├── comparisons/       # Model comparison tables
├── plots/            # Exported diagnostic plots
├── reports/          # Generated markdown/HTML reports
└── tables/           # Coefficient and summary tables

docs/
└── API_DOCUMENTATION.md  # Auto-generated API docs
```

---

## 🎯 Common Use Cases

### For Portfolio/Demo
1. Run `batch_run_all_models.py` to demonstrate all capabilities
2. Run `export_all_results.py` to create visualizations
3. Run `generate_all_reports.py` for shareable documentation

### For Development
1. Run `test_utils.py` after modifying utility functions
2. Run `validate_datasets.py` after data changes
3. Run `check_project_health.py` before commits

### For Analysis
1. Use `quick_analysis.py` for exploratory analysis
2. Use `compare_all_models.py` to select best models
3. Use `benchmark_performance.py` to optimize

---

## 💡 Tips

- All scripts provide verbose output with ✅/❌ indicators
- Use `--help` flag with CLI tools for detailed options
- Scripts are designed to work from project root directory
- Check `exports/` directory for all generated files
- Use `cli.py` as the main entry point for all operations

---

**Last Updated:** March 9, 2026  
**Total Scripts:** 15  
**Total Lines:** ~3,500+
