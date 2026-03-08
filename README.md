# 🧠 StatsmodelsMasterPro

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-darkgreen.svg)](https://www.python.org/)
[![Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-100%25-brightgreen.svg)](https://www.statsmodels.org/)
[![Streamlit Ready](https://img.shields.io/badge/Streamlit-Optional-blueviolet.svg)](https://streamlit.io/)
[![Synthetic Datasets](https://img.shields.io/badge/Data-Synthetic-lightblue.svg)](./synthetic_data/)

---

## 🎯 Project Goal

**StatsmodelsMasterPro** is a portfolio-grade project crafted to master **statistical modeling, inference, and diagnostics** using the [`statsmodels`](https://www.statsmodels.org/) library.

Instead of relying on real-world noise, this project uses **carefully designed synthetic data** to help you focus on **concepts, assumptions, syntax, and model diagnostics** — the way they were meant to be learned.

---

## 🚀 Key Features

✅ **25 Statistical Modules** covering foundational to advanced techniques  
✅ **18 Synthetic Datasets** carefully designed for each analysis type  
✅ **Complete Streamlit Dashboard** with 25 interactive pages  
✅ **Jupyter Notebooks** for in-depth exploration  
✅ **Production-Ready Code** with proper error handling and diagnostics  
✅ **Comprehensive Visualizations** for every analysis  
✅ **Export Functionality** - Download results as CSV  
✅ **Modular Utilities** - Reusable functions in `utils/`  
✅ **Comparison Studies** - statsmodels vs scipy.stats  
✅ **Docker Support** for reproducible environments

---

## 📚 Complete Module List

### 🔹 Foundational Statistics
1. **Descriptive Statistics** - Summary stats, distributions, correlations
2. **OLS Regression** - Ordinary Least Squares with diagnostics
3. **GLM Models** - Poisson, Binomial, Gaussian, Gamma families
4. **Hypothesis Testing** - t-tests, ANOVA, effect sizes

### 🔹 Time Series Analysis
5. **Basic Time Series** - ARIMA, ACF/PACF, ADF tests
6. **Advanced Time Series** ⭐ ENHANCED
   - SARIMAX (Seasonal ARIMA with exogenous variables)
   - VAR (Vector Autoregression)
   - VECM (Vector Error Correction for cointegrated series)
   - Granger Causality Tests

### 🔹 Multivariate Statistics
7. **Multivariate Stats** - Hotelling's T², MANOVA
8. **Model Diagnostics** ⭐ ENHANCED - VIF, residuals, influence, advanced tests
   - Durbin-Watson (autocorrelation)
   - Jarque-Bera (normality)
   - Rainbow Test (linearity)
   - RESET Test (specification)
9. **Model Selection** - AIC, BIC, likelihood ratio tests

### 🔹 Advanced Inference
10. **Inference & Interpretation** - Confidence intervals, standardized coefficients
11. **Posthoc Tests** - Tukey HSD, Bonferroni corrections
12. **Bootstrap CI** - Non-parametric confidence intervals

### 🔹 Comparison Studies
13. **t-test Comparison** - statsmodels vs scipy
14. **Correlation Comparison** - Different methods compared
15. **CI Comparison** - Parametric vs bootstrap
16. **Distribution Simulation** - Monte Carlo demonstrations
17. **Summary Dashboard** - Overview of all analyses

### 🔹 Advanced Statistical Methods ⭐ ALL NEW
18. **Robust Regression**
    - Weighted Least Squares (WLS)
    - Robust Linear Models (RLM)
    - Quantile Regression
    
19. **Nonparametric Tests**
    - Mann-Whitney U Test
    - Wilcoxon Signed-Rank Test
    - Kruskal-Wallis H Test
    - Friedman Test
    
20. **Power Analysis**
    - Sample size calculations
    - Power curves
    - Effect size determination
    - t-test, ANOVA, correlation power
    
21. **Survival Analysis**
    - Kaplan-Meier Curves
    - Log-Rank Tests
    - Cox Proportional Hazards Models
    
22. **Panel Data Analysis**
    - Pooled OLS
    - Fixed Effects Models
    - Random Effects Models
    
23. **GEE Models**
    - Generalized Estimating Equations
    - Multiple correlation structures
    - Population-averaged effects
    
24. **Mediation & Moderation**
    - Mediation analysis (Baron & Kenny)
    - Moderation/Interaction effects
    - Simple slopes analysis

25. **Zero-Inflated Models** ⭐ NEW
    - Zero-Inflated Poisson (ZIP)
    - Zero-Inflated Negative Binomial (ZINB)
    - Model comparison with standard count models
    - Handling excess zeros in count data

---

## 💡 Why Synthetic?

Most tutorials use real-world datasets with domain-specific quirks. This project intentionally uses **carefully constructed synthetic data** to:

- Focus on statistical **principles** and **assumptions**
- Understand **inference over noise**
- Debug with **full control** over the generating process
- Build **intuition** before diving into messy real data

## 💼 Who This Is For

Perfect for:

- 📊 **Data Scientists** refining statistical intuition
- 🧠 **AI/ML Engineers** grounding predictions with inference
- 🧪 **Researchers** needing reproducible modeling templates
- 🎓 **Students** prepping for interviews, PhD, or coursework

---

## 🌱 Learning Outcomes

By the end, you’ll confidently:

### Core Skills
- Model linear, logistic, and multivariate data
- Evaluate assumptions with professional-grade visuals
- Compare statistical methods between libraries
- Automate tests and diagnostics for any model
- Master the internals of `statsmodels` over black-box tools

### Advanced Techniques
- Handle **outliers and heteroskedasticity** with robust regression
- Model **seasonal patterns** and **multivariate time series**
- Analyze **time-to-event** data with survival analysis
- Work with **panel/longitudinal** data using fixed and random effects
- Model **correlated data** with GEE
- Perform **power analysis** for study design
- Understand **mediation and moderation** effects
- Apply **nonparametric tests** when assumptions fail
- Model **excess zeros** with zero-inflated count models
- Test for **cointegration** and apply VECM
- Perform **advanced diagnostics** (autocorrelation, linearity, specification)

---

## 📊 Synthetic Datasets

All datasets are carefully constructed to demonstrate specific statistical concepts:

| Dataset | Purpose | Key Features |
|---------|---------|--------------|
| `ols_data.csv` | Linear regression | Clean linear relationships |
| `ols_diagnostics.csv` | Model diagnostics | Outliers, multicollinearity, heteroskedasticity |
| `glm_poisson.csv` | Count data | Poisson-distributed outcomes |
| `glm_logistic.csv` | Binary classification | Logistic regression |
| `arima_series.csv` | Time series | AR/MA processes |
| `seasonal_ts_data.csv` ⭐ | Seasonal patterns | SARIMAX modeling |
| `var_data.csv` ⭐ | Multivariate TS | Vector autoregression |
| `manova_data.csv` | Multivariate analysis | Multiple DVs |
| `multivariate_group_data.csv` | Group comparisons | Hotelling's T² |
| `heteroskedastic_data.csv` | Non-constant variance | WLS demonstration |
| `robust_regression_data.csv` ⭐ | Outliers & heteroskedasticity | RLM, quantile regression |
| `posthoc_dataset.csv` | Multiple comparisons | Tukey, Bonferroni |
| `panel_data.csv` ⭐ | Longitudinal | Fixed/random effects |
| `survival_data.csv` ⭐ | Time-to-event | Censoring, Cox PH |
| `gee_data.csv` ⭐ | Clustered data | GEE models |
| `mediation_data.csv` ⭐ | Causal pathways | Mediation/moderation |
| `zero_inflated_count.csv` ⭐ | Excess zeros | ZIP/ZINB models |

⭐ = New advanced datasets

---

## 🧱 Project Structure

```bash
StatsmodelsMasterPro/
├── notebooks/                 # Concept notebooks (OLS, GLM, ANOVA, etc.)
├── common_tests/              # Shared logic with Scipy, manual tests
├── cheatsheets/               # Condensed syntax & theory guides (Markdown)
├── exports/                   # All model outputs (plots, tables)
│   ├── plots/
│   └── tables/
├── synthetic_data/            # Dataset generation scripts and saved CSVs
├── utils/                     # Custom utilities: diagnostics, bootstrap, etc.
├── scripts/                   # Extra scripts, one-time jobs or tools
├── pages/                     # Streamlit multi-page UI files (if used)
├── streamlit_app/             # Streamlit dashboard logic and utils
├── sm_env/                    # Optional: environment setup or virtualenv storage
├── Home.py                    # Streamlit entry point
├── entrypoint.sh              # Entrypoint script for Docker (Jupyter/Streamlit)
├── Dockerfile                 # Docker build instructions
├── docker-compose.yml         # Compose setup for Jupyter/Streamlit modes
├── requirements.txt           # Core project dependencies
├── requirements_dev.txt       # Extended dev dependencies
├── .gitignore                 # Git exclusions
├── .dockerignore              # Docker context exclusions
└── README.md                  # Main documentation and overview

```

---

Perfect — let’s replace the demo image placeholders with actual **local plot references** using relative paths from within your `README.md`. Here's a clean markdown block you can directly **copy-paste** into your `README.md`, assuming you've placed your images in the `exports/plots/` folder.

---

## 📁 Visual Preview

Following are some of the plots from the project.

### 📊 Q–Q Plot from Residual Diagnostics

Used to assess normality of residuals:

![Q–Q Plot](exports/plots/07_model_diagnostics/qqplot.png)

> _(This can come from `07_model_diagnostics.ipynb`.)_

---

### 📈 Residuals vs Fitted Plot

To identify heteroskedasticity or nonlinearity:

![Residuals vs Fitted](exports/plots/07_model_diagnostics/residuals_vs_fitted.png)

---

### 🔄 ACF/PACF for Time Series

Used in `05_time_series_analysis.ipynb`:

![ACF PACF Plot](exports/plots/05_time_series_analysis/acf_pacf.png)

---

## 🧪 Cheatsheets Summary

📁 `cheatsheets/` contains Markdown guides for quick revision:

| Cheatsheet               | Topics Covered                               |
| ------------------------ | -------------------------------------------- |
| `statsmodels_cheatsheet` | OLS, GLM, ANOVA, AIC/BIC, CI, diagnostics    |
| `docker_cheatsheet`      | Build, compose, volume, ports, entrypoints   |
| `streamlit_cheatsheet`   | Layouts, inputs, plot, export, session state |

---

## ⚙️ Installation & Development Setup

Follow these steps to set up your local development environment:

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/your-username/StatsmodelsMasterPro.git
cd StatsmodelsMasterPro
```

### 🐍 2. Create and Activate Virtual Environment

#### For macOS/Linux

```bash
python3 -m venv sm_env
source sm_env/bin/activate
```

#### For Windows (CMD)

```cmd
python -m venv sm_env
sm_env\Scripts\activate
```

#### For Windows (PowerShell)

```powershell
python -m venv sm_env
.\sm_env\Scripts\Activate.ps1
```

### 📦 3. Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 For development/testing extras, you can also install:

```bash
pip install -r requirements_dev.txt
```

### 🎲 4. Generate Synthetic Datasets

```bash
cd synthetic_data
python generate_datasets.py
cd ..
```

This will create 16+ CSV files in the `synthetic_data/` folder.

---

## 🚀 Usage

### Option 1: Streamlit Dashboard (Interactive)

Launch the interactive web dashboard:

```bash
streamlit run Home.py
```

Then navigate to `http://localhost:8501` in your browser.

**Features:**
- 24 interactive pages
- Real-time parameter adjustment
- Export results as CSV
- Comprehensive visualizations

### Option 2: Jupyter Notebooks (Deep Dive)

Launch Jupyter Lab for detailed exploration:

```bash
jupyter lab
```

Navigate to the `notebooks/` directory and open any `.ipynb` file.

### Option 3: Docker (Reproducible)

Run with Docker for a consistent environment:

```bash
# Build the image
docker-compose build

# Run Streamlit mode
docker-compose up streamlit

# Or run Jupyter mode
docker-compose up jupyter
```

---

## 🛠️ Tech Stack

### Core Libraries
- **statsmodels** 0.14+ - Statistical models and tests
- **scipy** 1.10+ - Scientific computing
- **pandas** 2.0+ - Data manipulation
- **numpy** 1.24+ - Numerical computing

### Visualization
- **matplotlib** 3.7+ - Plotting
- **seaborn** 0.12+ - Statistical visualizations
- **plotly** 5.15+ - Interactive plots

### Advanced Methods
- **lifelines** 0.27+ - Survival analysis
- **scikit-learn** 1.3+ - Preprocessing and utilities

### Interfaces
- **streamlit** 1.26+ - Web dashboard
- **jupyter lab** 4.0+ - Notebooks
- **ipython** - Enhanced REPL

---

## 📋 Requirements

**Python Version:** 3.9+

**Key Dependencies:**
```
statsmodels>=0.14
scipy>=1.10
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.26
lifelines>=0.27
scikit-learn>=1.3
jupyterlab>=4.0
```

See `requirements.txt` for complete list.

---

✅ You can now start exploring the Jupyter notebooks or launch the optional Streamlit dashboard!

---

## 🐳 Dockerized Interface

This project supports **Streamlit dashboards** and **JupyterLab notebooks** via Docker.

### 🔧 Prerequisites

- Docker installed ([Download](https://www.docker.com/products/docker-desktop/))
- Docker Compose v2

### 🚀 Build and Run

```bash
# Default (Streamlit Mode)
docker compose up --build

# JupyterLab Mode
APP_MODE=jupyter docker compose up --build
```

## 📘 Notebook Modules

| Notebook                       | Concepts Covered                                                    |
| ------------------------------ | ------------------------------------------------------------------- |
| `01_intro_descriptive`         | EDA, skew/kurtosis, summary stats, correlation heatmaps             |
| `02_linear_models`             | OLS (simple & multiple), predictions, R², interpretation            |
| `03_generalized_linear_models` | GLM families: Poisson, Binomial, Probit, Logit                      |
| `04_hypothesis_testing`        | t-tests, custom contrasts, assumptions                              |
| `05_time_series_analysis`      | AR/MA/ARIMA, ACF/PACF, diagnostics                                  |
| `06_multivariate_stats`        | MANOVA, fallback Hotelling’s T², ICC                                |
| `07_model_diagnostics`         | Residual plots, Q–Q, leverage, heteroskedasticity                   |
| `08_model_selection`           | AIC/BIC, nested models, stepwise hints                              |
| `09_inference_interpretation`  | CIs, marginal effects, statistical vs practical significance        |
| `10_posthoc_analysis`          | Tukey’s HSD, Bonferroni, pairwise CI plots, fallback Hotelling’s T² |

---

## 🔁 Shared Notebooks with SciPy (`common_tests/`)

We include head-to-head comparisons between `statsmodels` and `scipy.stats`, with clear visual and numerical comparisons.

| Notebook                         | Tests/Topics Covered                              |
| -------------------------------- | ------------------------------------------------- |
| `common_ttest_comparison`        | t-test (paired/unpaired, equal/unequal var)       |
| `common_kstest_comparison`       | KS test on synthetic distributions                |
| `common_correlation`             | Pearson, Spearman, Kendall, heatmaps              |
| `common_ci_comparison`           | Manual vs `statsmodels` CIs, side-by-side output  |
| `common_bootstrap_ci`            | Bootstrap CIs for group means + visual error bars |
| `common_distribution_simulation` | ECDF, KDE, normal comparison, resampling methods  |

---

## 🛠 Utility Modules

Our `utils/` directory contains reusable logic:

- 📊 `diagnostics.py` → Residual plots, leverage, heteroskedasticity, skew/kurt
- 🧮 `model_utils.py` → Model summaries, t/F-stats, CI bounds
- 🎨 `visual_utils.py` → Save plots, themes, CDF/ECDF helpers
- 📐 `hotelling_utils.py` → Manual Hotelling’s T² with NumPy fallback
- 🧪 `bootstrap_utils.py` → Bootstrap CI logic and formatting

> All outputs are routed to `exports/` and version-safe.

---

## 📊 Exports Example

```bash
exports/
├── plots/
│   ├── residuals_vs_fitted.png
│   ├── acf_comparison.png
│   └── cdf_comparison_Y1.png
├── tables/
│   ├── ols_summary_Y1.csv
│   ├── hotelling_t2_fallback.csv
│   └── bootstrap_ci_groupB.csv
```

---

## 📈 Model Auditing & Interpretation Support

Beyond modeling, this project emphasizes **interpretable inference and diagnostics**:

- 🧪 Visual assumption checks (Q–Q, leverage, residuals, skewness)
- 📊 Confidence & prediction intervals with bootstrap validation
- 🧩 Posthoc tests (Tukey, Bonferroni) and fallback Hotelling’s T²
- 🧠 Emphasis on **statistical vs practical significance**
- 🧾 Clear tabular exports (R², CI bounds, F-stats, effect sizes)

> Everything is built to **explain**, not just **compute**.

---

### 🧠 What You Get

| Interface  | Port | Description                                 |
| ---------- | ---- | ------------------------------------------- |
| Streamlit  | 8501 | Run `Home.py` interactive stats dashboard   |
| JupyterLab | 8899 | Full project access in notebook environment |

Project files are mounted inside the container with live sync.

---

## 🔄 Maintenance & Future Extensions

Even after completion, StatsmodelsMasterPro is designed to grow:

- 🔌 Add new models (Mixed Effects, GEE, Survival)
- 🔎 Include domain-specific use cases (Econometrics, Biostatistics)
- 🎯 Integrate `linearmodels`, `lifelines`, or `pingouin` modules
- 📊 Export results to LaTeX or publication-ready tables
- 🧪 Add test coverage for utilities (e.g., residual analysis, CI)

> This isn’t a one-off project — it’s a reusable and extensible stats toolkit.

---

## 🙌 Acknowledgements

Special thanks to the `statsmodels` and `scipy` teams for building libraries that make statistical modeling accessible, open, and extensible.

---

## 🔗 Related Projects

Looking to master other areas of data science and visualization?

Check out these companion repositories:

- 📊 [PandasPlayground](https://github.com/SatvikPraveen/PandasPlayground): Master pandas with real-world data pipelines and EDA workflows.
- 🔢 [NumPyMasterPro](https://github.com/SatvikPraveen/NumPyMasterPro): Dive deep into NumPy with structured notebooks, utilities, and broadcasting tricks.
- 🎨 [MatplotlibMasterPro](https://github.com/SatvikPraveen/MatplotlibMasterPro): Build static and animated visualizations with Matplotlib from scratch.
- 📘 [SeabornMasterPro](https://github.com/SatvikPraveen/SeabornMasterPro): Learn statistical visualization using Seaborn with layered plots and dashboards.
- 🌐 [PlotlyVizPro](https://github.com/SatvikPraveen/PlotlyVizPro): Create interactive dashboards using Plotly Express and Graph Objects.

Together, these projects offer a full-stack toolkit for mastering data science workflows.

### 📜 License

This project is licensed under the **GNU General Public License v3.0**.

> You are free to **use, study, share, and modify** this project under the terms of the GPLv3.
> Contributions are welcome and should also be licensed under GPLv3.
