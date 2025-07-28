# streamlit_app/utils/st_helpers.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import statsmodels.stats.api as sms

# === File Paths ===
SYN_DATA_PATH = Path("synthetic_data/ols_data.csv")
EXPORT_DIR = Path("exports/plots")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ========================
# 📁 Data Handling Utilities
# ========================
def load_default_data():
    """Load default synthetic dataset."""
    return pd.read_csv(SYN_DATA_PATH)


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include="object").columns.tolist()


def filter_columns(df: pd.DataFrame, cols: list):
    return df[cols]


# ===========================
# 🎨 Common Visualization Blocks
# ===========================
def show_corr_heatmap(df, cols):
    fig, ax = plt.subplots()
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def show_pairplot(df, cols):
    sns_plot = sns.pairplot(df[cols])
    st.pyplot(sns_plot)


def show_histogram(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)


def show_boxplot(df, col, group=None):
    fig, ax = plt.subplots()
    if group:
        sns.boxplot(x=df[group], y=df[col], ax=ax)
    else:
        sns.boxplot(y=df[col], ax=ax)
    st.pyplot(fig)


def save_plot_streamlit(fig, filename):
    fig.savefig(EXPORT_DIR / f"{filename}.png", bbox_inches="tight")


# ========================
# 🧮 OLS Modeling Helpers
# ========================
def run_ols_model(df, response, predictors):
    predictors_str = " + ".join(predictors)
    formula = f"{response} ~ {predictors_str}"
    model = smf.ols(formula=formula, data=df).fit()
    return model


def show_model_summary(model):
    st.text("Model Summary:")
    st.text(model.summary())


def plot_residuals(model):
    fig, ax = plt.subplots()
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    st.pyplot(fig)


# ========================
# 📊 Hypothesis Testing
# ========================
def run_ttest_ind(df, col, group_col):
    groups = df[group_col].unique()
    g1 = df[df[group_col] == groups[0]][col]
    g2 = df[df[group_col] == groups[1]][col]
    t_stat, p_value = stats.ttest_ind(g1, g2)
    return t_stat, p_value


def run_anova(df, response, group):
    formula = f"{response} ~ C({group})"
    model = smf.ols(formula, data=df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    return model, anova_result


# ========================
# 🔁 GLM Modeling Helpers
# ========================
def run_glm(df, response, predictors, family):
    predictors_str = " + ".join(predictors)
    formula = f"{response} ~ {predictors_str}"
    model = smf.glm(formula=formula, data=df, family=family).fit()
    return model


def get_glm_family(family_name):
    family_map = {
        "Poisson": sm.families.Poisson(),
        "Binomial": sm.families.Binomial(),
        "Gaussian": sm.families.Gaussian(),
        "Gamma": sm.families.Gamma(),
    }
    return family_map.get(family_name, sm.families.Gaussian())


# ========================
# 📉 CI Plot Helpers
# ========================
def plot_ci_barplot(df, x_col, y_col, lower_ci, upper_ci, title):
    fig, ax = plt.subplots()
    yerr = [df[y_col] - df[lower_ci], df[upper_ci] - df[y_col]]
    ax.bar(df[x_col], df[y_col], yerr=yerr, capsize=5)
    ax.set_title(title)
    ax.set_ylabel("Mean ± CI")
    st.pyplot(fig)


def display_glm_coefficients(model):
    """Styled GLM coefficient summary table"""
    summary_df = model.summary2().tables[1].reset_index()
    summary_df.columns = ["Predictor", "Coef", "Std Err", "z", "P>|z|", "[0.025", "0.975]"]
    st.dataframe(summary_df.style.format({
        "Coef": "{:.4f}",
        "Std Err": "{:.4f}",
        "z": "{:.2f}",
        "P>|z|": "{:.3f}",
        "[0.025": "{:.4f}",
        "0.975]": "{:.4f}"
    }))


import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def plot_glm_predictions(model, df, response_col):
    predicted = model.fittedvalues
    actual = df[response_col]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Check for prediction collapse
    if predicted.nunique() <= 2:
        st.warning("⚠️ Predictions are nearly constant — check GLM family or predictor configuration.")

        # Residual plot instead
        residuals = actual - predicted
        sns.residplot(x=predicted, y=residuals, lowess=True, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title("Residual Plot (Constant Prediction Detected)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
    else:
        # Normal actual vs predicted plot
        ax.scatter(actual, predicted, alpha=0.7)
        ax.plot([actual.min(), actual.max()],
                [actual.min(), actual.max()],
                'r--', lw=2, label="Ideal Fit")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("GLM: Actual vs Predicted")
        ax.legend()

    st.pyplot(fig)


# ========================
# 🧪 Utility Formatting
# ========================
def format_p_value(p):
    """Return formatted string for p-value"""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


# ===============================
# 📊 EDA Helpers for Streamlit Pages
# ===============================

def plot_histograms(df):
    """Plot histograms for each column in a DataFrame."""
    for col in df.columns:
        st.markdown(f"**Histogram for `{col}`**")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)


def plot_boxplots(df):
    """Plot boxplots for each column in a DataFrame."""
    for col in df.columns:
        st.markdown(f"**Boxplot for `{col}`**")
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax, color="orange")
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)


def plot_correlation_matrix(df):
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)


def compute_skew_kurtosis(df):
    """Return a dictionary of skewness and kurtosis for each column."""
    stats_dict = {}
    for col in df.columns:
        stats_dict[col] = {
            "Skewness": df[col].skew(),
            "Kurtosis": df[col].kurtosis()
        }
    return stats_dict


# ========================
# 📊 OLS Prediction & Plotting
# ========================
def plot_actual_vs_predicted(model, df, response):
    """Plot actual vs predicted values."""
    predictions = model.predict(df)
    fig, ax = plt.subplots()
    ax.scatter(df[response], predictions, alpha=0.7)
    ax.plot([df[response].min(), df[response].max()],
            [df[response].min(), df[response].max()],
            'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


def display_model_coefficients(model):
    """Display coefficients with p-values."""
    coef_df = pd.DataFrame({
        "Coefficient": model.params,
        "P-value": model.pvalues.apply(format_p_value)
    })
    st.dataframe(coef_df.style.format(precision=4))


# ========================
# 📊 Hypothesis Tests - utilities
# ========================
def plot_group_distributions(df, col, group_col):
    """Plot histograms of numeric column by group."""
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=col, hue=group_col, kde=True, ax=ax, palette="Set2")
    ax.set_title(f"Distribution of {col} by {group_col}")
    st.pyplot(fig)


def plot_anova_boxplot(df, response, group_col):
    """Boxplot for visualizing ANOVA group differences."""
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=group_col, y=response, ax=ax, palette="Set3")
    ax.set_title(f"{response} by {group_col}")
    st.pyplot(fig)


# Inside streamlit_app/utils/st_helpers.py
def display_ttest_result(t_stat, p_value, use_welch=False):
    test_type = "Welch's T-Test" if use_welch else "Standard T-Test"
    st.markdown(f"**{test_type}**")
    st.markdown(f"**t-statistic:** `{t_stat:.3f}`")
    st.markdown(f"**p-value:** `{p_value:.4f}`")



def display_anova_table(anova_result):
    """Display formatted ANOVA table."""
    st.dataframe(anova_result.style.format(precision=4))


# ========================
# ⏱ Time Series Utilities
# ========================
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

def check_stationarity(series, alpha=0.05):
    """ADF test for stationarity"""
    result = adfuller(series.dropna())
    stat, p_value = result[0], result[1]
    is_stationary = p_value < alpha
    return stat, p_value, is_stationary

def plot_time_series(df, col):
    fig, ax = plt.subplots()
    df[col].plot(ax=ax, title=f"Time Series Plot: {col}")
    st.pyplot(fig)

def plot_acf_pacf(series, lags=20):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sm.graphics.tsa.plot_acf(series.dropna(), lags=lags, ax=axs[0])
    sm.graphics.tsa.plot_pacf(series.dropna(), lags=lags, ax=axs[1])
    axs[0].set_title("ACF")
    axs[1].set_title("PACF")
    st.pyplot(fig)

def fit_arima_model(series, order=(1, 0, 0)):
    model = ARIMA(series.dropna(), order=order).fit()
    return model

def plot_forecast(series, model, steps=10):
    forecast = model.get_forecast(steps=steps)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(ax=ax, label='Observed')
    pred.plot(ax=ax, label='Forecast', color='red')
    ax.fill_between(pred.index, ci.iloc[:, 0], ci.iloc[:, 1], color='pink', alpha=0.3)
    ax.set_title("Forecast with Confidence Interval")
    ax.legend()
    st.pyplot(fig)


# ========================
# 🎯 Multivariate Statistics Helpers
# ========================
from scipy.stats import chi2
import numpy as np

def compute_hotelling_t2(group1_df, group2_df, cols):
    """
    Compute Hotelling's T² statistic for two groups across multiple variables.
    """
    X1 = group1_df[cols].values
    X2 = group2_df[cols].values
    n1, n2 = X1.shape[0], X2.shape[0]
    mean1, mean2 = X1.mean(axis=0), X2.mean(axis=0)
    pooled_cov = ((np.cov(X1, rowvar=False) * (n1 - 1)) +
                  (np.cov(X2, rowvar=False) * (n2 - 1))) / (n1 + n2 - 2)
    diff = mean1 - mean2
    T2 = (n1 * n2) / (n1 + n2) * diff.T @ np.linalg.inv(pooled_cov) @ diff
    df1 = len(cols)
    df2 = n1 + n2 - df1 - 1
    F = (df2 * T2) / (df1 * (n1 + n2 - 2))
    p_value = 1 - chi2.cdf(T2, df1)
    return {
        "T2": T2,
        "F": F,
        "p_value": p_value
    }

def show_multivariate_distributions(df, cols):
    """Pairplot for multivariate relationships"""
    group_col = list(set(df.columns) - set(cols))
    hue = group_col[0] if group_col else None
    sns_plot = sns.pairplot(df[cols + [hue]] if hue else df[cols], hue=hue)
    st.pyplot(sns_plot)


# ========================
# 🩺 Model Diagnostics Helpers
# ========================
import statsmodels.stats.outliers_influence as influence
from statsmodels.graphics.gofplots import qqplot

def plot_qq(model):
    """Q–Q plot of residuals."""
    fig = qqplot(model.resid, line="s")
    plt.title("Q–Q Plot of Residuals")
    st.pyplot(fig)

def plot_leverage(model):
    """Leverage vs Residual Squared plot."""
    infl = model.get_influence()
    leverage = infl.hat_matrix_diag
    studentized_resid = infl.resid_studentized_external

    fig, ax = plt.subplots()
    ax.scatter(leverage, studentized_resid**2)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentized Residual²")
    ax.set_title("Leverage vs Studentized Residual²")
    st.pyplot(fig)

def show_cooks_distance(model):
    """Cook’s distance plot."""
    infl = model.get_influence()
    cooks_d = infl.cooks_distance[0]
    fig, ax = plt.subplots()
    ax.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
    ax.axhline(4 / len(cooks_d), color="red", linestyle="--", label="4/n threshold")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Cook’s Distance")
    ax.set_title("Cook’s Distance per Observation")
    ax.legend()
    st.pyplot(fig)


# ========================
# 🧮 Model Selection Helpers
# ========================
def compute_model_metrics(model):
    """Extract key model metrics."""
    return {
        "AIC": model.aic,
        "BIC": model.bic,
        "R-squared": model.rsquared,
        "Adj. R-squared": model.rsquared_adj,
        "Log-Likelihood": model.llf
    }

def compare_models(models: dict):
    """Return DataFrame comparing AIC, BIC, and other scores."""
    records = []
    for name, mod in models.items():
        metrics = compute_model_metrics(mod)
        metrics["Model"] = name
        records.append(metrics)
    return pd.DataFrame(records).set_index("Model")


# ========================
# 🧠 Inference & Interpretation Helpers
# ========================
def extract_confidence_intervals(model, alpha=0.05):
    """Return confidence intervals DataFrame."""
    ci = model.conf_int(alpha=alpha)
    ci.columns = ['Lower CI', 'Upper CI']
    ci["Estimate"] = model.params
    ci["Variable"] = ci.index
    return ci.reset_index(drop=True)

def plot_ci_intervals(ci_df, title="Confidence Intervals"):
    fig, ax = plt.subplots()
    ax.errorbar(ci_df["Estimate"], ci_df["Variable"],
                xerr=[ci_df["Estimate"] - ci_df["Lower CI"], ci_df["Upper CI"] - ci_df["Estimate"]],
                fmt='o', capsize=5, color="navy")
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Coefficient Estimate")
    st.pyplot(fig)


# ========================
# 🧪 Posthoc Analysis Helpers
# ========================
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

def run_tukey_hsd(df, response, group):
    """Perform Tukey HSD test and return result DataFrame."""
    tukey = pairwise_tukeyhsd(endog=df[response], groups=df[group], alpha=0.05)
    return pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

def run_pairwise_tests(df, response, group):
    """Run pairwise t-tests with Bonferroni correction."""
    result = pg.pairwise_tests(dv=response, between=group, data=df, padjust="bonf")
    return result


# ========================
# 🔁 Shared Comparison Helpers (T-test)
# ========================
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind
from scipy.stats import ttest_ind as sp_ttest_ind

def compare_ttests(df, col, group_col):
    groups = df[group_col].unique()
    g1 = df[df[group_col] == groups[0]][col]
    g2 = df[df[group_col] == groups[1]][col]
    
    # Statsmodels
    sm_stat, sm_pval, _ = sm_ttest_ind(g1, g2, usevar='pooled')
    
    # SciPy
    sp_stat, sp_pval = sp_ttest_ind(g1, g2, equal_var=True)

    return {
        "Statsmodels": {"t-stat": sm_stat, "p-value": sm_pval},
        "SciPy": {"t-stat": sp_stat, "p-value": sp_pval}
    }


# ========================
# 🔁 Shared Comparison Helpers (Correlation)
# ========================
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr

def compare_correlations(df, col1, col2):
    # Statsmodels-style Pearson (using pandas corr)
    pearson_r_sm = df[[col1, col2]].corr(method="pearson").iloc[0, 1]

    # SciPy Pearson
    r_pearson, p_pearson = pearsonr(df[col1], df[col2])

    # SciPy Spearman
    r_spearman, p_spearman = spearmanr(df[col1], df[col2])

    return {
        "Statsmodels": {
            "Pearson r": pearson_r_sm,
            "p-value": None  # Not available via corr matrix
        },
        "SciPy": {
            "Pearson r": r_pearson,
            "Pearson p": p_pearson,
            "Spearman r": r_spearman,
            "Spearman p": p_spearman
        }
    }


# ========================
# 📏 Confidence Interval Comparison
# ========================
import statsmodels.stats.api as sms
from scipy import stats
import numpy as np

def get_confidence_interval_statsmodels(data, alpha=0.05):
    """
    Compute the confidence interval for the mean using statsmodels.
    
    Parameters:
        data (array-like): Numeric array or Series
        alpha (float): Significance level (default 0.05 → 95% CI)

    Returns:
        (float, float): Lower and upper bounds of the confidence interval
    """
    ci_low, ci_upp = sms.DescrStatsW(data).tconfint_mean(alpha=alpha)
    return ci_low, ci_upp

def get_confidence_interval_scipy(data, alpha=0.05):
    """
    Compute the confidence interval for the mean using SciPy.
    
    Parameters:
        data (array-like): Numeric array or Series
        alpha (float): Significance level (default 0.05 → 95% CI)

    Returns:
        (float, float): Lower and upper bounds of the confidence interval
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf(1 - alpha/2, len(data) - 1)
    return mean - margin, mean + margin


# ========================
# 🎯 Bootstrap Confidence Intervals
# ========================
#from statsmodels.stats.bootstrap import bs_ci

def get_bootstrap_ci(data, alpha=0.05, reps=1000, seed=42):
    np.random.seed(seed)
    n = len(data)
    boot_means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(reps)]
    ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return ci_lower, ci_upper


def get_bootstrap_ci_numpy(data, alpha=0.05, reps=1000, seed=42):
    np.random.seed(seed)
    n = len(data)
    boot_means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(reps)]
    ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return ci_lower, ci_upper


# ========================
# 🧪 Distribution Simulation Utilities
# ========================
from scipy import stats
import numpy as np

def simulate_distribution(dist_name, params, size=1000, random_state=None):
    rng = np.random.default_rng(random_state)

    dist_map = {
        "Normal": lambda p: stats.norm(loc=p[0], scale=p[1]),
        "Exponential": lambda p: stats.expon(loc=0, scale=p[1]),
        "Poisson": lambda p: stats.poisson(mu=p[0]),
        "Binomial": lambda p: stats.binom(n=p[0], p=p[1]),
        "Gamma": lambda p: stats.gamma(a=p[0], loc=0, scale=p[2]),
        "Beta": lambda p: stats.beta(a=p[0], b=p[1]),
        "Lognormal": lambda p: stats.lognorm(s=p[1], scale=np.exp(p[0])),  # mean = exp(mu)
        "Uniform": lambda p: stats.uniform(loc=p[0], scale=p[1] - p[0]) if p[1] > p[0] else None
    }

    dist_func = dist_map.get(dist_name)

    if dist_func is None:
        raise ValueError(f"Unsupported distribution: {dist_name}")

    dist = dist_func(params)
    if dist is None:
        raise ValueError(f"Invalid parameters for {dist_name}: {params}")

    try:
        data = dist.rvs(size=size, random_state=rng)
    except TypeError:
        # Fallback if rvs() doesn't accept random_state for some distributions
        data = dist.rvs(size=size)

    return data, dist



def plot_distribution_histogram(data, dist_name, bins=30):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, bins=bins, ax=ax, color="steelblue")
    ax.set_title(f"Simulated {dist_name} Distribution")
    st.pyplot(fig)


# ========================
# 📊 Project Summary Utilities
# ========================
def display_project_metrics(df):
    """Display basic dataset metrics using Streamlit columns."""
    num_rows, num_cols = df.shape
    num_numeric = len(get_numeric_columns(df))
    num_cat = len(get_categorical_columns(df))

    col1, col2, col3 = st.columns(3)
    col1.metric("🧮 Rows", num_rows)
    col2.metric("📊 Columns", num_cols)
    col3.metric("🔢 Numeric Columns", num_numeric)


def show_summary_statistics(df):
    """Display summary table."""
    st.markdown("#### 📋 Summary Statistics")
    st.dataframe(df.describe().T.style.format(precision=3))


def display_random_sample(df, n=5):
    """Show sample rows from dataset."""
    st.markdown("#### 🔍 Random Sample from Dataset")
    st.dataframe(df.sample(n=n).reset_index(drop=True))


# ========================
# 📊 Additional Utilities
# ========================
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st

def plot_qq(model):
    """Plot Q–Q plot of residuals for normality check."""
    fig = sm.qqplot(model.resid, line='45', fit=True)
    plt.title("Q–Q Plot of Residuals")
    st.pyplot(fig)


def plot_influence(model):
    """Plot influence plot (Cook’s distance and leverage)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    st.pyplot(fig)

import tempfile

def export_model_summary(model):
    """Exports the model summary as a downloadable .txt file."""
    summary_str = model.summary().as_text()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp:
        tmp.write(summary_str)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        st.download_button(
            label="📥 Download Summary",
            data=f,
            file_name="ols_model_summary.txt",
            mime="text/plain"
        )

import pandas as pd
import tempfile

def export_coefficients(model):
    """Export model coefficients and stats as downloadable CSV."""
    summary_frame = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-value": model.tvalues,
        "p-value": model.pvalues
    })

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', encoding='utf-8') as tmp:
        summary_frame.to_csv(tmp.name, index=True)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        st.download_button(
            label="📥 Download Coefficients (.csv)",
            data=f,
            file_name="ols_coefficients.csv",
            mime="text/csv"
        )


def plot_residuals(model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    residuals = model.resid
    fitted = model.fittedvalues

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=fitted, y=residuals, ax=ax)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    st.pyplot(fig)


def get_predictions_df(model, df, response_col):
    df_pred = df.copy()
    df_pred["Predicted"] = model.predict()
    df_pred["Residuals"] = df_pred[response_col] - df_pred["Predicted"]
    return df_pred


from scipy.stats import ttest_ind
import statsmodels.api as sm

def run_ttest_ind(df, col, group_col, equal_var=True):
    group_vals = df[group_col].dropna().unique()
    g1 = df[df[group_col] == group_vals[0]][col].dropna()
    g2 = df[df[group_col] == group_vals[1]][col].dropna()
    t_stat, p_value = ttest_ind(g1, g2, equal_var=equal_var)
    return t_stat, p_value

def compute_cohens_d(df, col, group_col):
    group_vals = df[group_col].dropna().unique()
    g1 = df[df[group_col] == group_vals[0]][col].dropna()
    g2 = df[df[group_col] == group_vals[1]][col].dropna()
    pooled_std = ((g1.std() ** 2 + g2.std() ** 2) / 2) ** 0.5
    return abs(g1.mean() - g2.mean()) / pooled_std

def compute_eta_squared(model):
    anova_table = sm.stats.anova_lm(model, typ=2)
    ss_between = anova_table["sum_sq"].iloc[0]
    ss_total = anova_table["sum_sq"].sum()
    return ss_between / ss_total


# 🔹 1. Mann–Whitney U Test (Independent Samples)
from scipy.stats import mannwhitneyu
import numpy as np

def run_mannwhitney_u(df, value_col, group_col):
    """Run Mann–Whitney U Test and compute Cliff's Delta."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Mann–Whitney requires exactly 2 groups.")

    g1 = df[df[group_col] == groups[0]][value_col]
    g2 = df[df[group_col] == groups[1]][value_col]

    u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
    d = compute_cliffs_delta(g1, g2)
    return u_stat, p_val, d


def compute_cliffs_delta(x, y):
    """Compute Cliff’s Delta (non-parametric effect size)."""
    m, n = len(x), len(y)
    x = np.array(x)
    y = np.array(y)
    greater = np.sum(x[:, None] > y)
    less = np.sum(x[:, None] < y)
    delta = (greater - less) / (m * n)
    return delta


def display_u_test_result(u_stat, p_val):
    st.markdown(f"**U Statistic**: `{u_stat:.3f}`")
    st.markdown(f"**p-value**: `{p_val:.5f}`")
    if p_val < 0.05:
        st.success("✅ Statistically significant difference between groups.")
    else:
        st.info("ℹ️ No significant difference detected between groups.")

# 2. Kruskal–Wallis H Test (Independent Groups >2)
from scipy.stats import kruskal

def run_kruskal(df, value_col, group_col):
    """Run Kruskal–Wallis H Test and compute η² effect size."""
    groups = [grp[value_col].dropna() for name, grp in df.groupby(group_col)]
    h_stat, p_val = kruskal(*groups)
    eta2 = compute_eta_squared_kruskal(h_stat, len(groups), len(df))
    return h_stat, p_val, eta2


def compute_eta_squared_kruskal(h, k, n):
    """Eta-squared effect size for Kruskal–Wallis."""
    return (h - k + 1) / (n - k)


def display_kruskal_result(h_stat, p_val):
    st.markdown(f"**H Statistic**: `{h_stat:.3f}`")
    st.markdown(f"**p-value**: `{p_val:.5f}`")
    if p_val < 0.05:
        st.success("✅ Statistically significant difference across groups.")
    else:
        st.info("ℹ️ No significant difference detected across groups.")


# 🔬 3. Tukey’s HSD Posthoc Test (After ANOVA)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

def run_tukey_test(df, value_col, group_col):
    """Run Tukey’s HSD and return result DataFrame + compact plot."""
    result = pairwise_tukeyhsd(df[value_col], df[group_col])
    summary_df = pd.DataFrame(data=result.summary().data[1:], columns=result.summary().data[0])

    # Plotting group means + confidence intervals
    fig, ax = plt.subplots(figsize=(8, 5))
    result.plot_simultaneous(ax=ax)
    plt.title("Tukey HSD: Group Comparison")
    plt.grid(True)

    return summary_df, fig

def plot_tukey_summary(tukey_fig):
    """Display Tukey summary plot in Streamlit."""
    st.pyplot(tukey_fig)

def get_kruskal_result_df(h_stat, p_val, eta2):
    return pd.DataFrame({
        "H Statistic": [h_stat],
        "p-value": [p_val],
        "Eta Squared": [eta2]
    })


def get_hotelling_summary_df(result_dict):
    """Convert Hotelling's T² result into a DataFrame for export."""
    return pd.DataFrame([{
        "T2_statistic": result_dict["T2"],
        "F_equivalent": result_dict["F"],
        "p_value": result_dict["p_value"]
    }])


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame.
    """
    X = df.dropna().astype(float)
    vif_data = {
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }
    return pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)


def get_influential_points_df(model, df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Return a DataFrame of points with high Cook’s distance or leverage.
    """
    infl = model.get_influence()
    cooks_d = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    standardized_resid = infl.resid_studentized_internal

    df_out = df.copy()
    df_out["Cooks_D"] = cooks_d
    df_out["Leverage"] = leverage
    df_out["Std_Resid"] = standardized_resid
    df_out["High_Influence"] = (cooks_d > threshold / len(df)) | (leverage > 2 * df.shape[1] / len(df))

    return df_out[df_out["High_Influence"]].sort_values(by="Cooks_D", ascending=False)


def get_standardized_coefficients(df, target, predictors):
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[predictors])
    y_scaled = scaler.fit_transform(df[[target]]).flatten()

    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y_scaled, X_scaled).fit()
    
    return pd.DataFrame({
        "Predictor": ["Intercept"] + predictors,
        "Standardized Coefficient": model.params.round(4)
    })


def get_bootstrap_ci_statsmodels(data, alpha=0.05, reps=1000):
    """
    Compute bootstrap confidence interval using statsmodels.

    Parameters:
        data (array-like): Input numeric data.
        alpha (float): Significance level (1 - confidence level).
        reps (int): Number of bootstrap samples.

    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval.
    """
    if len(data) < 10:
        raise ValueError("Bootstrap CI requires at least 10 data points.")

    ci_bounds = sms.DescrStatsW(data).bootstrap(
        reps=reps,
        method='percentile',
        alpha=alpha,
        func=np.mean
    )
    return ci_bounds[0], ci_bounds[1]


def get_bootstrap_ci_statsmodels(data, alpha=0.05, reps=1000):
    """
    Emulates bootstrap confidence interval (mean) similar to statsmodels,
    using manual resampling and percentile method.

    Parameters:
        data (array-like): Input numeric data.
        alpha (float): Significance level (1 - confidence level).
        reps (int): Number of bootstrap samples.

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if len(data) < 10:
        raise ValueError("Bootstrap CI requires at least 10 data points.")

    data = np.asarray(data, dtype=np.float64)

    rng = np.random.default_rng()
    boot_means = rng.choice(data, size=(reps, len(data)), replace=True).mean(axis=1)

    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

