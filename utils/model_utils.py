# model_utils.py

import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import kstest, norm
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Tuple, List
from scipy.stats import f


def summarize_stats(df, cols=None):
    if cols:
        return df[cols].describe().T
    return df.describe().T

def compute_central_tendency(df, cols):
    summary = {}
    for col in cols:
        summary[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'mode': df[col].mode().iloc[0]
        }
    return pd.DataFrame(summary).T


def summarize_model_coefficients(model):
    """
    Extracts a summary DataFrame of coefficients, p-values, and CIs from a fitted model.
    """
    summary_df = pd.DataFrame({
        'coef': model.params,
        'p_value': model.pvalues,
        'ci_lower': model.conf_int()[0],
        'ci_upper': model.conf_int()[1]
    })
    return summary_df


def export_model_summary_as_text(model, filepath: Path):
    """
    Save the model's summary output to a `.txt` file.
    """
    with open(filepath, "w") as f:
        f.write(model.summary().as_text())


def extract_anova_table(model, typ=2):
    """
    Returns the ANOVA table for a fitted model.
    Default type is II (can be changed to I or III).
    """
    return sm.stats.anova_lm(model, typ=typ)


def export_forecast_to_csv(df_forecast: pd.DataFrame, filepath: Path):
    """
    Saves forecast result with confidence intervals to CSV.
    """
    df_forecast.to_csv(filepath, index=True)


def extract_aic_bic(model):
    return {
        "AIC": model.aic,
        "BIC": model.bic,
        "Log-Likelihood": model.llf
    }

def compare_models_by_ic(*models):
    results = []
    for i, model in enumerate(models, 1):
        ic = extract_aic_bic(model)
        ic["Model"] = f"Model_{i}"
        results.append(ic)
    return pd.DataFrame(results).set_index("Model")


def run_ttest_model(df, formula):
    model = smf.ols(formula, data=df).fit()
    return model


def export_model_summary_as_text(model, file_path):
    with open(file_path, "w") as f:
        f.write(model.summary().as_text())


def extract_group_means(df, group_col, target_col):
    return df.groupby(group_col)[target_col].mean()


# utils/model_utils.py

def run_kstest_normality(series):
    """
    Runs a one-sample KS test to compare the sample with a standard normal distribution.
    Returns statistic and p-value.
    """
    standardized = (series - series.mean()) / series.std()
    stat, pval = kstest(standardized, 'norm')
    return stat, pval

def generate_normal_and_skewed_data(n=100):
    """
    Generate one normally distributed and one skewed dataset.
    """
    normal = np.random.normal(loc=0, scale=1, size=n)
    skewed = np.random.exponential(scale=1, size=n)
    return pd.DataFrame({'normal': normal, 'skewed': skewed})


def compute_correlations(df, col1, col2):
    """
    Returns Pearson, Spearman, and Kendall correlations and p-values
    between two columns of a DataFrame.
    """
    pearson_corr, pearson_p = pearsonr(df[col1], df[col2])
    spearman_corr, spearman_p = spearmanr(df[col1], df[col2])
    kendall_corr, kendall_p = kendalltau(df[col1], df[col2])
    
    return pd.DataFrame({
        "method": ["Pearson", "Spearman", "Kendall"],
        "correlation": [pearson_corr, spearman_corr, kendall_corr],
        "p-value": [pearson_p, spearman_p, kendall_p]
    })


def compute_confidence_interval(model, alpha=0.05):
    """Returns confidence intervals from a statsmodels fitted model."""
    return model.conf_int(alpha=alpha)


def fit_ols_model(formula, df):
    """Fits OLS model and returns the result."""
    return sm.OLS.from_formula(formula, data=df).fit()


def bootstrap_mean_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> Tuple[float, float]:
    """Return bootstrap confidence interval for the mean."""
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def bootstrap_groupwise_ci(df: pd.DataFrame, group_col: str, value_col: str,
                           n_bootstrap: int = 1000, ci: float = 95) -> pd.DataFrame:
    """Return bootstrap CI for each group."""
    result = []
    for group, subdf in df.groupby(group_col):
        lower, upper = bootstrap_mean_ci(subdf[value_col].values, n_bootstrap=n_bootstrap, ci=ci)
        mean_val = np.mean(subdf[value_col])
        result.append({'Group': group, 'Mean': mean_val, 'Lower': lower, 'Upper': upper})
    return pd.DataFrame(result)


def summarize_simulated_distributions(dist_data: dict) -> pd.DataFrame:
    """Return a dataframe of mean, std, skew, and kurtosis for each distribution."""
    from scipy.stats import skew, kurtosis
    summary = []
    for name, arr in dist_data.items():
        summary.append({
            "Distribution": name,
            "Mean": np.mean(arr),
            "Std": np.std(arr),
            "Skewness": skew(arr),
            "Kurtosis": kurtosis(arr)
        })
    return pd.DataFrame(summary)


def compute_hotelling_t2(group1_df, group2_df, cols):
    """
    Computes Hotelling’s T² statistic and F-approximation for two groups on multivariate data.

    Parameters:
        group1_df (pd.DataFrame): Data for group 1.
        group2_df (pd.DataFrame): Data for group 2.
        cols (list): List of column names to include in test.

    Returns:
        dict: {
            'T2': Hotelling’s T² statistic,
            'F': F-statistic approximation,
            'p_value': p-value from F-distribution
        }
    """
    X1 = group1_df[cols].to_numpy()
    X2 = group2_df[cols].to_numpy()

    n1, p = X1.shape
    n2 = X2.shape[0]

    # Means
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    diff = mean1 - mean2

    # Covariance
    cov1 = np.cov(X1, rowvar=False)
    cov2 = np.cov(X2, rowvar=False)

    pooled_cov = ((n1 - 1)*cov1 + (n2 - 1)*cov2) / (n1 + n2 - 2)
    inv_pooled_cov = np.linalg.inv(pooled_cov)

    # T² statistic
    T2 = (n1 * n2) / (n1 + n2) * (diff.T @ inv_pooled_cov @ diff)

    # F approximation
    df1 = p
    df2 = n1 + n2 - p - 1
    F_stat = T2 * df2 / (df1 * (n1 + n2 - 2))
    p_val = 1 - f.cdf(F_stat, df1, df2)

    return {
        'T2': T2,
        'F': F_stat,
        'p_value': p_val
    }

