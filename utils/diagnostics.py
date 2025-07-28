# diagnostics.py

"""
Diagnostic functions for computing distribution shape metrics using pandas,
to align with the StatsmodelsMasterPro philosophy — no scipy used.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as sm


def compute_skewness_kurtosis(df, cols):
    """
    Compute skewness and Pearson-style kurtosis for selected columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        cols (list): List of numeric column names

    Returns:
        dict: Dictionary with skewness and kurtosis per column
    """
    result = {}
    for col in cols:
        result[col] = {
            'skewness': df[col].skew(skipna=True),
            'kurtosis': df[col].kurt(skipna=True) + 3  # Convert to Pearson-style
        }
    return result


def plot_fitted_vs_actual(y_true, y_pred, title="Fitted vs Actual"):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=y_true)
    plt.xlabel("Fitted Values")
    plt.ylabel("Actual Values")
    plt.title(title)
    plt.axline((0, 0), slope=1, color='red', linestyle='--')
    plt.tight_layout()


def plot_residuals(model, title="Residuals vs Fitted"):
    plt.figure(figsize=(6, 4))
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()



def plot_residual_histogram(model, title="Residual Histogram"):
    plt.figure(figsize=(6, 4))
    sns.histplot(model.resid, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()


def plot_acf_pacf(series, lags=40, title_prefix=""):
    """
    Plot ACF and PACF side by side
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix} ACF")

    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title(f"{title_prefix} PACF")

    plt.tight_layout()


def plot_qq_residuals(model, title="Q–Q Plot of Residuals"):

    fig = sm.qqplot(model.resid, line='45', fit=True)
    plt.title(title)
    plt.tight_layout()


def plot_leverage_cooks(model, title="Influence Plot"):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    influence_plot(model, ax=ax)
    plt.title(title)
    plt.tight_layout()


def run_heteroskedasticity_tests(model):
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    residuals = model.resid
    exog = model.model.exog
    
    bp_test = het_breuschpagan(residuals, exog)
    white_test = het_white(residuals, exog)
    
    return {
        "Breusch-Pagan": {
            "LM Stat": bp_test[0],
            "p-value": bp_test[1]
        },
        "White": {
            "Stat": white_test[0],
            "p-value": white_test[1]
        }
    }


