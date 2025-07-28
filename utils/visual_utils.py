# visual_utils.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def plot_histograms(df, cols, bins=30):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f'Histogram of {col}')
        plt.tight_layout()
        plt.show()


def plot_boxplots(df, cols):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()


def plot_pairplot(df, hue=None):
    sns.pairplot(df, hue=hue)
    plt.suptitle("Pairwise Relationships", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, method='pearson'):
    corr = df.corr(method=method)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def save_and_show_plot(filename: str, path: Path):
    plt.tight_layout()
    plt.savefig(path / f"{filename}.png", dpi=300)
    plt.show()
    plt.close()


def plot_group_scatter_2d(df, x_col, y_col, group_col, title="Group-wise Scatter"):
    """
    Plot a 2D scatter of multivariate data colored by group.
    """
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, palette="Set2")
    plt.title(title)
    plt.tight_layout()


def plot_ic_barplot(ic_df, title="Model Comparison by AIC/BIC"):
    ic_df[["AIC", "BIC"]].plot(kind="bar", figsize=(8, 5))
    plt.title(title)
    plt.ylabel("Information Criterion Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_scatter_with_trend(df, x, y, method="reg"):
    """
    Creates a scatterplot with optional trend line.
    method: 'reg' (linear), 'lowess', or None
    """
    plt.figure(figsize=(6, 4))
    if method == "lowess":
        sns.regplot(data=df, x=x, y=y, lowess=True)
    elif method == "reg":
        sns.regplot(data=df, x=x, y=y)
    else:
        sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"{method.upper() if method else 'Scatter'} Plot: {x} vs {y}")
    plt.tight_layout()
    plt.show()


def plot_ci_errorbar(df, x_col, y_col, ci=95, title="Confidence Intervals"):
    """Plots mean with confidence intervals using seaborn."""
    plt.figure(figsize=(6, 4))
    sns.pointplot(data=df, x=x_col, y=y_col, ci=ci, capsize=0.2)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_ci_barplot(df, x_col, y_col, ci=95, title="Barplot with CI"):
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x=x_col, y=y_col, ci=ci, capsize=0.1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_distribution(data, title="Distribution", bins=30, kde=True, color='skyblue'):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, bins=bins, kde=kde, color=color)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_distributions(dist_data: dict, bins=30):
    """
    Plot multiple distributions side by side.
    dist_data: dict of form {'Label': np.array, ...}
    """
    n = len(dist_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, data) in zip(axes, dist_data.items()):
        sns.histplot(data, bins=bins, kde=True, ax=ax)
        ax.set_title(label)
    plt.tight_layout()
    plt.show()
