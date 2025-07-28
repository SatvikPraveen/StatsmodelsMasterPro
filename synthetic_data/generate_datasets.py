# generate_datasets.py

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, bernoulli, multivariate_normal
from pathlib import Path

# Create output path
DATA_PATH = Path(__file__).parent
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# ---------- 1. OLS Regression Data ----------
def generate_ols_data(n=200):
    X1 = np.random.normal(5, 2, n)
    X2 = np.random.normal(10, 3, n)
    noise = np.random.normal(0, 1.5, n)
    y = 2 + 1.5 * X1 - 0.7 * X2 + noise

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
    df.to_csv(DATA_PATH / "ols_data.csv", index=False)
    print("[✔] OLS data saved → ols_data.csv")

# ---------- 2. GLM Data (Poisson + Logistic) ----------
def generate_glm_data(n=300):
    # Poisson
    X = np.random.normal(2, 1, n)
    lambda_ = np.exp(0.5 + 0.9 * X)
    y_pois = poisson.rvs(mu=lambda_)

    df_pois = pd.DataFrame({'X': X, 'y': y_pois})
    df_pois.to_csv(DATA_PATH / "glm_poisson.csv", index=False)

    # Logistic (binary classification)
    X_log = np.random.normal(0, 1, n)
    logits = 0.8 * X_log - 0.5
    probs = 1 / (1 + np.exp(-logits))
    y_logit = bernoulli.rvs(probs)

    df_logit = pd.DataFrame({'X': X_log, 'y': y_logit})
    df_logit.to_csv(DATA_PATH / "glm_logistic.csv", index=False)

    print("[✔] GLM Poisson and Logistic data saved")

# ---------- 3. Time Series Data (ARIMA-like) ----------
def generate_time_series_data(n=250):
    ar_param = 0.8
    ma_param = 0.4
    noise = np.random.normal(0, 1, n)
    series = np.zeros(n)

    for t in range(1, n):
        series[t] = ar_param * series[t - 1] + noise[t] + ma_param * noise[t - 1]

    df = pd.DataFrame({'t': pd.date_range("2020-01-01", periods=n, freq='D'), 'value': series})
    df.to_csv(DATA_PATH / "arima_series.csv", index=False)
    print("[✔] Time series data saved → arima_series.csv")

# ---------- 4. MANOVA Data ----------
def generate_manova_data(n=150):
    means_group1 = [0, 0]
    means_group2 = [2, 2]
    cov = [[1, 0.5], [0.5, 1]]

    group1 = multivariate_normal.rvs(mean=means_group1, cov=cov, size=n)
    group2 = multivariate_normal.rvs(mean=means_group2, cov=cov, size=n)

    df1 = pd.DataFrame(group1, columns=['Y1', 'Y2'])
    df1['group'] = 'A'

    df2 = pd.DataFrame(group2, columns=['Y1', 'Y2'])
    df2['group'] = 'B'

    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(DATA_PATH / "manova_data.csv", index=False)
    print("[✔] MANOVA data saved → manova_data.csv")

# ---------- 5. Heteroskedasticity Data ----------
def generate_heteroskedastic_data(n=200):
    X = np.random.normal(0, 1, n)
    error = np.random.normal(0, 1 + 2 * np.abs(X), n)
    y = 3 + 2 * X + error

    df = pd.DataFrame({'X': X, 'y': y})
    df.to_csv(DATA_PATH / "heteroskedastic_data.csv", index=False)
    print("[✔] Heteroskedastic data saved → heteroskedastic_data.csv")

# ---------- 6. Multivariate Group Data ----------

def generate_multivariate_group_data(n=1000):
    # ===== Set seed for consistency =====
    np.random.seed(42)

    # ===== Create Numeric Columns for Group A =====
    a_num1 = np.random.normal(loc=50, scale=10, size=n//2)
    a_num2 = np.random.uniform(low=10, high=100, size=n//2)
    a_num3 = np.random.lognormal(mean=3, sigma=0.5, size=n//2)
    a_num4 = np.random.exponential(scale=1.5, size=n//2)
    a_num5 = np.random.normal(loc=0, scale=50, size=n//2)

    df_A = pd.DataFrame({
        "Num1": a_num1,
        "Num2": a_num2,
        "Num3": a_num3,
        "Num4": a_num4,
        "Num5": a_num5,
        "Group": "A"
    })

    # ===== Create Numeric Columns for Group B =====
    b_num1 = np.random.normal(loc=60, scale=15, size=n//2)
    b_num2 = np.random.uniform(low=20, high=120, size=n//2)
    b_num3 = np.random.lognormal(mean=4, sigma=0.8, size=n//2)
    b_num4 = np.random.exponential(scale=1.0, size=n//2)
    b_num5 = np.random.normal(loc=20, scale=60, size=n//2)

    df_B = pd.DataFrame({
        "Num1": b_num1,
        "Num2": b_num2,
        "Num3": b_num3,
        "Num4": b_num4,
        "Num5": b_num5,
        "Group": "B"
    })

    # ===== Combine and Shuffle =====
    df = pd.concat([df_A, df_B], ignore_index=True)

    # ===== Add Categorical Columns =====
    df["Cat1"] = np.random.choice(["Yes", "No"], size=n)
    df["Cat2"] = np.random.choice(["Low", "Medium", "High"], size=n)
    df["Cat3"] = np.random.choice(["A", "B", "C", "D"], size=n)
    df["Cat4"] = np.random.choice(["Urban", "Rural"], size=n)
    df["Cat5"] = np.random.choice(["X", "Y"], size=n)

    # ===== Save CSV =====
    df.to_csv(DATA_PATH / "multivariate_group_data.csv", index=False)
    print("[✔] Multivariate Group Data saved → multivariate_group_data.csv")

# ---------- 7. OLS Diagnostic Data ----------
def generate_ols_diagnostics_data(n=600, seed=42):
    np.random.seed(seed)

    # Core predictors
    X1 = np.random.normal(50, 10, n)
    X2 = np.random.normal(30, 5, n)
    X3 = np.random.uniform(10, 100, n)
    X4 = np.random.exponential(1.2, n)
    X5 = np.random.normal(0, 20, n)

    # Add multicollinearity
    X6 = X1 * 0.5 + np.random.normal(0, 2, n)  # correlated with X1
    X7 = X2 * -0.4 + np.random.normal(0, 1.5, n)  # correlated with X2

    # Add weak predictors
    X8 = np.random.normal(0, 1, n)
    X9 = np.random.uniform(0, 1, n)
    X10 = np.random.chisquare(2, n)

    # Response variable with linear combination + noise + outliers
    noise = np.random.normal(0, 10, n)
    y = 5 + 1.5 * X1 - 2.0 * X2 + 0.3 * X3 + noise

    # Inject some outliers in y
    outlier_indices = np.random.choice(n, size=10, replace=False)
    y[outlier_indices] += np.random.normal(100, 20, 10)

    df = pd.DataFrame({
        "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5,
        "X6": X6, "X7": X7, "X8": X8, "X9": X9, "X10": X10,
        "y": y
    })

    # Save CSV
    df.to_csv(DATA_PATH / "ols_diagnostics.csv", index=False)
    print("[✔] OLS Diagnostics data saved → ols_diagnostics.csv")


# ---------- 8. Posthoc Analysis Data ----------
def generate_posthoc_data(n=100):
    """
    Generate a dataset suitable for posthoc analysis (Tukey HSD, Bonferroni).
    It creates a categorical grouping variable ('Group') and a numeric response ('Score')
    with distinct group means.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate group labels: A, B, C
    groups = np.repeat(['A', 'B', 'C'], n)

    # Generate scores with different means for each group
    scores = np.concatenate([
        np.random.normal(loc=60, scale=10, size=n),   # Group A
        np.random.normal(loc=70, scale=12, size=n),   # Group B
        np.random.normal(loc=65, scale=8, size=n)     # Group C
    ])

    # Assemble DataFrame
    df = pd.DataFrame({
        "Group": groups,
        "Score": scores
    })

    # Save CSV
    df.to_csv(DATA_PATH / "posthoc_dataset.csv", index=False)
    print("[✔] Posthoc analysis data saved → posthoc_dataset.csv")


# ---------- Master Function ----------
def generate_all_datasets():
    generate_ols_data()
    generate_glm_data()
    generate_time_series_data()
    generate_manova_data()
    generate_heteroskedastic_data()
    generate_multivariate_group_data()
    generate_ols_diagnostics_data()
    generate_posthoc_data()
    print("\n✅ All synthetic datasets generated successfully!")


if __name__ == "__main__":
    generate_all_datasets()
