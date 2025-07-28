# 🧠 Statsmodels Cheatsheet

A comprehensive reference for mastering statistical modeling and inference using the `statsmodels` library. Tailored to the **StatsmodelsMasterPro** project.

---

## 📘 Model Types & Syntax

### 📌 OLS – Ordinary Least Squares

```python
import statsmodels.api as sm

X = df[['x1', 'x2']]
X = sm.add_constant(X)
y = df['y']

model = sm.OLS(y, X).fit()
model.summary()
```

---

### 📌 GLM – Generalized Linear Models

```python
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
model.summary()
```

> ✅ Families: `Poisson`, `Binomial`, `Gaussian`, `Gamma`, `InverseGaussian`

---

### 📌 ANOVA / MANOVA

```python
from statsmodels.formula.api import ols, manova

# ANOVA
model = ols("y ~ C(group)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# MANOVA
mv = manova.MANOVA.from_formula("Y1 + Y2 ~ group", data=df)
mv.mv_test()
```

---

## 🧪 Diagnostics & Assumption Checks

### 📌 Residual Plots

```python
from utils.diagnostics import plot_residuals
plot_residuals(model)
```

---

### 📌 QQ Plot

```python
from utils.diagnostics import plot_qq_residuals
plot_qq_residuals(model)
```

---

### 📌 Leverage & Influence

```python
from utils.diagnostics import plot_leverage_cooks
plot_leverage_cooks(model)
```

---

### 📌 Heteroskedasticity Tests

```python
from utils.diagnostics import run_heteroskedasticity_tests
run_heteroskedasticity_tests(model)
```

---

## 🔁 Bootstrap CI

```python
from utils.model_utils import bootstrap_ci
bootstrap_ci(df['y'], n_bootstrap=1000, ci=95)
```

---

## 📊 Model Selection & Metrics

### 📌 Metrics Access

```python
model.rsquared
model.rsquared_adj
model.aic
model.bic
```

---

### 📌 Likelihood Ratio Test

```python
lr_stat = 2 * (model_full.llf - model_reduced.llf)
```

---

## 📈 Time Series

### 📌 ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(ts_data, order=(1, 1, 1)).fit()
model.summary()
```

---

### 📌 ACF / PACF

```python
from utils.diagnostics import plot_acf_pacf
plot_acf_pacf(ts_data)
```

---

## 🧪 Hypothesis Testing

```python
from statsmodels.stats.weightstats import ttest_ind

t_stat, p_value, df = ttest_ind(data1, data2)
```

---

### 📌 Custom Contrasts

```python
from statsmodels.stats.multicomp import MultiComparison
mc = MultiComparison(df['y'], df['group'])
mc.tukeyhsd().summary()
```

---

### 📌 Hotelling’s T² (Fallback)

```python
from utils.stats_utils import compute_hotelling_t2
result = compute_hotelling_t2(groupA, groupB, ['Y1', 'Y2'])
```

---

## 📐 Skewness & Kurtosis

```python
from utils.diagnostics import compute_skewness_kurtosis
compute_skewness_kurtosis(df, ['Y1', 'Y2'])
```

---

## 📤 Export Tools

```python
from utils.visual_utils import save_and_show_plot
from utils.model_utils import export_model_summary_as_text

save_and_show_plot("plot_name", EXPORT_PLOTS)
export_model_summary_as_text(model, EXPORT_TABLES / "model_summary.txt")
```

---

## 📚 `statsmodels` vs `scipy.stats`

| Task                 | `statsmodels`                   | `scipy.stats`                    |
| -------------------- | ------------------------------- | -------------------------------- |
| Linear Models        | ✅ `OLS`, `GLM`                 | ❌ Not supported                 |
| ANOVA/MANOVA         | ✅ `anova_lm`, `MANOVA`         | ❌ Limited or absent             |
| t-tests, Correlation | ✅ `ttest_ind`, `pearsonr`      | ✅ `ttest_ind`, `pearsonr`       |
| Posthoc Tests        | ✅ `TukeyHSD`, `Bonferroni`     | ❌                               |
| Bootstrap CI         | ✅ Custom utilities or manually | ✅ With `resample` + aggregation |
| AIC/BIC              | ✅ Available in models          | ❌                               |

---

## 🧠 Best Practices

- Always check residuals visually.
- Use `sm.add_constant()` to include intercepts.
- Prefer model-based diagnostics over rule-of-thumb tests.
- Validate assumptions before interpreting p-values.

---

✅ Happy Modeling with `Statsmodels`!
