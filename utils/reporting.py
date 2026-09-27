"""Publication-ready reporting.

* ``tidy`` / ``glance`` – broom-style coefficient and model-level tables that
  work across OLS, GLM, discrete, and robust-linear results.
* ``regression_table`` – side-by-side comparison tables (stars, SEs) exported
  as DataFrame, LaTeX, Markdown, or HTML.
* ``format_p``, ``format_ci``, ``apa_coefficient``, ``apa_ttest``,
  ``apa_anova`` – APA 7 style reporting strings.
* ``coefficient_plot`` – forest plot of coefficients with confidence intervals.
* ``save_table`` – write a DataFrame to CSV / LaTeX / Markdown in one call.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.iolib.summary2 import summary_col

__all__ = [
    "tidy",
    "glance",
    "regression_table",
    "format_p",
    "format_ci",
    "apa_coefficient",
    "apa_ttest",
    "apa_anova",
    "coefficient_plot",
    "save_table",
    "to_markdown",
    "to_latex",
]


# --------------------------------------------------------------------------- #
# Tidy outputs
# --------------------------------------------------------------------------- #
def tidy(result, alpha: float = 0.05, exponentiate: bool = False) -> pd.DataFrame:
    """Coefficient table with one row per term (after R's ``broom::tidy``)."""
    ci = result.conf_int(alpha=alpha)
    out = pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.to_numpy(),
            "std_error": result.bse.to_numpy(),
            "statistic": result.tvalues.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "conf_low": ci.iloc[:, 0].to_numpy(),
            "conf_high": ci.iloc[:, 1].to_numpy(),
        }
    )
    if exponentiate:
        for c in ("estimate", "conf_low", "conf_high"):
            out[c] = np.exp(out[c])
        out = out.drop(columns="std_error")
    return out.reset_index(drop=True)


def _get(result, attr):
    try:
        val = getattr(result, attr)
        return float(val) if np.isscalar(val) or isinstance(val, (int, float, np.floating)) else val
    except Exception:
        return np.nan


def glance(result) -> pd.DataFrame:
    """One-row model-level summary (after R's ``broom::glance``)."""
    row = {
        "nobs": int(result.nobs),
        "df_model": _get(result, "df_model"),
        "df_resid": _get(result, "df_resid"),
        "log_likelihood": _get(result, "llf"),
        "aic": _get(result, "aic"),
        "bic": _get(result, "bic"),
    }
    for attr, name in [("rsquared", "r_squared"), ("rsquared_adj", "adj_r_squared"), ("fvalue", "f_statistic"), ("f_pvalue", "f_pvalue"), ("prsquared", "pseudo_r_squared"), ("deviance", "deviance"), ("scale", "scale")]:
        if hasattr(result, attr):
            row[name] = _get(result, attr)
    conv = getattr(getattr(result, "mle_retvals", None) or {}, "get", lambda *_: None)("converged")
    if conv is not None:
        row["converged"] = bool(conv)
    row["cov_type"] = getattr(result, "cov_type", "nonrobust")
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# Regression comparison tables
# --------------------------------------------------------------------------- #
def _default_info(result) -> dict:
    info = {"N": lambda r: f"{int(r.nobs):d}"}
    if hasattr(result, "rsquared"):
        info["R²"] = lambda r: f"{r.rsquared:.3f}"
        info["Adj. R²"] = lambda r: f"{r.rsquared_adj:.3f}"
    if hasattr(result, "prsquared"):
        info["Pseudo R²"] = lambda r: f"{r.prsquared:.3f}"
    info["AIC"] = lambda r: f"{r.aic:.1f}"
    info["BIC"] = lambda r: f"{r.bic:.1f}"
    return info


def regression_table(
    results: Sequence,
    model_names: Sequence[str] | None = None,
    stars: bool = True,
    float_format: str = "%.3f",
    info_dict: dict | None = None,
    regressor_order: Sequence[str] | None = None,
    drop_omitted: bool = False,
    to: str = "dataframe",
):
    """Side-by-side regression table in the style of ``stargazer`` / ``modelsummary``.

    Parameters
    ----------
    to
        ``"dataframe"`` (default), ``"latex"``, ``"markdown"``, ``"html"``, or ``"text"``.
    """
    results = list(results)
    names = list(model_names) if model_names is not None else [f"({i + 1})" for i in range(len(results))]
    info = info_dict if info_dict is not None else _default_info(results[0])
    summary = summary_col(
        results,
        model_names=names,
        stars=stars,
        float_format=float_format,
        info_dict=info,
        regressor_order=list(regressor_order) if regressor_order else None,
        drop_omitted=drop_omitted,
    )
    to = to.lower()
    if to == "dataframe":
        return summary.tables[0]
    if to == "latex":
        return summary.as_latex()
    if to == "html":
        return summary.as_html()
    if to == "text":
        return summary.as_text()
    if to == "markdown":
        return to_markdown(summary.tables[0])
    raise ValueError("to must be 'dataframe', 'latex', 'markdown', 'html', or 'text'.")


# --------------------------------------------------------------------------- #
# APA-style strings
# --------------------------------------------------------------------------- #
def _strip_leading_zero(x: float, digits: int) -> str:
    s = f"{x:.{digits}f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)


def format_p(p: float, digits: int = 3, threshold: float = 0.001) -> str:
    """APA-style p-value: ``p < .001`` or ``p = .034``."""
    if np.isnan(p):
        return "p = n/a"
    if p < threshold:
        return f"p < {_strip_leading_zero(threshold, digits)}"
    return f"p = {_strip_leading_zero(p, digits)}"


def format_ci(low: float, high: float, digits: int = 2, level: float = 95) -> str:
    return f"{level:g}% CI [{low:.{digits}f}, {high:.{digits}f}]"


def apa_coefficient(result, term: str, digits: int = 2, alpha: float = 0.05) -> str:
    """e.g. ``b = 1.50, SE = 0.05, t(197) = 30.10, p < .001, 95% CI [1.40, 1.60]``."""
    row = tidy(result, alpha=alpha).set_index("term").loc[term]
    df = getattr(result, "df_resid", None)
    stat_name = "t" if df is not None and result.use_t else "z"
    stat = f"{stat_name}({int(df)}) = {row['statistic']:.{digits}f}" if stat_name == "t" else f"z = {row['statistic']:.{digits}f}"
    return (
        f"b = {row['estimate']:.{digits}f}, SE = {row['std_error']:.{digits}f}, {stat}, "
        f"{format_p(row['p_value'])}, {format_ci(row['conf_low'], row['conf_high'], digits, 100 * (1 - alpha))}"
    )


def apa_ttest(t: float, df: float, p: float, d: float | None = None, digits: int = 2) -> str:
    s = f"t({df:g}) = {t:.{digits}f}, {format_p(p)}"
    if d is not None:
        s += f", d = {d:.{digits}f}"
    return s


def apa_anova(anova_table: pd.DataFrame, term: str, eta_squared: float | None = None, digits: int = 2) -> str:
    """e.g. ``F(2, 297) = 24.31, p < .001, η² = .14`` from an ``anova_lm`` table."""
    row = anova_table.loc[term]
    df_resid = anova_table.loc["Residual", "df"]
    s = f"F({row['df']:g}, {df_resid:g}) = {row['F']:.{digits}f}, {format_p(row['PR(>F)'])}"
    if eta_squared is not None:
        s += f", η² = {_strip_leading_zero(eta_squared, digits)}"
    return s


# --------------------------------------------------------------------------- #
# Plots and export
# --------------------------------------------------------------------------- #
def coefficient_plot(results, names: Sequence[str] | None = None, terms: Sequence[str] | None = None, alpha: float = 0.05, exclude_intercept: bool = True, ax=None):
    """Forest plot of coefficients with ``100(1-alpha)``% confidence intervals for one or more models."""
    import matplotlib.pyplot as plt

    if not isinstance(results, (list, tuple)):
        results = [results]
    names = list(names) if names else [f"Model {i + 1}" for i in range(len(results))]
    frames = [tidy(r, alpha=alpha).assign(model=n) for r, n in zip(results, names)]
    df = pd.concat(frames, ignore_index=True)
    if exclude_intercept:
        df = df[~df["term"].str.lower().isin(["intercept", "const"])]
    if terms:
        df = df[df["term"].isin(terms)]
    order = list(dict.fromkeys(df["term"]))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.5))
    offsets = np.linspace(-0.25, 0.25, len(names)) if len(names) > 1 else [0.0]
    for off, name in zip(offsets, names):
        sub = df[df["model"] == name].set_index("term").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(sub["estimate"], y, xerr=[sub["estimate"] - sub["conf_low"], sub["conf_high"] - sub["estimate"]], fmt="o", capsize=3, label=name)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel(f"Estimate with {100 * (1 - alpha):.0f}% CI")
    if len(names) > 1:
        ax.legend()
    return ax


def to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}", index: bool = True) -> str:
    """Minimal GitHub-flavoured Markdown table writer (no ``tabulate`` dependency)."""
    frame = df.reset_index() if index else df.copy()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else float_format.format(v)
        return str(v)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def to_latex(df: pd.DataFrame, float_format: str = "%.4f", index: bool = True, caption: str | None = None, label: str | None = None) -> str:
    """Booktabs-style LaTeX table without the ``jinja2`` dependency of ``DataFrame.to_latex``."""
    frame = df.reset_index() if index else df.copy()

    def esc(s: str) -> str:
        return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else float_format % v
        return esc(v)

    cols = "l" + "r" * (len(frame.columns) - 1)
    lines = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(esc(c) for c in frame.columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(fmt(v) for v in row) + " \\\\" for row in frame.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, path_stem: str | Path, formats: Sequence[str] = ("csv", "tex", "md"), index: bool = True, float_format: str = "%.4f") -> list[Path]:
    """Save a DataFrame as CSV, LaTeX, and/or Markdown next to each other."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=index, float_format=float_format)
        elif fmt == "tex":
            path.write_text(to_latex(df, float_format=float_format, index=index))
        elif fmt == "md":
            path.write_text(to_markdown(df, float_format="{:" + float_format[1:] + "}", index=index))
        else:
            raise ValueError(f"Unsupported format {fmt!r}.")
        written.append(path)
    return written
