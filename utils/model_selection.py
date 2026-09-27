"""Model selection: information criteria, cross-validation, and subset search.

* ``information_criteria_table`` – AIC, AICc, BIC with Δ values and Akaike /
  Schwarz weights (Burnham & Anderson, 2002; Wagenmakers & Farrell, 2004).
* ``cross_validate`` – k-fold cross-validation for formula-based models with
  out-of-sample RMSE, MAE, and R².
* ``best_subset`` – exhaustive search over predictor subsets.
* ``stepwise_selection`` – forward, backward, or bidirectional stepwise search
  with a full trace of every step.
* ``nested_f_test`` – F-test for nested OLS models.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.inference import likelihood_ratio_test

__all__ = [
    "information_criteria_table",
    "cross_validate",
    "best_subset",
    "stepwise_selection",
    "nested_f_test",
    "likelihood_ratio_test",
]


def _n_params(result) -> int:
    return int(len(result.params))


def information_criteria_table(models: dict) -> pd.DataFrame:
    """Compare models by AIC, AICc, and BIC with Δ and evidence weights.

    Parameters
    ----------
    models
        Mapping of model name → fitted results object. All models must be
        fitted to the same response and observations.
    """
    rows = []
    for name, res in models.items():
        k = _n_params(res)
        n = int(res.nobs)
        aic = float(res.aic)
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        rows.append({"model": name, "k": k, "n": n, "log_likelihood": float(res.llf), "AIC": aic, "AICc": aicc, "BIC": float(res.bic)})
    table = pd.DataFrame(rows).set_index("model")
    for crit in ("AIC", "AICc", "BIC"):
        delta = table[crit] - table[crit].min()
        w = np.exp(-0.5 * delta)
        table[f"delta_{crit}"] = delta
        table[f"weight_{crit}"] = w / w.sum()
    table["evidence_ratio_AIC"] = table["weight_AIC"].max() / table["weight_AIC"]
    return table.sort_values("AIC")


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2_oos": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else np.nan,
    }


def cross_validate(
    formula: str,
    data: pd.DataFrame,
    k: int = 5,
    seed: int | None = None,
    fit_fn: Callable = smf.ols,
    shuffle: bool = True,
    fit_kwargs: dict | None = None,
) -> pd.DataFrame:
    """k-fold cross-validation for a formula-based ``statsmodels`` model.

    Returns a DataFrame with one row per fold and the mean/SD in ``attrs``.
    Predictions use ``result.predict`` so GLMs return mean-scale predictions.
    """
    if k < 2 or k > len(data):
        raise ValueError("k must be between 2 and the number of observations.")
    fit_kwargs = fit_kwargs or {}
    response = formula.split("~")[0].strip()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data)) if shuffle else np.arange(len(data))
    folds = np.array_split(idx, k)
    rows = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        res = fit_fn(formula, data=train).fit(**fit_kwargs)
        pred = np.asarray(res.predict(test))
        scores = _score(np.asarray(test[response], dtype=float), pred)
        rows.append({"fold": i + 1, "n_train": len(train), "n_test": len(test), **scores})
    table = pd.DataFrame(rows).set_index("fold")
    table.attrs["mean"] = table[["rmse", "mae", "r2_oos"]].mean().to_dict()
    table.attrs["sd"] = table[["rmse", "mae", "r2_oos"]].std(ddof=1).to_dict()
    table.attrs["formula"] = formula
    return table


def _fit_subset(data, response, subset, fit_fn, fit_kwargs):
    rhs = " + ".join(subset) if subset else "1"
    return fit_fn(f"{response} ~ {rhs}", data=data).fit(**fit_kwargs)


def _crit_value(res, criterion: str) -> float:
    if criterion == "aic":
        return float(res.aic)
    if criterion == "bic":
        return float(res.bic)
    if criterion == "adj_r2":
        return -float(res.rsquared_adj)  # minimise the negative
    raise ValueError("criterion must be 'aic', 'bic', or 'adj_r2'.")


def best_subset(
    data: pd.DataFrame,
    response: str,
    candidates: Sequence[str],
    max_size: int | None = None,
    criterion: str = "aic",
    fit_fn: Callable = smf.ols,
    fit_kwargs: dict | None = None,
    include_empty: bool = True,
) -> pd.DataFrame:
    """Exhaustively fit every subset of ``candidates`` and rank by ``criterion``."""
    fit_kwargs = fit_kwargs or {}
    max_size = len(candidates) if max_size is None else max_size
    rows = []
    sizes = range(0 if include_empty else 1, max_size + 1)
    for size in sizes:
        for subset in combinations(candidates, size):
            res = _fit_subset(data, response, list(subset), fit_fn, fit_kwargs)
            row = {"predictors": " + ".join(subset) if subset else "(intercept only)", "size": size, "aic": float(res.aic), "bic": float(res.bic)}
            if hasattr(res, "rsquared_adj"):
                row["adj_r2"] = float(res.rsquared_adj)
            row["_crit"] = _crit_value(res, criterion)
            rows.append(row)
    table = pd.DataFrame(rows).sort_values("_crit").drop(columns="_crit").reset_index(drop=True)
    table.attrs["criterion"] = criterion
    return table


def stepwise_selection(
    data: pd.DataFrame,
    response: str,
    candidates: Sequence[str],
    direction: str = "both",
    criterion: str = "aic",
    fit_fn: Callable = smf.ols,
    fit_kwargs: dict | None = None,
    verbose: bool = False,
):
    """Stepwise predictor selection.

    Parameters
    ----------
    direction
        ``"forward"`` starts empty and adds; ``"backward"`` starts full and
        removes; ``"both"`` starts empty and considers additions and removals
        at every step.

    Returns
    -------
    (result, selected, trace)
        The final fitted model, the list of selected predictors, and a
        DataFrame recording every step.
    """
    if direction not in {"forward", "backward", "both"}:
        raise ValueError("direction must be 'forward', 'backward', or 'both'.")
    fit_kwargs = fit_kwargs or {}
    candidates = list(candidates)
    selected = candidates.copy() if direction == "backward" else []
    current = _crit_value(_fit_subset(data, response, selected, fit_fn, fit_kwargs), criterion)
    trace = [{"step": 0, "action": "start", "variable": None, "predictors": " + ".join(selected) or "(none)", criterion: current if criterion != "adj_r2" else -current}]

    step = 0
    while True:
        moves = []
        if direction in {"forward", "both"}:
            for var in candidates:
                if var not in selected:
                    val = _crit_value(_fit_subset(data, response, selected + [var], fit_fn, fit_kwargs), criterion)
                    moves.append((val, "add", var))
        if direction in {"backward", "both"}:
            for var in selected:
                trial = [v for v in selected if v != var]
                val = _crit_value(_fit_subset(data, response, trial, fit_fn, fit_kwargs), criterion)
                moves.append((val, "remove", var))
        if not moves:
            break
        best_val, action, var = min(moves, key=lambda m: m[0])
        if best_val >= current - 1e-10:
            break
        selected = selected + [var] if action == "add" else [v for v in selected if v != var]
        current = best_val
        step += 1
        trace.append({"step": step, "action": action, "variable": var, "predictors": " + ".join(selected) or "(none)", criterion: current if criterion != "adj_r2" else -current})
        if verbose:
            print(f"step {step}: {action} {var} -> {criterion}={trace[-1][criterion]:.3f}")

    result = _fit_subset(data, response, selected, fit_fn, fit_kwargs)
    return result, selected, pd.DataFrame(trace)


def nested_f_test(restricted, full) -> dict:
    """F-test comparing nested OLS models via ``anova_lm``."""
    table = sm.stats.anova_lm(restricted, full)
    return {
        "f_statistic": float(table["F"].iloc[1]),
        "df_num": float(table["df_diff"].iloc[1]),
        "df_denom": float(table["df_resid"].iloc[1]),
        "p_value": float(table["Pr(>F)"].iloc[1]),
    }
