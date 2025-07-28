# compare_models.py

import pandas as pd

def extract_model_metrics(model, name):
    return {
        'Model': name,
        'AIC': model.aic,
        'BIC': model.bic,
        'R_squared': model.rsquared,
        'Adj_R_squared': model.rsquared_adj
    }

def compare_model_metrics(models: dict):
    results = []
    for name, model in models.items():
        results.append(extract_model_metrics(model, name))
    return pd.DataFrame(results).sort_values(by='AIC')

def forward_stepwise_selection(data, response, predictors, verbose=True):
    import statsmodels.formula.api as smf

    remaining = list(predictors)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    formula_base = response + ' ~ {}'

    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = formula_base.format(' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            if verbose:
                print(f'✔ Added {best_candidate}, AIC = {current_score:.2f}')
        else:
            break

    formula = formula_base.format(' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model, selected
