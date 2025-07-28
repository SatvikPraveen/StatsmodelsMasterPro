# utils/mixed_effects_utils.py

from statsmodels.formula.api import mixedlm
import pandas as pd
from pathlib import Path

def fit_mixed_model(df, formula, groups_col, reml=True):
    """
    Fit a mixed effects model with random intercepts.
    """
    model = mixedlm(formula, data=df, groups=df[groups_col])
    result = model.fit(reml=reml)
    return result

def extract_random_effects(result):
    """
    Return random effects (group-level intercepts)
    """
    return result.random_effects

def export_mixed_summary(result, path: Path):
    """
    Export model summary as .txt
    """
    with open(path, "w") as f:
        f.write(result.summary().as_text())
