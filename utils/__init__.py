"""StatsmodelsMasterPro utilities.

Research-grade helpers layered on top of ``statsmodels``: resampling and robust
inference, effect sizes, causal inference, Monte Carlo validation, model
selection, time-series evaluation, publication-ready reporting, and diagnostics.

Modules are imported lazily by callers (``from utils import inference``) so that
optional dependencies such as ``lifelines`` or ``streamlit`` are only required
when the corresponding module is used.
"""

__version__ = "2.0.0"

__all__ = [
    "causal",
    "compare_models",
    "diagnostics",
    "effect_sizes",
    "inference",
    "mediation",
    "mixed_effects_utils",
    "model_selection",
    "model_utils",
    "power",
    "reporting",
    "simulation",
    "survival_utils",
    "time_series_utils",
    "visual_utils",
]
