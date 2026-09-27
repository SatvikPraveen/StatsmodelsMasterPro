"""Shared fixtures for the StatsmodelsMasterPro test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "synthetic_data"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20240926)


@pytest.fixture
def linear_df() -> pd.DataFrame:
    """Small, well-behaved linear dataset with known coefficients (2, 1.5, -0.7)."""
    gen = np.random.default_rng(42)
    n = 200
    x1 = gen.normal(5, 2, n)
    x2 = gen.normal(10, 3, n)
    y = 2 + 1.5 * x1 - 0.7 * x2 + gen.normal(0, 1.5, n)
    return pd.DataFrame({"X1": x1, "X2": x2, "y": y})


@pytest.fixture
def ols_model(linear_df):
    return smf.ols("y ~ X1 + X2", data=linear_df).fit()


@pytest.fixture
def hetero_df() -> pd.DataFrame:
    """Heteroskedastic data: error sd grows with |X|."""
    gen = np.random.default_rng(7)
    n = 300
    x = gen.normal(0, 1, n)
    y = 3 + 2 * x + gen.normal(0, 1 + 2 * np.abs(x), n)
    return pd.DataFrame({"X": x, "y": y})


@pytest.fixture
def two_groups():
    gen = np.random.default_rng(11)
    return gen.normal(0, 1, 60), gen.normal(0.8, 1, 60)


@pytest.fixture
def ols_data_csv(data_dir) -> pd.DataFrame:
    return pd.read_csv(data_dir / "ols_data.csv")


@pytest.fixture
def add_constant():
    return sm.add_constant
