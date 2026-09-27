"""Smoke tests for the Streamlit dashboard using streamlit.testing.v1.AppTest.

Each new research page is run headlessly and its main actions are driven;
the test fails on any uncaught exception or ``st.error``. Requires the ``app``
extra (``pip install -e ".[app]"``); skipped otherwise.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.app


def _assert_clean(at: AppTest, label: str) -> None:
    assert not at.exception, f"{label}: {[e.value for e in at.exception]}"
    errors = [e.value for e in at.error]
    assert not errors, f"{label}: st.error shown: {errors}"


def _page(project_root, name: str, timeout: int = 300) -> AppTest:
    return AppTest.from_file(str(project_root / name), default_timeout=timeout)


def test_home(project_root):
    at = _page(project_root, "Home.py", 120).run()
    _assert_clean(at, "Home")
    assert at.dataframe, "Home should show the dataset preview"


@pytest.mark.parametrize(
    "section",
    ["Robust standard errors", "Bootstrap CI for a statistic", "Bootstrap regression", "Permutation test & effect sizes", "Multiple testing"],
)
def test_robust_inference_sections(project_root, section):
    at = _page(project_root, "pages/26_Robust_Inference.py").run()
    _assert_clean(at, "26 initial")
    at.sidebar.radio[0].set_value(section).run()
    if section == "Bootstrap regression":
        at.sidebar.slider[0].set_value(200)
    at.button[0].click().run()
    _assert_clean(at, f"26 {section}")
    assert at.dataframe, f"26 {section}: expected at least one results table"


@pytest.mark.parametrize("key", ["ipw_btn", "match_btn", "did_btn", "iv_btn", "rd_btn", "ev_btn"])
def test_causal_inference_tabs(project_root, key):
    at = _page(project_root, "pages/27_Causal_Inference.py").run()
    _assert_clean(at, "27 initial")
    at.sidebar.slider[0].set_value(600).run()
    at.button(key=key).click().run()
    _assert_clean(at, f"27 {key}")


def test_monte_carlo_page(project_root):
    at = _page(project_root, "pages/28_Monte_Carlo_Validation.py").run()
    _assert_clean(at, "28 initial")
    at.sidebar.slider[1].set_value(150).run()
    at.button(key="mc_btn").click().run()
    _assert_clean(at, "28 mc")
    at.slider(key="pw_reps").set_value(80).run()
    at.button(key="pw_btn").click().run()
    _assert_clean(at, "28 power")
    at.sidebar.checkbox[0].set_value(True).run()
    at.button(key="mc_btn").click().run()
    _assert_clean(at, "28 mc heteroskedastic")


def test_publication_reporting_page(project_root):
    at = _page(project_root, "pages/29_Publication_Reporting.py").run()
    _assert_clean(at, "29 initial")
    at.button[0].click().run()
    _assert_clean(at, "29 fit")
    assert len(at.dataframe) >= 3
