"""Integrity and schema tests for the synthetic datasets and their registry."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from scripts.build_manifest import sha256
from synthetic_data.dgp_registry import DATASETS


@pytest.fixture(scope="module")
def manifest(data_dir):
    return json.loads((data_dir / "MANIFEST.json").read_text())["datasets"]


def test_registry_covers_every_csv(data_dir):
    files_on_disk = {p.name for p in data_dir.glob("*.csv")}
    assert files_on_disk == {m["file"] for m in DATASETS.values()}


def test_manifest_hashes_match_files(data_dir, manifest):
    for name, meta in manifest.items():
        assert sha256(data_dir / name) == meta["sha256"], f"{name} has been modified; rebuild the manifest"


@pytest.mark.parametrize("key", list(DATASETS))
def test_schema_and_shape(key, data_dir, manifest):
    meta = DATASETS[key]
    df = pd.read_csv(data_dir / meta["file"])
    assert len(df) == meta["n"] == manifest[meta["file"]]["rows"]
    assert list(df.columns) == manifest[meta["file"]]["columns"]
    assert not df.isna().any().any()
    for col in meta["columns"]:
        if "-" not in col:  # ranges like Cat1-Cat5 are documented collectively
            assert col in df.columns, f"{col} documented but missing from {meta['file']}"


def test_true_params_are_numeric():
    for key, meta in DATASETS.items():
        assert meta["true_params"], key
        assert all(isinstance(v, (int, float)) for v in meta["true_params"].values()), key


def test_verify_manifest_script_passes(project_root):
    proc = subprocess.run([sys.executable, "scripts/verify_manifest.py"], cwd=project_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "All datasets verified." in proc.stdout


@pytest.mark.slow
def test_parameter_recovery_strict(project_root):
    proc = subprocess.run([sys.executable, "scripts/parameter_recovery.py", "--strict"], cwd=project_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
