#!/usr/bin/env python3
"""Build synthetic_data/MANIFEST.json with SHA-256 hashes and shapes of every dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from synthetic_data.dgp_registry import DATASETS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "synthetic_data"
MANIFEST = DATA_DIR / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    entries = {}
    for key, meta in DATASETS.items():
        path = DATA_DIR / meta["file"]
        df = pd.read_csv(path)
        entries[meta["file"]] = {
            "registry_key": key,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "generator": meta["generator"],
        }
    return {"generated": str(date.today()), "generator_script": "synthetic_data/generate_datasets.py", "datasets": entries}


if __name__ == "__main__":
    manifest = build()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {MANIFEST.relative_to(PROJECT_ROOT)} with {len(manifest['datasets'])} datasets")
