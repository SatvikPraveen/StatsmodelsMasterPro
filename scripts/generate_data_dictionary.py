#!/usr/bin/env python3
"""Render docs/DATA_DICTIONARY.md from synthetic_data/dgp_registry.py and MANIFEST.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from synthetic_data.dgp_registry import DATASETS, ORDER_DEPENDENT  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "DATA_DICTIONARY.md"
MANIFEST = PROJECT_ROOT / "synthetic_data" / "MANIFEST.json"


def render() -> str:
    manifest = json.loads(MANIFEST.read_text())["datasets"] if MANIFEST.exists() else {}
    lines = [
        "# Data Dictionary",
        "",
        "Every dataset in `synthetic_data/` is simulated from a fully specified data-generating process (DGP)",
        "with a fixed seed, so the *true* parameter values are known. This document is generated from",
        "`synthetic_data/dgp_registry.py` by `scripts/generate_data_dictionary.py`; do not edit it by hand.",
        "",
        "Integrity hashes live in `synthetic_data/MANIFEST.json` and are checked by `scripts/verify_manifest.py`.",
        "Parameter recovery for every dataset is reported in `exports/tables/validation/parameter_recovery.csv`",
        "(produced by `scripts/parameter_recovery.py`).",
        "",
        "## Summary",
        "",
        "| Dataset | Rows | Generator | Intended models |",
        "|---|---:|---|---|",
    ]
    for key, meta in DATASETS.items():
        rows = manifest.get(meta["file"], {}).get("rows", meta["n"])
        lines.append(f"| `{meta['file']}` | {rows} | `{meta['generator']}` | {', '.join(meta['intended_models'])} |")
    lines += ["", "## Datasets", ""]
    for key, meta in DATASETS.items():
        lines += [f"### `{meta['file']}`", "", meta["description"], ""]
        lines += [f"- **Generator:** `{meta['generator']}` (n = {meta['n']}, seed: {meta['seed']})", f"- **DGP:** `{meta['dgp']}`"]
        if meta.get("note"):
            lines.append(f"- **Note:** {meta['note']}")
        if meta["file"] in manifest:
            lines.append(f"- **SHA-256:** `{manifest[meta['file']]['sha256']}`")
        lines += ["", "| Column | Definition |", "|---|---|"]
        lines += [f"| `{c}` | {d} |" for c, d in meta["columns"].items()]
        lines += ["", "| True parameter | Value |", "|---|---:|"]
        lines += [f"| `{p}` | {v} |" for p, v in meta["true_params"].items()]
        lines.append("")
    lines += [
        "## Reproducibility note",
        "",
        "The following generators do not reseed and therefore consume the module-level `np.random.seed(42)`",
        "stream in call order: " + ", ".join(f"`{g}`" for g in ORDER_DEPENDENT) + ".",
        "Changing their order in `generate_all_datasets()` changes those files; rebuild the manifest if you do.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render())
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}")
