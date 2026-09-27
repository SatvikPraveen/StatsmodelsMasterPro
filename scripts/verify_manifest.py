#!/usr/bin/env python3
"""Verify every dataset matches the SHA-256 hash and shape recorded in MANIFEST.json.

Exit status 0 when all datasets verify, 1 otherwise. Use ``--regenerate`` to
also regenerate the datasets into a temporary directory and confirm the
generator is still byte-for-byte reproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.build_manifest import DATA_DIR, MANIFEST, sha256  # noqa: E402


def verify_files() -> bool:
    manifest = json.loads(MANIFEST.read_text())
    ok = True
    for name, meta in manifest["datasets"].items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"MISSING  {name}")
            ok = False
            continue
        digest = sha256(path)
        status = "OK      " if digest == meta["sha256"] else "MODIFIED"
        ok &= digest == meta["sha256"]
        print(f"{status} {name}  ({meta['rows']} rows)")
    return ok


def verify_regeneration() -> bool:
    manifest = json.loads(MANIFEST.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "generate_datasets.py"
        script.write_text((DATA_DIR / "generate_datasets.py").read_text())
        subprocess.run([sys.executable, str(script)], check=True, cwd=tmp, capture_output=True)
        ok = True
        for name, meta in manifest["datasets"].items():
            digest = sha256(Path(tmp) / name)
            status = "REPRODUCED" if digest == meta["sha256"] else "DIVERGED  "
            ok &= digest == meta["sha256"]
            print(f"{status} {name}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="also re-run the generator and compare hashes")
    args = parser.parse_args()
    good = verify_files()
    if args.regenerate:
        print("\nRegenerating datasets in a temporary directory ...")
        good &= verify_regeneration()
    print("\nAll datasets verified." if good else "\nVerification FAILED.")
    sys.exit(0 if good else 1)
