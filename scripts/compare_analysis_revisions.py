"""Compare two analysis revisions table by table and list every changed cell.

Usage:
    python scripts/compare_analysis_revisions.py OLD_ANALYSIS_DIR NEW_ANALYSIS_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS = ["metric", "missing_rate", "target_pattern", "source_pattern", "classifier"]


def compare(old_dir: Path, new_dir: Path) -> int:
    old_tables = sorted((old_dir / "tables").glob("*.csv"))
    new_names = sorted(p.name for p in (new_dir / "tables").glob("*.csv"))
    if [p.name for p in old_tables] != new_names:
        raise SystemExit(f"Different table sets: {[p.name for p in old_tables]} vs {new_names}")
    changed = 0
    for old_path in old_tables:
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_dir / "tables" / old_path.name)
        if old.shape != new.shape or list(old.columns) != list(new.columns):
            raise SystemExit(f"{old_path.name}: table layout differs")
        keys = [c for c in KEY_COLUMNS if c in old.columns]
        for column in old.columns:
            a, b = old[column], new[column]
            if pd.api.types.is_numeric_dtype(a):
                differs = ~(np.isclose(a, b, rtol=0, atol=0, equal_nan=True))
            else:
                differs = a.astype(str) != b.astype(str)
            for i in np.flatnonzero(np.asarray(differs)):
                label = ", ".join(f"{k}={old.at[i, k]}" for k in keys)
                print(f"{old_path.name} [{label}] {column}: {a.iloc[i]!r} -> {b.iloc[i]!r}")
                changed += 1
    print(f"{changed} changed cells in {len(old_tables)} tables")
    return changed


if __name__ == "__main__":
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
