"""Label cleaning rules (documented, reproducible). Reads features.parquet, writes features_clean.parquet.

1. Dedupe: identical code (whitespace-normalised md5) within the same problem+language -> keep one.
2. Conflict: the same normalised code carrying BOTH an AC and a TLE label -> drop all copies (label noise).
3. Borderline: AC with cpu_time >= 0.8 x limit -> flagged `borderline` and excluded from the main set
   (threshold reported as an ablation, not hidden).
   TLE rows with cpu_time ratio < 1.0 (judge inconsistency) are flagged `tle_odd` and excluded too.

    python -I -m tleye.data.clean
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

BORDER = 0.8


def norm_hash(path: str) -> str:
    return hashlib.md5(re.sub(rb"\s+", b"", Path(path).read_bytes())).hexdigest()


def main() -> None:
    df = pd.read_parquet("data/processed/features.parquet")
    sel = pd.read_parquet("data/processed/selected.parquet", columns=["submission_id", "path"])
    df = df.merge(sel, on="submission_id")
    df["code_hash"] = [norm_hash(p) for p in df["path"]]
    log = {"start": len(df)}

    g = df.groupby(["problem_id", "language", "code_hash"])["y_tle"]
    conflict = g.transform("nunique") > 1
    log["dropped_conflicting_labels"] = int(conflict.sum())
    df = df[~conflict]
    dup = df.duplicated(["problem_id", "language", "code_hash"], keep="first")
    log["dropped_duplicates"] = int(dup.sum())
    df = df[~dup].copy()

    df["borderline"] = (df["y_tle"] == 0) & (df["ratio"] >= BORDER)
    df["tle_odd"] = (df["y_tle"] == 1) & (df["ratio"] < 1.0)
    log["flagged_borderline_ac"] = int(df["borderline"].sum())
    log["flagged_tle_odd"] = int(df["tle_odd"].sum())
    main_set = df[~df["borderline"] & ~df["tle_odd"]]
    log["final_main_set"] = len(main_set)
    df.to_parquet("data/processed/features_clean.parquet")
    print(pd.Series(log).to_string())
    print(main_set.groupby(["split", "language"])["y_tle"].agg(n="size", tle_rate="mean").round(3).to_string())


if __name__ == "__main__":
    main()
