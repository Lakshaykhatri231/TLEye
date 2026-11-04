"""Problem-disjoint train/val/test split (70/10/20). NEVER split by submission.

    python -I -m tleye.data.splits
Writes data/processed/splits.parquet (problem_id -> split) and prints per-split coverage.
Leave-one-language-out (LOLO) reuses the same problem splits: train on 2 languages' train
rows, evaluate on the held-out language's test rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}


def make_split(problem_ids: list[str], seed: int) -> pd.DataFrame:
    ids = np.array(sorted(set(problem_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(FRACS["train"] * n), int(FRACS["val"] * n)
    split = np.empty(n, dtype=object)
    split[:n_tr], split[n_tr:n_tr + n_va], split[n_tr + n_va:] = "train", "val", "test"
    return pd.DataFrame({"problem_id": ids, "split": split})


def coverage(sel: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    m = sel.merge(sp, on="problem_id")
    return (m.groupby(["split", "language", "status"])
              .agg(submissions=("submission_id", "size"), problems=("problem_id", "nunique"))
              .reset_index())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sel", type=Path, default=Path("data/processed/selected.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/splits.parquet"))
    a = ap.parse_args()
    sel = pd.read_parquet(a.sel)
    sp = make_split(sel["problem_id"].tolist(), a.seed)
    sp.to_parquet(a.out)
    print(sp["split"].value_counts().to_string())
    print(coverage(sel, sp).to_string(index=False))
