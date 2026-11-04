"""Choose which AtCoder submissions to extract (Gate 1 decision: AtCoder only, C++/Java/Python).

Per (problem, language, status): one submission per user first (seeded random), then fill
up to --cap. Keeps borderline-AC rows (flagged via ratio) so the 0.8x rule can be ablated later.

    python -I -m tleye.data.select --cap 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/extracted/Project_CodeNet")
LANGS = {"C++", "Java", "Python"}
STATUSES = {"Accepted", "Time Limit Exceeded"}


def pick(g: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    g = g.sample(frac=1.0, random_state=seed)            # seeded shuffle
    first = g.drop_duplicates("user_id")                 # one per user first
    rest = g.drop(first.index)
    return pd.concat([first, rest]).head(cap)


def main(cap: int, seed: int, out: Path) -> None:
    pl = pd.read_csv(ROOT / "metadata" / "problem_list.csv")
    at = pl[(pl["dataset"] == "AtCoder") & pl["time_limit"].notna()].set_index("id")
    frames = []
    for pid in at.index:
        f = ROOT / "metadata" / f"{pid}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=["submission_id", "problem_id", "user_id", "language",
                                    "original_language", "filename_ext", "status", "cpu_time", "code_size"])
        d = d[d["language"].isin(LANGS) & d["status"].isin(STATUSES)]
        if d.empty:
            continue
        has = d.groupby("language")["status"].agg(lambda s: set(s) == STATUSES)
        d = d[d["language"].map(has)]                   # only languages with both AC and TLE
        for _, g in d.groupby(["language", "status"], sort=False):
            frames.append(pick(g, cap, seed))
    sel = pd.concat(frames, ignore_index=True)
    sel["time_limit_ms"] = sel["problem_id"].map(at["time_limit"])
    sel["ratio"] = sel["cpu_time"] / sel["time_limit_ms"]
    sel["y_tle"] = (sel["status"] == "Time Limit Exceeded").astype(int)
    sel["runtime"] = sel["original_language"].str.extract(r"^(PyPy\d?)", expand=False).fillna(sel["language"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(out)
    print(sel.groupby(["language", "status"]).agg(n=("submission_id", "size"),
                                                  problems=("problem_id", "nunique")).to_string())
    print("total", len(sel), "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/processed/selected.parquet"))
    a = ap.parse_args()
    main(a.cap, a.seed, a.out)
