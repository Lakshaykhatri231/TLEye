"""Phase 1 feasibility spike: aggregate CodeNet metadata (text only, no code executed).

Streams per-problem CSVs one at a time and keeps only small aggregates, so memory
stays low on an 8 GB machine. Writes data/processed/spike_agg.json for the report.

    python -I -m tleye.data.spike --root data/raw/extracted/Project_CodeNet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LANGS = ["C++", "Java", "Python"]
RATIO_BINS = [-np.inf, 0, 0.1, 0.25, 0.5, 0.8, 0.95, 1.0, 1.05, 1.5, 3, np.inf]


def main(root: Path, out: Path) -> None:
    plist = pd.read_csv(root / "metadata" / "problem_list.csv")
    limits = plist.set_index("id")["time_limit"]
    dataset = plist.set_index("id")["dataset"]

    status_all: dict[str, int] = {}
    cells: list[pd.DataFrame] = []  # (problem, language, original_language, status) -> n
    ratio_hist: dict[str, np.ndarray] = {}
    cpu_stats: dict[str, dict[str, int]] = {}

    files = sorted((root / "metadata").glob("p[0-9]*.csv"))  # excludes problem_list.csv
    for i, f in enumerate(files):
        d = pd.read_csv(
            f,
            usecols=["problem_id", "language", "original_language", "status", "cpu_time"],
            dtype={"problem_id": "string", "language": "string",
                   "original_language": "string", "status": "string"},
        )
        for s, n in d["status"].value_counts().items():
            status_all[s] = status_all.get(s, 0) + int(n)
        d = d[d["language"].isin(LANGS) & d["status"].isin(["Accepted", "Time Limit Exceeded"])]
        if d.empty:
            continue
        pid = d["problem_id"].iloc[0]
        lim = limits.get(pid, np.nan)
        cells.append(
            d.groupby(["problem_id", "language", "original_language", "status"])
            .size().rename("n").reset_index()
        )
        ds = dataset.get(pid, "?")
        for (lang, st), g in d.groupby(["language", "status"]):
            key = f"{ds}|{lang}|{st}"
            cpu = g["cpu_time"]
            c = cpu_stats.setdefault(key, {"n": 0, "cpu_nan": 0, "cpu_le0": 0, "limit_nan": 0})
            c["n"] += len(g)
            c["cpu_nan"] += int(cpu.isna().sum())
            c["cpu_le0"] += int((cpu <= 0).sum())
            if np.isnan(lim):
                c["limit_nan"] += len(g)
                continue
            r = (cpu.dropna() / lim).to_numpy()
            h, _ = np.histogram(r, bins=RATIO_BINS)
            ratio_hist[key] = ratio_hist.get(key, np.zeros(len(RATIO_BINS) - 1, dtype=np.int64)) + h
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(files)} files", flush=True)

    agg = pd.concat(cells, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out.parent / "spike_cells.parquet")

    # Statements
    desc_dir = root / "problem_descriptions"
    have_desc = {p.stem for p in desc_dir.glob("p*.html")}
    ids = set(plist["id"])

    result = {
        "n_problems_listed": len(plist),
        "n_metadata_files": len(files),
        "datasets": plist["dataset"].value_counts().to_dict(),
        "time_limit_missing": int(plist["time_limit"].isna().sum()),
        "time_limit_describe_ms": plist["time_limit"].describe().to_dict(),
        "time_limit_by_dataset_ms": plist.groupby("dataset")["time_limit"].describe().to_dict("index"),
        "status_all": dict(sorted(status_all.items(), key=lambda kv: -kv[1])),
        "descriptions_found": len(have_desc & ids),
        "descriptions_missing": len(ids - have_desc),
        "ratio_bins": [str(b) for b in RATIO_BINS],
        "ratio_hist": {k: v.tolist() for k, v in ratio_hist.items()},
        "cpu_stats": cpu_stats,
    }
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/extracted/Project_CodeNet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/spike_agg.json"))
    a = ap.parse_args()
    main(a.root, a.out)
