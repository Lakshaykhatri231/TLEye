"""Build data/processed/features.parquet: AST features + parsed constraints + labels + split.

    python -I -m tleye.features.build
Reads source files as BYTES and parses them with tree-sitter; nothing is executed.
"""
from __future__ import annotations

import math
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from tleye.constraints.parser import parse_statement
from tleye.features.ast_features import extract

ROOT = Path("data/raw/extracted/Project_CodeNet")


def _work(args: tuple[str, str]) -> dict:
    path, lang = args
    try:
        return extract(Path(path).read_bytes()[:400_000], lang)
    except Exception as e:  # keep going; record failure
        return {"_error": repr(e)[:80]}


def constraints_table(problem_ids: list[str]) -> pd.DataFrame:
    rows = []
    for pid in problem_ids:
        c = parse_statement((ROOT / "problem_descriptions" / f"{pid}.html").read_text(errors="replace"))
        rows.append(dict(
            problem_id=pid, c_section=int(c.found_section), c_n_upper=len(c.upper),
            c_max_bound=c.max_bound, c_primary=c.primary_size_bound, c_size=c.size_bound,
        ))
    d = pd.DataFrame(rows)
    for col in ("c_max_bound", "c_primary", "c_size"):
        d["log10_" + col[2:]] = d[col].map(lambda v: math.log10(min(v, 1e300)) if pd.notna(v) and v > 0 else float("nan"))
    return d


def main() -> None:
    sel = pd.read_parquet("data/processed/selected.parquet")
    sp = pd.read_parquet("data/processed/splits.parquet")
    jobs = list(zip(sel["path"], sel["language"]))
    with Pool(6) as pool:
        feats = pool.map(_work, jobs, chunksize=2000)
    f = pd.DataFrame(feats)
    errs = int(f["_error"].notna().sum()) if "_error" in f else 0
    f = f.drop(columns=["_error"], errors="ignore")
    keep = ["submission_id", "problem_id", "language", "runtime", "status", "y_tle", "cpu_time",
            "time_limit_ms", "ratio", "code_size", "user_id"]
    out = pd.concat([sel[keep].reset_index(drop=True), f], axis=1)
    out = out.merge(sp, on="problem_id").merge(constraints_table(sorted(out["problem_id"].unique())), on="problem_id")
    out.to_parquet("data/processed/features.parquet")
    print(f"rows={len(out)} parse_errors={errs} cols={out.shape[1]}")
    print(out.groupby(["split", "language"])["y_tle"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
