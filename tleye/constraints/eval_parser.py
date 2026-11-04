"""Evaluate the constraint parser against hand labels (val/test problems only, never tuned on).

    python -I -m tleye.constraints.eval_parser
Fields: max_bound (largest literal numeric upper bound), n_bound (bound of the primary size variable).
Match = both missing, or both present and equal (relative 1e-9). Wilson 95% CI.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tleye.constraints.parser import parse_statement

R = Path("data/raw/extracted/Project_CodeNet/problem_descriptions")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def same(a, b) -> bool:
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na or b_na:
        return a_na and b_na
    return math.isclose(float(a), float(b), rel_tol=1e-9)


def main() -> None:
    lab = pd.read_csv("data/labels/constraints_labels.csv")
    rows = []
    for pid, mb, nb in zip(lab.problem_id, lab.max_bound, lab.n_bound):
        c = parse_statement((R / f"{pid}.html").read_text(errors="replace"))
        rows.append(dict(problem_id=pid, true_max=mb, pred_max=c.max_bound, true_n=nb, pred_n=c.primary_size_bound))
    d = pd.DataFrame(rows)
    d["ok_max"] = [same(a, b) for a, b in zip(d.true_max, d.pred_max)]
    d["ok_n"] = [same(a, b) for a, b in zip(d.true_n, d.pred_n)]
    n = len(d)
    print(f"n = {n} problems (val/test only)")
    for name, col in [("max_bound", "ok_max"), ("n_bound", "ok_n")]:
        k = int(d[col].sum())
        lo, hi = wilson(k, n)
        print(f"{name:10s} exact match {k}/{n} = {k / n:.1%}  (95% CI {lo:.1%} - {hi:.1%})")
    # detection of 'has any size/bound' where truth exists
    has = d.true_n.notna()
    k = int((d.ok_n & has).sum())
    lo, hi = wilson(k, int(has.sum()))
    print(f"n_bound recall on problems that have one: {k}/{int(has.sum())} = {k / has.sum():.1%} (95% CI {lo:.1%} - {hi:.1%})")
    fp = int((d.true_n.isna() & d.pred_n.notna()).sum())
    print(f"n_bound false positives (parser invents a size): {fp}/{int((~has).sum())}")
    print("\nMismatches:")
    print(d[~(d.ok_max & d.ok_n)][["problem_id", "true_max", "pred_max", "true_n", "pred_n"]].to_string(index=False))
    d.to_csv("reports/constraints_eval.csv", index=False)


if __name__ == "__main__":
    main()
