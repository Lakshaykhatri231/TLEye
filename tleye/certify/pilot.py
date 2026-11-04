"""Certificate pilot: run real TEST-split submissions (labelled AC / TLE by the original judge) against generated inputs.

    python -I -m tleye.certify.pilot --problems 25 --k 2 --workers 2
Writes data/processed/cert_pilot.jsonl (one JSON line per submission, appended as it goes) so an interrupted run keeps results.
Untrusted code only ever runs inside the Docker sandbox (tleye.certify.sandbox).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from tleye.certify.bounds import parse as cert_parse
from tleye.certify.generate import GenError, build_inputs
from tleye.certify.inputspec import parse_input_spec
from tleye.certify.runner import certify_submission
from tleye.constraints.parser import constraints_text

R = Path("data/raw/extracted/Project_CodeNet")
OUT = Path("data/processed/cert_pilot.jsonl")


def main(n_prob: int, k: int, workers: int, seed: int, out: Path = OUT, exclude: list[Path] | None = None, new_only: bool = False, all_simple: bool = False, baseline: Path = Path("data/processed/cert_coverage_v1.parquet")) -> None:
    global OUT
    OUT = out
    cov = pd.read_parquet("data/processed/cert_coverage.parquet")
    if all_simple:      # every currently supported problem, any split (certificates do not involve the model)
        simple = cov[cov.category == "simple"].problem_id.tolist()
    elif new_only:      # problems supported by the v2 bounds reader but NOT by v1: any split (certificates do not involve the model)
        v1 = pd.read_parquet(baseline).set_index("problem_id")["category"]
        simple = [p for p in cov[cov.category == "simple"].problem_id if v1.get(p) != "simple"]
    else:
        simple = cov[(cov.category == "simple") & (cov.split == "test")].problem_id.tolist()
    d = pd.read_parquet("data/processed/features_clean.parquet",
                        columns=["submission_id", "problem_id", "language", "runtime", "y_tle", "split", "time_limit_ms", "borderline", "tle_odd"])
    sel = pd.read_parquet("data/processed/selected.parquet", columns=["submission_id", "path"])
    d = d[(d.split == "test" if not (new_only or all_simple) else True) & ~d.borderline & ~d.tle_odd].merge(sel, on="submission_id")
    d["rt"] = d.runtime.where(~d.runtime.str.startswith("PyPy"), "PyPy")
    used = {json.loads(line)["problem_id"] for f in (exclude or []) if f.exists() for line in f.read_text().splitlines()}
    simple = [p for p in simple if p not in used]
    probs = pd.Series(sorted(simple)).sample(min(n_prob, len(simple)), random_state=seed).tolist()

    batteries: dict[str, dict | str] = {}
    for pid in probs:
        html = (R / "problem_descriptions" / f"{pid}.html").read_text(errors="replace")
        sec = constraints_text(html) or ""
        try:
            batteries[pid] = build_inputs(parse_input_spec(html), cert_parse(sec), sec)
        except GenError as e:
            batteries[pid] = str(e)
    ok = [p for p, b in batteries.items() if isinstance(b, dict)]
    print(f"problems sampled {len(probs)}, generator ok {len(ok)}, failed {[(p, b) for p, b in batteries.items() if isinstance(b, str)]}", flush=True)

    tasks = []
    for pid in ok:
        for (rt, y), g in d[d.problem_id == pid].groupby(["rt", "y_tle"]):
            for r in g.sample(min(k, len(g)), random_state=seed).itertuples():
                tasks.append(r)
    print("tasks", len(tasks), flush=True)
    done = {json.loads(line)["submission_id"] for line in OUT.read_text().splitlines()} if OUT.exists() else set()
    tasks = [t for t in tasks if t.submission_id not in done]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def work(t):
        t0 = time.time()
        c = certify_submission(t.path, t.rt, batteries[t.problem_id], t.time_limit_ms)
        return {"submission_id": t.submission_id, "problem_id": t.problem_id, "language": t.language, "rt": t.rt,
                "y_tle": int(t.y_tle), "limit_ms": float(t.time_limit_ms), "status": c.status, "max_cpu_s": c.max_cpu_s,
                "runs": c.runs, "secs": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(workers) as ex, OUT.open("a") as f:
        for i, rec in enumerate(ex.map(work, tasks), 1):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 10 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=25); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--exclude", type=Path, nargs="*", default=None)
    ap.add_argument("--new-only", action="store_true"); ap.add_argument("--baseline", type=Path, default=Path("data/processed/cert_coverage_v1.parquet")); ap.add_argument("--all-simple", action="store_true")
    a = ap.parse_args()
    main(a.problems, a.k, a.workers, a.seed, a.out, a.exclude, a.new_only, a.all_simple, a.baseline)
