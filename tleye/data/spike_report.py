"""Build reports/spike.md from spike aggregates. Every number is computed here, none typed by hand.

    python -I -m tleye.data.spike_report
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/extracted/Project_CodeNet")
BIN_NAMES = ["<0", "0-.1", ".1-.25", ".25-.5", ".5-.8", ".8-.95", ".95-1", "1-1.05", "1.05-1.5", "1.5-3", ">3"]
CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
BOUND = re.compile(r"\\leq?\s*[\d,]+|10\s*\^\s*\{?\d+\}?|\d[\d,]{2,}")


def stmt_probe(plist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, ds in zip(plist["id"], plist["dataset"]):
        f = ROOT / "problem_descriptions" / f"{pid}.html"
        if not f.exists():
            rows.append((ds, False, False, False, False))
            continue
        t = html.unescape(re.sub(r"<[^>]+>", " ", f.read_text(errors="replace")))
        m = re.search(r"Constraints(.*?)(?:Input|$)", t, re.DOTALL)
        sec = m.group(1) if m else ""
        rows.append((ds, True, bool(m), bool(BOUND.search(sec)), bool(CJK.search(t))))
    return pd.DataFrame(rows, columns=["dataset", "has_stmt", "has_constraints_sec", "has_bound", "has_cjk"])


def pct(a, b) -> str:
    return f"{100 * a / b:.2f}%" if b else "n/a"


def main() -> None:
    r = json.load(open("data/processed/spike_agg.json"))
    c = pd.read_parquet("data/processed/spike_cells.parquet")
    pl = pd.read_csv(ROOT / "metadata" / "problem_list.csv")
    ds_of = pl.set_index("id")["dataset"]
    c["dataset"] = c["problem_id"].map(ds_of)
    c["pypy"] = c["original_language"].str.contains("PyPy")
    L = []
    w = L.append

    w("# Phase 1 data feasibility spike (Gate 1)\n")
    w("All numbers computed by `tleye/data/spike.py` + `spike_report.py` from CodeNet metadata. "
      "Scope: languages C++ / Java / Python, statuses Accepted vs Time Limit Exceeded only.\n")

    w("## 1. Dataset basics")
    w(f"- Problems listed: {r['n_problems_listed']} ({r['datasets']}); time_limit missing: {r['time_limit_missing']}")
    w(f"- Statements found: {r['descriptions_found']} / {r['n_problems_listed']}\n")

    w("## 2. Label quality: cpu_time / limit ratio (the key finding)")
    w("Ratio bins: " + ", ".join(BIN_NAMES) + "\n")
    w("| dataset | lang | status | n | share >= 1.0x limit | histogram |")
    w("|---|---|---|---:|---:|---|")
    for k, v in sorted(r["ratio_hist"].items()):
        ds, lang, st = k.split("|")
        n = sum(v)
        w(f"| {ds} | {lang} | {st} | {n} | {pct(sum(v[7:]), n)} | {v} |")
    w("")

    w("## 3. Problems and submissions available (problems having both AC and TLE)")
    g = c.pivot_table(index=["problem_id", "language", "dataset"], columns="status",
                      values="n", aggfunc="sum", fill_value=0).reset_index()
    g["both"] = (g["Accepted"] > 0) & (g["Time Limit Exceeded"] > 0)
    g["both5"] = (g["Accepted"] >= 5) & (g["Time Limit Exceeded"] >= 5)
    t = g.groupby(["dataset", "language"]).agg(
        problems=("problem_id", "nunique"), with_tle=("Time Limit Exceeded", lambda x: int((x > 0).sum())),
        both=("both", "sum"), both_ge5=("both5", "sum"),
        ac_sub=("Accepted", "sum"), tle_sub=("Time Limit Exceeded", "sum")).reset_index()
    w(t.to_markdown(index=False) + "\n")

    w("## 4. Python runtime split (TLE submissions)")
    py = c[(c.language == "Python") & (c.status == "Time Limit Exceeded")]
    w(py.groupby(["dataset", "pypy"])["n"].sum().reset_index().to_markdown(index=False) + "\n")

    w("## 5. Statement parseability (crude regex probe, NOT the Phase 2 parser)")
    sp = stmt_probe(pl)
    s = sp.groupby("dataset").agg(problems=("has_stmt", "size"), with_stmt=("has_stmt", "sum"),
                                   constraints_section=("has_constraints_sec", "sum"),
                                   numeric_bound=("has_bound", "sum"), contains_cjk=("has_cjk", "sum")).reset_index()
    w(s.to_markdown(index=False) + "\n")

    w("## 6. Borderline AC rule: share of AC rows with cpu_time >= 0.8x limit")
    for k, v in sorted(r["ratio_hist"].items()):
        ds, lang, st = k.split("|")
        if st == "Accepted":
            w(f"- {ds} {lang}: {pct(sum(v[5:]), sum(v))} of {sum(v)}")
    w("")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/spike.md").write_text("\n".join(L))
    print("wrote reports/spike.md")


if __name__ == "__main__":
    main()
