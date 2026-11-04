"""Re-run the INCONCLUSIVE (crashed) submissions of a pilot with stderr capture and group the error messages.
    python -I -m tleye.certify.diagnose_inconclusive data/processed/cert_pilot3.jsonl"""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from tleye.certify.bounds import parse as cert_parse
from tleye.certify.generate import build_inputs
from tleye.certify.inputspec import parse_input_spec
from tleye.certify.runner import certify_submission
from tleye.constraints.parser import constraints_text

R = Path("data/raw/extracted/Project_CodeNet")
rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines()]
rows = [r for r in rows if r["status"] == "inconclusive"]
sel = pd.read_parquet("data/processed/selected.parquet", columns=["submission_id", "path"]).set_index("submission_id")["path"]
bat = {}
for pid in {r["problem_id"] for r in rows}:
    h = (R / "problem_descriptions" / f"{pid}.html").read_text(errors="replace"); sec = constraints_text(h) or ""
    bat[pid] = build_inputs(parse_input_spec(h), cert_parse(sec), sec)


def work(r):
    c = certify_submission(sel[r["submission_id"]], r["rt"], bat[r["problem_id"]], r["limit_ms"])
    return r, c


def kind(err: str) -> str:
    e = err.lower()
    for k, pat in [("ValueError/int parse", r"valueerror|invalid literal"), ("IndexError", r"indexerror|out of range"), ("EOF / missing input", r"eoferror|eof"),
                   ("segfault/abort", r"segmentation|core dumped|aborted|terminate"), ("recursion/stack", r"recursion|stack"), ("MemoryError/OOM", r"memoryerror|out of memory|killed"),
                   ("NumberFormat (Java)", r"numberformat"), ("Java exception", r"exception")]:
        if re.search(pat, e):
            return k
    return "other/empty: " + err[:60]


out, kinds = [], Counter()
with ThreadPoolExecutor(2) as ex:
    for r, c in ex.map(work, rows):
        bad = [x for x in c.runs if x["rc"] not in (0, 124)]
        k = kind(bad[0].get("err", "")) if bad else "now-" + c.status
        kinds[(r["rt"], k)] += 1
        out.append({"problem_id": r["problem_id"], "rt": r["rt"], "status_now": c.status, "kind": k, "err": (bad[0].get("err", "") if bad else "")[:200]})
print(f"{len(out)} inconclusive submissions re-run")
for (rt, k), n in kinds.most_common():
    print(f"  {n:3d}  {rt:7s} {k}")
pd.DataFrame(out).to_csv("reports/inconclusive_diagnosis.csv", index=False)
print("by problem (top):", pd.DataFrame(out).groupby("problem_id").size().sort_values(ascending=False).head(8).to_dict())
