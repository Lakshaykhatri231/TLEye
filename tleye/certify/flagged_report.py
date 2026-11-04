"""What happens to the solutions the LoRA model FLAGS? Crosses the two certificate pilots (real sandbox outcomes) with the model's test logits.
    python -I -m tleye.certify.flagged_report   ->  reports/phase4_flagged.md
Caveat built into the design: the pilots sampled 2 Accepted + 2 TLE submissions per problem/runtime irrespective of the model, so TLE prevalence in this
sample is ~47% (not real-world) and intervals ignore problem clustering. Strict rule: certified = CPU time >= the limit on some generated input."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

meta = json.loads(Path("models/lora/meta.json").read_text()); T, THR = meta["T"], meta["thr_f1_val"]
z = np.load("reports/kaggle_confirm/cfgN/test_logits.npz")
p = pd.Series(1 / (1 + np.exp(-z["logit"] / float(z["T"]))), index=z["submission_id"])


def wil(k, n, zz=1.96):
    if n == 0:
        return (float("nan"),) * 2
    q = k / n; d = 1 + zz * zz / n; c = (q + zz * zz / (2 * n)) / d; h = zz * math.sqrt(q * (1 - q) / n + zz * zz / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    lo, hi = wil(k, n); return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]" if n else "0/0"


L = ["# Certificates among the solutions the LoRA model flags (script-generated)", "",
     f"Flag = P(TLE) >= {THR:.3f} (F1-optimal on val, capped prevalence). Rows = test-split submissions from the two certificate pilots (supported problems only). "
     "Sandbox outcome strict rule (CPU >= limit). Wilson intervals ignore problem clustering; sample TLE share is by design ~47%, not real-world.", ""]
for name, f in [("pilot 1", "cert_pilot.jsonl"), ("pilot 2 (fresh problems, final battery)", "cert_pilot_confirm.jsonl"), ("both pilots pooled", None)]:
    d = pd.concat([pd.DataFrame([json.loads(x) for x in Path("data/processed", g).read_text().splitlines()]) for g in (["cert_pilot.jsonl", "cert_pilot_confirm.jsonl"] if f is None else [f])])
    d = d[d.status != "compile_error"].copy()
    d["p"] = d.submission_id.map(p); d = d.dropna(subset=["p"])
    d["flag"] = d.p >= THR
    d["cert"] = [any(x["rc"] in (0, 124) and x["cpu_s"] * 1000 >= lim for x in rs) for rs, lim in zip(d.runs, d.limit_ms)]
    fl, tle, ac = d[d.flag], d[(d.flag) & (d.y_tle == 1)], d[(d.flag) & (d.y_tle == 0)]
    L += [f"## {name}", "", f"{len(d)} submissions ({d.problem_id.nunique()} problems), {len(fl)} flagged by the model.", "",
          "| question | answer |", "|---|---|",
          f"| share of true TLE solutions the model flags (recall) | {pct(int(((d.y_tle == 1) & d.flag).sum()), int((d.y_tle == 1).sum()))} |",
          f"| share of true Accepted solutions the model flags (false alarms) | {pct(len(ac), int((d.y_tle == 0).sum()))} |",
          f"| flagged AND truly TLE: confirmed by a failing input | {pct(int(tle.cert.sum()), len(tle))} |",
          f"| flagged BUT truly Accepted: wrongly certified (false certificate) | {pct(int(ac.cert.sum()), len(ac))} |",
          f"| all flagged: certified share | {pct(int(fl.cert.sum()), len(fl))} |",
          f"| certified-TLE precision among flagged (certified that are truly TLE) | {pct(int(tle.cert.sum()), int(fl.cert.sum()))} |",
          f"| flagged TLE solutions left 'suspected only' (not confirmed) | {pct(int((~tle.cert).sum()), len(tle))} |", ""]
L += ["## Reading it",
      "- A flag alone has many false alarms; the certificate is what supplies precision. Certified solutions are almost all truly TLE, but a large share of true TLE solutions stay 'suspected only' because no generated input slowed them down.",
      "- No claim about real-world precision: the sample is label-balanced by design and covers only problems the generator fully supports."]
Path("reports/phase4_flagged.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
