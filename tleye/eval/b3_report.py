"""B3 (prompted LLM) vs the fine-tuned models on the SAME 400 items; follows reports/b3_predeclaration.md. -> reports/b3_test.md
    python -I -m tleye.eval.b3_report [--dir reports/kaggle_b3] [--out reports/b3_test.md]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import cluster_bootstrap

ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_b3")); ap.add_argument("--out", type=Path, default=Path("reports/b3_test.md"))
a = ap.parse_args()
z = np.load(a.dir / "b3_scores.npz")
run = json.loads((a.dir / "run.json").read_text())
meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle"])
lz = np.load("reports/kaggle_confirm/cfgN/test_logits.npz")
d = pd.DataFrame({"submission_id": z["submission_id"], "prompt_A": z["sA"], "prompt_B": z["sB"], "massA": z["mA"], "massB": z["mB"]}) \
    .merge(meta, on="submission_id") \
    .merge(pd.DataFrame({"submission_id": lz["submission_id"], "LoRA": lz["logit"]}), on="submission_id") \
    .merge(pd.read_parquet("data/processed/b2m_test_preds.parquet").rename(columns={"p_raw": "B2m"}), on="submission_id")
assert len(d) == len(z["y"]) and (d.y_tle.values == pd.Series(z["y"], index=z["submission_id"]).loc[d.submission_id].values).all(), "item/label mismatch"
y, g = d.y_tle.values, d.problem_id.values
names = [("Prompted Qwen2.5-Coder-7B-Instruct, variant A (matched info)", "prompt_A"), ("Prompted, variant B (+ problem statement)", "prompt_B"),
         ("B2m LightGBM", "B2m"), ("Fine-tuned LoRA 0.5B", "LoRA")]


def auc(col, i=None):
    return roc_auc_score(y if i is None else y[i], d[col].values if i is None else d[col].values[i])


def ci(fn):
    return cluster_bootstrap(g, fn, n_boot=500)


L = ["# Prompted-LLM baseline B3 (script-generated; follows reports/b3_predeclaration.md)", "",
     f"{len(d)} fixed test items ({d.problem_id.nunique()} problems; {int((d.language == 'C++').sum())} C++ / {int((d.language == 'Java').sum())} Java / {int((d.language == 'Python').sum())} Python; "
     f"TLE share {y.mean():.3f}). Model {run['model']}, one forward pass per item, direct YES/NO answer (no chain-of-thought). "
     "Problem-clustered bootstrap (500). Small sample: intervals are wide.", "",
     f"Instruction following: mean probability mass on YES/NO tokens, variant A {d.massA.mean():.2f}, variant B {d.massB.mean():.2f} (1.0 = always answers YES or NO).", "",
     "| model | AUROC [95% CI] |", "|---|---|"]
for lab, col in names:
    lo, hi = ci(lambda i, c=col: auc(c, i)); L.append(f"| {lab} | {auc(col):.3f} [{lo:.3f}, {hi:.3f}] |")
L += ["", "## Paired AUROC differences (first minus second), same items", "", "| comparison | difference [95% CI] |", "|---|---|"]
res = {}
for x, w in [("LoRA", "prompt_A"), ("LoRA", "prompt_B"), ("B2m", "prompt_A"), ("B2m", "prompt_B"), ("LoRA", "B2m"), ("prompt_B", "prompt_A")]:
    lo, hi = ci(lambda i, x=x, w=w: auc(x, i) - auc(w, i)); res[(x, w)] = (auc(x) - auc(w), lo, hi)
    L.append(f"| {x} - {w} | {auc(x) - auc(w):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
L += ["", "## AUROC by language", "", "| language | n | prompt A | prompt B | B2m | LoRA |", "|---|---|---|---|---|---|"]
for lang, s in d.groupby("language"):
    L.append(f"| {lang} | {len(s)} | " + " | ".join(f"{roc_auc_score(s.y_tle, s[c]):.3f}" for _, c in names) + " |")
beatA, beatB = res[("LoRA", "prompt_A")][1] > 0, res[("LoRA", "prompt_B")][1] > 0
claim = ("fine-tuning beats prompting (LoRA - prompted CI excludes 0 for both variants)" if beatA and beatB else
         "PARTIAL: LoRA clearly beats variant " + ("A" if beatA else "B") + " only" if beatA or beatB else
         "NOT SHOWN: LoRA - prompted CI contains 0 for both variants on these 400 items")
L += ["", "## Claim by the pre-declared rule", f"- **{claim}**.",
      "- Limits: one open 7B model with direct answers only (no chain-of-thought, no few-shot, no larger/proprietary model); n=400 from about 200 problems."]
a.out.write_text("\n".join(L) + "\n"); print("\n".join(L))
