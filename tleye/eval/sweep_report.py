"""Learning-rate sweep report (VALIDATION ONLY; the test split was not scored by these runs): reports/lr_sweep.md
    python -I -m tleye.eval.sweep_report
Paired, problem-clustered bootstrap on the same 8,000 validation rows for every pair."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import cluster_bootstrap

D = Path("reports/kaggle_sweep")
meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language"])
b = pd.read_parquet("data/processed/b2_val_preds.parquet").rename(columns={"p_raw": "B2"}).merge(
    pd.read_parquet("data/processed/b2m_val_preds.parquet").rename(columns={"p_raw": "B2m"}), on="submission_id")
d = None
cfg = {}
for c in "ABC":
    m = json.load(open(D / f"cfg{c}" / "metrics.json")); z = np.load(D / f"cfg{c}" / "val_logits.npz")
    cfg[c] = m
    x = pd.DataFrame({"submission_id": z["submission_id"], c: z["logit"], "y": z["y"]})
    d = x if d is None else d.merge(x[["submission_id", c]], on="submission_id")
d = d.merge(b, on="submission_id").merge(meta, on="submission_id")
y, g = d.y.values, d.problem_id.values
cols = ["A", "B", "C", "B2", "B2m"]


def auc(col, i=None):
    return roc_auc_score(y if i is None else y[i], d[col].values if i is None else d[col].values[i])


def ci(fn):
    lo, hi = cluster_bootstrap(g, fn, n_boot=300)
    return lo, hi


L = ["# Learning-rate sweep (script-generated; VALIDATION only)", "",
     f"{len(d)} validation rows from {d.problem_id.nunique()} problems, identical for every model. 12,000 training rows, 1,500 optimizer steps each. "
     "Intervals: problem-clustered bootstrap (300). The test split was NOT scored by these runs.", "",
     "| model | learning rate | symbolic prefix | val AUROC [95% CI] | val BCE after temperature | training minutes |", "|---|---|---|---|---|---|"]
for c in "ABC":
    lo, hi = ci(lambda i, c=c: auc(c, i)); m = cfg[c]
    L.append(f"| LoRA {c} | {m['lr']:g} | {m['sym']} | {auc(c):.3f} [{lo:.3f}, {hi:.3f}] | {m['val_bce_after_T']:.3f} (constant guess {m['val_const_guess_bce']:.3f}) | {m['minutes']:.0f} |")
for c in ["B2", "B2m"]:
    lo, hi = ci(lambda i, c=c: auc(c, i)); L.append(f"| {c} | - | - | {auc(c):.3f} [{lo:.3f}, {hi:.3f}] | - | - |")
L += ["", "## Paired AUROC differences (first minus second), same rows, problems resampled together", "", "| comparison | difference [95% CI] |", "|---|---|"]
for a_, b_ in [("A", "B2m"), ("B", "B2m"), ("C", "B2m"), ("C", "B2"), ("B", "A"), ("C", "B")]:
    lo, hi = ci(lambda i, a_=a_, b_=b_: auc(a_, i) - auc(b_, i))
    L.append(f"| {a_} - {b_} | {auc(a_) - auc(b_):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
L += ["", "## Validation AUROC by language", "", "| language | n | A | B | C | B2 | B2m |", "|---|---|---|---|---|---|---|"]
for lang, s in d.groupby("language"):
    L.append(f"| {lang} | {len(s)} | " + " | ".join(f"{roc_auc_score(s.y, s[c]):.3f}" for c in cols) + " |")
L += ["", "## Reading it",
      "- Configs A, B, C use identical data/steps and differ only in learning rate and the prefix, so their pairwise differences (last two rows above) say how much each choice matters; if the interval contains 0, the sweep cannot separate them.",
      "- A,B,C selected on validation among 3 options: the best of them is mildly optimistic. The confirmatory number must come from a longer run scored once on the TEST split.",
      "- Run 1 (lr 2e-4, 40k rows) reached only 0.663 on test; here lr 5e-5 to 1e-4 learns clearly on 12k rows. Validation is easier than test for B2 as well (0.776 vs 0.759)."]
Path("reports/lr_sweep.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
