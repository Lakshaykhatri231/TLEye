"""Seed-robustness report for the confirmed LoRA recipe; follows reports/seeds_predeclaration.md. -> reports/seeds_test.md
    python -I -m tleye.eval.seeds_report [--dir reports/kaggle_seeds] [--out reports/seeds_test.md]
Seed 0 = the original confirmatory run (reports/kaggle_confirm/cfgN); seeds 1 and 2 = the extra runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import cluster_bootstrap, metrics_with_ci

ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_seeds")); ap.add_argument("--out", type=Path, default=Path("reports/seeds_test.md"))
a = ap.parse_args()
meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle"])
d = meta.merge(pd.read_parquet("data/processed/b2m_test_preds.parquet").rename(columns={"p_raw": "B2m"}), on="submission_id")
SRC = {"seed 0": Path("reports/kaggle_confirm/cfgN/test_logits.npz"), "seed 1": a.dir / "cfg_s1" / "test_logits.npz", "seed 2": a.dir / "cfg_s2" / "test_logits.npz"}
T = {}
for k, f in SRC.items():
    z = np.load(f); T[k] = float(z["T"])
    d = d.merge(pd.DataFrame({"submission_id": z["submission_id"], k: 1 / (1 + np.exp(-z["logit"] / float(z["T"]))), f"y_{k}": z["y"]}), on="submission_id")
    assert (d[f"y_{k}"].values == d["y_tle"].values).all(), f"label mismatch {k}"
y, g = d.y_tle.values, d.problem_id.values
fm = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
L = ["# Seed robustness of the confirmed LoRA recipe (script-generated; follows reports/seeds_predeclaration.md)", "",
     f"{len(d)} test rows, {d.problem_id.nunique()} problems. Same recipe (lr 1e-4, no prefix, 40k rows); only the seed differs. Problem-clustered bootstrap (300).", "",
     "| run | temperature T | AUROC [95% CI] | AUPRC [95% CI] | ECE [95% CI] |", "|---|---|---|---|---|"]
aurocs, ok = {}, {}
for k in SRC:
    r = metrics_with_ci(y, d[k].values, g, prob=True, n_boot=300); aurocs[k] = r["auroc"][0]
    L.append(f"| {k} | {T[k]:.3f} | {fm(r['auroc'])} | {fm(r['auprc'])} | {fm(r['ece'])} |")
r = metrics_with_ci(y, d["B2m"].values, g, prob=True, n_boot=300)
L.append(f"| B2m (reference) | - | {fm(r['auroc'])} | {fm(r['auprc'])} | {fm(r['ece'])} |")
vals = np.array(list(aurocs.values()))
L += ["", f"AUROC across seeds: min {vals.min():.3f}, max {vals.max():.3f}, spread (max - min) {vals.max() - vals.min():.3f}, mean {vals.mean():.3f}. "
      "Three seeds only: no interval over seeds is claimed.", "", "## Paired AUROC differences (first minus second), same rows", "", "| comparison | difference [95% CI] |", "|---|---|"]


def pair(x, w):
    lo, hi = cluster_bootstrap(g, lambda i: roc_auc_score(y[i], d[x].values[i]) - roc_auc_score(y[i], d[w].values[i]), n_boot=300)
    return roc_auc_score(y, d[x]) - roc_auc_score(y, d[w]), lo, hi


for k in SRC:
    pt, lo, hi = pair(k, "B2m"); ok[k] = lo > 0; L.append(f"| {k} - B2m | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
for x, w in [("seed 1", "seed 0"), ("seed 2", "seed 0"), ("seed 2", "seed 1")]:
    pt, lo, hi = pair(x, w); L.append(f"| {x} - {w} | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
L += ["", "## AUROC by language", "", "| language | n | seed 0 | seed 1 | seed 2 | B2m |", "|---|---|---|---|---|---|"]
for lang, s in d.groupby("language"):
    L.append(f"| {lang} | {len(s)} | " + " | ".join(f"{roc_auc_score(s.y_tle, s[c]):.3f}" for c in list(SRC) + ["B2m"]) + " |")
allok = all(ok.values())
L += ["", "## Claim by the pre-declared rule",
      f"- **{'The LoRA advantage over B2m reproduces across all three seeds (every paired CI excludes 0)' if allok else 'NOT robust: seed(s) ' + ', '.join(k for k, v in ok.items() if not v) + ' do not clear B2m by the paired-CI rule'}**.",
      f"- Run-to-run spread in AUROC: {vals.max() - vals.min():.3f} (three seeds)."]
a.out.write_text("\n".join(L) + "\n"); print("\n".join(L))
