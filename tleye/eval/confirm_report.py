"""Confirmatory-run report, following reports/confirm_predeclaration.md exactly (script-generated: reports/confirm_test.md).
    python -I -m tleye.eval.confirm_report [--dir reports/kaggle_confirm] [--out reports/confirm_test.md]
Both configs (P = with prefix, N = without) are reported; nothing is selected on the test split. Claim wording follows the pre-declared rule."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tleye.eval.metrics import cluster_bootstrap, metrics_with_ci


def sig(z):
    return 1 / (1 + np.exp(-z))


ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_confirm"))
ap.add_argument("--out", type=Path, default=Path("reports/confirm_test.md")); a = ap.parse_args()
meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle"])
d = meta.merge(pd.read_parquet("data/processed/b2_test_preds.parquet").rename(columns={"p_raw": "B2"})[["submission_id", "B2"]], on="submission_id") \
        .merge(pd.read_parquet("data/processed/b2m_test_preds.parquet").rename(columns={"p_raw": "B2m"}), on="submission_id")
T = {}
for c in "PN":
    z = np.load(a.dir / f"cfg{c}" / "test_logits.npz"); T[c] = float(z["T"])
    d = d.merge(pd.DataFrame({"submission_id": z["submission_id"], c: sig(z["logit"] / float(z["T"])), f"y{c}": z["y"]}), on="submission_id")
    assert (d[f"y{c}"].values == d["y_tle"].values).all(), "label mismatch with local data"
y, g = d.y_tle.values, d.problem_id.values
L = ["# Confirmatory test-split result (script-generated; follows reports/confirm_predeclaration.md)", "",
     f"{len(d)} test rows, {d.problem_id.nunique()} problems, prevalence {y.mean():.3f} (capped-sample artifact). P = lr 1e-4 WITH symbolic prefix (T={T['P']:.3f}), N = lr 1e-4 WITHOUT (T={T['N']:.3f}). "
     "Problem-clustered bootstrap, 300 resamples. One seed per config.", "", "| model | AUROC [95% CI] | AUPRC [95% CI] | ECE [95% CI] | Brier |", "|---|---|---|---|---|"]
for lab, col in [("B2 LightGBM", "B2"), ("B2m monotone LightGBM", "B2m"), ("LoRA N (no prefix)", "N"), ("LoRA P (prefix)", "P")]:
    r = metrics_with_ci(y, d[col].values, g, prob=True, n_boot=300)
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
    L.append(f"| {lab} | {f(r['auroc'])} | {f(r['auprc'])} | {f(r['ece'])} | {np.mean((d[col].values - y) ** 2):.4f} |")


def paired(m, x, z, fn):
    gg, yy, a_, b_ = g[m], y[m], d[x].values[m], d[z].values[m]
    pt = fn(yy, a_) - fn(yy, b_)
    lo, hi = cluster_bootstrap(gg, lambda i: fn(yy[i], a_[i]) - fn(yy[i], b_[i]), n_boot=300)
    return pt, lo, hi


L += ["", "## Paired differences (first minus second), same rows", "", "| comparison | metric | difference [95% CI] |", "|---|---|---|"]
res = {}
for x, z in [("P", "B2m"), ("N", "B2m"), ("P", "N"), ("P", "B2"), ("N", "B2")]:
    for mn, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score)]:
        pt, lo, hi = paired(np.ones(len(d), bool), x, z, fn); res[(x, z, mn)] = (pt, lo, hi)
        L.append(f"| {x} - {z} | {mn} | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
L += ["", "## Per-language test AUROC (paired difference vs B2m in brackets)", "", "| language | n | B2m | N | P |", "|---|---|---|---|---|"]
for lang in ["C++", "Java", "Python"]:
    m = (d.language == lang).values; row = [f"{roc_auc_score(y[m], d.B2m.values[m]):.3f}"]
    for c in "NP":
        pt, lo, hi = paired(m, c, "B2m", roc_auc_score); row.append(f"{roc_auc_score(y[m], d[c].values[m]):.3f} ({pt:+.3f} [{lo:+.3f}, {hi:+.3f}])")
    L.append(f"| {lang} | {int(m.sum())} | " + " | ".join(row) + " |")
ex = lambda k: res[k][1] > 0  # noqa: E731
pP, pN, pPN = ex(("P", "B2m", "AUROC")), ex(("N", "B2m", "AUROC")), res[("P", "N", "AUROC")][1] > 0 or res[("P", "N", "AUROC")][2] < 0
claim = ("LoRA beats B2m on the test split (both configs' paired CIs exclude 0)" if pP and pN else
         "PARTIAL: only " + ("P" if pP else "N") + " has a paired CI that excludes 0 versus B2m; report as partial" if pP or pN else
         "NOT CONFIRMED: neither config's paired AUROC CI excludes 0 versus B2m on the test split")
L += ["", "## Claim by the pre-declared rule", f"- Verdict model: **{claim}**.",
      "- Prefix: " + ("claimed useful (P - N CI excludes 0)." if pPN else "NOT claimed: the P - N paired CI contains 0, so this run cannot say whether the prefix helps.")]
a.out.write_text("\n".join(L) + "\n"); print("\n".join(L))
