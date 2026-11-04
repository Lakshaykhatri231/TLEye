"""LoRA vs B2 on the SAME test rows: metrics with problem-clustered CIs + paired bootstrap of differences.

    python -I -m tleye.eval.compare_lora --dir reports/kaggle_train [--out reports/lora_vs_b2.md]
Inputs: <dir>/test_logits.npz, <dir>/val_logits.npz (from kaggle/train/train.py), data/processed/b2_test_preds.parquet.
Probabilities for LoRA = sigmoid(logit / T), T fit on the val subset. 'avg' = fixed-rule average of the two
probabilities (no tuning). F1 is omitted here because it depends on a threshold; AUROC/AUPRC/ECE/Brier are reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tleye.eval.metrics import metrics_with_ci


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def paired_ci(groups, y, a, b, fn, n_boot=500, seed=0):
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        try:
            d.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
        except ValueError:
            continue
    d = np.array(d)
    return float(fn(y, a) - fn(y, b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main(dirp: Path, out: Path) -> None:
    lt = np.load(dirp / "test_logits.npz")
    T = float(lt["T"])
    df = pd.DataFrame({"submission_id": lt["submission_id"], "logit": lt["logit"], "y": lt["y"]})
    meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle"])
    b2 = pd.read_parquet("data/processed/b2_test_preds.parquet")
    d = df.merge(meta, on="submission_id").merge(b2, on="submission_id")
    assert (d["y"].values == d["y_tle"].values).all(), "label mismatch between Kaggle output and local data"
    d["p_lora"] = sigmoid(d["logit"] / T)
    d["p_avg"] = (d["p_lora"] + d["p_raw"]) / 2
    y, g = d["y"].values, d["problem_id"].values
    L = [f"# LoRA (Qwen2.5-Coder-0.5B) vs B2 LightGBM - same test rows (n={len(d)}, problems={d.problem_id.nunique()})", "",
         f"Temperature T = {T:.3f} (fit on val subset). Point [95% CI], problem-clustered bootstrap (500). Prevalence {y.mean():.3f} (capped-sample artifact).", ""]
    rows = []
    for scope, m in [("ALL", np.ones(len(d), bool))] + [(l, (d.language == l).values) for l in ["C++", "Java", "Python"]]:
        for name, col in [("B2 LightGBM (raw)", "p_raw"), ("LoRA 0.5B (T-scaled)", "p_lora"), ("avg(LoRA,B2) fixed rule", "p_avg")]:
            r = metrics_with_ci(y[m], d[col].values[m], g[m], prob=True, n_boot=500)
            rows.append({"scope": scope, "model": name, "n": r["n"], "auroc": fmt(r["auroc"]), "auprc": fmt(r["auprc"]),
                         "ece": fmt(r["ece"]), "brier": round(float(np.mean((d[col].values[m] - y[m]) ** 2)), 4)})
    L += [pd.DataFrame(rows).to_markdown(index=False), "", "## Paired differences (LoRA minus B2), same resampled problems", ""]
    prow = []
    for scope, m in [("ALL", np.ones(len(d), bool))] + [(l, (d.language == l).values) for l in ["C++", "Java", "Python"]]:
        for mname, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score)]:
            pt, lo, hi, p0 = paired_ci(g[m], y[m], d["p_lora"].values[m], d["p_raw"].values[m], fn)
            prow.append({"scope": scope, "metric": mname, "diff": f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                         "share of resamples where LoRA <= B2": round(p0, 3)})
    L += [pd.DataFrame(prow).to_markdown(index=False), "",
          "A CI that excludes 0 supports a real difference on THESE test problems; it is not a claim about other datasets."]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_train"))
    ap.add_argument("--out", type=Path, default=Path("reports/lora_vs_b2.md"))
    a = ap.parse_args()
    main(a.dir, a.out)
