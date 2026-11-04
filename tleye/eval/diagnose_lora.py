"""Where does the LoRA model lose to B2? AUROC by code-length quartile and loop depth (test split).
    python -I -m tleye.eval.diagnose_lora --dir reports/kaggle_train > reports/lora_diagnostics.txt"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_train")); a = ap.parse_args()
lt = np.load(a.dir / "test_logits.npz")
d = pd.DataFrame({"submission_id": lt["submission_id"], "lora": lt["logit"]})
m = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "y_tle", "n_chars", "language", "max_loop_depth"])
d = d.merge(m, on="submission_id").merge(pd.read_parquet("data/processed/b2_test_preds.parquet"), on="submission_id")
print(f"n {len(d)} | Spearman(LoRA, B2) = {spearmanr(d.lora, d.p_raw)[0]:.3f}")
d["q"] = pd.qcut(d.n_chars, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
print("\nAUROC by code-length quartile (test):")
for q, g in d.groupby("q", observed=True):
    print(f"  {q:12s} n={len(g):5d} chars {int(g.n_chars.min()):>6d}-{int(g.n_chars.max()):>6d}  LoRA {roc_auc_score(g.y_tle, g.lora):.3f}  B2 {roc_auc_score(g.y_tle, g.p_raw):.3f}")
print("\nAUROC by loop depth (test):")
for k, g in d.assign(dep=d.max_loop_depth.clip(upper=3)).groupby("dep"):
    if g.y_tle.nunique() > 1:
        print(f"  depth {k}: n={len(g):5d}  LoRA {roc_auc_score(g.y_tle, g.lora):.3f}  B2 {roc_auc_score(g.y_tle, g.p_raw):.3f}")
