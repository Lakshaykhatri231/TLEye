"""LoRA leave-one-language-out report, following reports/lolo_predeclaration.md (script-generated: reports/lolo_test.md).
    python -I -m tleye.eval.lolo_report [--dir reports/kaggle_lolo] [--out reports/lolo_test.md]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import cluster_bootstrap, metrics_with_ci

ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_lolo"))
ap.add_argument("--out", type=Path, default=Path("reports/lolo_test.md")); a = ap.parse_args()
HELD = {"cpp": "C++", "java": "Java", "py": "Python"}
meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle"])
b2 = pd.read_parquet("data/processed/b2_lolo_test_preds.parquet")
conf = np.load("reports/kaggle_confirm/cfgN/test_logits.npz") if Path("reports/kaggle_confirm/cfgN/test_logits.npz").exists() else None
indist = {}
if conf is not None:
    ci_ = pd.DataFrame({"submission_id": conf["submission_id"], "p": 1 / (1 + np.exp(-conf["logit"] / float(conf["T"])))}).merge(meta, on="submission_id")
    indist = {lang: roc_auc_score(g.y_tle, g.p) for lang, g in ci_.groupby("language")}
fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
L = ["# LoRA unseen-language (leave-one-language-out) test result (script-generated; follows reports/lolo_predeclaration.md)", "",
     "Each run: 20,000 training rows from the OTHER two languages, T fit on their val rows, FULL test rows of the held-out language scored once. lr 1e-4, no prefix, one seed. "
     "Problem-clustered bootstrap (300). In-distribution LoRA used 40,000 rows of all three languages, so the drop mixes 'unseen language' with 'less data'.", "",
     "| held-out | n | problems | LoRA LOLO AUROC [CI] | LightGBM LOLO AUROC [CI] | paired LoRA - LightGBM | LoRA AUPRC | LoRA ECE | in-distribution LoRA AUROC (40k rows, 3 langs) |", "|---|---|---|---|---|---|---|---|---|"]
verdict = {}
for key, lang in HELD.items():
    z = np.load(a.dir / f"cfg_{key}" / "test_logits.npz"); T = float(z["T"])
    d = meta.merge(pd.DataFrame({"submission_id": z["submission_id"], "lora": 1 / (1 + np.exp(-z["logit"] / T)), "yl": z["y"]}), on="submission_id") \
            .merge(b2[b2.heldout == lang][["submission_id", "p_raw"]], on="submission_id")
    assert (d.yl.values == d.y_tle.values).all() and (d.language == lang).all(), "label/language mismatch"
    y, g = d.y_tle.values, d.problem_id.values
    rl, rb = metrics_with_ci(y, d.lora.values, g, prob=True, n_boot=300), metrics_with_ci(y, d.p_raw.values, g, n_boot=300)
    pt = roc_auc_score(y, d.lora) - roc_auc_score(y, d.p_raw)
    lo, hi = cluster_bootstrap(g, lambda i: roc_auc_score(y[i], d.lora.values[i]) - roc_auc_score(y[i], d.p_raw.values[i]), n_boot=300)
    verdict[lang] = lo > 0
    L.append(f"| {lang} | {len(d)} | {d.problem_id.nunique()} | {fmt(rl['auroc'])} | {fmt(rb['auroc'])} | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] | {rl['auprc'][0]:.3f} | {rl['ece'][0]:.3f} | "
             + (f"{indist[lang]:.3f}" if lang in indist else "n/a") + f" (T={T:.3f}) |")
allok = all(verdict.values())
L += ["", "## Claim by the pre-declared rule",
      "- Per language, LoRA transfers better than LightGBM only where the paired CI excludes 0: " + "; ".join(f"{k}: {'YES' if v else 'NOT shown'}" for k, v in verdict.items()) + ".",
      f"- Overall statement ('LoRA transfers to an unseen language better than LightGBM'): **{'supported for all three languages' if allok else 'NOT supported for all three languages'}**.",
      "- Calibration: T was fit on the training languages, so ECE on the held-out language reflects calibration under language shift."]
a.out.write_text("\n".join(L) + "\n"); print("\n".join(L))
