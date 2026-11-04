"""Metamorphic checks for the verdict model.

 (1) raise every input-size bound x10 (log10 bounds +1) -> P(TLE) must NOT decrease
 (2) double the time limit                            -> P(TLE) must NOT increase
Reports the violation rate (+ problem-clustered 95% CI) for plain B2 and for B2m (LightGBM monotone constraints on the
size / estimate / time-limit features), plus B2m's test AUROC cost. Test split only; B2m trained on train, early-stopped on val.
    python -I -m tleye.eval.metamorphic
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import best_f1_threshold, cluster_bootstrap
from tleye.models.baselines import add_est, feature_cols, fit_lgb, load

UP = {"log10_primary", "log10_max_bound", "log10_size", "est_log_ops", "b1_score"}   # P(TLE) should not decrease when these rise
DOWN = {"log_tl"}                                                                      # ... and should not increase when this rises
TOL = 1e-9


def perturb_n(d: pd.DataFrame, fill: float) -> pd.DataFrame:
    d = d.copy()
    for c in ("log10_primary", "log10_max_bound", "log10_size"):
        d[c] = d[c] + 1.0                  # NaN stays NaN
    return add_est(d, fill + 0.0)


def perturb_tl(d: pd.DataFrame, fill: float) -> pd.DataFrame:
    d = d.copy(); d["log_tl"] = d["log_tl"] + math.log10(2.0)
    return add_est(d, fill)


def violation(model_predict, d, fill, kind):
    p0 = model_predict(d)
    p1 = model_predict(perturb_n(d, fill) if kind == "n_up" else perturb_tl(d, fill))
    v = (p1 < p0 - TOL) if kind == "n_up" else (p1 > p0 + TOL)
    return v.astype(float), np.abs(p1 - p0)


def main() -> None:
    d = load(); fill = float(d[d.split == "train"]["log10_primary"].median()); d = add_est(d, fill)
    d["lang_cpp"], d["lang_java"], d["lang_py"] = (d.language == "C++").astype(int), (d.language == "Java").astype(int), (d.language == "Python").astype(int)
    tr, va, te = (d[d.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    cols = feature_cols(False, True)
    mono = [1 if c in UP else (-1 if c in DOWN else 0) for c in cols]
    plain = fit_lgb(tr, va, cols)
    mono_m = fit_lgb(tr, va, cols, monotone=mono)
    rows, g = [], te.problem_id.values
    for name, m in [("B2 (plain)", plain), ("B2m (monotone constraints)", mono_m)]:
        pred = lambda x, m=m: m.predict(x[cols])      # noqa: E731
        auc = roc_auc_score(te.y_tle, pred(te))
        row = {"model": name, "test_auroc": round(float(auc), 4), "best_iter": int(m.best_iteration)}
        for kind, label in [("n_up", "N x10 lowers P(TLE)"), ("tl_up", "time limit x2 raises P(TLE)")]:
            v, mag = violation(pred, te, fill, kind)
            lo, hi = cluster_bootstrap(g, lambda i, v=v: float(v[i].mean()), n_boot=300)
            row[label] = f"{v.mean():.2%} [{lo:.2%}, {hi:.2%}]"
            row[label + " (mean abs dP)"] = round(float(mag.mean()), 4)
        rows.append(row)
    pa, pb = plain.predict(te[cols]), mono_m.predict(te[cols])
    diff = roc_auc_score(te.y_tle, pb) - roc_auc_score(te.y_tle, pa)
    def f(i): return roc_auc_score(te.y_tle.values[i], pb[i]) - roc_auc_score(te.y_tle.values[i], pa[i])
    lo, hi = cluster_bootstrap(g, f, n_boot=200)
    out = ["# Metamorphic checks (test split, script-generated)", "",
           f"n = {len(te)} rows, {te.problem_id.nunique()} problems. Violation = probability moves the wrong way by > {TOL}.", "",
           pd.DataFrame(rows).to_markdown(index=False), "",
           f"Paired AUROC change from adding monotone constraints (B2m - B2): {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]", ""]
    Path("reports/metamorphic.md").write_text("\n".join(out)); print("\n".join(out))
    mono_m.save_model("models/b2/model_monotone.txt")
    pv = mono_m.predict(va[cols])
    json.dump({"cols": cols, "monotone": dict(zip(cols, mono)), "fill_log10_primary": fill, "best_iter": int(mono_m.best_iteration),
               "thr_f1_raw_val": best_f1_threshold(va.y_tle.values, pv), "train_prevalence": float(tr.y_tle.mean()), "n_train": len(tr),
               "note": "B2m: LightGBM with monotone constraints (size/estimate up -> P(TLE) up; time limit up -> down); raw probabilities"},
              open("models/b2/meta_monotone.json", "w"), indent=1)
    pd.DataFrame({"submission_id": te["submission_id"].values, "p_raw": mono_m.predict(te[cols])}).to_parquet("data/processed/b2m_test_preds.parquet")
    pd.DataFrame({"submission_id": va["submission_id"].values, "p_raw": pv}).to_parquet("data/processed/b2m_val_preds.parquet")


if __name__ == "__main__":
    main()
