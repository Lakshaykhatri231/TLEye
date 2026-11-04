"""Baselines B1 (loop-depth x constraint rule) and B2 (LightGBM on AST + constraint features).

Protocol: train on `train`, early-stop / calibrate / pick threshold on `val`, report on `test`.
Leave-one-language-out (LOLO): train+val on two languages, test on the held-out language's test rows.
NEVER used as features: cpu_time, ratio (label leakage). Prevalence is a sampling artifact (per-class cap).

    python -I -m tleye.models.baselines
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tleye.eval.metrics import best_f1_threshold, metrics_with_ci
from tleye.features.ast_features import FEATURE_NAMES

LANGS = ["C++", "Java", "Python"]
LEAK = {"cpu_time", "ratio", "status", "y_tle"}
CONSTRAINT_COLS = ["log10_primary", "log10_max_bound", "log10_size", "c_n_upper", "c_section", "log_tl"]


def load() -> pd.DataFrame:
    d = pd.read_parquet("data/processed/features_clean.parquet")
    d = d[~d["borderline"] & ~d["tle_odd"]].reset_index(drop=True)
    d["rt"] = np.where(d["runtime"].str.startswith("PyPy"), "PyPy", d["language"])
    d["is_pypy"] = (d["rt"] == "PyPy").astype(int)
    d["log_tl"] = np.log10(d["time_limit_ms"] / 1000.0)
    d["log_code"] = np.log10(d["code_size"].clip(lower=1))
    return d


def add_est(d: pd.DataFrame, fill: float) -> pd.DataFrame:
    n = d["log10_primary"].fillna(d["log10_max_bound"]).fillna(fill)
    depth = d["max_loop_depth"].clip(upper=4).clip(lower=1)
    d["est_log_ops"] = depth * n
    d["b1_score"] = d["est_log_ops"] - d["log_tl"]          # more ops / less time => more TLE-like
    return d


def feature_cols(lolo: bool, constraints: bool = True) -> list[str]:
    cols = FEATURE_NAMES + ["log_code"]
    if constraints:
        cols += CONSTRAINT_COLS + ["est_log_ops", "b1_score"]
    if not lolo:
        cols += ["lang_cpp", "lang_java", "lang_py", "is_pypy"]
    return cols


def fit_lgb(tr: pd.DataFrame, va: pd.DataFrame, cols: list[str], monotone: list[int] | None = None):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=0,
                  deterministic=True, force_row_wise=True, verbose=-1, num_threads=4)
    if monotone is not None:
        params["monotone_constraints"] = monotone
        params["monotone_constraints_method"] = "advanced"
    dtr, dva = lgb.Dataset(tr[cols], tr["y_tle"]), lgb.Dataset(va[cols], va["y_tle"])
    return lgb.train(params, dtr, 600, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def platt(raw_va: np.ndarray, y_va: np.ndarray) -> LogisticRegression:
    z = np.log(np.clip(raw_va, 1e-6, 1 - 1e-6) / (1 - np.clip(raw_va, 1e-6, 1 - 1e-6)))
    return LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y_va)


def apply_platt(m: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    return m.predict_proba(z.reshape(-1, 1))[:, 1]


def fmt(t) -> str:
    return "n/a" if t is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]" if isinstance(t, tuple) else str(t)


def row(name: str, scope: str, res: dict) -> dict:
    return {"model": name, "scope": scope, "n": res["n"], "problems": res["n_problems"],
            "prevalence": round(res["prevalence"], 3), "auroc": fmt(res.get("auroc")),
            "auprc": fmt(res.get("auprc")), "f1": fmt(res.get("f1")), "ece": fmt(res.get("ece"))}


def main() -> None:
    d = load()
    tr0 = d[d.split == "train"]
    fill = float(tr0["log10_primary"].median())
    d = add_est(d, fill)
    d["lang_cpp"], d["lang_java"], d["lang_py"] = (d.language == "C++").astype(int), (d.language == "Java").astype(int), (d.language == "Python").astype(int)
    tr, va, te = (d[d.split == s] for s in ["train", "val", "test"])
    rows, store = [], {}

    # ---- B1: rule, one threshold per runtime group tuned on val
    thr = {g: best_f1_threshold(va[va.rt == g]["y_tle"].values, va[va.rt == g]["b1_score"].values) for g in va.rt.unique()}
    te_thr = te["rt"].map(thr).values
    s = te["b1_score"].values
    for scope, m in [("ALL", np.ones(len(te), bool))] + [(l, (te.language == l).values) for l in LANGS]:
        res = metrics_with_ci(te.y_tle.values[m], s[m] - te_thr[m], te.problem_id.values[m], thr=0.0, n_boot=300)
        rows.append(row("B1 rule", scope, res))
    store["B1_thresholds"] = thr

    # ---- B2 variants
    for name, cons in [("B2 LightGBM (AST+constraints)", True), ("B2 LightGBM (AST only)", False)]:
        cols = feature_cols(False, cons)
        mdl = fit_lgb(tr, va, cols)
        raw_va, raw_te = mdl.predict(va[cols]), mdl.predict(te[cols])
        pl = platt(raw_va, va.y_tle.values)
        p_va, p_te = apply_platt(pl, raw_va), apply_platt(pl, raw_te)
        t = best_f1_threshold(va.y_tle.values, p_va)
        if cons:   # keep per-row test predictions for paired comparison with the LoRA model
            pd.DataFrame({"submission_id": te["submission_id"].values, "p_raw": raw_te, "p_platt": p_te}).to_parquet(
                "data/processed/b2_test_preds.parquet")
            pd.DataFrame({"submission_id": va["submission_id"].values, "p_raw": raw_va, "p_platt": p_va}).to_parquet(
                "data/processed/b2_val_preds.parquet")
        for scope, m in [("ALL", np.ones(len(te), bool))] + [(l, (te.language == l).values) for l in LANGS]:
            res = metrics_with_ci(te.y_tle.values[m], p_te[m], te.problem_id.values[m], thr=t, prob=True, n_boot=300)
            rows.append(row(name + " +Platt", scope, res))
        if cons:
            res = metrics_with_ci(te.y_tle.values, raw_te, te.problem_id.values, thr=t, prob=True, n_boot=300)
            rows.append(row(name + " raw (uncalibrated)", "ALL", res))
            imp = pd.Series(mdl.feature_importance("gain"), index=cols).sort_values(ascending=False)
            store["B2_top_features_gain"] = (imp / imp.sum()).round(4).head(12).to_dict()
            store["B2_best_iter"] = int(mdl.best_iteration)
            Path("models/b2").mkdir(parents=True, exist_ok=True)           # persisted for the CLI / tests
            mdl.save_model("models/b2/model.txt")
            Path("models/b2/meta.json").write_text(json.dumps({
                "cols": cols, "fill_log10_primary": fill, "best_iter": int(mdl.best_iteration),
                "thr_f1_raw_val": best_f1_threshold(va.y_tle.values, raw_va), "train_prevalence": float(tr.y_tle.mean()),
                "n_train": len(tr), "note": "raw LightGBM probabilities (better calibrated than Platt on this data, see reports/gate2.md)"}, indent=1))

    # ---- LOLO for B1 (single threshold from other langs' val) and B2
    lolo = []
    lolo_preds: list[pd.DataFrame] = []
    for held in LANGS:
        others = [l for l in LANGS if l != held]
        trL, vaL, teL = tr[tr.language.isin(others)], va[va.language.isin(others)], te[te.language == held]
        t1 = best_f1_threshold(vaL.y_tle.values, vaL.b1_score.values)
        res = metrics_with_ci(teL.y_tle.values, teL.b1_score.values - t1, teL.problem_id.values, thr=0.0, n_boot=300)
        lolo.append(row("B1 rule", f"held-out {held}", res))
        cols = feature_cols(True, True)
        mdl = fit_lgb(trL, vaL, cols)
        pl = platt(mdl.predict(vaL[cols]), vaL.y_tle.values)
        raw_L = mdl.predict(teL[cols])
        p = apply_platt(pl, raw_L)
        lolo_preds.append(pd.DataFrame({"submission_id": teL["submission_id"].values, "heldout": held, "p_raw": raw_L, "p_platt": p}))
        tt = best_f1_threshold(vaL.y_tle.values, apply_platt(pl, mdl.predict(vaL[cols])))
        res = metrics_with_ci(teL.y_tle.values, p, teL.problem_id.values, thr=tt, prob=True, n_boot=300)
        lolo.append(row("B2 LightGBM LOLO +Platt", f"held-out {held}", res))

    pd.concat(lolo_preds).to_parquet("data/processed/b2_lolo_test_preds.parquet")
    out = Path("reports"); out.mkdir(exist_ok=True)
    md = ["# Baselines (B1 rule, B2 LightGBM) - test split, problem-disjoint", "",
          "Point estimate [95% CI]; CIs from a problem-clustered bootstrap (300 resamples).",
          "Prevalence = share of TLE in the CAPPED sample (40 per class per problem/language): "
          "AUPRC, F1 and ECE depend on it and must not be read as real-world rates; AUROC does not.", "",
          "## Main (train on all 3 languages)", pd.DataFrame(rows).to_markdown(index=False), "",
          "## Leave-one-language-out (train+val on 2 languages, test on the 3rd; language one-hots removed)",
          pd.DataFrame(lolo).to_markdown(index=False), "",
          "## B2 details", "```", json.dumps({k: v for k, v in store.items() if k != "B1_thresholds"}, indent=1), "```",
          f"B1 thresholds (val-tuned, per runtime group): {json.dumps({k: round(v, 3) for k, v in thr.items()})}"]
    (out / "baselines.md").write_text("\n".join(md))
    Path("data/processed/baselines.json").write_text(json.dumps({"main": rows, "lolo": lolo, **store}, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
