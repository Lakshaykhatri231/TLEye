"""Stage 4 explanation gate report; follows reports/stage4_predeclaration.md. Validation only. -> reports/sft_gate.md
    python -I -m tleye.eval.sft_report [--dir reports/kaggle_sft_gate] [--out reports/sft_gate.md]
Field scores compare the model's generated value with the tree-sitter value on the visible token window."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import cluster_bootstrap
from tleye.features.visible_window import FIELD_RE, parse_target

MIN_AUROC = 0.80
DEPTH_CAP = 4        # depths >= 4 share one class, so a handful of rare rows cannot dominate balanced accuracy


def depth_class(values) -> np.ndarray:
    """Generated or true depth -> class 0..DEPTH_CAP; anything unparsable -> -1, which is always wrong."""
    out = []
    for v in values:
        s = str(v).strip()
        out.append(min(int(s), DEPTH_CAP) if s.isdecimal() else -1)
    return np.array(out)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def majority_by_language(train: pd.DataFrame, col: str) -> dict:
    return train.groupby("language")[col].agg(lambda s: s.value_counts().index[0]).to_dict()


def gate_decision(results: dict) -> dict:
    """results: {cfg: {"best_auroc": float, "depth_diff_lo": float}}. The rule in reports/stage4_predeclaration.md."""
    passed = {k: bool(r["best_auroc"] >= MIN_AUROC and r["depth_diff_lo"] > 0) for k, r in results.items()}
    if passed.get("dual"):
        return {"passed": passed, "stage5": "dual"}
    if passed.get("v1only"):
        return {"passed": passed, "stage5": "v1only"}
    return {"passed": passed, "stage5": None}


def load_shards(folder: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(folder.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        parts.append(pd.DataFrame({k: z[k] for k in z.files}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def values(text: str) -> dict:
    """Field name -> generated string. Generation forces the template, so a split on the field names is reliable even for odd values."""
    kv = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def ci_pair(g, fn):
    lo, hi = cluster_bootstrap(g, fn, n_boot=300)
    return lo, hi


def score_cfg(name: str, run: Path, meta: pd.DataFrame, side: pd.DataFrame, maj: dict, L: list) -> dict:
    gen, rest = load_shards(run / "val_gen"), load_shards(run / "val_v0")
    both = pd.concat([gen[["submission_id", "v0"]], rest[["submission_id", "v0"]]], ignore_index=True).merge(meta, on="submission_id")
    gen = gen.merge(meta, on="submission_id").merge(side, on="submission_id")
    assert len(gen) and gen.submission_id.is_unique and both.submission_id.is_unique
    fm = lambda v, c: f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    y, g = gen.y_tle.values, gen.problem_id.values
    L += [f"## Config `{name}`", "", f"Rows: {len(both)} for v0, {len(gen)} for generation; {both.problem_id.nunique()} and {gen.problem_id.nunique()} problems. Problem-clustered bootstrap (300).", "",
          "| verdict readout | rows | AUROC [95% CI] |", "|---|---|---|"]
    aurocs = {}

    def add(label, yy, ss, gg):
        v = roc_auc_score(yy, ss)
        ci = cluster_bootstrap(gg, lambda i: roc_auc_score(yy[i], ss[i]), n_boot=300)
        L.append(f"| {label} | {len(yy)} | {fm(v, ci)} |")
        return v
    if name == "dual":                  # v1only never trains head0: its v0 is a constant 0 and is not a readout
        aurocs["v0"] = add("v0, at the classifier's read point", both.y_tle.values, both.v0.values, both.problem_id.values)
        add("v0, generation rows only", y, gen.v0.values, g)
    aurocs["v1_gen"] = add("v1, after the model's own analysis", y, gen.v1_gen.values, g)
    add("v1, after the TRUE analysis (not available in real use)", y, gen.v1_true.values, g)
    pairs = [("v1 true minus v1 generated", "v1_true", "v1_gen")] + ([("v1 generated minus v0", "v1_gen", "v0")] if name == "dual" else [])
    L += ["", "| paired AUROC difference, same rows | difference [95% CI] |", "|---|---|"]
    for lab, a_, b_ in pairs:
        sa, sb = gen[a_].values, gen[b_].values
        d_ = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        lo, hi = ci_pair(g, lambda i: roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        L.append(f"| {lab} | {d_:+.3f} [{lo:+.3f}, {hi:+.3f}] |")

    pred = [values(t) for t in gen.text]
    true = [values(t) for t in gen.target]
    ok_parse = np.mean([parse_target(t) is not None for t in gen.text])
    trunc = gen.truncated.values.astype(bool)
    ty = depth_class([t["loop_depth"] for t in true]); py = depth_class([p.get("loop_depth", "") for p in pred])
    base = gen.language.map(maj["loop_depth_class"]).values.astype(int)
    ba = balanced_accuracy(ty, py); bb = balanced_accuracy(ty, base)
    dlo, dhi = ci_pair(g, lambda i: balanced_accuracy(ty[i], py[i]) - balanced_accuracy(ty[i], base[i]))
    L += ["", f"Generated analyses that parse as well-formed: {ok_parse:.1%}.", "",
          f"**Loop depth (the gate's field):** balanced accuracy {ba:.3f} vs per-language majority guess {bb:.3f}, paired difference {ba - bb:+.3f} [{dlo:+.3f}, {dhi:+.3f}]. Depths >= {DEPTH_CAP} are one class.",
          f"Plain accuracy {np.mean(ty == py):.3f}; on truncated rows ({trunc.sum()}) {np.mean(ty[trunc] == py[trunc]) if trunc.any() else float('nan'):.3f}, on untruncated rows ({(~trunc).sum()}) {np.mean(ty[~trunc] == py[~trunc]) if (~trunc).any() else float('nan'):.3f}.", "",
          "| field | exact match, generated | exact match, per-language majority guess |", "|---|---|---|"]
    for f in ("loops", "recursion", "sorts", "sort_in_loop", "listed_calls_in_loop", "n_var", "log10_n", "est_log10_ops"):
        t_ = np.array([t[f] for t in true]); p_ = np.array([p.get(f, "") for p in pred])
        b_ = gen.language.map(maj[f]).values.astype(str)
        L.append(f"| {f} | {np.mean(t_ == p_):.3f} | {np.mean(t_ == b_):.3f} |")
    errs = {}
    for f in ("log10_n", "est_log10_ops"):
        d_ = [abs(float(p[f]) - float(t[f])) for p, t in zip(pred, true) if t[f] != "unknown" and _is_num(p.get(f, ""))]
        errs[f] = (float(np.mean(d_)), len(d_)) if d_ else (float("nan"), 0)
    arith = [abs(max(int(p["loop_depth"]), 1) * float(p["log10_n"]) - float(p["est_log10_ops"])) <= 0.15 for p in pred
             if p.get("loop_depth", "").isdecimal() and _is_num(p.get("log10_n", "")) and _is_num(p.get("est_log10_ops", ""))]
    L += ["", "Mean absolute error where N is known and a number was generated: " + ", ".join(f"{k} {v[0]:.2f} (n={v[1]})" for k, v in errs.items()) + ".",
          f"Self-consistency, stated estimate equals depth x log10 N within 0.15: {np.mean(arith) if arith else float('nan'):.3f} (n={len(arith)}).", ""]

    log = json.loads((run / "train_log.json").read_text())
    recs = [r for r in log if "verdict_bce" in r]; gn = [r["grad_norms"] for r in log if "grad_norms" in r]
    sc = json.loads((run / "selfcheck.json").read_text()); rj = json.loads((run / "run.json").read_text())
    if recs:
        pts = [recs[int(q * (len(recs) - 1))] for q in (0, .25, .5, .75, 1)]
        L += ["Training curve (step: verdict BCE / value cross-entropy): " + ", ".join(f"{r['step']}: {r['verdict_bce']:.3f} / {r['ce']:.3f}" for r in pts) + ".",]
    if gn:
        L.append("Mean gradient norm per loss on the LoRA weights: " + ", ".join(f"{k} {np.mean([x[k] for x in gn]):.3f}" for k in gn[0]) + ".")
    L += [f"Self-checks: reload max change {sc['reload_max_abs_v0']:.4f}; batched equals single generation {sc['gen_batched_equals_single']} (v1 max change {sc['gen_v1_max_abs']:.4f}); forced vs stepwise v1 max change {sc['forced_vs_stepwise_max_abs']:.4f}; steps {sc['steps']}/{sc['total_steps']}.",
          f"Run: {rj['n_train']} training rows, {rj['minutes']} minutes, peak {rj['peak_mem_GB']} GB, script sha {rj['script_sha256'][:12]}.", ""]
    return {"best_auroc": max(aurocs.values()), "depth_diff_lo": dlo, "dry": rj["dry"]}


def _is_num(s: str) -> bool:
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False


def main(root: Path, out: Path) -> None:
    meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle", "split"])
    side = pd.read_parquet("data/processed/kaggle_sidecar/sft_targets.parquet", columns=["submission_id", "target", "truncated"])
    assert side.target.map(lambda t: FIELD_RE.search(t) is not None).all()
    tr = meta[meta.split == "train"].merge(side, on="submission_id")
    kv = pd.DataFrame([values(t) for t in tr.target]); kv["language"] = tr.language.values
    kv["loop_depth_class"] = depth_class(kv.loop_depth)
    maj = {c: majority_by_language(kv, c) for c in kv.columns if c != "language"}
    L = ["# Stage 4 explanation gate (script-generated; follows reports/stage4_predeclaration.md)", "", "VALIDATION rows only. The test split was not touched.", ""]
    res = {n: score_cfg(n, root / f"cfg_{n}", meta, side, maj, L) for n in ("dual", "v1only")}
    dec = gate_decision({k: v for k, v in res.items()})
    if any(r["dry"] for r in res.values()):
        L.insert(2, "**DRY RUN on a tiny sample: this checks the pipeline. Nothing here is a result.**\n")
    L += ["## Decision by the pre-declared rule", "",
          f"Pass requires best verdict AUROC >= {MIN_AUROC} and a loop-depth paired CI above 0.", ""]
    for k, r in res.items():
        L.append(f"- `{k}`: best verdict AUROC {r['best_auroc']:.3f}, loop-depth difference lower bound {r['depth_diff_lo']:+.3f}: **{'pass' if dec['passed'][k] else 'fail'}**.")
    L += ["", f"**Stage 5 config: `{dec['stage5']}`.**" if dec["stage5"] else "**Neither config passes: stop and report; no Stage 5.**", "",
          "Context, not a pass mark: sweep A 0.826 (lr 5e-5, no prefix) and C 0.836 (lr 1e-4, with prefix) on the same 8,000 val rows, classifiers trained on the same 12,000 rows."]
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_sft_gate")); ap.add_argument("--out", type=Path, default=Path("reports/sft_gate.md"))
    a = ap.parse_args(); main(a.dir, a.out)
