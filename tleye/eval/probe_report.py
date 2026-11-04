"""Stage 2 report, following reports/stage2_predeclaration.md -> reports/probes.md
    python -I -m tleye.eval.probe_report [--val reports/kaggle_probes_val] [--test reports/kaggle_probes_test]"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from tleye.eval.metrics import cluster_bootstrap

_s = importlib.util.spec_from_file_location("probe_lib", "kaggle/lib/probe_lib.py"); pl = importlib.util.module_from_spec(_s); _s.loader.exec_module(pl)
FAMILY = ["depth_ge2", "recursive", "has_sort", "sort_in_loop", "listed_any", "loops_log1p", "log10_n", "est"]
KIND = dict.fromkeys(["depth_ge2", "recursive", "has_sort", "sort_in_loop", "listed_any", "lang_python", "tle"], "binary") | dict.fromkeys(["loops_log1p", "log10_n", "est", "log_tl"], "numeric")
PROBLEM_LEVEL = {"log10_n", "est", "log_tl"}


def metric(kind, y, p):
    return roc_auc_score(y, p) if kind == "binary" else r2_score(y, p)


def truth(ids):
    pack = pd.read_parquet("data/processed/kaggle_pack/tleye_data.parquet", columns=["submission_id", "problem_id", "language", "y_tle", "time_limit_ms"])
    d = pack.merge(pd.read_parquet("data/processed/kaggle_sidecar/sft_targets.parquet"), on="submission_id").set_index("submission_id").loc[ids]
    t = {"depth_ge2": (d.vis_loop_depth >= 2).astype(float).values, "recursive": d.vis_recursive.astype(float).values, "has_sort": (d.vis_sorts > 0).astype(float).values, "sort_in_loop": (d.vis_sort_in_loop > 0).astype(float).values,
         "listed_any": (d.vis_listed > 0).astype(float).values, "loops_log1p": np.log1p(d.vis_loops.values.astype(float)), "log10_n": d.log10_n.values.astype(float), "est": (d.vis_loop_depth.clip(lower=1) * d.log10_n).clip(upper=30).values.astype(float),
         "lang_python": (d.language == "Python").astype(float).values, "tle": d.y_tle.astype(float).values, "log_tl": np.log10(d.time_limit_ms.values / 1000.0)}
    return d, t


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def ci(g, fn, n=500):
    lo, hi = cluster_bootstrap(g, fn, n_boot=n)
    return fn(np.arange(len(g))), lo, hi


def boot_p(g, fn, n=500):
    """Two-sided bootstrap p for a difference."""
    u, inv = np.unique(g, return_inverse=True); rows = [np.where(inv == i)[0] for i in range(len(u))]; rng = np.random.default_rng(1); v = []
    for _ in range(n):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(u), len(u))])
        try:
            v.append(fn(idx))
        except ValueError:
            continue
    v = np.array(v); return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / n))


def main(val: Path, test: Path, out: Path) -> None:
    L = ["# Linear probes (script-generated; follows reports/stage2_predeclaration.md)", "", "Probes show what is **linearly decodable** from the model's internal state, not what it uses to decide.", ""]
    ch = json.loads((val / "val_choices.json").read_text())
    L += ["## Validation: chosen sites (committed before test was opened)", "", "| target | tuned site | tuned CV score | base site | base CV score |", "|---|---|---|---|---|"]
    for k, v in ch["choices"].items():
        L.append(f"| {k} | {v['tuned']['site']} | {v['tuned']['cv_score']:.3f} | {v['base']['site']} | {v['base']['cv_score']:.3f} |")
    L += ["", f"Pipeline checks: {json.dumps({k: round(v, 4) for k, v in ch['checks'].items()})}", ""]
    if (test / "test_preds.npz").exists():
        z = np.load(test / "test_preds.npz"); ids = z["submission_id"]; d, t = truth(ids); g = d.problem_id.values; res, ps = {}, []
        L += [f"## Test split ({len(ids)} rows, {d.problem_id.nunique()} problems; AUROC for binary targets, R2 for numeric; problem-clustered 95% CI)", "",
              "| target | tuned | base | metadata | bag of tokens | shuffled control | tuned minus metadata | tuned minus bag of tokens | tuned minus base | demeaned corr (tuned) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k in KIND:
            kind, y = KIND[k], t[k]; ok = np.isfinite(y); yy, gg = y[ok], g[ok]
            P = {s: z[f"{k}|{s}"][ok] for s in ("tuned", "base", "metadata", "bag_of_tokens", "tuned_shuffled")}
            one = lambda s: (lambda i: metric(kind, yy[i], P[s][i]))  # noqa: E731
            dif = lambda a, b: (lambda i: metric(kind, yy[i], P[a][i]) - metric(kind, yy[i], P[b][i]))  # noqa: E731
            r = {"tuned": ci(gg, one("tuned")), "base": ci(gg, one("base")), "meta": ci(gg, one("metadata")), "bow": ci(gg, one("bag_of_tokens")), "shuf": metric(kind, yy, P["tuned_shuffled"]),
                 "t_m": ci(gg, dif("tuned", "metadata")), "t_b": ci(gg, dif("tuned", "bag_of_tokens")), "t_base": ci(gg, dif("tuned", "base"))}
            if k not in PROBLEM_LEVEL:
                r["dm"] = ci(gg, lambda i, p=P["tuned"]: pl.demeaned_corr(yy[i], p[i], gg[i]), n=200)
            res[k] = r
            if k in FAMILY:
                ps.append(boot_p(gg, dif("tuned", "base")))
            L.append(f"| {k} | {fmt(r['tuned'])} | {fmt(r['base'])} | {r['meta'][0]:.3f} | {r['bow'][0]:.3f} | {r['shuf']:.3f} | {r['t_m'][0]:+.3f} [{r['t_m'][1]:+.3f}, {r['t_m'][2]:+.3f}] | {r['t_b'][0]:+.3f} [{r['t_b'][1]:+.3f}, {r['t_b'][2]:+.3f}] | {r['t_base'][0]:+.3f} [{r['t_base'][1]:+.3f}, {r['t_base'][2]:+.3f}] | " + (fmt(r["dm"]) if "dm" in r else "-") + " |")
        adj = pl.holm(ps)
        L += ["", "## Claims by the pre-declared rules", "", "| target | decodable beyond metadata | beyond bag of tokens | fine-tuning changed it (Holm p) |", "|---|---|---|---|"]
        for k, a in zip(FAMILY, adj):
            r = res[k]; beyond = r["t_m"][1] > 0 and ("dm" not in r or r["dm"][1] > 0)
            L.append(f"| {k} | {'yes' if beyond else 'no'} | {'yes' if r['t_b'][1] > 0 else 'no'} | {'yes' if a < 0.05 else 'no'} ({a:.3f}, {'+' if r['t_base'][0] > 0 else '-'}) |")
        L += ["", "Caveat: the base model was never trained to summarize at the final `:`, so a tuned gain at the last token may only mean information moved there.", ""]
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--val", type=Path, default=Path("reports/kaggle_probes_val")); ap.add_argument("--test", type=Path, default=Path("reports/kaggle_probes_test")); ap.add_argument("--out", type=Path, default=Path("reports/probes.md"))
    a = ap.parse_args(); main(a.val, a.test, a.out)
