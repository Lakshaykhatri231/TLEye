"""Stage 1 report, following reports/stage1_predeclaration.md -> reports/metamorphic_lora.md
    python -I -m tleye.eval.metamorphic_lora [--dir reports/kaggle_interp] [--out reports/metamorphic_lora.md]
LightGBM is loaded here for the B2m comparison, so this process must never load the LoRA model (macOS OpenMP clash)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tleye.eval.metrics import cluster_bootstrap


def claims(n_lo, n_mean, placebo_abs, tl_mean, tl_hi, gate_n, gate_tl) -> dict:
    """The pre-declared rules as a pure function (tested on synthetic numbers)."""
    return {"C1_responds_to_N": bool(n_mean > 0 and n_lo > 0), "C2_beyond_noise": bool(n_lo > placebo_abs), "C3_responds_to_time_limit": bool(tl_mean < 0 and tl_hi < 0),
            "patch_gate_N": bool(gate_n >= 0.5), "patch_gate_time_limit": bool(abs(gate_tl) >= 0.5)}


def load_variant(d: Path, split: str, name: str):
    z = np.load(d / "meta" / f"{split}_{name}.npz")
    return pd.Series(z["logit"], index=z["submission_id"])


def main(d: Path, out: Path) -> None:
    meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle", "max_loop_depth"]).set_index("submission_id")
    L = ["# LoRA metamorphic checks (script-generated; follows reports/stage1_predeclaration.md)", ""]
    nz = np.load(d / "meta" / "noise.npz"); gap = np.abs(nz["a"] - nz["b"])
    L += [f"Noise floor (same rows rebatched, n={len(gap)}): median |change| {np.median(gap):.3f}, 99th percentile {np.percentile(gap, 99):.3f} logit.", ""]

    def row(split, name, want):
        base = load_variant(d, split, "base"); v = load_variant(d, split, name); idx = v.index
        x = pd.DataFrame({"delta": v.values - base.loc[idx].values}, index=idx).join(meta)
        x["bucket"] = x.max_loop_depth.clip(upper=3)
        g = x.problem_id.values; dl = x.delta.values
        lo, hi = cluster_bootstrap(g, lambda i: float(dl[i].mean()), n_boot=300) if len(x) else (float("nan"),) * 2
        res = {"n": len(x), "problems": x.problem_id.nunique(), "mean": float(dl.mean()) if len(x) else float("nan"), "lo": lo, "hi": hi, "abs": float(np.abs(dl).mean()) if len(x) else float("nan")}
        deep = x[x.bucket >= 2]
        if len(deep):
            gg, dd = deep.problem_id.values, deep.delta.values
            res["deep_mean"] = float(dd.mean()); res["deep_lo"], res["deep_hi"] = cluster_bootstrap(gg, lambda i: float(dd[i].mean()), n_boot=300)
        res["by_depth"] = {int(k): round(float(v_.delta.mean()), 3) for k, v_ in x.groupby("bucket")}
        if want:
            s = dl * want
            res["viol"] = {t: float((s < -t).mean()) for t in (0.0, 0.1, 0.5)}
        return res

    def table(split, spec):
        out_ = {}
        L.extend([f"## {split} split", "", "| change | n | problems | mean logit change [95% CI] | mean abs change | by loop depth 0 / 1 / 2 / 3+ | wrong-direction share at 0 / 0.1 / 0.5 logit |", "|---|---|---|---|---|---|---|"])
        for name, want in spec:
            if not (d / "meta" / f"{split}_{name}.npz").exists():
                continue
            r = row(split, name, want); out_[name] = r
            dep = " / ".join(str(r["by_depth"].get(k, "-")) for k in range(4))
            vi = " / ".join(f"{r['viol'][t]:.1%}" for t in (0.0, 0.1, 0.5)) if want else "-"
            L.append(f"| {name} | {r['n']} | {r['problems']} | {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['abs']:.3f} | {dep} | {vi} |")
        L.append("")
        return out_

    va = table("val", [("n_div100", -1), ("n_div10", -1), ("n_x10", 1), ("n_x100", 1), ("n_x1000", 1), ("tl_x2", -1), ("tl_div2", 1), ("placebo_reorder", 0), ("placebo_tl", 0)])
    te = table("test", [("n_x10", 1), ("tl_x2", -1), ("placebo_reorder", 0), ("placebo_tl", 0)])
    if {"n_x10", "tl_x2", "placebo_reorder"} <= set(te) and {"n_x1000", "tl_x2"} <= set(va):
        c = claims(te["n_x10"]["deep_lo"], te["n_x10"]["deep_mean"], te["placebo_reorder"]["abs"], te["tl_x2"]["mean"], te["tl_x2"]["hi"], va["n_x1000"].get("deep_mean", 0.0), va["tl_x2"]["mean"])
        L += ["## Claims by the pre-declared rules", "",
              f"- At depth >= 2, N x10 mean logit change on test: {te['n_x10']['deep_mean']:+.3f} [{te['n_x10']['deep_lo']:+.3f}, {te['n_x10']['deep_hi']:+.3f}]. Reorder placebo mean absolute change: {te['placebo_reorder']['abs']:.3f}.",
              f"- C1 responds to N: **{c['C1_responds_to_N']}**. C2 beyond noise: **{c['C2_beyond_noise']}**. C3 responds to the time limit: **{c['C3_responds_to_time_limit']}**.",
              f"- Stage 3 gates (val): N patching **{c['patch_gate_N']}** (N x1000 at depth >= 2: {va['n_x1000'].get('deep_mean', float('nan')):+.3f}), time-limit patching **{c['patch_gate_time_limit']}** (time limit x2: {va['tl_x2']['mean']:+.3f}).", ""]
        Path(str(out) + ".claims.json").write_text(json.dumps(c, indent=1))
    pf = d / "meta" / "prefix_test.npz"
    if pf.exists():
        z = np.load(pf); t = z["true"]; yy = meta.loc[z["submission_id"]].y_tle.values
        from sklearn.metrics import roc_auc_score
        L += ["## Prefix-edit pre-test (val, adapter P, descriptive only)", "", f"n={len(t)}; logit SD {t.std():.2f}.", "", "| edit | AUROC | mean change | mean abs change | flips at logit 0 |", "|---|---|---|---|---|"]
        L.append(f"| true analysis | {roc_auc_score(yy, t):.3f} | - | - | - |")
        for k in ("neutral", "depth_up", "depth_down", "donor"):
            e = z[k]; L.append(f"| {k} | {roc_auc_score(yy, e):.3f} | {(e - t).mean():+.3f} | {np.abs(e - t).mean():.3f} | {((e > 0) != (t > 0)).mean():.1%} |")
        L.append("")
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_interp")); ap.add_argument("--out", type=Path, default=Path("reports/metamorphic_lora.md"))
    a = ap.parse_args(); main(a.dir, a.out)
