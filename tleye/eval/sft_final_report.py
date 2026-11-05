"""Stage 5 report: the explanation model on the test split; follows reports/stage5_predeclaration.md. -> reports/sft_test.md
    python -I -m tleye.eval.sft_final_report [--dir reports/kaggle_sft_full] [--out reports/sft_test.md]
Written before any real test-split explanation output existed; exercised on dry-run outputs only."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from tleye.eval.metrics import best_f1_threshold, cluster_bootstrap
from tleye.eval.sft_report import balanced_accuracy, depth_class, load_shards, majority_by_language, values

NO_LOSS = -0.02
RELEVANT_SHIFT, RELEVANT_FLIP, DECORATIVE_SHIFT = 0.5, 0.05, 0.2
RANK = {"decorative": 0, "weak": 1, "decision-relevant": 2}
SEEDS = {"s0": "reports/kaggle_confirm/cfgN", "s1": "reports/kaggle_seeds/cfg_s1"}
N_BOOT = 300
PILOTS = ["cert_pilot", "cert_pilot_confirm", "cert_pilot3", "cert_pilot4", "cert_pilot5", "cert_pilot6"]


def label(mean_abs: float, flip: float, direction_lo: float | None = None) -> str:
    """Pre-declared label for one counterfactual. direction_lo: lower CI bound of the mean shift, only for edits with a declared direction."""
    if mean_abs < DECORATIVE_SHIFT:
        return "decorative"
    if mean_abs >= RELEVANT_SHIFT and flip >= RELEVANT_FLIP and (direction_lo is None or direction_lo > 0):
        return "decision-relevant"
    return "weak"


def lower_label(labels: list[str]) -> str:
    return min(labels, key=RANK.__getitem__)


def naming(donor: str, neutral: str) -> str:
    both = donor == neutral == "decision-relevant"
    return "the reason for the verdict" if both else "a checkable code summary"


def verdict_claims(lows: list[float]) -> dict:
    return {"no_meaningful_loss": all(x > NO_LOSS for x in lows), "beats_classifier": all(x > 0 for x in lows)}


def is_num(s: str) -> bool:
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False


def regex_baseline(code: str, constraints: str) -> dict:
    """The pre-declared regex baselines. Text matching only; dataset code is never executed."""
    out = {"loops": str(len(re.findall(r"\b(?:for|while)\b", code))), "sorts": str(len(re.findall(r"sort", code))), "n_var": "unknown", "log10_n": "unknown"}
    first = constraints.split(", ")[0] if constraints != "none" else ""
    if "<=" in first:
        k, v = first.split("<=", 1)
        out["n_var"] = k
        if is_num(v) and float(v) > 0:
            out["log10_n"] = f"{math.log10(float(v)):.1f}"
    return out


def fm(v, c, sign=False):
    f = "+.3f" if sign else ".3f"
    return f"{v:{f}} [{c[0]:{f}}, {c[1]:{f}}]"


def score_fields(gen: pd.DataFrame, maj: dict, L: list) -> dict:
    """Per-field accuracy against each baseline. Returns {field: {baseline: paired lower bound}}."""
    g = gen.problem_id.values
    pred = pd.DataFrame([values(t) for t in gen.text]).reindex(columns=["loop_depth", "loops", "recursion", "sorts", "sort_in_loop", "listed_calls_in_loop", "n_var", "log10_n", "est_log10_ops"]).fillna("")
    true = pd.DataFrame([values(t) for t in gen.target])
    rx = pd.DataFrame([regex_baseline(c, k) for c, k in zip(gen.code, gen.constraints)])
    lang = gen.language
    guess = {f: lang.map(maj[f]).values for f in maj}
    trunc = gen.truncated.values.astype(bool)
    claims = {}
    L += ["| field | metric | generated | majority guess | regex | minus majority [95% CI] | minus regex [95% CI] | generated: truncated / untruncated |", "|---|---|---|---|---|---|---|---|"]

    def row(name, metric, mod, base_m, base_r, fn):
        """fn(pred_array, idx) -> score, higher is better for the paired test."""
        v = fn(mod, np.arange(len(gen)))
        cells, lows = [], {}
        for key, b in (("majority", base_m), ("regex", base_r)):
            if b is None:
                cells += ["-", "-"]
                continue
            lo, hi = cluster_bootstrap(g, lambda i: fn(mod, i) - fn(b, i), n_boot=N_BOOT)
            cells += [f"{fn(b, np.arange(len(gen))):.3f}", f"{v - fn(b, np.arange(len(gen))):+.3f} [{lo:+.3f}, {hi:+.3f}]"]
            lows[key] = lo
        tr = [fn(mod, np.where(m)[0]) if m.any() else float("nan") for m in (trunc, ~trunc)]
        L.append(f"| {name} | {metric} | {v:.3f} | {cells[0]} | {cells[2]} | {cells[1]} | {cells[3]} | {tr[0]:.3f} / {tr[1]:.3f} |")
        claims[name] = lows

    t_d = depth_class(true.loop_depth); row("loop_depth", "balanced accuracy", depth_class(pred.loop_depth), guess["loop_depth_class"].astype(int), None, lambda p, i: balanced_accuracy(t_d[i], p[i]))
    t_r = (true.recursion == "yes").values; row("recursion", "balanced accuracy", (pred.recursion == "yes").values, (guess["recursion"] == "yes"), None, lambda p, i: balanced_accuracy(t_r[i], p[i]))

    def flag(s):
        return np.array([x.isdecimal() and int(x) > 0 for x in s])
    t_s = flag(true.sort_in_loop); row("sort_in_loop (any)", "balanced accuracy", flag(pred.sort_in_loop), flag(guess["sort_in_loop"]), None, lambda p, i: balanced_accuracy(t_s[i], p[i]))
    for f in ("loops", "sorts", "listed_calls_in_loop", "n_var"):
        t_ = true[f].values
        row(f, "exact match", pred[f].values, guess[f], rx[f].values if f in rx else None, lambda p, i, t_=t_: float(np.mean(t_[i] == p[i])))
    for f, base_col in (("log10_n", "log10_n"), ("est_log10_ops", None)):
        known = (true[f] != "unknown").values
        t_ = np.array([float(x) if k else np.nan for x, k in zip(true[f], known)])
        def num(s):
            return np.array([float(x) if is_num(x) else np.nan for x in s])
        modal = lang.map(maj[f + "_known"]).values.astype(float)

        def neg_mae(p, i, t_=t_):
            e = np.abs(p[i] - t_[i])
            return -float(np.nanmean(e)) if np.isfinite(e).any() else float("nan")
        row(f, "minus mean abs error", num(pred[f]), modal, num(rx[base_col]) if base_col else None, neg_mae)
    return claims


def load_verdicts(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_shards(run / "test_v0"), load_shards(run / "test_gen"), load_shards(run / "test_gen_cf"), load_shards(run / "test_sensitivity")


def faithfulness(cf: pd.DataFrame, thr: float, L: list) -> dict:
    g = cf.problem_id.values; base = cf.v1_gen.values; y = cf.y_tle.values; kv = [values(t) for t in cf.text]
    can_depth = np.array([k.get("loop_depth", "").isdecimal() and k.get("sorts", "").isdecimal() for k in kv])
    can_sort = can_depth & np.array([k.get("sort_in_loop") != "1" or k.get("sorts") == "0" for k in kv])
    out = {}
    L += [f"Flip threshold (val best-F1 of v1): {thr:.3f}. Rows: {len(cf)}, {cf.problem_id.nunique()} problems.", "",
          "| replacement | rows | excluded | mean shift [95% CI] | mean absolute shift | flip rate | label |", "|---|---|---|---|---|---|---|"]
    for name, col, mask, directed in (("placebo: same text forced again", "v1_gen_forced", np.ones(len(cf), bool), False),
                                      ("neutral", "v1_neutral", np.ones(len(cf), bool), False), ("donor", "v1_donor", np.ones(len(cf), bool), False),
                                      ("depth_up", "v1_depth_up", can_depth, True), ("sort_on", "v1_sort_on", can_sort, True)):
        d = (cf[col].values - base)
        idx = np.where(mask)[0]
        lo, hi = cluster_bootstrap(g[idx], lambda i: float(d[idx][i].mean()), n_boot=N_BOOT)
        ma = float(np.abs(d[idx]).mean()); flip = float(np.mean((cf[col].values[idx] >= thr) != (base[idx] >= thr)))
        lab = label(ma, flip, lo if directed else None)
        out[name.split(":")[0]] = lab
        L.append(f"| {name} | {len(idx)} | {len(cf) - len(idx)} | {fm(float(d[idx].mean()), (lo, hi), sign=True)} | {ma:.3f} | {flip:.1%} | {'noise floor' if name.startswith('placebo') else lab} |")
    a_gen, a_neu = roc_auc_score(y, base), roc_auc_score(y, cf.v1_neutral.values)
    lo, hi = cluster_bootstrap(g, lambda i: roc_auc_score(y[i], base[i]) - roc_auc_score(y[i], cf.v1_neutral.values[i]), n_boot=N_BOOT)
    L += ["", f"Paired AUROC drop, generated minus neutral: {fm(a_gen - a_neu, (lo, hi), sign=True)} (generated {a_gen:.3f}, neutral {a_neu:.3f}).", "",
          "Donor text was not saved, so a donor that happens to equal the generated text is not excluded; such rows pull the donor shift toward 0.", ""]
    X = np.array([[float(k[f]) if is_num(k.get(f, "")) else 0.0 for f in ("loop_depth", "loops", "sorts", "sort_in_loop", "listed_calls_in_loop", "log10_n", "est_log10_ops")]
                  + [float(k.get("recursion") == "yes"), float(is_num(k.get("log10_n", "")))] for k in kv])
    n_split = min(5, cf.problem_id.nunique())
    if n_split >= 2:
        pr = np.zeros(len(cf))
        for tr_, te_ in GroupKFold(n_split).split(X, base, g):
            pr[te_] = Ridge(alpha=1.0).fit(X[tr_], base[tr_]).predict(X[te_])
        L += [f"Sufficiency (descriptive): cross-validated linear model on the generated fields reproduces the v1 logit with R2 {r2_score(base, pr):.3f}.", ""]
    return out


def sensitivity(cf: pd.DataFrame, sens: pd.DataFrame, L: list) -> None:
    m = cf[["submission_id", "text", "v1_gen"]].merge(sens[["submission_id", "text", "v1_gen"]], on="submission_id", suffixes=("", "_x"))
    a, b = [values(t) for t in m.text], [values(t) for t in m.text_x]
    same = [x.get("loop_depth") == y.get("loop_depth") for x, y in zip(a, b)]
    ok_e, ok_n = [], []
    for x, y in zip(a, b):
        if all(is_num(z) for z in (x.get("est_log10_ops", ""), y.get("est_log10_ops", ""), x.get("loop_depth", ""))):
            ok_e.append(abs(float(y["est_log10_ops"]) - float(x["est_log10_ops"]) - max(float(x["loop_depth"]), 1) * 3) <= 0.15)
        if is_num(x.get("log10_n", "")) and is_num(y.get("log10_n", "")):
            ok_n.append(abs(float(y["log10_n"]) - float(x["log10_n"]) - 3) <= 0.15)
    sh = lambda v: f"{np.mean(v):.3f} (n={len(v)})" if v else "no rows"  # noqa: E731
    L += [f"N x1000 in the constraints line, {len(m)} rows. Depth unchanged: {sh(same)}. log10 N moved by 3: {sh(ok_n)}. Estimate moved by max(depth, 1) x 3: {sh(ok_e)}. Mean v1 change {float((m.v1_gen_x - m.v1_gen).mean()):+.3f}.", ""]


def certificates(gen: pd.DataFrame, thr: float, L: list) -> None:
    rows = []
    for f in PILOTS:
        p = Path("data/processed") / f"{f}.jsonl"
        if p.exists():
            rows += [{k: r[k] for k in ("submission_id", "status")} for r in map(json.loads, p.read_text().splitlines())]
    pil = pd.DataFrame(rows).drop_duplicates("submission_id", keep="last")
    d = gen.merge(pil[pil.status.isin(["certified", "not_certified", "inconclusive"])], on="submission_id")
    L += ["Descriptive only; differences between problems are confounded.", "", "| outcome | true label | rows | problems | mean v1 logit | flagged at threshold | mean generated depth |", "|---|---|---|---|---|---|---|"]
    for (st, y), s in d.groupby(["status", "y_tle"]):
        dep = [float(values(t).get("loop_depth")) for t in s.text if is_num(values(t).get("loop_depth", ""))]
        L.append(f"| {st} | {'TLE' if y else 'AC'} | {len(s)} | {s.problem_id.nunique()} | {s.v1_gen.mean():+.2f} | {(s.v1_gen >= thr).mean():.1%} | {np.mean(dep) if dep else float('nan'):.2f} |")
    L.append("")


def majorities(meta: pd.DataFrame, side: pd.DataFrame) -> dict:
    tr = meta[meta.split == "train"].merge(side[["submission_id", "target"]], on="submission_id")
    kv = pd.DataFrame([values(t) for t in tr.target]); kv["language"] = tr.language.values
    kv["loop_depth_class"] = depth_class(kv.loop_depth)
    maj = {c: majority_by_language(kv, c) for c in kv.columns if c != "language"}
    for f in ("log10_n", "est_log10_ops"):
        maj[f + "_known"] = majority_by_language(kv[kv[f] != "unknown"], f)
    return maj


def main(root: Path, out: Path) -> None:
    meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle", "split"])
    side = pd.read_parquet("data/processed/kaggle_sidecar/sft_targets.parquet", columns=["submission_id", "target", "truncated", "n_var"])
    pack = pd.read_parquet("data/processed/kaggle_pack/tleye_data.parquet", columns=["submission_id", "code", "constraints"], filters=[("split", "==", "test")])
    maj = majorities(meta, side)
    L, lows_v0, lows_v1, field_claims, faith, dry = [], [], [], {}, {}, False
    L_sec = {k: [] for k in ("verdict", "fields", "self", "faith", "sens", "cert", "run")}
    for s, comp in SEEDS.items():
        run = root / f"cfg_{s}"
        rj, sc = json.loads((run / "run.json").read_text()), json.loads((run / "selfcheck.json").read_text()); dry = dry or rj["dry"]
        v0, g1, g2, sens = load_verdicts(run)
        z = np.load(Path(comp) / "test_logits.npz"); clf = pd.DataFrame({"submission_id": z["submission_id"].astype(str), "clf": z["logit"]})
        v0 = v0.merge(meta, on="submission_id").merge(clf, on="submission_id")
        gen = pd.concat([g1, g2], ignore_index=True).merge(meta, on="submission_id").merge(clf, on="submission_id").merge(side, on="submission_id").merge(pack, on="submission_id")
        cf = g2.merge(meta, on="submission_id")
        assert v0.submission_id.is_unique and gen.submission_id.is_unique
        if not rj["dry"]:
            zv = np.load(Path(comp) / "val_logits.npz")["submission_id"].astype(str)
            have = pd.concat([load_shards(run / n)[["submission_id"]] for n in ("val_gen", "val_v0")]).submission_id.astype(str).values
            assert set(zv) == set(have), "val rows differ from the same-seed classifier's"
        vg = load_shards(run / "val_gen").merge(meta, on="submission_id")
        thr = best_f1_threshold(vg.y_tle.values, vg.v1_gen.values)

        def paired(d, col):
            y, g, a_, b_ = d.y_tle.values, d.problem_id.values, d[col].values, d.clf.values
            lo, hi = cluster_bootstrap(g, lambda i: roc_auc_score(y[i], a_[i]) - roc_auc_score(y[i], b_[i]), n_boot=N_BOOT)
            return roc_auc_score(y, a_), roc_auc_score(y, b_), roc_auc_score(y, a_) - roc_auc_score(y, b_), lo, hi
        a0, b0, d0, lo0, hi0 = paired(v0, "v0"); a1, b1, d1, lo1, hi1 = paired(gen, "v1_gen")
        lows_v0.append(lo0); lows_v1.append(lo1)
        L_sec["verdict"] += [f"| {s} | v0 | {len(v0)} | {v0.problem_id.nunique()} | {a0:.3f} | {b0:.3f} | {fm(d0, (lo0, hi0), sign=True)} |",
                             f"| {s} | v1 | {len(gen)} | {gen.problem_id.nunique()} | {a1:.3f} | {b1:.3f} | {fm(d1, (lo1, hi1), sign=True)} |"]
        L_sec["fields"] += [f"### Seed {s[1]} ({len(gen)} rows, {gen.problem_id.nunique()} problems)", ""]
        field_claims[s] = score_fields(gen, maj, L_sec["fields"]); L_sec["fields"].append("")
        kv = [values(t) for t in gen.text]
        src = gen.n_var.map(lambda v: "N or n" if v in ("N", "n") else ("length" if v.startswith("|") else ("unknown" if v == "unknown" else "other")))
        good = np.array([k.get("loop_depth", "").isdecimal() and is_num(k.get("log10_n", "")) and is_num(k.get("est_log10_ops", "")) for k in kv])
        hit = np.array([abs(max(int(k["loop_depth"]), 1) * float(k["log10_n"]) - float(k["est_log10_ops"])) <= 0.15 if ok else False for k, ok in zip(kv, good)])
        parts = ", ".join(f"{n} {hit[(src == n).values & good].mean():.3f} (n={int(((src == n).values & good).sum())})" for n in sorted(src.unique()) if ((src == n).values & good).any())
        L_sec["self"].append(f"- Seed {s[1]}: overall {hit[good].mean() if good.any() else float('nan'):.3f} (n={int(good.sum())}); by source of N: {parts}.")
        L_sec["faith"] += [f"### Seed {s[1]}", ""]; faith[s] = faithfulness(cf, thr, L_sec["faith"])
        L_sec["sens"].append(f"Seed {s[1]}:"); sensitivity(cf, sens, L_sec["sens"])
        L_sec["cert"].append(f"### Seed {s[1]}"); L_sec["cert"].append(""); certificates(gen, thr, L_sec["cert"])
        L_sec["run"].append(f"- Seed {s[1]}: {rj['n_train']} training rows, {rj['steps']} steps, {rj['minutes']} minutes, peak {rj['peak_mem_GB']} GB, script sha {rj['script_sha256'][:12]}. Self-checks: reload {sc['reload_max_abs_v0']:.4f}; text differs on {sc['gen_text_differs_share']:.1%} of {sc['gen_checked_rows']} rows (v1 max change {sc['gen_v1_max_abs']:.4f}); forced vs stepwise {sc['forced_vs_stepwise_max_abs']:.4f}.")

    v0c, v1c = verdict_claims(lows_v0), verdict_claims(lows_v1)
    better = {f: all(all(lo > 0 for lo in field_claims[s][f].values()) and field_claims[s][f] for s in SEEDS) for f in field_claims["s0"]}
    final = {k: lower_label([faith[s][k] for s in SEEDS]) for k in ("neutral", "donor", "depth_up", "sort_on")}
    overall = lower_label([final["donor"], final["neutral"]])
    name = naming(final["donor"], final["neutral"])
    L = ["# Explanation model on the test split (script-generated; follows reports/stage5_predeclaration.md)", ""]
    if dry:
        L += ["**DRY RUN on a tiny sample: this checks the pipeline. Nothing here is a result.**", ""]
    L += ["All CIs: problem-clustered bootstrap, 95%, 300 resamples. Same-seed classifier: same recipe and training rows, no explanation.", "", "## 1. Verdict quality", "",
          "| seed | readout | rows | problems | explanation model AUROC | classifier AUROC, same rows | paired difference [95% CI] |", "|---|---|---|---|---|---|---|"] + L_sec["verdict"]
    L += ["", "## 2. Accuracy against tree-sitter on the visible window", "", "`sort_in_loop` is scored as present or absent (values above 1 are rare). Regex baselines read the full code, so they can see code the model's window cuts off. "
          "`n_var` and `log10_n` are mostly copied from the prompt's constraints line.", ""] + L_sec["fields"]
    L += ["## 3. Self-consistency (estimate equals max(depth, 1) x log10 N within 0.15)", ""] + L_sec["self"] + ["", "## 4. Decision faithfulness", ""] + L_sec["faith"]
    L += ["## 5. Sensitivity", ""] + L_sec["sens"] + ["## 6. Certificates", ""] + L_sec["cert"] + ["## Runs", ""] + L_sec["run"]
    L += ["", "## Claims by the pre-declared rules", "",
          f"- **v0 verdict:** {'no meaningful loss against the classifier' if v0c['no_meaningful_loss'] else 'the no-meaningful-loss claim FAILS'} (lower bounds {', '.join(f'{x:+.3f}' for x in lows_v0)}; limit {NO_LOSS}). Beats the classifier: {'yes' if v0c['beats_classifier'] else 'no'}.",
          f"- **v1 verdict:** {'no meaningful loss against the classifier' if v1c['no_meaningful_loss'] else 'the no-meaningful-loss claim FAILS'} (lower bounds {', '.join(f'{x:+.3f}' for x in lows_v1)}). Beats the classifier: {'yes' if v1c['beats_classifier'] else 'no'}.",
          "- **Better than every baseline, both seeds:** " + (", ".join(f for f, ok in better.items() if ok) or "no field") + ". Not established: " + (", ".join(f for f, ok in better.items() if not ok) or "none") + ".",
          f"- **Faithfulness labels (lower of the two seeds):** neutral {final['neutral']}, donor {final['donor']}, depth_up {final['depth_up']}, sort_on {final['sort_on']}. **Overall: {overall}.**",
          f"- **Naming:** this must be called {name}.", ""]
    out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, default=Path("reports/kaggle_sft_full")); ap.add_argument("--out", type=Path, default=Path("reports/sft_test.md"))
    a = ap.parse_args(); main(a.dir, a.out)
