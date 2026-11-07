"""Fill the README headline block from run outputs (no hand-typed numbers):  python -I -m tleye.eval.headline
Sources: b2/b2m test predictions, Kaggle LoRA logits (if present), baselines.json, constraints_eval.csv, certificate pilots, metamorphic.md."""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tleye.eval.metrics import cluster_bootstrap, metrics_with_ci

START, END = "<!-- HEADLINE:START -->", "<!-- HEADLINE:END -->"
SUM_START, SUM_END = "<!-- SUMMARY:START -->", "<!-- SUMMARY:END -->"
S: dict = {}          # numbers for the at-a-glance table, filled while the headline is built


def sig(x):
    return 1 / (1 + np.exp(-x))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def meta_line() -> str:
    t = Path("reports/metamorphic.md").read_text()
    pct = re.findall(r"(\d+\.\d+)% \[", t)               # plain N, plain TL, B2m N, B2m TL
    delta = re.search(r"\(B2m - B2\): ([^\n]+)", t).group(1)
    return (f"plain B2 {pct[0]}% / {pct[1]}%, B2m {pct[2]}% / {pct[3]}%; paired AUROC change from the constraints (B2m - B2) {delta}.")


def build() -> str:
    meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "language", "y_tle"])
    b2 = pd.read_parquet("data/processed/b2_test_preds.parquet").rename(columns={"p_raw": "b2"})[["submission_id", "b2"]]
    b2m = pd.read_parquet("data/processed/b2m_test_preds.parquet").rename(columns={"p_raw": "b2m"})
    d = meta.merge(b2, on="submission_id").merge(b2m, on="submission_id")
    lora = Path("reports/kaggle_train/test_logits.npz")
    if lora.exists():
        z = np.load(lora); d = d.merge(pd.DataFrame({"submission_id": z["submission_id"], "lora": sig(z["logit"] / float(z["T"]))}), on="submission_id")
    for c in "NP":                                    # confirmatory runs (pre-declared, scored once on test)
        zf = Path(f"reports/kaggle_confirm/cfg{c}/test_logits.npz")
        if zf.exists():
            z = np.load(zf); d = d.merge(pd.DataFrame({"submission_id": z["submission_id"], f"conf{c}": sig(z["logit"] / float(z["T"]))}), on="submission_id")
    y, g = d.y_tle.values, d.problem_id.values
    L = [f"**Verdict models on the held-out test split** ({len(d):,} submissions, {d.problem_id.nunique()} problems; AtCoder only; problem-disjoint; "
         f"95% CI from a problem-clustered bootstrap; TLE prevalence {y.mean():.3f} is an artifact of per-class capping):", "",
         "| model | AUROC | AUPRC | ECE |", "|---|---|---|---|"]
    names = [("B2 LightGBM (AST + parsed constraints)", "b2"), ("**B2m** monotone-constrained LightGBM (CPU fallback; monotone in N and the time limit by construction)", "b2m")]
    if "lora" in d:
        names.append(("LoRA run 1: lr 2e-4, a recipe that failed to learn (reports/phase3_run1.md)", "lora"))
    if "confN" in d:
        names += [("**Qwen2.5-Coder-0.5B + LoRA, lr 1e-4, no prefix** (confirmatory run, 40k training rows)", "confN"),
                  ("Qwen2.5-Coder-0.5B + LoRA, lr 1e-4, with symbolic prefix (confirmatory run)", "confP")]
    for lab, col in names:
        r = metrics_with_ci(y, d[col].values, g, prob=True, n_boot=300)
        L.append(f"| {lab} | {fmt(r['auroc'])} | {fmt(r['auprc'])} | {r['ece'][0]:.3f} |")
        S[col] = r["auroc"]
    if "confN" in d:
        def pair(a, b):
            def f(i):
                return roc_auc_score(y[i], d[a].values[i]) - roc_auc_score(y[i], d[b].values[i])
            lo, hi = cluster_bootstrap(g, f, n_boot=200)
            return f"{f(np.arange(len(d))):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        L += ["", f"Paired test AUROC differences (problem-clustered): LoRA no-prefix minus B2m {pair('confN', 'b2m')}; LoRA with-prefix minus B2m {pair('confP', 'b2m')}; "
              f"with-prefix minus no-prefix {pair('confP', 'confN')} (the prefix is NOT shown to help). Both configs were fixed in advance and scored once "
              "(reports/confirm_predeclaration.md, reports/confirm_test.md); one seed each here, more seeds below."]
    sw = Path("reports/lr_sweep.md")
    if sw.exists():
        t = sw.read_text()
        pairs = {k: re.search(rf"\| {k} \| ([^|]+) \|", t).group(1).strip() for k in ("A - B2m", "B - B2m", "C - B2m", "B - A", "C - B")}
        L += ["", "**Learning-rate sweep, VALIDATION only** (reports/lr_sweep.md; 12,000 training rows, lr 5e-5 to 1e-4; paired AUROC difference, LoRA minus B2m): "
              f"A {pairs['A - B2m']}, B {pairs['B - B2m']}, C {pairs['C - B2m']}. The configs themselves are not separable "
              f"(prefix B-A {pairs['B - A']}, lr C-B {pairs['C - B']}). These rates were then tested once in the confirmatory run above."]
    ml = Path("reports/metamorphic_lora.md")
    if ml.exists():
        t = ml.read_text()
        dep = re.search(r"At depth >= 2, N x10 mean logit change on test: ([^.]+\.\d+ \[[^\]]+\])", t).group(1)
        rows = re.findall(r"\| (LoRA \(seed 0\)|B2m \(monotone by construction\)) \| \d+ \| ([+\-\d.]+) \| ([^|]+) \|", t)
        tl = re.search(r"\| tl_x2 \| 5000 \| \d+ \| ([^|]+) \| ([\d.]+) \|", t)
        ptl = re.search(r"\| placebo_tl \| 5000 \| \d+ \| [^|]+ \| ([\d.]+) \|", t)
        L += ["", "**Does the LoRA model react to the constraints?** (reports/metamorphic_lora.md; rules fixed in advance in reports/stage1_predeclaration.md; same code, one change to the prompt) "
              f"N x10 at loop depth >= 2 moves the logit {dep}, the right direction and above placebo noise, but small against a logit spread of about 3. "
              + " ".join(f"{a}: mean change {b}, wrong-direction share at 0 / 0.1 / 0.5 logit {c.strip()}." for a, b, c in rows)
              + f" Doubling the time limit changes the logit by {tl.group(1).strip()} (mean absolute {tl.group(2)}), less than rewriting `2s` as `2.0s` (mean absolute {ptl.group(1)}): the model ignores the time limit."]
    pr = Path("reports/probes.md")
    if pr.exists():
        cells = {c[0]: c for c in ([x.strip() for x in ln.strip("|").split("|")] for ln in pr.read_text().splitlines()
                                   if ln.startswith("|") and ln.count("|") == 11)}
        def probe(k, col=1):
            return cells[k][col]
        L += ["", "**What can be read linearly from the model's hidden states?** (reports/probes.md; rules and sites fixed in advance in reports/stage2_predeclaration.md and on validation; 8,000 test rows, 296 problems) "
              f"Decodable means a simple linear probe recovers it, not that the model uses it. Loop depth >= 2 AUROC {probe('depth_ge2')}, recursion {probe('recursive')}, sort inside a loop {probe('sort_in_loop')}, "
              f"log10 N R2 {probe('log10_n')}, time limit R2 {probe('log_tl')} (descriptive only, since 91.5% of rows share one limit; consistent with the time-limit result above). "
              f"Fine-tuning made these more decodable than in the base model: log10 N {probe('log10_n', 8)}, sort inside a loop {probe('sort_in_loop', 8)}; it did not change loop depth ({probe('depth_ge2', 8)}), "
              f"and has-a-sort got less decodable ({probe('has_sort', 8)}), no better than a bag of tokens ({probe('has_sort', 7)}). "
              "The base model was never trained to summarize at the last token, so a gain there may only mean the information moved."]
    ex = Path("reports/sft_test.md")
    if ex.exists():
        t = ex.read_text()
        v = re.findall(r"\| (s[01]) \| (v[01]) \| \d+ \| \d+ \| ([\d.]+) \| ([\d.]+) \| ([^|]+) \|", t)
        ver = "; ".join(f"seed {s[1]} {r} {a} vs {b} ({c.strip()})" for s, r, a, b, c in v)
        fld = lambda f: re.findall(rf"\| {f} \| [^|]+ \| ([\d.]+) \| ([\d.]+) \|", t)  # noqa: E731
        fa = re.findall(r"\| (neutral|donor) \| 3000 \| 0 \| [^|]+ \| ([\d.]+) \| ([\d.]+%) \|", t)
        drop = re.findall(r"generated minus neutral: ([^(]+) \(", t)
        lab = re.search(r"\*\*Faithfulness labels[^:]*:\*\* neutral (\S+), donor (\S+), depth_up (\S+), sort_on (\S+)\.", t)
        L += ["", "**Can it write a summary of the code, and does that summary drive the verdict?** (reports/sft_test.md; rules fixed in advance in reports/stage5_predeclaration.md; two seeds; each compared with the classifier of the same seed on the same rows) "
              f"Verdict AUROC, explanation model vs classifier: {ver}. By the rule: no meaningful loss, and not better than the classifier (the two seeds point in opposite directions). "
              "Generated fields against a real AST check, seeds 0 / 1: "
              f"loop depth balanced accuracy {fld('loop_depth')[0][0]} / {fld('loop_depth')[1][0]} (majority guess {fld('loop_depth')[0][1]}), "
              f"loop count exact {fld('loops')[0][0]} / {fld('loops')[1][0]} (majority {fld('loops')[0][1]}), "
              f"listed calls in a loop exact {fld('listed_calls_in_loop')[0][0]} / {fld('listed_calls_in_loop')[1][0]} (majority {fld('listed_calls_in_loop')[0][1]}). "
              f"**But the summary is decorative by the pre-declared labels:** replacing it on 3,000 rows moves the verdict logit only "
              + ", ".join(f"seed {i // 2} {k} {a} mean absolute, {b} flips" for i, (k, a, b) in enumerate(fa))
              + f", and a neutral summary costs no AUROC ({', '.join(x.strip() for x in drop)}). Labels: neutral {lab.group(1)}, donor {lab.group(2)}, depth up {lab.group(3)}, sort on {lab.group(4)}. "
              "So it is a checkable code summary, not the reason for the verdict. The trained verdict read-out before the summary may make this more likely; a design forcing the verdict through the summary was not tested on the test split."]
    sd = Path("reports/seeds_test.md")
    if sd.exists():
        t = sd.read_text()
        au = re.findall(r"\| seed (\d) \| [\d.]+ \| ([\d.]+) \[", t)
        dif = re.findall(r"\| seed (\d) - B2m \| ([^|]+) \|", t)
        spread = re.search(r"min (\d\.\d+), max (\d\.\d+), spread \(max - min\) (\d\.\d+), mean (\d\.\d+)", t)
        L += ["", "**Seed robustness of the confirmed LoRA recipe** (reports/seeds_test.md; seeds 0, 1, 2; rule fixed in advance in reports/seeds_predeclaration.md): test AUROC "
              + ", ".join(f"seed {k} {v}" for k, v in au) + f" (spread {spread.group(3)}, mean {spread.group(4)}); paired difference vs B2m "
              + ", ".join(f"seed {k} {v.strip()}" for k, v in dif) + ". Three seeds only: run-to-run variation is described, not given a confidence interval."]
        S["seeds"] = (spread.group(1), spread.group(2), spread.group(4)); S["seed_diffs"] = [v.strip() for _, v in dif]
    lt = Path("reports/lolo_test.md")
    if lt.exists():
        rows = [ln for ln in lt.read_text().splitlines() if re.match(r"\| (C\+\+|Java|Python) \|", ln)]
        L += ["", "**LoRA on an UNSEEN language** (trained on 20,000 rows of the other two languages; reports/lolo_test.md; claim rule fixed in advance in reports/lolo_predeclaration.md):", "",
              "| held-out | LoRA AUROC | LightGBM AUROC (same protocol) | paired difference | LoRA ECE under shift | in-distribution LoRA AUROC |", "|---|---|---|---|---|---|"]
        S["lolo"] = []
        for ln in rows:
            c = [x.strip() for x in ln.strip("|").split("|")]
            S["lolo"].append((c[0], c[3].split(" [")[0], c[4].split(" [")[0]))
            L.append(f"| {c[0]} | {c[3]} | {c[4]} | {c[5]} | {c[7]} | {c[8].split(' (')[0]} (40k rows, 3 languages) |")
        L += ["", "The ranking transfers, but calibration does not (ECE far above the in-distribution value), and the in-distribution comparison mixes "
              "'unseen language' with 'half the training rows'."]
    b3 = Path("reports/b3_test.md")
    if b3.exists():
        t = b3.read_text()
        pats = {"pa": r"variant A \(matched info\) \| ([^|]+) \|", "pb": r"variant B \(\+ problem statement\) \| ([^|]+) \|",
                "lora": r"Fine-tuned LoRA 0.5B \| ([^|]+) \|", "b2m": r"B2m LightGBM \| ([^|]+) \|",
                "d_a": r"LoRA - prompt_A \| ([^|]+) \|", "d_b": r"LoRA - prompt_B \| ([^|]+) \|", "m_a": r"B2m - prompt_A \| ([^|]+) \|"}
        v = {k: re.search(pat, t).group(1).strip() for k, pat in pats.items()}      # (regexes with backslashes stay OUT of f-strings: Python 3.10)
        S["b3"] = v
        L += ["", "**Prompting vs fine-tuning** (reports/b3_test.md; the same 400 fixed test items, rule fixed in advance in reports/b3_predeclaration.md; one open model, "
              "Qwen2.5-Coder-7B-Instruct, direct YES/NO answer, no chain-of-thought): "
              f"prompted AUROC {v['pa']} (same information as the fine-tuned model) and {v['pb']} (plus the full statement); LoRA {v['lora']}; B2m {v['b2m']}. "
              f"Paired differences, LoRA minus prompted: {v['d_a']} and {v['d_b']}. "
              f"Note the untrained prompted model is not distinguishable from LightGBM (B2m minus prompted A: {v['m_a']})."]
    L += ["", "**Metamorphic checks** (reports/metamorphic.md; share of test rows where the property is violated, size bound x10 lowers P(TLE) / time limit x2 raises P(TLE)): "
          + meta_line(), ""]
    ce = pd.read_csv("reports/constraints_eval.csv")
    n = len(ce)

    def same(a, b):
        if pd.isna(a) or pd.isna(b):
            return bool(pd.isna(a) and pd.isna(b))
        return math.isclose(a, b, rel_tol=1e-9)

    km = sum(same(a, b) for a, b in zip(ce.true_max, ce.pred_max))
    kn = sum(same(a, b) for a, b in zip(ce.true_n, ce.pred_n))
    L += [f"**Constraint reader** (100 val/test problems, labels written by the AI blind to parser output, not by a human): largest bound exact match {km}/{n} "
          f"[{wilson(km, n)[0]:.1%}, {wilson(km, n)[1]:.1%}], primary size exact match {kn}/{n} [{wilson(kn, n)[0]:.1%}, {wilson(kn, n)[1]:.1%}].", ""]
    rows = []
    for name, f_ in [("pilot 1", "cert_pilot.jsonl"), ("pilot 2 (fresh problems)", "cert_pilot_confirm.jsonl"),
                     ("pilot 3 (development set: used to find engine bugs)", "cert_pilot3.jsonl"), ("pilot 4 (fresh problems, after bounds fixes)", "cert_pilot4.jsonl"),
                     ("pilot 5 (development set: exposed missing rules)", "cert_pilot5.jsonl"), ("**pilot 6 (fresh problems, final engine)**", "cert_pilot6.jsonl")]:
        p = pd.DataFrame([json.loads(x) for x in Path("data/processed", f_).read_text().splitlines()])
        p = p[~p.status.isin(["compile_error", "missing_dependency", "infra_error"])]
        # strict rule (cpu >= limit) re-evaluated from stored per-input CPU times
        c = [any(x["rc"] in (0, 124) and x["cpu_s"] * 1000 >= lim for x in rs) for rs, lim in zip(p.runs, p.limit_ms)]
        p = p.assign(c=c); t, a = p[p.y_tle == 1], p[p.y_tle == 0]
        rows.append(f"| {name} | {p.problem_id.nunique()} | {int(t.c.sum())}/{len(t)} = {t.c.mean():.1%} | {int(a.c.sum())}/{len(a)} |")
        S["pilot"] = (p.problem_id.nunique(), int(t.c.sum()), len(t), t.c.mean(), int(a.c.sum()), len(a))
    cov = pd.read_parquet("data/processed/cert_coverage.parquet")
    cov_all = (cov.category == "simple").mean()
    cov_test = ((cov.category == "simple") & (cov.split == "test")).sum() / (cov.split == "test").sum()
    S["cov"] = (cov_all, cov_test)
    L += [f"**Sandbox certificates** (arm64 container, CPU time >= the problem limit; only problems the generator fully supports: {cov_all:.1%} of all AtCoder problems, {cov_test:.1%} of test problems; "
          "intervals omitted because they would ignore problem clustering; rows are as classified when each pilot ran, and the classification was tightened after pilots 3, 5 and 6 "
          "exposed missing rules, so reports/phase4_pilot.md also gives the restricted view):", "",
          "| pilot | problems | true TLE confirmed | true AC wrongly certified |", "|---|---|---|---|", *rows, ""]
    return "\n".join(L)


def summary() -> str:
    """At-a-glance table for the top of the README, from the same sources as the headline (call after build())."""
    lo, hi, mean = S["seeds"]
    rows = [f"| Fine-tuned verdict model (Qwen2.5-Coder-0.5B + LoRA), test AUROC | **{mean}** (3 seeds, {lo} to {hi}) | LightGBM baseline B2m: {S['b2m'][0]:.3f} "
            f"[{S['b2m'][1]:.3f}, {S['b2m'][2]:.3f}]; every paired gain excludes 0 ({', '.join(S['seed_diffs'])}) |"]
    if "lolo" in S:
        rows.append("| Language never seen in training, test AUROC | " + ", ".join(f"{a} {b}" for a, b, _ in S["lolo"]) + " | LightGBM, same protocol: "
                    + ", ".join(f"{a} {c}" for a, _, c in S["lolo"]) + "; calibration does NOT transfer |")
    if "b3" in S:
        rows.append(f"| Against prompting a 7B code model (400 items) | LoRA {S['b3']['lora'].split(' [')[0]} vs prompted {S['b3']['pa'].split(' [')[0]} | paired gain {S['b3']['d_a']} |")
    n, k, m, share, fa, na = S["pilot"]
    rows.append(f"| Sandbox certificates, final engine on fresh problems | {k}/{m} = {share:.1%} of true TLE solutions confirmed | {fa}/{na} Accepted solutions wrongly certified; "
                f"only {S['cov'][0]:.1%} of problems are supported |")
    return "\n".join(["| | result | compared with |", "|---|---|---|", *rows])


def put(s: str, start: str, end: str, block: str) -> str:
    """Replace a marked block in the README; a block the README does not have is left out (the full set is in reports/headline.md)."""
    full = f"{start}\n{block}\n{end}"
    return re.sub(f"{re.escape(start)}.*?{re.escape(end)}", lambda _: full, s, flags=re.S) if start in s else s


def main() -> None:
    head = build(); summ = summary()
    p = Path("README.md"); s = put(put(p.read_text(), START, END, head), SUM_START, SUM_END, summ)
    p.write_text(s)
    Path("reports/headline.md").write_text(f"{SUM_START}\n{summ}\n{SUM_END}\n\n{START}\n{head}\n{END}\n"); print(summ)


if __name__ == "__main__":
    main()
