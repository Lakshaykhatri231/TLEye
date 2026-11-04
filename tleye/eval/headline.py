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
    names = [("B2 LightGBM (AST + parsed constraints)", "b2"), ("**B2m** monotone-constrained LightGBM (default verdict model)", "b2m")]
    if "lora" in d:
        names.append(("LoRA run 1: lr 2e-4, a recipe that failed to learn (reports/phase3_run1.md)", "lora"))
    if "confN" in d:
        names += [("**Qwen2.5-Coder-0.5B + LoRA, lr 1e-4, no prefix** (confirmatory run, 40k training rows)", "confN"),
                  ("Qwen2.5-Coder-0.5B + LoRA, lr 1e-4, with symbolic prefix (confirmatory run)", "confP")]
    for lab, col in names:
        r = metrics_with_ci(y, d[col].values, g, prob=True, n_boot=300)
        L.append(f"| {lab} | {fmt(r['auroc'])} | {fmt(r['auprc'])} | {r['ece'][0]:.3f} |")
    if "confN" in d:
        def pair(a, b):
            def f(i):
                return roc_auc_score(y[i], d[a].values[i]) - roc_auc_score(y[i], d[b].values[i])
            lo, hi = cluster_bootstrap(g, f, n_boot=200)
            return f"{f(np.arange(len(d))):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        L += ["", f"Paired test AUROC differences (problem-clustered): LoRA no-prefix minus B2m {pair('confN', 'b2m')}; LoRA with-prefix minus B2m {pair('confP', 'b2m')}; "
              f"with-prefix minus no-prefix {pair('confP', 'confN')} (the prefix is NOT shown to help). Both configs were fixed in advance and scored once "
              "(reports/confirm_predeclaration.md, reports/confirm_test.md); one seed each."]
    sw = Path("reports/lr_sweep.md")
    if sw.exists():
        t = sw.read_text()
        pairs = {k: re.search(rf"\| {k} \| ([^|]+) \|", t).group(1).strip() for k in ("A - B2m", "B - B2m", "C - B2m", "B - A", "C - B")}
        L += ["", "**Learning-rate sweep, VALIDATION only** (reports/lr_sweep.md; 12,000 training rows, lr 5e-5 to 1e-4; paired AUROC difference, LoRA minus B2m): "
              f"A {pairs['A - B2m']}, B {pairs['B - B2m']}, C {pairs['C - B2m']}. The configs themselves are not separable "
              f"(prefix B-A {pairs['B - A']}, lr C-B {pairs['C - B']}). A confirmatory test-split run is pending."]
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
    sd = Path("reports/seeds_test.md")
    if sd.exists():
        t = sd.read_text()
        au = re.findall(r"\| seed (\d) \| [\d.]+ \| ([\d.]+) \[", t)
        dif = re.findall(r"\| seed (\d) - B2m \| ([^|]+) \|", t)
        spread = re.search(r"min (\d\.\d+), max (\d\.\d+), spread \(max - min\) (\d\.\d+), mean (\d\.\d+)", t)
        L += ["", "**Seed robustness of the confirmed LoRA recipe** (reports/seeds_test.md; seeds 0, 1, 2; rule fixed in advance in reports/seeds_predeclaration.md): test AUROC "
              + ", ".join(f"seed {k} {v}" for k, v in au) + f" (spread {spread.group(3)}, mean {spread.group(4)}); paired difference vs B2m "
              + ", ".join(f"seed {k} {v.strip()}" for k, v in dif) + ". Three seeds only: run-to-run variation is described, not given a confidence interval."]
    lt = Path("reports/lolo_test.md")
    if lt.exists():
        rows = [ln for ln in lt.read_text().splitlines() if re.match(r"\| (C\+\+|Java|Python) \|", ln)]
        L += ["", "**LoRA on an UNSEEN language** (trained on 20,000 rows of the other two languages; reports/lolo_test.md; claim rule fixed in advance in reports/lolo_predeclaration.md):", "",
              "| held-out | LoRA AUROC | LightGBM AUROC (same protocol) | paired difference | LoRA ECE under shift | in-distribution LoRA AUROC |", "|---|---|---|---|---|---|"]
        for ln in rows:
            c = [x.strip() for x in ln.strip("|").split("|")]
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
        L += ["", "**Prompting vs fine-tuning** (reports/b3_test.md; the same 400 fixed test items, rule fixed in advance in reports/b3_predeclaration.md; one open model, "
              "Qwen2.5-Coder-7B-Instruct, direct YES/NO answer, no chain-of-thought): "
              f"prompted AUROC {v['pa']} (same information as the fine-tuned model) and {v['pb']} (plus the full statement); LoRA {v['lora']}; B2m {v['b2m']}. "
              f"Paired differences, LoRA minus prompted: {v['d_a']} and {v['d_b']}. "
              f"Note the untrained prompted model is not distinguishable from LightGBM (B2m minus prompted A: {v['m_a']})."]
    bj = json.load(open("data/processed/baselines.json"))
    lolo = [r for r in bj["lolo"] if r["model"].startswith("B2")]
    L += ["", "**Unseen-language transfer** (B2, train on two languages, test on the third; plain B2 not B2m): " +
          "; ".join(f"{r['scope'].replace('held-out ', '')} AUROC {r['auroc']}" for r in lolo), "",
          "**Metamorphic checks** (reports/metamorphic.md; share of test rows where the property is violated, size bound x10 lowers P(TLE) / time limit x2 raises P(TLE)): "
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
    cov = pd.read_parquet("data/processed/cert_coverage.parquet")
    cov_all = (cov.category == "simple").mean()
    cov_test = ((cov.category == "simple") & (cov.split == "test")).sum() / (cov.split == "test").sum()
    L += [f"**Sandbox certificates** (arm64 container, CPU time >= the problem limit; only problems the generator fully supports: {cov_all:.1%} of all AtCoder problems, {cov_test:.1%} of test problems; "
          "intervals omitted because they would ignore problem clustering; rows are as classified when each pilot ran, and the classification was tightened after pilots 3, 5 and 6 "
          "exposed missing rules, so reports/phase4_pilot.md also gives the restricted view):", "",
          "| pilot | problems | true TLE confirmed | true AC wrongly certified |", "|---|---|---|---|", *rows, ""]
    return "\n".join(L)


def main() -> None:
    block = f"{START}\n{build()}\n{END}"
    p = Path("README.md"); s = p.read_text()
    p.write_text(re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _: block, s, flags=re.S) if START in s else s + "\n" + block + "\n")
    Path("reports/headline.md").write_text(block + "\n"); print(block)


if __name__ == "__main__":
    main()
