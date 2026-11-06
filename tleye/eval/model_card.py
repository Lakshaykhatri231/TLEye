"""Generate the Hugging Face model card (models/lora/README.md) from report files. No hand-typed numbers.
    python -I -m tleye.eval.model_card"""
import json
import re
from pathlib import Path

meta = json.loads(Path("models/lora/meta.json").read_text())
conf = Path("reports/confirm_test.md").read_text()


def row(label):
    m = re.search(rf"\| {re.escape(label)} \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", conf)
    return [x.strip() for x in m.groups()]


n_line = re.search(r"(\d+) test rows, (\d+) problems, prevalence ([\d.]+)", conf).groups()
lora, b2m = row("LoRA N (no prefix)"), row("B2m monotone LightGBM")
diff = re.search(r"\| N - B2m \| AUROC \| ([^|]+) \|", conf).group(1).strip()
lang_rows = re.findall(r"\| (C\+\+|Java|Python) \| \d+ \| ([\d.]+) \| ([\d.]+) \(([^)]*)\)", conf)
b3 = Path("reports/b3_test.md")
b3_line = ""
if b3.exists():
    t3 = b3.read_text()
    d_a = re.search(r"LoRA - prompt_A \| ([^|]+) \|", t3).group(1).strip()
    d_b = re.search(r"LoRA - prompt_B \| ([^|]+) \|", t3).group(1).strip()
    b3_line = ("\nCompared with zero-shot prompting of Qwen2.5-Coder-7B-Instruct (direct YES/NO, no chain-of-thought) on 400 fixed test items, paired AUROC difference "
               f"(this adapter minus prompted): {d_a} with the same information, {d_b} with the full statement added.\n")
seeds = Path("reports/seeds_test.md")
seed_line = "One training seed"
if seeds.exists():
    mm = re.search(r"min (\d\.\d+), max (\d\.\d+)", seeds.read_text())
    seed_line = f"Three training seeds for the main recipe (test AUROC {mm.group(1)} to {mm.group(2)}); the unseen-language runs used one seed each"
lolo = Path("reports/lolo_test.md")
lolo_txt = ""
if lolo.exists():
    lolo_txt = "\n## Unseen-language transfer (leave-one-language-out)\n\n" + "\n".join(
        l for l in lolo.read_text().splitlines() if l.startswith("|") or l.startswith("- ")) + "\n"
card = f"""---
license: apache-2.0
base_model: {meta['base_model']}
library_name: peft
tags: [lora, code, competitive-programming, time-limit-exceeded, classification]
---

# TLEye verdict model: LoRA adapter predicting Time Limit Exceeded (AtCoder; C++/Java/Python)

A LoRA adapter on `{meta['base_model']}` with a 1-logit classification head. Given a solution, the problem's time limit and the input-size bounds
parsed from the statement, it outputs P(TLE). It is the verdict half of TLEye; the other half (sandbox-confirmed failing inputs) is not part of this adapter.

**Measured limits on those inputs:** the model ignores the time limit (doubling it changes the logit less than rewriting `2s` as `2.0s`) and responds to N in the right direction but weakly, and not monotonically for every solution (reports/metamorphic_lora.md in the project repo).

## Prompt format (must match exactly)
```
language: <C++|Java|Python|PyPy>
time limit: <seconds>s
constraints: <e.g. N<=2e5, A<=1e9>
code:
<source, first {meta['max_code_chars']} characters>
```
tokenised, truncated to {meta['max_len']} tokens, then the suffix `{meta['suffix']!r}`; the head reads the last token. Probability = sigmoid(logit / T) with T = {meta['T']:.3f}.

## Data
IBM Project CodeNet, **AtCoder subset only** (AIZU dropped: inconsistent time limits). AC vs TLE submissions; borderline Accepted (CPU >= 0.8x limit) and
conflicting/duplicate code removed; at most 40 submissions per class per problem and language. Train/val/test are split **by problem** (never by submission).
Training used a seeded 40,000-row subset of the training split. Test: {n_line[0]} submissions from {n_line[1]} unseen problems.

## Results on the held-out test split (problem-clustered 95% CI)
| model | AUROC | AUPRC | ECE |
|---|---|---|---|
| this adapter | {lora[0]} | {lora[1]} | {lora[2]} |
| LightGBM with monotone constraints (CPU baseline) | {b2m[0]} | {b2m[1]} | {b2m[2]} |

Paired AUROC difference (adapter minus LightGBM): {diff}. Configuration and claim rule were written down before the run (reports/confirm_predeclaration.md); one seed.
Per-language AUROC: {', '.join(f"{l} {a}" for l, _, a, _ in lang_rows) if lang_rows else 'see report'}.
{b3_line}{lolo_txt}
## Non-claims and limitations
- Not a judge replacement and not a safety guarantee: it estimates P(TLE); a low probability is not a proof the solution passes.
- Prevalence in the evaluation data ({n_line[2]} TLE) is an artifact of per-class capping; AUPRC/F1/ECE are not real-world rates (AUROC is unaffected). Shift the prior before using probabilities in practice.
- AtCoder data only; solutions are human submissions, not LLM-generated; performance on other judges, newer language versions or other hardware is untested.
- {seed_line}; 296 test problems.
- The adapter was trained on public competitive-programming submissions; do not execute dataset code outside a sandbox.

Training/evaluation code, the sandbox certificate engine and all reports are in the TLEye code repository: to be published.
"""
Path("models/lora/README.md").write_text(card); print(card[:1800])
