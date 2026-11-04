---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B
library_name: peft
tags: [lora, code, competitive-programming, time-limit-exceeded, classification]
---

# TLEye verdict model: LoRA adapter predicting Time Limit Exceeded (AtCoder; C++/Java/Python)

A LoRA adapter on `Qwen/Qwen2.5-Coder-0.5B` with a 1-logit classification head. Given a solution, the problem's time limit and the input-size bounds
parsed from the statement, it outputs P(TLE). It is the verdict half of TLEye; the other half (sandbox-confirmed failing inputs) is not part of this adapter.

## Prompt format (must match exactly)
```
language: <C++|Java|Python|PyPy>
time limit: <seconds>s
constraints: <e.g. N<=2e5, A<=1e9>
code:
<source, first 24000 characters>
```
tokenised, truncated to 1024 tokens, then the suffix `'\n### verdict:'`; the head reads the last token. Probability = sigmoid(logit / T) with T = 1.290.

## Data
IBM Project CodeNet, **AtCoder subset only** (AIZU dropped: inconsistent time limits). AC vs TLE submissions; borderline Accepted (CPU >= 0.8x limit) and
conflicting/duplicate code removed; at most 40 submissions per class per problem and language. Train/val/test are split **by problem** (never by submission).
Training used a seeded 40,000-row subset of the training split. Test: 42990 submissions from 296 unseen problems.

## Results on the held-out test split (problem-clustered 95% CI)
| model | AUROC | AUPRC | ECE |
|---|---|---|---|
| this adapter | 0.863 [0.847, 0.881] | 0.812 [0.785, 0.839] | 0.014 [0.010, 0.023] |
| LightGBM with monotone constraints (CPU baseline) | 0.756 [0.735, 0.774] | 0.667 [0.638, 0.697] | 0.008 [0.008, 0.021] |

Paired AUROC difference (adapter minus LightGBM): +0.107 [+0.095, +0.120]. Configuration and claim rule were written down before the run (reports/confirm_predeclaration.md); one seed.
Per-language AUROC: C++ 0.854, Java 0.886, Python 0.859.

Compared with zero-shot prompting of Qwen2.5-Coder-7B-Instruct (direct YES/NO, no chain-of-thought) on 400 fixed test items, paired AUROC difference (this adapter minus prompted): +0.085 [+0.039, +0.128] with the same information, +0.101 [+0.057, +0.139] with the full statement added.

## Unseen-language transfer (leave-one-language-out)

| held-out | n | problems | LoRA LOLO AUROC [CI] | LightGBM LOLO AUROC [CI] | paired LoRA - LightGBM | LoRA AUPRC | LoRA ECE | in-distribution LoRA AUROC (40k rows, 3 langs) |
|---|---|---|---|---|---|---|---|---|
| C++ | 19779 | 293 | 0.790 [0.771, 0.810] | 0.707 [0.688, 0.726] | +0.084 [+0.070, +0.098] | 0.752 | 0.120 | 0.854 (T=1.347) |
| Java | 9759 | 198 | 0.846 [0.821, 0.868] | 0.733 [0.703, 0.761] | +0.113 [+0.094, +0.136] | 0.755 | 0.200 | 0.886 (T=1.245) |
| Python | 13452 | 240 | 0.775 [0.755, 0.795] | 0.693 [0.670, 0.717] | +0.082 [+0.065, +0.102] | 0.718 | 0.127 | 0.859 (T=1.370) |
- Per language, LoRA transfers better than LightGBM only where the paired CI excludes 0: C++: YES; Java: YES; Python: YES.
- Overall statement ('LoRA transfers to an unseen language better than LightGBM'): **supported for all three languages**.
- Calibration: T was fit on the training languages, so ECE on the held-out language reflects calibration under language shift.

## Non-claims and limitations
- Not a judge replacement and not a safety guarantee: it estimates P(TLE); a low probability is not a proof the solution passes.
- Prevalence in the evaluation data (0.401 TLE) is an artifact of per-class capping; AUPRC/F1/ECE are not real-world rates (AUROC is unaffected). Shift the prior before using probabilities in practice.
- AtCoder data only; solutions are human submissions, not LLM-generated; performance on other judges, newer language versions or other hardware is untested.
- Three training seeds for the main recipe (test AUROC 0.857 to 0.863); the unseen-language runs used one seed each; 296 test problems.
- The adapter was trained on public competitive-programming submissions; do not execute dataset code outside a sandbox.

Training/evaluation code, the sandbox certificate engine and all reports are in the TLEye repository (add its link when publishing).
