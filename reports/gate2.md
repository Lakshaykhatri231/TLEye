# Gate 2 assessment (constraint reader + baselines)

Numbers are copied from `reports/baselines.md`, `reports/constraints_eval.csv` and the build logs (all script-generated).
Data: AtCoder only, 206,472 cleaned submissions (train 142,400 / val 21,082 / test 42,990 rows; 1,036 / 148 / 296 problems, split by PROBLEM).

## 1. Constraint reader (frozen before evaluation; evaluated on 100 random val/test problems it was never tuned on)
| field | exact match | 95% CI (Wilson) |
|---|---|---|
| max_bound (largest literal upper bound) | 94/100 = 94.0% | 87.5% - 97.2% |
| n_bound (primary size variable) | 96/100 = 96.0% | 90.2% - 98.4% |

- Labels: `data/labels/constraints_labels.csv`. **They were written by the model (Claude), blind to parser output, not by a human** - a spot check by you would strengthen this number.
- 16.5% of training rows have no parsed primary size (problems with no explicit N, or unparsed phrasing); a fallback to max_bound is used where available.
- Known failure classes (kept as limitations, NOT patched, to keep the number honest): trailing parentheticals on the same line (`... \leq 999 (1 \leq i \leq N)`), `|a_{i,j}|`, product bounds `N \times M \leq ...`, chains inside prose, and hours/minutes variables (`H`, `M`) mistaken for sizes (2 false positives out of 11 problems with no size).

## 2. Baselines on the test split (point [95% CI], problem-clustered bootstrap)
| model | AUROC | AUPRC | F1 | ECE |
|---|---|---|---|---|
| B1 rule (loop depth x log N - log time limit) | 0.632 [0.613, 0.650] | 0.504 [0.483, 0.526] | 0.585 [0.571, 0.598] | - |
| B2 LightGBM, AST only | 0.727 [0.707, 0.746] | 0.620 [0.594, 0.645] | 0.630 [0.615, 0.644] | 0.025 [0.018, 0.041] |
| B2 LightGBM, AST + constraints | 0.759 [0.738, 0.775] | 0.675 [0.643, 0.703] | 0.650 [0.637, 0.664] | 0.020 [0.013, 0.033] (Platt) / 0.008 raw |

n = 42,990 test submissions from 296 problems; prevalence 0.401.
- **Prevalence is an artifact of the per-class cap.** AUROC is unaffected; AUPRC, F1 and ECE are not real-world rates.
- **Trivial reference:** always predicting TLE gives F1 = 0.573 at this prevalence. B1's F1 (0.585) is barely above it; B1 is a weak baseline, mostly informative through AUROC.
- Constraints add +0.032 AUROC over AST-only (0.727 -> 0.759). The CIs partly overlap and I did not run a paired test, so treat this as suggestive, not established.
- Per-language AUROC (B2 AST+constraints): C++ 0.744, Java 0.780, Python 0.757 (CIs in `reports/baselines.md`).

## 3. Calibration finding (a negative result)
Platt scaling fit on the val split made B2 slightly WORSE: ECE 0.020 (Platt) vs 0.008 (raw LightGBM). The val set is only 148 problems and its TLE prevalence differs from test, which adds noise. Percentile-bootstrap CIs for ECE are biased (some point estimates fall outside their own interval, e.g. Python 0.016 [0.018, 0.035]); treat ECE CIs as approximate. Plan: use raw / temperature-scaled outputs for the LoRA model and compare on the same footing.

## 4. Leave-one-language-out (B2, no language features)
| held-out | AUROC | in-distribution AUROC | ECE |
|---|---|---|---|
| C++ | 0.707 [0.688, 0.726] | 0.744 | 0.094 |
| Java | 0.733 [0.703, 0.761] | 0.780 | 0.105 |
| Python | 0.693 [0.670, 0.717] | 0.757 | 0.075 |

Ranking transfers across languages with a drop of about 0.04-0.06 AUROC; calibration degrades clearly (ECE 0.075-0.105 vs ~0.01-0.03). B1 transfers at its in-distribution level (it has no learned language-specific part).

## 5. Gate decision
- Plan's gate: "if B2 AUROC >= 0.95, LoRA must beat it meaningfully, otherwise reframe". **Not triggered:** B2 is 0.759, so there is real headroom, but structural features alone leave most of the signal unexplained. Unknown: how much is irreducible (same algorithm, different constants, I/O style, judge noise).
- **Bar the LoRA model has to beat:** overall AUROC 0.759 [0.738, 0.775]; LOLO AUROC 0.707 / 0.733 / 0.693 (C++ / Java / Python held out).

## 6. Risks and open items before Phase 3
1. Compute: a 1.5B model on a free Kaggle T4 cannot see all ~142k training rows in a session; plan is a throughput benchmark first (0.5B vs 1.5B), then a training subset sized to the budget. (No throughput number measured yet.)
2. Prior shift for deployment: real-world TLE rates are far below the capped-sample 33-42%; a prior-correction step is needed for any user-facing probability. Not implemented yet.
3. The Kaggle token authenticates (account lists existing notebooks). The key shown in your earlier screenshot is still exposed in this chat and should be rotated.
4. Constraint-reader labels are model-written (see section 1).
5. **Baseline B3 (prompted open LLM, CodeJudge-Eval style, 400-item fixed test sample) is NOT done:** it needs a Kaggle GPU, so it moves into the first Kaggle session in Phase 3.
