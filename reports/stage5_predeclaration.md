# Pre-declaration: Stage 5, full explanation runs (written BEFORE any test-split explanation output exists)

Date: 2026-11-05. Kernel: khatrilakshay2004/tleye-sft (kaggle/sft/sft.py, STAGE="full", FULL_MODE="dual", chosen by the Stage 4 rule; reports/sft_gate.md). Runs only after the user's OK.

**Runs:** `dual`, seeds 0 and 1 side by side, 40,000 training rows, the same draw as the same-seed classifier (seed 0: reports/kaggle_confirm/cfgN; seed 1: reports/kaggle_seeds/cfg_s1). The val rows are checked to equal the classifier's. Each run is scored once on test.
**Test rows:** v0 on the full test split. Generation (v1) on the fixed subset from `tleye/data/sft_subset.py`: 12,263 rows, at most 40 per problem plus all 766 usable certificate-pilot rows. 3,000 of them (`in_B`) also get counterfactuals. Sensitivity (N x1000) on `in_B` rows with a known N and a finite bound.

**All CIs:** problem-clustered bootstrap, 95%. All comparisons on identical rows.

## 1. Verdict quality
Paired AUROC against the same-seed classifier: v0 on the full test split, v1 on the subset, reported separately. "No meaningful loss" holds for a readout only if the lower CI bound is above -0.02 for BOTH seeds. A claim that the explanation model beats the classifier requires a lower bound above 0 for both seeds; it is not expected.

## 2. Accuracy against tree-sitter on the visible window
Per field, for generated values, on the subset: balanced accuracy for depth (>= 4 one class), recursion and `sort_in_loop`; exact match plus mean absolute error for counts; mean absolute error for `log10_n` and `est_log10_ops` where N is known. Split by truncated or not. Baselines on the same rows:
- per-language majority guess, learned on the train split;
- regex on the full code: loops = count of `for`/`while` keywords, sorts = count of `sort`, recursion = always the majority; regex on the constraints line: `n_var` and `log10_n` from its first item.
A field is claimed "better than the baselines" only if the paired CI against EACH baseline excludes 0, for both seeds. High `n_var` and `log10_n` scores are described as copying from the prompt.

## 3. Self-consistency
Share where the stated estimate equals max(depth, 1) x log10 N within 0.15, split by where N came from.

## 4. Decision faithfulness (does the summary drive the verdict?)
On `in_B`, the v1 logit is recomputed with the analysis replaced, everything else fixed:
- `neutral`: fixed uninformative values; also the paired AUROC drop against generated;
- `donor`: the analysis of another submission to the same problem in the same language;
- `depth_up`: loop depth + 1 with the estimate recomputed (declared direction: logit up);
- `sort_on`: `sort_in_loop=1` (declared direction: logit up);
- placebo: the identical generated string forced again, which measures the noise floor.

For each counterfactual, against the generated v1:
- **decision-relevant:** mean absolute shift >= 0.5 logit AND flip rate >= 5%, AND, for `depth_up` and `sort_on`, mean shift in the declared direction with a CI excluding 0;
- **decorative:** mean absolute shift < 0.2 logit;
- **weak:** anything else.
A flip is a change of side at the val best-F1 threshold of that run's v1, fixed on val. Rows an edit could not change are excluded and counted. If the two seeds disagree, the report uses the lower label (decorative < weak < decision-relevant). `donor` and `neutral` set the overall label; the edits are reported beside it.
Sufficiency, descriptive: how well a problem-grouped cross-validated linear model on the generated fields reproduces the v1 logit (R2).

**Naming rule:** it is called a "checkable code summary", not "the reason for the verdict", unless BOTH `donor` and `neutral` are decision-relevant.

## 5. Sensitivity
With N x1000 in the constraints line: the generated estimate should move by max(depth, 1) x 3 and depth should stay unchanged. Reported as shares, with `none`/`huge` rows excluded.

## 6. Certificates
Descriptive only: agreement of fields and verdict with certificate outcomes on pilot rows. 364 TLE rows in 65 problems give power only for large effects, and differences between problems are confounded.

## Amendment after Stage 4 (self-check)
Stage 4 listed "batched equals single generation" as a check that fails the run, but the kernel enforced only the v1 difference, and `v1only` differed in text on at least one of 8 rows (v1 change under 0.006, likely rounding flipping a close token choice; the cause was not verified). For Stage 5 the check runs on 64 rows and fails the run if more than 5% of rows differ in text or any v1 differs by 0.2 or more; the measured share is reported.

**Self-checks that fail the run:** exact step count; reload equals the in-memory model including the heads; the check above; forced continuation equals stepwise v1; worker exit codes propagated. Abort: verdict BCE above 0.680 at step 750.
