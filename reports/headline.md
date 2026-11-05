<!-- HEADLINE:START -->
**Verdict models on the held-out test split** (42,990 submissions, 296 problems; AtCoder only; problem-disjoint; 95% CI from a problem-clustered bootstrap; TLE prevalence 0.401 is an artifact of per-class capping):

| model | AUROC | AUPRC | ECE |
|---|---|---|---|
| B2 LightGBM (AST + parsed constraints) | 0.759 [0.738, 0.775] | 0.675 [0.643, 0.703] | 0.008 |
| **B2m** monotone-constrained LightGBM (default verdict model) | 0.756 [0.735, 0.774] | 0.667 [0.638, 0.697] | 0.008 |
| LoRA run 1: lr 2e-4, a recipe that failed to learn (reports/phase3_run1.md) | 0.663 [0.646, 0.679] | 0.544 [0.526, 0.563] | 0.025 |
| **Qwen2.5-Coder-0.5B + LoRA, lr 1e-4, no prefix** (confirmatory run, 40k training rows) | 0.863 [0.847, 0.881] | 0.812 [0.785, 0.839] | 0.014 |
| Qwen2.5-Coder-0.5B + LoRA, lr 1e-4, with symbolic prefix (confirmatory run) | 0.860 [0.842, 0.878] | 0.811 [0.787, 0.837] | 0.017 |

Paired test AUROC differences (problem-clustered): LoRA no-prefix minus B2m +0.107 [+0.095, +0.120]; LoRA with-prefix minus B2m +0.104 [+0.092, +0.117]; with-prefix minus no-prefix -0.003 [-0.008, +0.001] (the prefix is NOT shown to help). Both configs were fixed in advance and scored once (reports/confirm_predeclaration.md, reports/confirm_test.md); one seed each.

**Learning-rate sweep, VALIDATION only** (reports/lr_sweep.md; 12,000 training rows, lr 5e-5 to 1e-4; paired AUROC difference, LoRA minus B2m): A +0.051 [+0.030, +0.073], B +0.055 [+0.034, +0.077], C +0.060 [+0.040, +0.083]. The configs themselves are not separable (prefix B-A +0.005 [-0.001, +0.010], lr C-B +0.005 [-0.001, +0.011]). A confirmatory test-split run is pending.

**Does the LoRA model react to the constraints?** (reports/metamorphic_lora.md; rules fixed in advance in reports/stage1_predeclaration.md; same code, one change to the prompt) N x10 at loop depth >= 2 moves the logit +0.216 [+0.138, +0.294], the right direction and above placebo noise, but small against a logit spread of about 3. LoRA (seed 0): mean change +0.142, wrong-direction share at 0 / 0.1 / 0.5 logit 41.8% / 12.0% / 1.1%. B2m (monotone by construction): mean change +0.158, wrong-direction share at 0 / 0.1 / 0.5 logit 0.0% / 0.0% / 0.0%. Doubling the time limit changes the logit by +0.001 [+0.001, +0.002] (mean absolute 0.008), less than rewriting `2s` as `2.0s` (mean absolute 0.027): the model ignores the time limit.

**What can be read linearly from the model's hidden states?** (reports/probes.md; rules and sites fixed in advance in reports/stage2_predeclaration.md and on validation; 8,000 test rows, 296 problems) Decodable means a simple linear probe recovers it, not that the model uses it. Loop depth >= 2 AUROC 0.959 [0.953, 0.965], recursion 0.965 [0.960, 0.971], sort inside a loop 0.926 [0.903, 0.946], log10 N R2 0.383 [0.228, 0.464], time limit R2 -0.002 [-0.074, 0.047] (descriptive only, since 91.5% of rows share one limit; consistent with the time-limit result above). Fine-tuning made these more decodable than in the base model: log10 N +0.319 [+0.234, +0.482], sort inside a loop +0.042 [+0.014, +0.072]; it did not change loop depth (-0.002 [-0.005, +0.000]), and has-a-sort got less decodable (-0.026 [-0.034, -0.018]), no better than a bag of tokens (-0.004 [-0.015, +0.007]). The base model was never trained to summarize at the last token, so a gain there may only mean the information moved.

**Can it write a summary of the code, and does that summary drive the verdict?** (reports/sft_test.md; rules fixed in advance in reports/stage5_predeclaration.md; two seeds; each compared with the classifier of the same seed on the same rows) Verdict AUROC, explanation model vs classifier: seed 0 v0 0.858 vs 0.863 (-0.005 [-0.010, -0.001]); seed 0 v1 0.852 vs 0.858 (-0.007 [-0.012, -0.002]); seed 1 v0 0.864 vs 0.858 (+0.006 [+0.001, +0.011]); seed 1 v1 0.860 vs 0.856 (+0.004 [-0.001, +0.009]). By the rule: no meaningful loss, and not better than the classifier (the two seeds point in opposite directions). Generated fields against a real AST check, seeds 0 / 1: loop depth balanced accuracy 0.918 / 0.945 (majority guess 0.200), loop count exact 0.847 / 0.821 (majority 0.175), listed calls in a loop exact 0.787 / 0.797 (majority 0.673). **But the summary is decorative by the pre-declared labels:** replacing it on 3,000 rows moves the verdict logit only seed 0 neutral 0.136 mean absolute, 0.7% flips, seed 0 donor 0.160 mean absolute, 0.7% flips, seed 1 neutral 0.104 mean absolute, 1.0% flips, seed 1 donor 0.110 mean absolute, 0.5% flips, and a neutral summary costs no AUROC (+0.000 [-0.000, +0.001], +0.000 [-0.000, +0.001]). Labels: neutral decorative, donor decorative, depth up decorative, sort on decorative. So it is a checkable code summary, not the reason for the verdict. The trained verdict read-out before the summary may make this more likely; a design forcing the verdict through the summary was not tested on the test split.

**Seed robustness of the confirmed LoRA recipe** (reports/seeds_test.md; seeds 0, 1, 2; rule fixed in advance in reports/seeds_predeclaration.md): test AUROC seed 0 0.863, seed 1 0.858, seed 2 0.857 (spread 0.006, mean 0.860); paired difference vs B2m seed 0 +0.107 [+0.095, +0.120], seed 1 +0.102 [+0.091, +0.115], seed 2 +0.101 [+0.088, +0.115]. Three seeds only: run-to-run variation is described, not given a confidence interval.

**LoRA on an UNSEEN language** (trained on 20,000 rows of the other two languages; reports/lolo_test.md; claim rule fixed in advance in reports/lolo_predeclaration.md):

| held-out | LoRA AUROC | LightGBM AUROC (same protocol) | paired difference | LoRA ECE under shift | in-distribution LoRA AUROC |
|---|---|---|---|---|---|
| C++ | 0.790 [0.771, 0.810] | 0.707 [0.688, 0.726] | +0.084 [+0.070, +0.098] | 0.120 | 0.854 (40k rows, 3 languages) |
| Java | 0.846 [0.821, 0.868] | 0.733 [0.703, 0.761] | +0.113 [+0.094, +0.136] | 0.200 | 0.886 (40k rows, 3 languages) |
| Python | 0.775 [0.755, 0.795] | 0.693 [0.670, 0.717] | +0.082 [+0.065, +0.102] | 0.127 | 0.859 (40k rows, 3 languages) |

The ranking transfers, but calibration does not (ECE far above the in-distribution value), and the in-distribution comparison mixes 'unseen language' with 'half the training rows'.

**Prompting vs fine-tuning** (reports/b3_test.md; the same 400 fixed test items, rule fixed in advance in reports/b3_predeclaration.md; one open model, Qwen2.5-Coder-7B-Instruct, direct YES/NO answer, no chain-of-thought): prompted AUROC 0.799 [0.748, 0.850] (same information as the fine-tuned model) and 0.783 [0.734, 0.834] (plus the full statement); LoRA 0.884 [0.844, 0.918]; B2m 0.775 [0.729, 0.822]. Paired differences, LoRA minus prompted: +0.085 [+0.039, +0.128] and +0.101 [+0.057, +0.139]. Note the untrained prompted model is not distinguishable from LightGBM (B2m minus prompted A: -0.024 [-0.074, +0.028]).

**Unseen-language transfer** (B2, train on two languages, test on the third; plain B2 not B2m): C++ AUROC 0.707 [0.688, 0.726]; Java AUROC 0.733 [0.703, 0.761]; Python AUROC 0.693 [0.670, 0.717]

**Metamorphic checks** (reports/metamorphic.md; share of test rows where the property is violated, size bound x10 lowers P(TLE) / time limit x2 raises P(TLE)): plain B2 36.53% / 71.56%, B2m 0.02% / 0.00%; paired AUROC change from the constraints (B2m - B2) -0.0030 [-0.0062, +0.0012].

**Constraint reader** (100 val/test problems, labels written by the AI blind to parser output, not by a human): largest bound exact match 94/100 [87.5%, 97.2%], primary size exact match 96/100 [90.2%, 98.4%].

**Sandbox certificates** (arm64 container, CPU time >= the problem limit; only problems the generator fully supports: 44.7% of all AtCoder problems, 42.9% of test problems; intervals omitted because they would ignore problem clustering; rows are as classified when each pilot ran, and the classification was tightened after pilots 3, 5 and 6 exposed missing rules, so reports/phase4_pilot.md also gives the restricted view):

| pilot | problems | true TLE confirmed | true AC wrongly certified |
|---|---|---|---|
| pilot 1 | 24 | 91/135 = 67.4% | 0/154 |
| pilot 2 (fresh problems) | 24 | 69/128 = 53.9% | 0/140 |
| pilot 3 (development set: used to find engine bugs) | 23 | 80/152 = 52.6% | 0/159 |
| pilot 4 (fresh problems, after bounds fixes) | 29 | 104/171 = 60.8% | 0/181 |
| pilot 5 (development set: exposed missing rules) | 29 | 95/170 = 55.9% | 2/165 |
| **pilot 6 (fresh problems, final engine)** | 30 | 109/179 = 60.9% | 1/196 |

<!-- HEADLINE:END -->
