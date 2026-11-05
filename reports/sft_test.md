# Explanation model on the test split (script-generated; follows reports/stage5_predeclaration.md)

All CIs: problem-clustered bootstrap, 95%, 300 resamples. Same-seed classifier: same recipe and training rows, no explanation.

## 1. Verdict quality

| seed | readout | rows | problems | explanation model AUROC | classifier AUROC, same rows | paired difference [95% CI] |
|---|---|---|---|---|---|---|
| s0 | v0 | 42990 | 296 | 0.858 | 0.863 | -0.005 [-0.010, -0.001] |
| s0 | v1 | 12263 | 296 | 0.852 | 0.858 | -0.007 [-0.012, -0.002] |
| s1 | v0 | 42990 | 296 | 0.864 | 0.858 | +0.006 [+0.001, +0.011] |
| s1 | v1 | 12263 | 296 | 0.860 | 0.856 | +0.004 [-0.001, +0.009] |

## 2. Accuracy against tree-sitter on the visible window

`sort_in_loop` is scored as present or absent (values above 1 are rare). Regex baselines read the full code, so they can see code the model's window cuts off. `n_var` and `log10_n` are mostly copied from the prompt's constraints line.

### Seed 0 (12263 rows, 296 problems)

| field | metric | generated | majority guess | regex | minus majority [95% CI] | minus regex [95% CI] | generated: truncated / untruncated |
|---|---|---|---|---|---|---|---|
| loop_depth | balanced accuracy | 0.918 | 0.200 | - | +0.718 [+0.698, +0.735] | - | 0.849 / 0.931 |
| recursion | balanced accuracy | 0.941 | 0.500 | - | +0.441 [+0.434, +0.448] | - | 0.897 / 0.943 |
| sort_in_loop (any) | balanced accuracy | 0.876 | 0.500 | - | +0.376 [+0.339, +0.417] | - | 0.754 / 0.894 |
| loops | exact match | 0.847 | 0.175 | 0.672 | +0.672 [+0.654, +0.687] | +0.175 [+0.162, +0.188] | 0.558 / 0.900 |
| sorts | exact match | 0.992 | 0.824 | 0.933 | +0.167 [+0.139, +0.195] | +0.059 [+0.052, +0.068] | 0.985 / 0.993 |
| listed_calls_in_loop | exact match | 0.787 | 0.673 | - | +0.114 [+0.100, +0.129] | - | 0.755 / 0.792 |
| n_var | exact match | 0.995 | 0.702 | 0.524 | +0.293 [+0.248, +0.357] | +0.471 [+0.418, +0.524] | 0.997 / 0.994 |
| log10_n | minus mean abs error | -0.012 | -1.459 | -1.908 | +1.447 [+1.230, +1.662] | +1.896 [+1.600, +2.251] | -0.001 / -0.014 |
| est_log10_ops | minus mean abs error | -0.141 | -2.760 | - | +2.620 [+2.345, +2.897] | - | -0.334 / -0.102 |

### Seed 1 (12263 rows, 296 problems)

| field | metric | generated | majority guess | regex | minus majority [95% CI] | minus regex [95% CI] | generated: truncated / untruncated |
|---|---|---|---|---|---|---|---|
| loop_depth | balanced accuracy | 0.945 | 0.200 | - | +0.745 [+0.732, +0.756] | - | 0.864 / 0.961 |
| recursion | balanced accuracy | 0.934 | 0.500 | - | +0.434 [+0.427, +0.442] | - | 0.883 / 0.941 |
| sort_in_loop (any) | balanced accuracy | 0.903 | 0.500 | - | +0.403 [+0.357, +0.436] | - | 0.787 / 0.920 |
| loops | exact match | 0.821 | 0.175 | 0.672 | +0.646 [+0.629, +0.661] | +0.148 [+0.133, +0.161] | 0.539 / 0.872 |
| sorts | exact match | 0.990 | 0.824 | 0.933 | +0.166 [+0.138, +0.194] | +0.058 [+0.051, +0.066] | 0.981 / 0.992 |
| listed_calls_in_loop | exact match | 0.797 | 0.673 | - | +0.124 [+0.109, +0.139] | - | 0.765 / 0.803 |
| n_var | exact match | 0.993 | 0.702 | 0.524 | +0.291 [+0.245, +0.357] | +0.469 [+0.416, +0.522] | 0.994 / 0.993 |
| log10_n | minus mean abs error | -0.002 | -1.459 | -1.908 | +1.457 [+1.237, +1.682] | +1.907 [+1.610, +2.259] | -0.000 / -0.002 |
| est_log10_ops | minus mean abs error | -0.126 | -2.760 | - | +2.634 [+2.346, +2.897] | - | -0.337 / -0.085 |

## 3. Self-consistency (estimate equals max(depth, 1) x log10 N within 0.15)

- Seed 0: overall 0.999 (n=10319); by source of N: N or n 0.999 (n=8940), length 1.000 (n=712), other 1.000 (n=626), unknown 1.000 (n=41).
- Seed 1: overall 0.998 (n=10328); by source of N: N or n 0.997 (n=8940), length 1.000 (n=712), other 1.000 (n=635), unknown 1.000 (n=41).

## 4. Decision faithfulness

### Seed 0

Flip threshold (val best-F1 of v1): -0.649. Rows: 3000, 296 problems.

| replacement | rows | excluded | mean shift [95% CI] | mean absolute shift | flip rate | label |
|---|---|---|---|---|---|---|
| placebo: same text forced again | 3000 | 0 | -0.000 [-0.000, +0.000] | 0.001 | 0.0% | noise floor |
| neutral | 3000 | 0 | +0.095 [+0.073, +0.123] | 0.136 | 0.7% | decorative |
| donor | 3000 | 0 | +0.082 [+0.058, +0.109] | 0.160 | 0.7% | decorative |
| depth_up | 3000 | 0 | +0.134 [+0.114, +0.158] | 0.140 | 0.4% | decorative |
| sort_on | 2955 | 45 | -0.000 [-0.001, +0.000] | 0.005 | 0.1% | decorative |

Paired AUROC drop, generated minus neutral: +0.000 [-0.000, +0.001] (generated 0.854, neutral 0.854).

Donor text was not saved, so a donor that happens to equal the generated text is not excluded; such rows pull the donor shift toward 0.

Sufficiency (descriptive): cross-validated linear model on the generated fields reproduces the v1 logit with R2 0.228.

### Seed 1

Flip threshold (val best-F1 of v1): -0.578. Rows: 3000, 296 problems.

| replacement | rows | excluded | mean shift [95% CI] | mean absolute shift | flip rate | label |
|---|---|---|---|---|---|---|
| placebo: same text forced again | 3000 | 0 | +0.000 [-0.000, +0.000] | 0.001 | 0.0% | noise floor |
| neutral | 3000 | 0 | +0.027 [+0.013, +0.043] | 0.104 | 1.0% | decorative |
| donor | 3000 | 0 | +0.043 [+0.031, +0.058] | 0.110 | 0.5% | decorative |
| depth_up | 3000 | 0 | +0.091 [+0.078, +0.106] | 0.099 | 0.4% | decorative |
| sort_on | 2946 | 54 | +0.001 [+0.001, +0.001] | 0.004 | 0.1% | decorative |

Paired AUROC drop, generated minus neutral: +0.000 [-0.000, +0.001] (generated 0.866, neutral 0.866).

Donor text was not saved, so a donor that happens to equal the generated text is not excluded; such rows pull the donor shift toward 0.

Sufficiency (descriptive): cross-validated linear model on the generated fields reproduces the v1 logit with R2 0.243.

## 5. Sensitivity

Seed 0:
N x1000 in the constraints line, 2524 rows. Depth unchanged: 0.998 (n=2524). log10 N moved by 3: 0.988 (n=2522). Estimate moved by max(depth, 1) x 3: 0.974 (n=2522). Mean v1 change +0.274.

Seed 1:
N x1000 in the constraints line, 2524 rows. Depth unchanged: 0.997 (n=2524). log10 N moved by 3: 0.974 (n=2523). Estimate moved by max(depth, 1) x 3: 0.964 (n=2523). Mean v1 change +0.312.

## 6. Certificates

### Seed 0

Descriptive only; differences between problems are confounded.

| outcome | true label | rows | problems | mean v1 logit | flagged at threshold | mean generated depth |
|---|---|---|---|---|---|---|
| certified | AC | 1 | 1 | -4.25 | 0.0% | 1.00 |
| certified | TLE | 223 | 57 | +1.82 | 82.1% | 1.53 |
| inconclusive | AC | 20 | 16 | -0.69 | 45.0% | 1.20 |
| inconclusive | TLE | 35 | 22 | +1.23 | 80.0% | 1.80 |
| not_certified | AC | 381 | 65 | -2.66 | 29.1% | 1.27 |
| not_certified | TLE | 106 | 36 | +1.29 | 82.1% | 1.80 |

### Seed 1

Descriptive only; differences between problems are confounded.

| outcome | true label | rows | problems | mean v1 logit | flagged at threshold | mean generated depth |
|---|---|---|---|---|---|---|
| certified | AC | 1 | 1 | -4.59 | 0.0% | 1.00 |
| certified | TLE | 223 | 57 | +1.65 | 82.1% | 1.56 |
| inconclusive | AC | 20 | 16 | -1.16 | 45.0% | 1.25 |
| inconclusive | TLE | 35 | 22 | +1.00 | 80.0% | 1.89 |
| not_certified | AC | 381 | 65 | -3.06 | 24.7% | 1.28 |
| not_certified | TLE | 106 | 36 | +1.20 | 75.5% | 1.81 |

## Runs

- Seed 0: 40000 training rows, 5000 steps, 259.0 minutes, peak 4.79 GB, script sha 2f5af52f3a7e. Self-checks: reload 0.0000; text differs on 0.0% of 64 rows (v1 max change 0.0117); forced vs stepwise 0.0039.
- Seed 1: 40000 training rows, 5000 steps, 269.4 minutes, peak 4.79 GB, script sha 2f5af52f3a7e. Self-checks: reload 0.0000; text differs on 1.6% of 64 rows (v1 max change 0.0112); forced vs stepwise 0.0078.

## Claims by the pre-declared rules

- **v0 verdict:** no meaningful loss against the classifier (lower bounds -0.010, +0.001; limit -0.02). Beats the classifier: no.
- **v1 verdict:** no meaningful loss against the classifier (lower bounds -0.012, -0.001). Beats the classifier: no.
- **Better than every baseline, both seeds:** loop_depth, recursion, sort_in_loop (any), loops, sorts, listed_calls_in_loop, n_var, log10_n, est_log10_ops. Not established: none.
- **Faithfulness labels (lower of the two seeds):** neutral decorative, donor decorative, depth_up decorative, sort_on decorative. **Overall: decorative.**
- **Naming:** this must be called a checkable code summary.

