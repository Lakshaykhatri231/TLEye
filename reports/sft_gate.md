# Stage 4 explanation gate (script-generated; follows reports/stage4_predeclaration.md)

VALIDATION rows only. The test split was not touched.

## Config `dual`

Rows: 8000 for v0, 3000 for generation; 148 and 148 problems. Problem-clustered bootstrap (300).

| verdict readout | rows | AUROC [95% CI] |
|---|---|---|
| v0, at the classifier's read point | 8000 | 0.836 [0.809, 0.860] |
| v0, generation rows only | 3000 | 0.840 [0.813, 0.868] |
| v1, after the model's own analysis | 3000 | 0.840 [0.813, 0.869] |
| v1, after the TRUE analysis (not available in real use) | 3000 | 0.840 [0.813, 0.869] |

| paired AUROC difference, same rows | difference [95% CI] |
|---|---|
| v1 true minus v1 generated | -0.000 [-0.000, +0.000] |
| v1 generated minus v0 | +0.000 [-0.003, +0.003] |

Generated analyses that parse as well-formed: 100.0%.

**Loop depth (the gate's field):** balanced accuracy 0.800 vs per-language majority guess 0.200, paired difference +0.600 [+0.554, +0.638]. Depths >= 4 are one class.
Plain accuracy 0.915; on truncated rows (347) 0.833, on untruncated rows (2653) 0.926.

| field | exact match, generated | exact match, per-language majority guess |
|---|---|---|
| loops | 0.619 | 0.192 |
| recursion | 0.951 | 0.810 |
| sorts | 0.985 | 0.872 |
| sort_in_loop | 0.985 | 0.972 |
| listed_calls_in_loop | 0.714 | 0.710 |
| n_var | 0.995 | 0.623 |
| log10_n | 0.979 | 0.323 |
| est_log10_ops | 0.912 | 0.226 |

Little or no gain over the majority guess (under 0.02): sort_in_loop, listed_calls_in_loop. `n_var` and `log10_n` are mostly copied from the constraints line in the prompt, so high scores there say little about understanding the code.

Mean absolute error where N is known and a number was generated: log10_n 0.02 (n=2381), est_log10_ops 0.34 (n=2381).
Self-consistency, stated estimate equals depth x log10 N within 0.15: 1.000 (n=2381).

Training curve (step: verdict BCE / value cross-entropy): 25: 0.694 / 4.548, 375: 0.606 / 0.191, 750: 0.563 / 0.141, 1125: 0.510 / 0.110, 1500: 0.504 / 0.103.
Mean gradient norm per loss on the LoRA weights: ce 2.952, b1 4.636, b0 4.806.
Self-checks: reload max change 0.0000; batched equals single generation True (v1 max change 0.0054); forced vs stepwise v1 max change 0.0039; steps 1500/1500.
Run: 12000 training rows, 56.9 minutes, peak 4.57 GB, script sha 925bb19220c3.

## Config `v1only`

Rows: 8000 for v0, 3000 for generation; 148 and 148 problems. Problem-clustered bootstrap (300).

| verdict readout | rows | AUROC [95% CI] |
|---|---|---|
| v1, after the model's own analysis | 3000 | 0.832 [0.803, 0.860] |
| v1, after the TRUE analysis (not available in real use) | 3000 | 0.832 [0.803, 0.860] |

| paired AUROC difference, same rows | difference [95% CI] |
|---|---|
| v1 true minus v1 generated | -0.000 [-0.001, +0.000] |

Generated analyses that parse as well-formed: 100.0%.

**Loop depth (the gate's field):** balanced accuracy 0.869 vs per-language majority guess 0.200, paired difference +0.669 [+0.629, +0.697]. Depths >= 4 are one class.
Plain accuracy 0.939; on truncated rows (347) 0.847, on untruncated rows (2653) 0.951.

| field | exact match, generated | exact match, per-language majority guess |
|---|---|---|
| loops | 0.707 | 0.192 |
| recursion | 0.952 | 0.810 |
| sorts | 0.985 | 0.872 |
| sort_in_loop | 0.987 | 0.972 |
| listed_calls_in_loop | 0.716 | 0.710 |
| n_var | 0.995 | 0.623 |
| log10_n | 0.985 | 0.323 |
| est_log10_ops | 0.927 | 0.226 |

Little or no gain over the majority guess (under 0.02): sort_in_loop, listed_calls_in_loop. `n_var` and `log10_n` are mostly copied from the constraints line in the prompt, so high scores there say little about understanding the code.

Mean absolute error where N is known and a number was generated: log10_n 0.00 (n=2369), est_log10_ops 0.28 (n=2369).
Self-consistency, stated estimate equals depth x log10 N within 0.15: 0.999 (n=2369).

Training curve (step: verdict BCE / value cross-entropy): 25: 0.694 / 4.542, 375: 0.576 / 0.181, 750: 0.553 / 0.132, 1125: 0.529 / 0.094, 1500: 0.486 / 0.089.
Mean gradient norm per loss on the LoRA weights: ce 2.515, b1 3.836.
**Deviation from the pre-declaration:** batched and one-at-a-time generation produced different text on at least one of 8 rows. The pre-declaration lists this as a check that fails the run, but the kernel enforced only the v1 difference, so the run continued.

Self-checks: reload max change 0.0000; batched equals single generation False (v1 max change 0.0049); forced vs stepwise v1 max change 0.0039; steps 1500/1500.
Run: 12000 training rows, 57.8 minutes, peak 4.56 GB, script sha 925bb19220c3.

## Decision by the pre-declared rule

Pass requires best verdict AUROC >= 0.8 and a loop-depth paired CI above 0.

- `dual`: best verdict AUROC 0.840, loop-depth difference lower bound +0.554: **pass**.
- `v1only`: best verdict AUROC 0.832, loop-depth difference lower bound +0.629: **pass**.

**Stage 5 config: `dual`.**

Context, not a pass mark: sweep A 0.826 (lr 5e-5, no prefix) and C 0.836 (lr 1e-4, with prefix) on the same 8,000 val rows, classifiers trained on the same 12,000 rows.
