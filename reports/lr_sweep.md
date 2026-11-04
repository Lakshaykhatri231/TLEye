# Learning-rate sweep (script-generated; VALIDATION only)

8000 validation rows from 148 problems, identical for every model. 12,000 training rows, 1,500 optimizer steps each. Intervals: problem-clustered bootstrap (300). The test split was NOT scored by these runs.

| model | learning rate | symbolic prefix | val AUROC [95% CI] | val BCE after temperature | training minutes |
|---|---|---|---|---|---|
| LoRA A | 5e-05 | False | 0.826 [0.798, 0.850] | 0.489 (constant guess 0.661) | 47 |
| LoRA B | 5e-05 | True | 0.831 [0.805, 0.854] | 0.485 (constant guess 0.661) | 51 |
| LoRA C | 0.0001 | True | 0.836 [0.811, 0.860] | 0.480 (constant guess 0.661) | 48 |
| B2 | - | - | 0.781 [0.748, 0.810] | - | - |
| B2m | - | - | 0.775 [0.743, 0.802] | - | - |

## Paired AUROC differences (first minus second), same rows, problems resampled together

| comparison | difference [95% CI] |
|---|---|
| A - B2m | +0.051 [+0.030, +0.073] |
| B - B2m | +0.055 [+0.034, +0.077] |
| C - B2m | +0.060 [+0.040, +0.083] |
| C - B2 | +0.055 [+0.031, +0.081] |
| B - A | +0.005 [-0.001, +0.010] |
| C - B | +0.005 [-0.001, +0.011] |

## Validation AUROC by language

| language | n | A | B | C | B2 | B2m |
|---|---|---|---|---|---|---|
| C++ | 3592 | 0.822 | 0.829 | 0.829 | 0.766 | 0.764 |
| Java | 1835 | 0.841 | 0.845 | 0.858 | 0.800 | 0.799 |
| Python | 2573 | 0.809 | 0.811 | 0.821 | 0.767 | 0.757 |

## Reading it
- Configs A, B, C use identical data/steps and differ only in learning rate and the prefix, so their pairwise differences (last two rows above) say how much each choice matters; if the interval contains 0, the sweep cannot separate them.
- A,B,C selected on validation among 3 options: the best of them is mildly optimistic. The confirmatory number must come from a longer run scored once on the TEST split.
- Run 1 (lr 2e-4, 40k rows) reached only 0.663 on test; here lr 5e-5 to 1e-4 learns clearly on 12k rows. Validation is easier than test for B2 as well (0.776 vs 0.759).
