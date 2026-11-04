# Seed robustness of the confirmed LoRA recipe (script-generated; follows reports/seeds_predeclaration.md)

42990 test rows, 296 problems. Same recipe (lr 1e-4, no prefix, 40k rows); only the seed differs. Problem-clustered bootstrap (300).

| run | temperature T | AUROC [95% CI] | AUPRC [95% CI] | ECE [95% CI] |
|---|---|---|---|---|
| seed 0 | 1.290 | 0.863 [0.847, 0.881] | 0.812 [0.785, 0.839] | 0.014 [0.010, 0.023] |
| seed 1 | 1.403 | 0.858 [0.841, 0.876] | 0.808 [0.782, 0.833] | 0.016 [0.011, 0.028] |
| seed 2 | 1.368 | 0.857 [0.841, 0.875] | 0.806 [0.782, 0.833] | 0.010 [0.008, 0.021] |
| B2m (reference) | - | 0.756 [0.735, 0.774] | 0.667 [0.638, 0.697] | 0.008 [0.008, 0.021] |

AUROC across seeds: min 0.857, max 0.863, spread (max - min) 0.006, mean 0.860. Three seeds only: no interval over seeds is claimed.

## Paired AUROC differences (first minus second), same rows

| comparison | difference [95% CI] |
|---|---|
| seed 0 - B2m | +0.107 [+0.095, +0.120] |
| seed 1 - B2m | +0.102 [+0.091, +0.115] |
| seed 2 - B2m | +0.101 [+0.088, +0.115] |
| seed 1 - seed 0 | -0.005 [-0.010, +0.000] |
| seed 2 - seed 0 | -0.006 [-0.011, -0.001] |
| seed 2 - seed 1 | -0.002 [-0.007, +0.004] |

## AUROC by language

| language | n | seed 0 | seed 1 | seed 2 | B2m |
|---|---|---|---|---|---|
| C++ | 19779 | 0.854 | 0.850 | 0.845 | 0.742 |
| Java | 9759 | 0.886 | 0.876 | 0.877 | 0.776 |
| Python | 13452 | 0.859 | 0.855 | 0.856 | 0.753 |

## Claim by the pre-declared rule
- **The LoRA advantage over B2m reproduces across all three seeds (every paired CI excludes 0)**.
- Run-to-run spread in AUROC: 0.006 (three seeds).
