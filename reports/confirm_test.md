# Confirmatory test-split result (script-generated; follows reports/confirm_predeclaration.md)

42990 test rows, 296 problems, prevalence 0.401 (capped-sample artifact). P = lr 1e-4 WITH symbolic prefix (T=1.285), N = lr 1e-4 WITHOUT (T=1.290). Problem-clustered bootstrap, 300 resamples. One seed per config.

| model | AUROC [95% CI] | AUPRC [95% CI] | ECE [95% CI] | Brier |
|---|---|---|---|---|
| B2 LightGBM | 0.759 [0.738, 0.775] | 0.675 [0.643, 0.703] | 0.008 [0.008, 0.021] | 0.1928 |
| B2m monotone LightGBM | 0.756 [0.735, 0.774] | 0.667 [0.638, 0.697] | 0.008 [0.008, 0.021] | 0.1939 |
| LoRA N (no prefix) | 0.863 [0.847, 0.881] | 0.812 [0.785, 0.839] | 0.014 [0.010, 0.023] | 0.1475 |
| LoRA P (prefix) | 0.860 [0.842, 0.878] | 0.811 [0.787, 0.837] | 0.017 [0.012, 0.029] | 0.1491 |

## Paired differences (first minus second), same rows

| comparison | metric | difference [95% CI] |
|---|---|---|
| P - B2m | AUROC | +0.104 [+0.094, +0.116] |
| P - B2m | AUPRC | +0.143 [+0.125, +0.163] |
| N - B2m | AUROC | +0.107 [+0.095, +0.120] |
| N - B2m | AUPRC | +0.144 [+0.123, +0.166] |
| P - N | AUROC | -0.003 [-0.008, +0.001] |
| P - N | AUPRC | -0.001 [-0.007, +0.005] |
| P - B2 | AUROC | +0.101 [+0.090, +0.113] |
| P - B2 | AUPRC | +0.135 [+0.116, +0.157] |
| N - B2 | AUROC | +0.104 [+0.093, +0.117] |
| N - B2 | AUPRC | +0.136 [+0.115, +0.162] |

## Per-language test AUROC (paired difference vs B2m in brackets)

| language | n | B2m | N | P |
|---|---|---|---|---|
| C++ | 19779 | 0.742 | 0.854 (+0.112 [+0.099, +0.124]) | 0.851 (+0.109 [+0.096, +0.120]) |
| Java | 9759 | 0.776 | 0.886 (+0.109 [+0.087, +0.133]) | 0.880 (+0.104 [+0.084, +0.126]) |
| Python | 13452 | 0.753 | 0.859 (+0.106 [+0.084, +0.125]) | 0.859 (+0.106 [+0.084, +0.124]) |

## Claim by the pre-declared rule
- Verdict model: **LoRA beats B2m on the test split (both configs' paired CIs exclude 0)**.
- Prefix: NOT claimed: the P - N paired CI contains 0, so this run cannot say whether the prefix helps.
