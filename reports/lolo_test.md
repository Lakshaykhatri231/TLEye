# LoRA unseen-language (leave-one-language-out) test result (script-generated; follows reports/lolo_predeclaration.md)

Each run: 20,000 training rows from the OTHER two languages, T fit on their val rows, FULL test rows of the held-out language scored once. lr 1e-4, no prefix, one seed. Problem-clustered bootstrap (300). In-distribution LoRA used 40,000 rows of all three languages, so the drop mixes 'unseen language' with 'less data'.

| held-out | n | problems | LoRA LOLO AUROC [CI] | LightGBM LOLO AUROC [CI] | paired LoRA - LightGBM | LoRA AUPRC | LoRA ECE | in-distribution LoRA AUROC (40k rows, 3 langs) |
|---|---|---|---|---|---|---|---|---|
| C++ | 19779 | 293 | 0.790 [0.771, 0.810] | 0.707 [0.688, 0.726] | +0.084 [+0.070, +0.098] | 0.752 | 0.120 | 0.854 (T=1.347) |
| Java | 9759 | 198 | 0.846 [0.821, 0.868] | 0.733 [0.703, 0.761] | +0.113 [+0.094, +0.136] | 0.755 | 0.200 | 0.886 (T=1.245) |
| Python | 13452 | 240 | 0.775 [0.755, 0.795] | 0.693 [0.670, 0.717] | +0.082 [+0.065, +0.102] | 0.718 | 0.127 | 0.859 (T=1.370) |

## Claim by the pre-declared rule
- Per language, LoRA transfers better than LightGBM only where the paired CI excludes 0: C++: YES; Java: YES; Python: YES.
- Overall statement ('LoRA transfers to an unseen language better than LightGBM'): **supported for all three languages**.
- Calibration: T was fit on the training languages, so ECE on the held-out language reflects calibration under language shift.
