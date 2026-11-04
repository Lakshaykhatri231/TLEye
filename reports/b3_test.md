# Prompted-LLM baseline B3 (script-generated; follows reports/b3_predeclaration.md)

400 fixed test items (201 problems; 140 C++ / 120 Java / 140 Python; TLE share 0.383). Model Qwen/Qwen2.5-Coder-7B-Instruct, one forward pass per item, direct YES/NO answer (no chain-of-thought). Problem-clustered bootstrap (500). Small sample: intervals are wide.

Instruction following: mean probability mass on YES/NO tokens, variant A 1.00, variant B 1.00 (1.0 = always answers YES or NO).

| model | AUROC [95% CI] |
|---|---|
| Prompted Qwen2.5-Coder-7B-Instruct, variant A (matched info) | 0.799 [0.748, 0.850] |
| Prompted, variant B (+ problem statement) | 0.783 [0.734, 0.834] |
| B2m LightGBM | 0.775 [0.729, 0.822] |
| Fine-tuned LoRA 0.5B | 0.884 [0.844, 0.918] |

## Paired AUROC differences (first minus second), same items

| comparison | difference [95% CI] |
|---|---|
| LoRA - prompt_A | +0.085 [+0.039, +0.128] |
| LoRA - prompt_B | +0.101 [+0.057, +0.139] |
| B2m - prompt_A | -0.024 [-0.074, +0.028] |
| B2m - prompt_B | -0.008 [-0.058, +0.046] |
| LoRA - B2m | +0.109 [+0.068, +0.148] |
| prompt_B - prompt_A | -0.016 [-0.045, +0.012] |

## AUROC by language

| language | n | prompt A | prompt B | B2m | LoRA |
|---|---|---|---|---|---|
| C++ | 140 | 0.736 | 0.742 | 0.737 | 0.866 |
| Java | 120 | 0.864 | 0.861 | 0.834 | 0.932 |
| Python | 140 | 0.835 | 0.784 | 0.762 | 0.867 |

## Claim by the pre-declared rule
- **fine-tuning beats prompting (LoRA - prompted CI excludes 0 for both variants)**.
- Limits: one open 7B model with direct answers only (no chain-of-thought, no few-shot, no larger/proprietary model); n=400 from about 200 problems.
