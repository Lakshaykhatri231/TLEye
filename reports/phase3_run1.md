# Phase 3, run 1: LoRA Qwen2.5-Coder-0.5B on 40,000 training rows (NEGATIVE RESULT)

Sources: reports/kaggle_train/run.json, reports/lora_vs_b2.md, reports/lora_diagnostics.txt (all script-generated).

## Setup
Qwen2.5-Coder-0.5B + LoRA (r=16, all attention+MLP projections, 8.8M trainable params), zero-initialised linear head, BCE loss,
1 epoch over a seeded 40,000-row subset of the 142,400-row train split (1,035 problems), effective batch 8, lr 2e-4 with cosine decay,
max 1024 tokens (12.8% of training prompts truncated), prompt = language, time limit, parsed constraints text, code.
Calibration: temperature scaling on an 8,000-row val subset (T = 1.385). Wall-clock 173 min on one Tesla T4 (127 min training).

## Result on the full test split (42,990 rows, 296 problems), same rows as B2
| model | AUROC | AUPRC | ECE |
|---|---|---|---|
| B2 LightGBM (AST + constraints) | 0.759 [0.739, 0.777] | 0.675 [0.645, 0.704] | 0.008 |
| LoRA 0.5B | 0.663 [0.645, 0.680] | 0.544 [0.523, 0.563] | 0.025 |
| fixed average of the two | 0.754 [0.735, 0.770] | 0.671 [0.643, 0.698] | 0.050 |

Paired difference LoRA - B2 (AUROC): -0.096 [-0.110, -0.081]; LoRA is worse in 100% of bootstrap resamples and in every language
(C++ -0.078, Java -0.142, Python -0.101). The LoRA model did learn a signal (above chance, and above the B1 rule's 0.632, not paired-tested).

## What the diagnostics rule out / suggest
- Not a truncation artefact: LoRA also loses on the shortest quartile (never truncated): 0.739 vs 0.835.
- Loses at every loop depth (0: 0.795 vs 0.863; 1: 0.631 vs 0.726; 2: 0.605 vs 0.723; 3+: 0.611 vs 0.739).
- Rank correlation between the two models is only moderate (Spearman 0.539), so they capture partly different information, yet a plain average does not beat B2.
- Training loss stayed near the constant-guess level (~0.67) until roughly step 2000 and ended near 0.60 while still falling at the final (lowest) learning rate: likely undertrained. The early-abort check passed narrowly (0.6759 vs threshold 0.680).

## Untested hypotheses (do NOT treat as findings)
1. B2 is handed est_log_ops = loop depth x log10(N) explicitly; the LoRA model must learn that numeric interaction from 40k rows.
2. One epoch / 40k rows / 0.5B / zero-init head is too little optimisation.
3. A symbolic-feature text prefix (planned in Phase 3) might close the gap.

## Not done
No LOLO runs for the LoRA model, no B3 prompted-LLM baseline, no 1.5B run. GPU time used so far: ~3 h (benchmark + dry run + this run); weekly Kaggle quota not verified.

## Addendum after run 2 aborted (2026-11-02)
Run 2 (same recipe + symbolic `analysis:` prefix) was stopped by the built-in early check at step 750 with mean loss 0.6929 (about ln 2, i.e. no learning;
constant-guess BCE is 0.671). That cost ~20 minutes of GPU time and means **run 2 produced no test result**.
reports/lr_vs_loss.md shows that in run 1 the loss only fell clearly BELOW the constant guess once the learning rate had decayed
(mean loss 0.680 at lr >= 1.5e-4 -> 0.641 at lr < 3e-5); run 2 at lr ~2e-4 stayed at 0.6885. lr and training progress are confounded
within a run, so this is suggestive only, but it adds a hypothesis that is at least as plausible as the earlier ones:
4. **The recipe (lr 2e-4, batch 8, zero-init head, fp16) was poorly conditioned**, so run 1's AUROC 0.663 may understate what 0.5B + LoRA can do.
Run 1's negative result should therefore be read as "this recipe did not beat B2", NOT "LoRA fine-tuning cannot beat structural features".

## Addendum 2 after the learning-rate sweep (validation only)
reports/lr_sweep.md: with lr 5e-5 or 1e-4 the same 0.5B LoRA recipe learns from the start and beats B2m on the same validation rows
(paired AUROC difference +0.05 to +0.06, intervals exclude 0). This supports hypothesis 4 above (run 1's recipe was poorly conditioned), but the sweep
also used fewer rows (12k) and a shorter schedule, so lr is not proven to be the only difference. Run 1's test AUROC (0.663) stays reported as is.
