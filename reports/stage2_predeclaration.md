# Pre-declaration: Stage 2, linear probes (written BEFORE any hidden state was analysed)

Date: 2026-11-04, plan user-approved. Kernel: khatrilakshay2004/tleye-probes (kaggle/probes/probes.py, kaggle/lib/probe_lib.py), CPU only. Inputs: hidden states from Stage 1 (8,000 val and 8,000 test rows; tuned adapter N and untuned base model).

**Question:** which code facts can a linear model read from the model's internal state, and did fine-tuning change that? Probes show what is linearly decodable, NOT what the model uses. All wording says "linearly decodable".

**Targets (computed on the visible code window):** depth_ge2, recursive, has_sort, sort_in_loop, listed_any (binary, code-level); loops_log1p, est (numeric, est clipped at 30, rows with known N); log10_n (numeric, problem-level, known N). Pre-declared family for multiple comparisons: these 8. Also reported, outside the family: language is Python, TLE label, log time limit (too few effective problems; descriptive only).

**Fitting:** ridge on standardized features; alpha chosen by 5-fold CV with folds grouped by problem; top 50 principal components for the problem-level targets log10_n and est. Scores: AUROC for binary, R^2 for numeric; problem-demeaned correlation also for code-level targets, because it removes problem identity.

**Sites:** last token at layers 4, 8, 12, 16, 20, 24 and mean-pooled at layers 12, 24, separately for tuned and base.
**Process:** Phase 1 uses val only: choose one site per target per model, write val_choices.json, COMMIT it. Phase 2 then opens test shards for the first time, fits on all val rows at the chosen settings, and scores test once. Other layers stay descriptive and val only.

**Controls:**
- Shuffled target: within problem for code-level targets; across problems for log10_n and est (they barely vary within a problem), scored on test against true labels. Should be near chance.
- Metadata baseline: log prompt length, truncated flag, language. Bag-of-tokens baseline: mean-pooled layer 0 of the base model.

**Claims (test split, problem-clustered bootstrap, 500 resamples):**
- "Linearly decodable beyond metadata": the paired CI of (tuned probe minus metadata baseline) excludes 0 and is positive. For code-level targets, the problem-demeaned correlation's CI must also exclude 0.
- "Beyond bag of tokens" is reported the same way, separately. A concept that is only as decodable as token counts is not evidence of more than token presence.
- "Fine-tuning changed decodability": Holm-adjusted p < 0.05 over the 8-target family on the paired tuned-minus-base difference, reported in either direction. Base was never trained to summarize at the final `:`, so a tuned gain at the last token may only mean information moved there; this caveat is stated with the result.

**Pipeline tests, not findings:** last-token layer 0 has zero variance; the tuned layer 24 TLE probe tracks the model's logit (it reads the same state); hidden state 24 is post-norm while others are pre-norm, so a jump there is not interpreted.

**Caveats in advance:** 148 val problems to fit, 296 test problems at most; one seed; capped-sample prevalence affects binary base rates, not AUROC.
