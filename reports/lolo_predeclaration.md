# Pre-declaration: LoRA unseen-language (leave-one-language-out) runs (written BEFORE launching)

Date: 2026-11-02, user-approved. Kernel: khatrilakshay2004/tleye-lolo (kaggle/lolo/lolo.py).

**Design (fixed):** three runs. For held-out language H in {C++, Java, Python}: train on 20,000 seeded TRAIN rows of the OTHER two languages only; the
8,000-row val sample (also only from the other two languages) is used only to fit the temperature T; then the FULL test rows of H (and only H) are scored once.
Recipe = the confirmed one: Qwen2.5-Coder-0.5B + LoRA r=16, lr 1e-4, no symbolic prefix, effective batch 8, max 1024 tokens, one epoch, one seed.
The language name appears only as text ("language: Java") so the model has never seen the held-out language token in training. Held-out Python includes PyPy rows.

**Comparator:** plain B2 LightGBM trained the same way (two languages, test on the third), per-row predictions in data/processed/b2_lolo_test_preds.parquet (AUROC is unaffected by its Platt scaling).

**Reported whatever happens:** per held-out language: LoRA AUROC/AUPRC/ECE with problem-clustered CIs; paired AUROC difference vs the B2 LOLO comparator;
and, for context, the in-distribution LoRA numbers from reports/confirm_test.md.

**Claim rule:** "LoRA transfers to an unseen language better than LightGBM" per language only if the paired AUROC CI (LoRA_LOLO - B2_LOLO) excludes 0 for that language;
the overall statement needs all three. If a language's CI contains 0 or is negative, say so for that language.

**Caveats stated in advance:** (1) LOLO runs use 20,000 rows vs 40,000 in the in-distribution run and two languages instead of three, so any drop from the
in-distribution AUROC mixes "unseen language" with "less data"; no clean attribution is claimed. (2) One seed per run. (3) T is fit on the training languages, so calibration (ECE)
on the held-out language is expected to be worse than in-distribution. (4) Capped-sample prevalence: AUPRC/ECE are not real-world rates.
