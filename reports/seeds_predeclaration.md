# Pre-declaration: extra training seeds for the confirmed LoRA recipe (written BEFORE launching)

Date: 2026-11-03, user-approved ("perform all these steps"). Kernel: khatrilakshay2004/tleye-seeds (kaggle/seeds/seeds.py).

**What is repeated:** exactly the confirmed recipe (reports/confirm_predeclaration.md, config N): Qwen2.5-Coder-0.5B + LoRA r=16, lr 1e-4, NO symbolic prefix, 40,000 seeded training rows,
1 epoch, effective batch 8, max 1024 tokens, full 42,990-row test split scored once. Only the SEED changes: seed 1 and seed 2 (the original run used seed 0). The seed changes
which 40,000 training rows are drawn, the LoRA initialisation, dropout, batch order and the val sample used for the temperature. Both seeds run at the same time, one per T4.

**Reported whatever happens:** per seed (0, 1, 2): test AUROC / AUPRC / ECE with problem-clustered CIs; the spread (max minus min AUROC); paired AUROC difference vs B2m for every seed;
paired seed-vs-seed differences.
**Claim rule:** "the LoRA advantage over B2m reproduces across seeds" only if the paired AUROC CI (seed - B2m) excludes 0 for ALL THREE seeds. If any seed fails, report that seed and say the result is not robust.
Run-to-run variation is described by the three AUROCs and their range; with 3 seeds no confidence interval over seeds is claimed.
**Caveats in advance:** 3 seeds is a small sample of run-to-run variation; the same 296 test problems for every seed; capped-sample prevalence (AUPRC/ECE are not real-world rates).
