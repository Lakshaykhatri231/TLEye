# Pre-declaration for the confirmatory LoRA runs (written BEFORE launching; test split not yet scored by any tuned config)

Date: 2026-11-02 (before launch). Kernel: khatrilakshay2004/tleye-confirm (kaggle/confirm/confirm.py), user-approved.

**Configs (fixed):** Qwen2.5-Coder-0.5B + LoRA r=16, lr 1e-4, 40,000 seeded training rows, 1 epoch, effective batch 8, max 1024 tokens.
P = with the symbolic `analysis:` prefix, N = without. Both scored ONCE on the full test split (42,990 rows, 296 problems); temperature T fit on the 8,000-row val sample.

**What will be reported, whatever happens:**
1. Test AUROC / AUPRC / ECE with problem-clustered CIs for P and N, next to B2 and B2m (same rows).
2. Paired AUROC differences: P - B2m, N - B2m, P - N (problem-clustered bootstrap).
3. Per-language test AUROC.
No config is chosen using test scores. The early-abort check (loss > 0.680 at step 750) stays on; an aborted run is reported as aborted.

**How the claim will be worded:** "LoRA beats B2m on test" only if the paired CI of (P - B2m) AND (N - B2m) each exclude 0; if only one does, say so;
if neither does, report the validation result as not confirmed. The prefix is claimed useful only if the paired CI of (P - N) excludes 0.

**Known caveats stated in advance:** one seed per config (no run-to-run variance estimate); 296 test problems; capped-sample prevalence (AUPRC/ECE not real-world);
lr and prefix were chosen from a 3-config sweep on validation, so validation numbers are mildly optimistic.
