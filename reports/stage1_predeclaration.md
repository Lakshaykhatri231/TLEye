# Pre-declaration: Stage 1, does the LoRA model respond to N and the time limit? (written BEFORE launching)

Date: 2026-11-04, plan user-approved. Kernel: khatrilakshay2004/tleye-interp (kaggle/interp/interp.py, kaggle/lib/kernel_lib.py). Model: confirmed adapter N (seed 0), scored in inference only. No training.

**Why:** the README calls the model "constraint- and time-limit-aware", but metamorphic checks have only been run on LightGBM. This tests the claim directly, and the claim gets reworded if it fails.

**Text changes (same code, one change to the prompt):** the primary size variable's bound scaled by /100, /10, x10, x100, x1000; time limit x2 and /2. Only that one variable changes; rows with no known size variable are excluded from N tests.
**Placebos (meaning unchanged):** constraint items reversed; `2s` written `2.0s`. **Noise floor:** the same rows rebatched (256 rows).
**Rows:** dose response on the first 3,000 of the confirmed run's 8,000 val rows. Test: N x10, time limit x2 and both placebos on the first 5,000 of a fixed seeded 8,000-row test sample (random_state=123), scored once.

**Reported whatever happens:** mean logit change with problem-clustered CIs, overall and by loop depth (0, 1, 2, 3+); violation rates at 0, 0.1 and 0.5 logit; the same N x10 violation rates for B2m on the same test rows.

**Claim rules (test split):**
- C1 "responds to N": at loop depth >= 2, the mean logit change for N x10 is positive and its CI excludes 0.
- C2 "beyond noise": the lower CI bound in C1 exceeds the absolute mean shift of the reorder placebo on the same rows. Both C1 and C2 are needed to keep "constraint-aware" in the README without qualification.
- C3 "responds to the time limit": the mean logit change for time limit x2 is negative and its CI excludes 0. Evidence is weak by design: 91.5% of rows have a 2 s limit, so doubling is largely outside the training distribution.
- If a rule fails, the README says so, and the wording becomes "receives the constraint text" with the measured response stated.

**Patching gate (Stage 3, val only):** N x1000 mean logit change at depth >= 2 must be at least 0.5 logit (logit SD is about 3). Time-limit patching needs an absolute mean change of at least 0.5 for time limit x2.

**Prefix-edit pre-test (val only, adapter P, descriptive):** true analysis vs neutral constant, depth +1, depth -1 (estimate recomputed coherently), and a donor from the same problem and language. It informs Stage 4 and makes no claim.

**Hidden states:** dumped for 8,000 val and 8,000 test rows, tuned and base. Test shards stay unopened until Stage 2's val choices are committed.

**Self-checks that fail the run:** adapter reproduces stored confirmatory logits on 256 rows (correlation above 0.99, max difference below 0.5); 25 hidden states per forward pass; worker exit codes propagated.

**Caveats in advance:** one seed; single-variable edits can create combinations unseen in training (for example A<=N with N changed), which is itself a limit of the test; capped-sample prevalence does not affect logit changes but does affect base rates.
