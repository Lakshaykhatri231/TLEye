# Pre-declaration: Stage 4, explanation gate (written BEFORE launching; validation only)

Date: 2026-11-04, plan user-approved. Kernel: khatrilakshay2004/tleye-sft (kaggle/sft/sft.py, STAGE="gate"). The test split is not touched.

**Model:** Qwen2.5-Coder-0.5B + LoRA r=16 on the causal LM, lr 1e-4, 12,000 seeded training rows (the same draw as the sweep), effective batch 8, same prompt byte for byte through `\n### verdict:`, then a fixed template with model-filled values:
`loop_depth= loops= recursion= sorts= sort_in_loop= listed_calls_in_loop= n_var= log10_n= est_log10_ops=` and `tle:`. Values come from tree-sitter on the visible token window; unknown N stays `unknown`.
**Readouts:** v0 at the classifier's read point (no generation needed), v1 after the analysis. Verdict heads are zero-initialized linear layers. Loss: cross-entropy on value tokens and terminators only, plus verdict BCE.

**Two configs side by side, seed 0:** `dual` trains v0 and v1; `v1only` trains only v1 (so v1 cannot simply copy a trained v0).

**Reported on 3,000 val rows (generation) and 8,000 (v0):** v0 and v1 AUROC with problem-clustered CIs, v1 with true vs generated analysis, per-field accuracy against a per-language majority guess, and the training curves.

**Decision rule, fixed now:**
- Pass if at least one config's best verdict reaches val AUROC >= 0.80 AND its generated loop_depth balanced accuracy beats the per-language majority guess with a paired CI excluding 0. If neither passes, stop and report; no Stage 5.
- Stage 5 uses `dual` by default, since it alone gives a v0 readout on the full test split. It switches to `v1only` only if `dual` fails the rule above and `v1only` passes.
- Stated limit: no exact 12k classifier comparator exists at this lr. The nearest are sweep A (0.826, lr 5e-5, no prefix) and C (0.836, lr 1e-4, with prefix) on the same 8,000 rows; they are context, not a pass mark.
- The Stage 1 pre-test showed edits to analysis text barely move adapter P's verdict, so v1 is not expected to outperform v0.

**Abort rule:** verdict BCE above 0.680 at step 750 stops the run.
**Self-checks that fail the run:** exact step count; reload equals the in-memory model including the hand-added heads; batched equals single generation; forced continuation equals stepwise v1; worker exit codes propagated.
