# Pre-declaration: prompted-LLM baseline B3 (written BEFORE launching)

Date: 2026-11-03, user-approved. Kernel: khatrilakshay2004/tleye-b3 (kaggle/b3/b3.py).

**Question:** does the fine-tuned LoRA verdict model beat simply PROMPTING a strong open code model, zero-shot, with no training?

**Model:** Qwen/Qwen2.5-Coder-7B-Instruct (open weights, $0 budget), fp16 across the two T4s, chat template, ONE forward pass per item (no sampling).
**Score:** logsumexp of the next-token logits for {YES, Yes, " YES", " Yes"} minus logsumexp for {NO, No, " NO", " No"} after the question
"Will this solution exceed the time limit on the judge's worst-case tests? Answer with exactly one word: YES or NO." Higher = more TLE-like. We evaluate AUROC (threshold-free).
**Items:** the fixed 400-item test sample (140 C++ / 120 Java / 140 Python, seeded, drawn before any model result existed; column `b3_sample` of the pack).
**Two prompt variants, both reported, neither chosen on results:**
 - A (matched information): language, time limit, the same parsed `constraints:` line and code the fine-tuned model sees.
 - B (more information): A plus the full problem statement text (closer to CodeJudge-Eval-style prompting).
**Not tried (stated limitation):** chain-of-thought or few-shot prompting, which can help prompted models; larger or proprietary models. So this baseline is 'direct-answer prompting of one open 7B model', not the best possible prompting.

**Reported whatever happens:** AUROC with problem-clustered CIs for A and B; the same items scored by LoRA (confirmed run, no prefix) and by B2m; paired AUROC differences;
and the share of probability mass the model puts on YES/NO tokens (instruction following).
**Claim rule:** 'fine-tuning beats prompting' only if the paired AUROC CI (LoRA - prompted) excludes 0 for BOTH A and B. If a prompted variant matches or beats LoRA, say so.
**Caveats in advance:** n=400 so intervals are wide (about 250 problems, clustered); one open model; no CoT; the LoRA/B2m numbers on this subset can differ from their full-test numbers.

## Amendment 1 (2026-11-03, BEFORE any B3 score was seen)
The first full attempt crashed with a CUDA out-of-memory error on GPU 0 at item ~260 and saved nothing (the script only wrote results at the end).
The model, prompts, items, score and claim rule are UNCHANGED. Engineering changes only: (1) `device_map="balanced"` so both T4s hold about half the weights and keep room for long prompts;
(2) partial results are saved every 20 items; (3) if a prompt still runs out of memory it is retried once with the code cut to 4,000 characters and the statement to 2,000, and the item is flagged
`trunc` (the report states how many items, if any, were affected). The dry run now uses the 8 LONGEST prompts to exercise memory.
