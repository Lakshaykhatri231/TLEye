"""B3: zero-shot PROMPTED baseline (pre-declared in reports/b3_predeclaration.md). Qwen2.5-Coder-7B-Instruct, one forward pass per item,
score = logsumexp(YES-variants) - logsumexp(NO-variants) at the first answer token. Two prompt variants: A matched info, B + problem statement.
Nothing from the dataset is executed; code is only placed in a prompt."""
import glob, hashlib, json, math, os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DRY = False                     # True = 8 items; False = all 400
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_CODE_CHARS = 12000
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


df = pd.read_parquet(glob.glob("/kaggle/input/**/b3_items.parquet", recursive=True)[0])
if DRY:   # the 8 LONGEST prompts, to exercise GPU memory
    df = df.assign(_len=df.code.str.len().clip(upper=12000) + df.statement.str.len()).nlargest(8, "_len").drop(columns="_len").reset_index(drop=True)
log("items", len(df), "| problems", df.problem_id.nunique(), "| gpus", torch.cuda.device_count())
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="balanced", max_memory={0: "11GiB", 1: "11GiB"}).eval()
log("model loaded", {str(k): v for k, v in list(getattr(model, "hf_device_map", {}).items())[:2]}, "...")
ids = lambda words: sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})  # noqa: E731
YES, NO = ids(["YES", "Yes", " YES", " Yes"]), ids(["NO", "No", " NO", " No"])
log("yes ids", YES, "no ids", NO)
SYS = "You are an expert competitive programmer acting as an automated judge."
Q = "Will this solution exceed the time limit on the judge's worst-case tests? Answer with exactly one word: YES or NO."


def user_msg(r, with_statement, short=False):
    code, stmt = (r.code[:4000], r.statement[:2000]) if short else (r.code[:MAX_CODE_CHARS], r.statement)
    parts = []
    if with_statement:
        parts.append(f"Problem statement:\n{stmt}\n")
    parts.append(f"Language: {r.rt}\nTime limit: {r.time_limit_ms / 1000:g}s\nInput-size bounds parsed from the statement: {r.constraints}\n")
    parts.append(f"Solution code:\n```\n{code}\n```\n")
    parts.append(Q)
    return "\n".join(parts)


@torch.no_grad()
def score(r, with_statement, short=False):
    text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user_msg(r, with_statement, short)}],
                                   tokenize=False, add_generation_prompt=True)
    x = tok(text, return_tensors="pt").to(model.device)
    logits = model(**x, logits_to_keep=1).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    y, n = torch.logsumexp(lp[YES], 0), torch.logsumexp(lp[NO], 0)
    return float(y - n), float(torch.exp(y) + torch.exp(n)), int(x["input_ids"].shape[1])


os.makedirs("/kaggle/working", exist_ok=True)
out = {k: [] for k in ["sA", "mA", "tA", "sB", "mB", "tB", "trunc"]}


def save_partial():
    n = len(out["sA"])
    np.savez("/kaggle/working/b3_scores.npz", submission_id=df.submission_id.values.astype(str)[:n], y=df.y_tle.values[:n], **{k: np.array(v) for k, v in out.items()})


for i, r in enumerate(df.itertuples()):
    tr = 0
    for tag, ws in (("A", False), ("B", True)):
        try:
            s, m, t = score(r, ws)
        except torch.OutOfMemoryError:           # rare very long prompt: retry once trimmed, and record it
            torch.cuda.empty_cache(); s, m, t = score(r, ws, short=True); tr = 1
        out["s" + tag].append(s); out["m" + tag].append(m); out["t" + tag].append(t)
    out["trunc"].append(tr)
    if i % 20 == 0:
        log(f"item {i}/{len(df)}  scoreA {out['sA'][-1]:+.2f} massA {out['mA'][-1]:.2f} tokensB {out['tB'][-1]}  trunc so far {sum(out['trunc'])}")
        save_partial()
save_partial()
json.dump({"model": MODEL, "dry": DRY, "n": len(df), "minutes": round((time.time() - T0) / 60, 1),
           "mean_yes_no_mass_A": float(np.mean(out["mA"])), "mean_yes_no_mass_B": float(np.mean(out["mB"])), "n_trunc_retries": int(sum(out["trunc"])),
           "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}, open("/kaggle/working/run.json", "w"), indent=1)
log("DONE")
