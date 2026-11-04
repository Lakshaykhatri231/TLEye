"""STAGE 1 (pre-declared in reports/stage1_predeclaration.md): LoRA metamorphic checks, prefix-edit pre-test, hidden-state dump.
Two workers: `tuned` (confirmed adapter N) on one T4; `base` (untuned model, then adapter P prefix test) on the other. Inference only; dataset code is only tokenized."""
import glob, json, os, subprocess, sys, time

DRY = False
ROLE = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else None
OUT = "/kaggle/working"
if ROLE is None:                 # ---------- LAUNCHER: exit codes are propagated, so a crashed worker fails the kernel ----------
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)  # peft/torchao clash on Kaggle
    def launch(role, gpu):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--role", role], env=env, stdout=open(f"{OUT}/log_{role}.txt", "w"), stderr=subprocess.STDOUT)
    a, b = launch("tuned", 0), launch("base", 1)
    rc = {"tuned": a.wait(), "base": b.wait()}
    print(rc, flush=True)
    for n in rc:
        print(f"----- log_{n} (tail) -----"); print("".join(open(f"{OUT}/log_{n}.txt").readlines()[-8:]))
    sys.exit(max(rc.values()))

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/kernel_lib.py", recursive=True)[0]))
import kernel_lib as kl

MODEL = "Qwen/Qwen2.5-Coder-0.5B"
T0 = time.time()
LAYERS, POOL_LAYERS = [0, 4, 8, 12, 16, 20, 24], [0, 12, 24]
N_META_VAL, N_META_TEST, N_HID, N_PREFIX = (48, 48, 64, 32) if DRY else (3000, 5000, 8000, 2000)
SHARD = 16 if DRY else 1000
os.makedirs(f"{OUT}/meta", exist_ok=True); os.makedirs(f"{OUT}/hidden", exist_ok=True)


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]  # noqa: E731
df = pd.read_parquet(find("tleye_data.parquet"))
side = pd.read_parquet(find("sft_targets.parquet"))[["submission_id", "n_var", "vis_loop_depth", "truncated"]]
df = df.merge(side, on="submission_id", how="left", validate="one_to_one")
assert len(df) == 206472 and df.n_var.notna().all(), "sidecar did not join on every row"
VAL = df[df.split == "val"].sample(8000, random_state=0).reset_index(drop=True)            # the same 8,000 rows the confirmed run used
TEST = df[df.split == "test"].sample(8000, random_state=123).reset_index(drop=True)        # fixed before any result
if DRY:                          # include the LONGEST prompts to exercise memory
    pick = lambda d: pd.concat([d.assign(_l=d.code.str.len()).nlargest(16, "_l").drop(columns="_l"), d.head(N_HID)]).drop_duplicates("submission_id").reset_index(drop=True)  # noqa: E731
    VAL, TEST = pick(VAL), pick(TEST)
log("val", len(VAL), "test", len(TEST), "| role", ROLE)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
SUF = tok(kl.SUFFIX, add_special_tokens=False)["input_ids"]


def load(adapter=None):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float16)
    m.config.pad_token_id = tok.pad_token_id
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    return m.cuda().eval()


def collate(seqs):
    m_, pad = max(map(len, seqs)), tok.pad_token_id
    x = torch.full((len(seqs), m_), pad, dtype=torch.long); att = torch.zeros((len(seqs), m_), dtype=torch.long)
    for i, e in enumerate(seqs):
        x[i, :len(e)] = torch.tensor(e); att[i, :len(e)] = 1
    return x.cuda(), att.cuda()


@torch.no_grad()
def logits(model, texts, bs=16):
    ids = kl.encode(tok, texts, SUF)
    order = np.argsort([-len(e) for e in ids]); out = np.zeros(len(ids), np.float32)
    for i in range(0, len(ids), bs):
        sel = order[i:i + bs]; x, att = collate([ids[j] for j in sel])
        with torch.autocast("cuda", dtype=torch.float16):
            out[sel] = model(input_ids=x, attention_mask=att).logits.squeeze(-1).float().cpu().numpy()
    return out


def base_texts(d):
    return [kl.prompt(r.rt, kl.tl_text(r.time_limit_ms), r.constraints, r.code) for r in d.itertuples()]


@torch.no_grad()
def dump_hidden(model, d, name, want_logit):
    """Shards of last-token states at LAYERS and mean-pooled states at POOL_LAYERS, in row order."""
    ids = kl.encode(tok, base_texts(d), SUF); row = 0
    for s in range(0, len(d), SHARD):
        idx = list(range(s, min(s + SHARD, len(d))))
        order = sorted(idx, key=lambda j: -len(ids[j])); last, pool, lg = {}, {}, {}
        for b in range(0, len(order), 16):
            sel = order[b:b + 16]; x, att = collate([ids[j] for j in sel])
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(input_ids=x, attention_mask=att, output_hidden_states=True)
            hs = o.hidden_states
            assert len(hs) == 25, f"expected 25 hidden states, got {len(hs)}"
            pos = att.sum(1) - 1
            for k, j in enumerate(sel):
                last[j] = np.stack([hs[L][k, pos[k]].float().cpu().numpy() for L in LAYERS]).astype(np.float16)
                pool[j] = np.stack([(hs[L][k].float() * att[k].unsqueeze(-1).float()).sum(0).cpu().numpy() / float(att[k].sum()) for L in POOL_LAYERS]).astype(np.float16)
                lg[j] = float(o.logits[k].float().cpu().item()) if want_logit else 0.0
        kl.atomic_savez(f"{OUT}/hidden/{name}_{s:06d}.npz", submission_id=d.submission_id.values[idx].astype(str), last=np.stack([last[j] for j in idx]),
                        pool=np.stack([pool[j] for j in idx]), logit=np.array([lg[j] for j in idx], np.float32))
        row += len(idx); log(f"  hidden {name} {row}/{len(d)}")


check = {"role": ROLE, "dry": DRY}
if ROLE == "tuned":
    m = load(os.path.dirname(find("cfgN/adapter/adapter_config.json")))
    # ---- self-check 1: reproduces the stored confirmatory logits
    z = np.load(find("cfgN/test_logits.npz")); kag = pd.Series(z["logit"], index=z["submission_id"])
    chk = df[df.split == "test"].sample(32 if DRY else 256, random_state=7).reset_index(drop=True)
    now = logits(m, base_texts(chk)); ref = kag.loc[chk.submission_id].values
    check["parity_max_abs"] = float(np.abs(now - ref).max()); check["parity_corr"] = float(np.corrcoef(now, ref)[0, 1])
    log("parity", check["parity_max_abs"], check["parity_corr"])
    assert check["parity_corr"] > 0.99 and check["parity_max_abs"] < 0.5, "adapter does not reproduce the confirmatory logits"
    # ---- self-check 2 / noise floor: same rows, different batching
    single = logits(m, base_texts(chk), bs=1); check["noise_rebatch_median"] = float(np.median(np.abs(single - now))); check["noise_rebatch_p99"] = float(np.percentile(np.abs(single - now), 99))
    kl.atomic_savez(f"{OUT}/meta/noise.npz", submission_id=chk.submission_id.values.astype(str), a=now, b=single)
    kl.atomic_json(f"{OUT}/selfcheck_tuned.json", check)

    def run_variants(d, split, variants):
        kl.atomic_savez(f"{OUT}/meta/{split}_base.npz", submission_id=d.submission_id.values.astype(str), logit=logits(m, base_texts(d)))
        for name, fn in variants.items():
            keep, txt = [], []
            for i, r in enumerate(d.itertuples()):
                t = fn(r)
                if t is not None:
                    keep.append(i); txt.append(t)
            kl.atomic_savez(f"{OUT}/meta/{split}_{name}.npz", submission_id=d.submission_id.values[keep].astype(str), logit=logits(m, txt) if txt else np.zeros(0))
            log(f"  {split} {name}: {len(keep)} rows")

    def n_var(f):
        def g(r):
            if r.n_var == "unknown":
                return None
            c = kl.scale_var(r.constraints, r.n_var, f)
            return None if c is None else kl.prompt(r.rt, kl.tl_text(r.time_limit_ms), c, r.code)
        return g

    def tl_var(f, placebo=False):
        return lambda r: kl.prompt(r.rt, kl.tl_text(r.time_limit_ms, f, placebo), r.constraints, r.code)

    def reorder(r):
        c = kl.reorder_items(r.constraints)
        return None if c is None else kl.prompt(r.rt, kl.tl_text(r.time_limit_ms), c, r.code)

    placebos = {"placebo_reorder": reorder, "placebo_tl": tl_var(1.0, True)}
    run_variants(VAL.head(N_META_VAL), "val", {**{f"n_{k}": n_var(f) for k, f in {"div100": .01, "div10": .1, "x10": 10, "x100": 100, "x1000": 1000}.items()},
                                               "tl_x2": tl_var(2), "tl_div2": tl_var(.5), **placebos})
    run_variants(TEST.head(N_META_TEST), "test", {"n_x10": n_var(10), "tl_x2": tl_var(2), **placebos})
    dump_hidden(m, VAL.head(N_HID + 16 if DRY else N_HID), "tuned_val", True)
    dump_hidden(m, TEST.head(N_HID + 16 if DRY else N_HID), "tuned_test", True)
else:
    m = load()
    dump_hidden(m, VAL.head(N_HID + 16 if DRY else N_HID), "base_val", False)
    dump_hidden(m, TEST.head(N_HID + 16 if DRY else N_HID), "base_test", False)
    del m; torch.cuda.empty_cache()
    # ---- prefix-edit pre-test on adapter P (VALIDATION ONLY)
    p = load(os.path.dirname(find("cfgP/adapter/adapter_config.json")))
    d = VAL.head(N_PREFIX).reset_index(drop=True)
    txt = lambda sy: [kl.prompt(r.rt, kl.tl_text(r.time_limit_ms), r.constraints, r.code, analysis=s) for r, s in zip(d.itertuples(), sy)]  # noqa: E731
    neutral = "loop_depth=1 loops=1 recursion=no sorts=0 sort_in_loop=0 linear_ops_in_loop=0 est_log10_ops=5.0 budget_score=5.0"

    def edit_depth(s, delta):
        kv = dict(x.split("=") for x in s.split()); dep = int(kv["loop_depth"]); new = max(0, dep + delta)
        lg = float(kv["est_log10_ops"]) / max(dep, 1); gap = float(kv["est_log10_ops"]) - float(kv["budget_score"]); est = max(new, 1) * lg
        kv.update(loop_depth=str(new), est_log10_ops=f"{est:.1f}", budget_score=f"{est - gap:.1f}")
        return " ".join(f"{k}={v}" for k, v in kv.items())

    pool_val = df[df.split == "val"]
    rng = np.random.default_rng(0); donors = []
    groups = {k: g.sym.values for k, g in pool_val.groupby(["problem_id", "language"])}
    for r in d.itertuples():
        c = [s for s in groups[(r.problem_id, r.language)] if s != r.sym]
        donors.append(c[int(rng.integers(len(c)))] if c else r.sym)
    out = {"true": logits(p, txt(d.sym.tolist())), "neutral": logits(p, txt([neutral] * len(d))), "depth_up": logits(p, txt([edit_depth(s, 1) for s in d.sym])),
           "depth_down": logits(p, txt([edit_depth(s, -1) for s in d.sym])), "donor": logits(p, txt(donors))}
    kl.atomic_savez(f"{OUT}/meta/prefix_test.npz", submission_id=d.submission_id.values.astype(str), has_donor=np.array([a != b for a, b in zip(donors, d.sym)]), **out)
    log("prefix test done")
    kl.atomic_json(f"{OUT}/selfcheck_base.json", check)
kl.atomic_json(f"{OUT}/run_{ROLE}.json", {"role": ROLE, "dry": DRY, "minutes": round((time.time() - T0) / 60, 1), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                          "script_sha256": kl.sha(os.path.abspath(__file__)), "lib_sha256": kl.sha(os.path.join(os.path.dirname(kl.__file__), "kernel_lib.py")), "layers": LAYERS, "pool_layers": POOL_LAYERS})
log("DONE")
