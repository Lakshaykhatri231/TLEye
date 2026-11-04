"""STAGES 4-5 explanation model (pre-declared in reports/stage4_predeclaration.md / stage5_predeclaration.md). LoRA on the causal LM, same prompt byte for byte through the
classifier's `\\n### verdict:`. Readout v0 at that `:` (zero-initialized head). Then a forced template with model-filled values, `\\ntle:`, readout v1. Dataset code is only tokenized."""
import copy, glob, json, math, os, random, subprocess, sys, time

DRY = True
STAGE = "gate"                     # "gate" = 12k rows, val only; "full" = 40k rows, val then test
FULL_MODE = "dual"
CFGS = {"gate": {"dual": dict(mode="dual", seed=0), "v1only": dict(mode="v1only", seed=0)},
        "full": {"s0": dict(mode=FULL_MODE, seed=0), "s1": dict(mode=FULL_MODE, seed=1)}}[STAGE]
OUT = "/kaggle/working"
if "--cfg" not in sys.argv:        # ---------- LAUNCHER: worker exit codes are propagated ----------
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)  # peft/torchao clash on Kaggle
    def launch(n, g):
        return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--cfg", n], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}, stdout=open(f"{OUT}/log_{n}.txt", "w"), stderr=subprocess.STDOUT)
    names = list(CFGS); ps = [launch(n, i) for i, n in enumerate(names)]
    rc = {n: p.wait() for n, p in zip(names, ps)}; print(rc, flush=True)
    for n in names:
        print(f"----- log_{n} (tail) -----"); print("".join(open(f"{OUT}/log_{n}.txt").readlines()[-8:]))
    sys.exit(max(rc.values()))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]  # noqa: E731
sys.path.insert(0, os.path.dirname(find("kernel_lib.py")))
import kernel_lib as kl

NAME = sys.argv[sys.argv.index("--cfg") + 1]; CFG = CFGS[NAME]; SEED, MODE = CFG["seed"], CFG["mode"]
MODEL = "Qwen/Qwen2.5-Coder-0.5B"
MICRO_BS, ACCUM, WARMUP, LR = 4, 2, 100, 1e-4
N_TRAIN = 96 if DRY else (12000 if STAGE == "gate" else 40000)
N_VAL, N_GEN_VAL = (40, 24) if DRY else (8000, 3000)
MAX_TRAIN_H = 0.2 if DRY else (2.0 if STAGE == "gate" else 4.0)
CHECK_STEP = 6 if DRY else 750; CHECK_LIMIT = 99.0 if DRY else 0.680    # dry runs still execute the abort path
RUN = f"{OUT}/cfg_{NAME}"; os.makedirs(RUN, exist_ok=True)
T0 = time.time(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


df = pd.read_parquet(find("tleye_data.parquet"))
side = pd.read_parquet(find("sft_targets.parquet")).set_index("submission_id")
tr = df[df.split == "train"].sample(N_TRAIN, random_state=SEED).reset_index(drop=True)      # same rows as the same-seed classifier (seed 0: 40k)
va = df[df.split == "val"].sample(8000, random_state=SEED).reset_index(drop=True)
if not DRY and SEED == 0:
    z = np.load(find("cfgN/val_logits.npz")) if glob.glob("/kaggle/input/**/cfgN/val_logits.npz", recursive=True) else None
    if z is not None:
        assert (va.submission_id.values.astype(str) == z["submission_id"]).all(), "val rows differ from the confirmed classifier's"
va = va.head(N_VAL) if not DRY else pd.concat([va.assign(_l=va.code.str.len()).nlargest(12, "_l").drop(columns="_l"), va.head(N_VAL)]).drop_duplicates("submission_id").reset_index(drop=True)
for d_ in (tr, va):
    d_["target"] = side.target.loc[d_.submission_id].values
log("cfg", NAME, CFG, "| train", len(tr), "problems", tr.problem_id.nunique(), "| val", len(va))
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
SUF = tok(kl.SUFFIX, add_special_tokens=False)["input_ids"]; comp = kl.Completion(tok); PAD = tok.pad_token_id
texts = lambda d: [kl.prompt(r.rt, kl.tl_text(r.time_limit_ms), r.constraints, r.code) for r in d.itertuples()]  # noqa: E731


def build(peft_dir=None):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    if peft_dir:
        m = PeftModel.from_pretrained(base, peft_dir)
    else:
        m = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return m.cuda()


model = build()
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
core = model.base_model.model; inner, lm_head = core.model, core.lm_head
H = core.config.hidden_size
head0, head1 = torch.nn.Linear(H, 1).cuda(), torch.nn.Linear(H, 1).cuda()
for h_ in (head0, head1):
    torch.nn.init.zeros_(h_.weight); torch.nn.init.zeros_(h_.bias)     # verdicts start at logit 0 (a random head caused large loss spikes in the classifier)
lora = [p for p in model.parameters() if p.requires_grad]
params = lora + list(head1.parameters()) + (list(head0.parameters()) if MODE == "dual" else [])
log("trainable (M):", round(sum(p.numel() for p in params) / 1e6, 2), "| mode", MODE)

ids_tr = kl.encode(tok, texts(tr), SUF)
cmp_tr = [comp.encode(kl.target_values(t)) for t in tr.target]
log("tokenized | median completion", int(np.median([len(c[0]) for c in cmp_tr])), "| max", max(len(c[0]) for c in cmp_tr))


def make_batches(lens, bs, seed):       # same scheme as the classifier, on PROMPT lengths
    rng = random.Random(seed); idx = list(range(len(lens))); rng.shuffle(idx); out = []
    for i in range(0, len(idx), bs * 64):
        c = sorted(idx[i:i + bs * 64], key=lambda j: lens[j]); out += [c[k:k + bs] for k in range(0, len(c), bs)]
    rng.shuffle(out)
    return out


batches = make_batches([len(e) for e in ids_tr], MICRO_BS, SEED); total_steps = len(batches) // ACCUM
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda")
y_tr = tr.y_tle.values.astype(np.float32); hist, run_b, run_ce, step, checked, bad = [], [], [], 0, False, 0
model.train(); log(f"training {len(batches)} micro-batches, {total_steps} steps")
for mi, b in enumerate(batches):
    seqs = [ids_tr[j] + cmp_tr[j][0] for j in b]; L_ = max(map(len, seqs))
    x = torch.full((len(b), L_), PAD, dtype=torch.long); att = torch.zeros((len(b), L_), dtype=torch.long); rows, poss, tgts = [], [], []
    p0, p1 = [], []
    for i, j in enumerate(b):
        s = seqs[i]; x[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1; n = len(ids_tr[j]); p0.append(n - 1); p1.append(len(s) - 1)
        for t, (tid, is_v) in enumerate(zip(*cmp_tr[j])):
            if is_v:
                rows.append(i); poss.append(n + t - 1); tgts.append(tid)
    x, att = x.cuda(), att.cuda(); y = torch.tensor(y_tr[b], device="cuda"); ri = torch.arange(len(b), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        h = inner(input_ids=x, attention_mask=att).last_hidden_state
        l0, l1 = head0(h[ri, torch.tensor(p0).cuda()]).squeeze(-1), head1(h[ri, torch.tensor(p1).cuda()]).squeeze(-1)
        ce_logits = lm_head(h[torch.tensor(rows).cuda(), torch.tensor(poss).cuda()])
    ce = F.cross_entropy(ce_logits.float(), torch.tensor(tgts, device="cuda")); b0, b1 = F.binary_cross_entropy_with_logits(l0.float(), y), F.binary_cross_entropy_with_logits(l1.float(), y)
    loss = ce + b1 + (b0 if MODE == "dual" else 0.0)
    if not torch.isfinite(loss):
        bad += 1
        if bad > 20:
            raise RuntimeError("too many non-finite losses")
        opt.zero_grad(set_to_none=True); continue
    verdict_bce = float(b0 if MODE == "dual" else b1)
    if (mi + 1) % (ACCUM * (2 if DRY else 250)) == 0:           # gradient norm of each loss on the LoRA weights
        gn = {}
        for nm, lv in (("ce", ce), ("b1", b1)) + ((("b0", b0),) if MODE == "dual" else ()):
            gs = torch.autograd.grad(lv, lora, retain_graph=True, allow_unused=True)
            gn[nm] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        hist.append({"step": step, "grad_norms": gn})
    scaler.scale(loss / ACCUM).backward(); run_b.append(verdict_bce); run_ce.append(float(ce))
    if (mi + 1) % ACCUM == 0:
        lr = LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0 or step == total_steps:
            rec = {"step": step, "verdict_bce": float(np.mean(run_b[-50:])), "ce": float(np.mean(run_ce[-50:])), "lr": lr, "min": round((time.time() - T0) / 60, 1)}
            hist.append(rec); log(rec)
        if step >= CHECK_STEP and not checked:
            checked = True; m_ = float(np.mean(run_b[-200:])); log(f"early check at step {step}: verdict BCE {m_:.4f} (abort if > {CHECK_LIMIT})")
            if m_ > CHECK_LIMIT:
                kl.atomic_json(f"{RUN}/train_log.json", hist); raise RuntimeError(f"no verdict signal by step {step}: {m_:.4f}")
        if (time.time() - T0) / 3600 > MAX_TRAIN_H:
            kl.atomic_json(f"{RUN}/train_log.json", hist); raise RuntimeError("wall-clock guard hit before training finished")
assert step == total_steps, (step, total_steps)
assert checked, "the early check never ran"
model.save_pretrained(f"{RUN}/adapter"); torch.save({"head0": head0.state_dict(), "head1": head1.state_dict(), "H": H}, f"{RUN}/heads.pt"); kl.atomic_json(f"{RUN}/train_log.json", hist)
log("saved; peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

# ================= inference =================
model.gradient_checkpointing_disable(); model.eval()


def leftpad(seqs):
    m_ = max(map(len, seqs)); x = torch.full((len(seqs), m_), PAD, dtype=torch.long); a = torch.zeros((len(seqs), m_), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, m_ - len(s):] = torch.tensor(s); a[i, m_ - len(s):] = 1
    return x.cuda(), a.cuda()


def prompt_pass(seqs, mdl_inner=None, h0=None):
    """Left-padded prompt pass: returns v0, last hidden state, cache, mask."""
    x, a = leftpad(seqs); pos = (a.cumsum(1) - 1).clamp(min=0)
    with torch.autocast("cuda", dtype=torch.float16):
        o = (mdl_inner or inner)(input_ids=x, attention_mask=a, position_ids=pos, use_cache=True)
        h = o.last_hidden_state[:, -1]; v0 = (h0 or head0)(h).squeeze(-1)
    return v0.float().cpu().numpy(), h, o.past_key_values, a


def greedy(h, cache, a):
    """Stepwise generation with the template forced. Every row parses by construction. Returns (values per row, v1)."""
    n = h.size(0); ms = [comp.machine() for _ in range(n)]; v1 = np.full(n, np.nan, np.float32); nextpos = a.sum(1)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = lm_head(h).argmax(-1).tolist()
    for _ in range(200):
        active = [not m.done for m in ms]; feed = [m.next(p) if act else PAD for m, p, act in zip(ms, pred, active)]
        x = torch.tensor(feed, device="cuda").unsqueeze(1); act = torch.tensor(active, device="cuda").long().unsqueeze(1); a = torch.cat([a, act], 1)
        with torch.autocast("cuda", dtype=torch.float16):
            o = inner(input_ids=x, attention_mask=a, position_ids=nextpos.unsqueeze(1), past_key_values=cache, use_cache=True)
            hh = o.last_hidden_state[:, -1]; pred = lm_head(hh).argmax(-1).tolist()
            for i, (m, was) in enumerate(zip(ms, active)):
                if was and m.done:
                    v1[i] = float(head1(hh[i]).float().item())
        cache = o.past_key_values; nextpos = nextpos + act.squeeze(1)
        if all(m.done for m in ms):
            break
    assert all(m.done for m in ms), "generation did not finish"
    return [[v.strip() for v in m.values] for m in ms], v1


def forced(cache, a, seqs):
    """v1 given fixed analysis tokens, continuing from a copied prompt cache (the prompt is not recomputed)."""
    c = copy.deepcopy(cache); n, m_ = len(seqs), max(map(len, seqs))
    x = torch.full((n, m_), PAD, dtype=torch.long); ac = torch.zeros((n, m_), dtype=torch.long); last = []
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); ac[i, :len(s)] = 1; last.append(len(s) - 1)
    x, ac = x.cuda(), ac.cuda(); start = a.sum(1, keepdim=True); pos = start + (ac.cumsum(1) - 1).clamp(min=0)
    with torch.autocast("cuda", dtype=torch.float16):
        o = inner(input_ids=x, attention_mask=torch.cat([a, ac], 1), position_ids=pos, past_key_values=c, use_cache=False)
        hh = o.last_hidden_state[torch.arange(n, device="cuda"), torch.tensor(last, device="cuda")]; v = head1(hh).squeeze(-1)
    return v.float().cpu().numpy()


NEUTRAL = ["1", "1", "no", "0", "0", "0", "unknown", "unknown", "unknown"]


def edit(values, field):
    """Coherent single-field edits: 'depth_up' recomputes the estimate; 'sort_on' sets sort_in_loop=1 (and at least one sort)."""
    kv = dict(zip(kl.FIELDS, values))
    if field == "depth_up":
        d = int(kv["loop_depth"]); kv["loop_depth"] = str(d + 1)
        if kv["log10_n"] != "unknown":
            kv["est_log10_ops"] = f"{max(d + 1, 1) * float(kv['log10_n']):.1f}"
    elif field == "sort_on":
        kv["sort_in_loop"] = "1"; kv["sorts"] = str(max(1, int(kv["sorts"])))
    return [kv[f] for f in kl.FIELDS]


@torch.no_grad()
def score_rows(d, name, gen=True, variants=False, donors=None, chunk=1024, bs=16):
    """Atomic shards in row order. gen: greedy analysis and v1, plus v1 with the TRUE analysis. variants: counterfactual v1s."""
    ids = kl.encode(tok, texts(d), SUF); targ = d.target.tolist() if "target" in d else None; os.makedirs(f"{RUN}/{name}", exist_ok=True)
    for s in range(0, len(d), chunk):
        idx = list(range(s, min(s + chunk, len(d)))); f = f"{RUN}/{name}/{s:07d}.npz"
        order = sorted(idx, key=lambda j: -len(ids[j])); r = {k: {} for k in ("v0", "v1_gen", "v1_true", "v1_gen_forced", "v1_neutral", "v1_donor", "v1_depth_up", "v1_sort_on", "text")}
        for b in range(0, len(order), bs):
            sel = order[b:b + bs]; v0, h, c, a = prompt_pass([ids[j] for j in sel])
            for k, j in enumerate(sel):
                r["v0"][j] = v0[k]
            if not gen:
                continue
            c0 = copy.deepcopy(c) if variants or targ else None
            vals, v1 = greedy(h, c, a)
            for k, j in enumerate(sel):
                r["v1_gen"][j] = v1[k]; r["text"][j] = kl.values_to_target(vals[k])
            if targ:
                vt = forced(c0, a, [comp.encode(kl.target_values(targ[j]))[0] for j in sel])
                for k, j in enumerate(sel):
                    r["v1_true"][j] = vt[k]
            if variants:
                def fv(key, make):
                    out = forced(c0, a, [comp.encode(make(k, j))[0] for k, j in enumerate(sel)])
                    for k, j in enumerate(sel):
                        r[key][j] = out[k]
                fv("v1_gen_forced", lambda k, j: vals[k]); fv("v1_neutral", lambda k, j: NEUTRAL)
                fv("v1_donor", lambda k, j: kl.target_values(donors[j])); fv("v1_depth_up", lambda k, j: edit(vals[k], "depth_up")); fv("v1_sort_on", lambda k, j: edit(vals[k], "sort_on"))
        arrs = {"submission_id": d.submission_id.values[idx].astype(str), "v0": np.array([r["v0"][j] for j in idx], np.float32)}
        for k in ("v1_gen", "v1_true", "v1_gen_forced", "v1_neutral", "v1_donor", "v1_depth_up", "v1_sort_on"):
            if r[k]:
                arrs[k] = np.array([r[k][j] for j in idx], np.float32)
        if r["text"]:
            arrs["text"] = np.array([r["text"][j] for j in idx])
        kl.atomic_savez(f, **arrs); log(f"  {name} {s + len(idx)}/{len(d)}")


check = {"cfg": NAME, "dry": DRY, "steps": step, "total_steps": total_steps}
# ---- self-check 1: reload equals the in-memory model, for adapter AND hand-added heads
with torch.no_grad():
    samp = kl.encode(tok, texts(va.head(16)), SUF); trained = head0 if MODE == "dual" else head1          # compare a head that was actually trained
    before, _, _, _ = prompt_pass(samp, None, trained)
    rm = build(f"{RUN}/adapter").eval(); hs = torch.load(f"{RUN}/heads.pt"); h2 = torch.nn.Linear(H, 1).cuda(); h2.load_state_dict(hs["head0"]); h3 = torch.nn.Linear(H, 1).cuda(); h3.load_state_dict(hs["head1"])
    after, _, _, _ = prompt_pass(samp, rm.base_model.model.model, h2 if MODE == "dual" else h3)
    check["reload_max_abs_v0"] = float(np.abs(before - after).max())
    assert check["reload_max_abs_v0"] < 0.05, "reloaded model differs"
    del rm, h2, h3; torch.cuda.empty_cache()
    # ---- self-check 2: batched generation equals one-at-a-time generation
    few = samp[:8]; _, h, c, a = prompt_pass(few); vb, v1b = greedy(h, c, a); solo, v1s = [], []
    for s in few:
        _, h, c, a = prompt_pass([s]); w, v = greedy(h, c, a); solo.append(w[0]); v1s.append(v[0])
    check["gen_batched_equals_single"] = vb == solo; check["gen_v1_max_abs"] = float(np.abs(v1b - np.array(v1s)).max())
    assert check["gen_v1_max_abs"] < 0.2, "batched and single v1 disagree"
    # ---- self-check 3: forcing the generated analysis reproduces the stepwise v1
    _, h, c, a = prompt_pass(few); c0 = copy.deepcopy(c); vb, v1b = greedy(h, c, a); vf = forced(c0, a, [comp.encode(v)[0] for v in vb])
    check["forced_vs_stepwise_max_abs"] = float(np.abs(vf - v1b).max()); assert check["forced_vs_stepwise_max_abs"] < 0.2, "forced and stepwise v1 disagree"
kl.atomic_json(f"{RUN}/selfcheck.json", check); log("self-checks", check)

va["target"] = side.target.loc[va.submission_id].values
score_rows(va.head(N_GEN_VAL), "val_gen", gen=True, variants=False)
score_rows(va.iloc[N_GEN_VAL:], "val_v0", gen=False)
if STAGE == "full":
    te = df[df.split == "test"].reset_index(drop=True); sub = pd.read_parquet(find("sft_test_subset.parquet"))
    if DRY:
        te = pd.concat([te.assign(_l=te.code.str.len()).nlargest(8, "_l").drop(columns="_l"), te.head(32)]).drop_duplicates("submission_id").reset_index(drop=True)
    te["target"] = side.target.loc[te.submission_id].values
    score_rows(te, "test_v0", gen=False)
    A = te[te.submission_id.isin(sub.submission_id)] if not DRY else te.head(24)
    inb = set(sub[sub.in_B].submission_id) if not DRY else set(A.submission_id.head(12))
    A_a, A_b = A[~A.submission_id.isin(inb)].reset_index(drop=True), A[A.submission_id.isin(inb)].reset_index(drop=True)
    score_rows(A_a, "test_gen", gen=True)
    pool = {k: g.target.values for k, g in df.assign(target=side.target.loc[df.submission_id].values)[df.split == "test"].groupby(["problem_id", "language"])}
    rng = np.random.default_rng(SEED + 100); donors = []
    for r_ in A_b.itertuples():
        c_ = [t for t in pool[(r_.problem_id, r_.language)] if t != r_.target]; donors.append(c_[int(rng.integers(len(c_)))] if c_ else r_.target)
    score_rows(A_b, "test_gen_cf", gen=True, variants=True, donors=donors)
    sens = A_b[A_b.n_var != "unknown"] if "n_var" in A_b else A_b.assign(n_var=side.n_var.loc[A_b.submission_id].values)
    sens = sens[sens.n_var != "unknown"].reset_index(drop=True)
    c2 = [kl.scale_var(c, v, 1000) for c, v in zip(sens.constraints, sens.n_var)]; keep = [c is not None for c in c2]
    sens = sens[keep].reset_index(drop=True); sens["constraints"] = [c for c in c2 if c is not None]
    if len(sens):
        sens = sens.drop(columns=["target"]); score_rows(sens, "test_sensitivity", gen=True)
kl.atomic_json(f"{RUN}/run.json", {"cfg": NAME, **CFG, "stage": STAGE, "dry": DRY, "n_train": len(tr), "steps": step, "minutes": round((time.time() - T0) / 60, 1), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                   "script_sha256": kl.sha(os.path.abspath(__file__)), "lib_sha256": kl.sha(os.path.join(os.path.dirname(kl.__file__), "kernel_lib.py"))})
log("DONE")
