"""CONFIRMATORY RUN (pre-declared, no selection on test): lr 1e-4, 40k training rows, with (P) and without (N) the symbolic `analysis:` prefix,
each scored ONCE on the full test split. Both results are reported whatever they are.
TLEye LoRA verdict model, Kaggle GPU. Qwen2.5-Coder-0.5B + LoRA -> logit(P(TLE)).
Trains on a seeded subset of the problem-disjoint train split, then writes logits for a val subset and the FULL
test split (metrics + CIs are computed locally). Nothing from the dataset is executed; code is only tokenized."""
import glob, hashlib, json, math, os, random, subprocess, sys, time

DRY = False                     # True = tiny end-to-end smoke test; False = the real run
CFGS = {"P": dict(lr=1e-4, sym=True), "N": dict(lr=1e-4, sym=False)}

if "--cfg" not in sys.argv:     # ---------- LAUNCHER: P and N in parallel (one T4 each) ----------
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)  # peft/torchao clash on Kaggle
    os.makedirs("/kaggle/working", exist_ok=True)
    def launch(name, gpu):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--cfg", name], env=env,
                                stdout=open(f"/kaggle/working/log_{name}.txt", "w"), stderr=subprocess.STDOUT)
    pa, pb = launch("P", 0), launch("N", 1)
    print("P rc", pa.wait(), "| N rc", pb.wait(), flush=True)
    for n in "PN":
        print(f"----- log_{n}.txt (tail) -----"); print("".join(open(f"/kaggle/working/log_{n}.txt").readlines()[-6:]))
    sys.exit(0)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-0.5B"
CFG_NAME = sys.argv[sys.argv.index("--cfg") + 1]; CFG = CFGS[CFG_NAME]; LR, USE_SYM = CFG["lr"], CFG["sym"]
MAX_LEN, MICRO_BS, ACCUM, WARMUP, SEED = 1024, 4, 2, 100, 0
N_TRAIN, N_VAL, MAX_TRAIN_H = (96, 64, 0.2) if DRY else (40000, 8000, 5.0)
N_TEST = 96 if DRY else None
OUT = f"/kaggle/working/cfg{CFG_NAME}"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


path = glob.glob("/kaggle/input/**/tleye_data.parquet", recursive=True)[0]
df = pd.read_parquet(path)
assert "sym" in df.columns, "dataset version without the sym column (old version mounted?)"
tr = df[df.split == "train"].sample(N_TRAIN, random_state=SEED).reset_index(drop=True)
va = df[df.split == "val"].sample(N_VAL, random_state=SEED).reset_index(drop=True)
te = df[df.split == "test"]
te = (te.sample(N_TEST, random_state=SEED) if N_TEST else te).reset_index(drop=True)
log("cfg", CFG_NAME, CFG, "| rows train/val/test", len(tr), len(va), len(te), "| train problems", tr.problem_id.nunique(),
    "| train TLE rate %.3f" % tr.y_tle.mean(), "| langs", tr.language.value_counts().to_dict())

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
SUFFIX = tok("\n### verdict:", add_special_tokens=False)["input_ids"]   # fixed final tokens the head reads from


def prompt(r):
    return (f"language: {r.rt}\ntime limit: {r.time_limit_ms / 1000:g}s\nconstraints: {r.constraints}\n" +
            (f"analysis: {r.sym}\n" if USE_SYM else "") + f"code:\n{r.code}")


def encode(frame):
    texts = [prompt(r) for r in frame.itertuples()]
    out = []
    for i in range(0, len(texts), 2000):
        enc = tok(texts[i:i + 2000], truncation=True, max_length=MAX_LEN - len(SUFFIX), add_special_tokens=False)["input_ids"]
        out += [e + SUFFIX for e in enc]
    return out


ids_tr, ids_va, ids_te = encode(tr), encode(va), encode(te)
log("tokenized | train median len", int(np.median([len(e) for e in ids_tr])),
    "| share truncated %.3f" % np.mean([len(e) >= MAX_LEN for e in ids_tr]))

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float16)
model.config.pad_token_id = tok.pad_token_id
model.score.weight.data.zero_()                  # head starts at logit 0 (benchmark showed huge loss spikes with random head)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg).cuda()
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
log("trainable params (M):", round(sum(p.numel() for p in params) / 1e6, 2))


def collate(seqs):
    m, pad = max(map(len, seqs)), tok.pad_token_id
    x = torch.full((len(seqs), m), pad, dtype=torch.long); att = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, e in enumerate(seqs):
        x[i, :len(e)] = torch.tensor(e); att[i, :len(e)] = 1
    return x.cuda(), att.cuda()


def make_batches(lens, bs, seed):
    rng = random.Random(seed); idx = list(range(len(lens))); rng.shuffle(idx)
    batches = []
    for i in range(0, len(idx), bs * 64):           # sort by length inside chunks -> less padding
        c = sorted(idx[i:i + bs * 64], key=lambda j: lens[j])
        batches += [c[k:k + bs] for k in range(0, len(c), bs)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(ids, bs=16):
    model.eval()
    order = np.argsort([-len(e) for e in ids]); out = np.zeros(len(ids), np.float32)
    for i in range(0, len(ids), bs):
        sel = order[i:i + bs]; x, att = collate([ids[j] for j in sel])
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(input_ids=x, attention_mask=att).logits.squeeze(-1)
        out[sel] = lg.float().cpu().numpy()
        if (i // bs) % 200 == 0:
            log(f"  predict {i}/{len(ids)}")
    return out


def fit_temperature(logit, y):
    z, t = torch.tensor(logit), torch.tensor(y, dtype=torch.float32)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(z / T.clamp(min=0.05), t); l.backward(); return l
    opt.step(closure)
    return float(T.clamp(min=0.05))


# ---------------- training ----------------
batches = make_batches([len(e) for e in ids_tr], MICRO_BS, SEED)
total_steps = len(batches) // ACCUM
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
y_tr = tr.y_tle.values.astype(np.float32)
trainlog, run_loss, step, bad, checked = [], [], 0, 0, False
model.train()
log(f"training: {len(batches)} micro-batches, {total_steps} optimizer steps")
for mi, b in enumerate(batches):
    x, att = collate([ids_tr[j] for j in b]); y = torch.tensor(y_tr[b], device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logit = model(input_ids=x, attention_mask=att).logits.squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logit.float(), y)
    if not torch.isfinite(loss):
        bad += 1
        if bad > 20:
            raise RuntimeError("too many non-finite losses")
        opt.zero_grad(set_to_none=True); continue
    scaler.scale(loss / ACCUM).backward(); run_loss.append(float(loss))
    if (mi + 1) % ACCUM == 0:
        lr = LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0 or step == total_steps:
            rec = {"step": step, "loss": float(np.mean(run_loss[-50:])), "lr": lr, "min": round((time.time() - T0) / 60, 1)}
            trainlog.append(rec); log(rec)
            if not DRY and step >= 750 and not checked:
                checked = True                        # a constant-prior guess scores ~0.67-0.68 BCE; must beat it by now
                m = float(np.mean(run_loss[-200:]))
                log(f"early check at step {step}: mean loss of last 200 micro-batches = {m:.4f} (abort if > 0.680)")
                if m > 0.680:
                    model.save_pretrained(f"{OUT}/adapter"); json.dump(trainlog, open(f"{OUT}/train_log.json", "w"))
                    raise RuntimeError(f"no learning signal by step {step} (loss {m:.4f}); aborting to save GPU quota")
        if (time.time() - T0) / 3600 > MAX_TRAIN_H:
            log("wall-clock guard hit, stopping training early"); break
model.save_pretrained(f"{OUT}/adapter")
json.dump(trainlog, open(f"{OUT}/train_log.json", "w"))
log("adapter saved; peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

# ---------------- predictions: val subset (temperature) then the FULL test split, once ----------------
from sklearn.metrics import average_precision_score, roc_auc_score
lv = predict(ids_va); yv = va.y_tle.values; T = fit_temperature(lv, yv)
np.savez(f"{OUT}/val_logits.npz", submission_id=va.submission_id.values.astype(str), logit=lv, y=yv, T=T)
bce = float(F.binary_cross_entropy_with_logits(torch.tensor(lv) / T, torch.tensor(yv, dtype=torch.float32)))
pv = float(np.mean(yv)); const = -(pv * math.log(pv) + (1 - pv) * math.log(1 - pv))
res = {"cfg": CFG_NAME, **CFG, "dry": DRY, "n_train": len(tr), "n_val": len(va), "optimizer_steps": step,
       "val_auroc": float(roc_auc_score(yv, lv)), "val_auprc": float(average_precision_score(yv, lv)),
       "val_bce_after_T": bce, "val_const_guess_bce": const, "T": T,
       "final_train_loss_last100": float(np.mean(run_loss[-100:])), "minutes": round((time.time() - T0) / 60, 1),
       "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
       "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1)
log("RESULT", json.dumps(res))
lt = predict(ids_te)
np.savez(f"{OUT}/test_logits.npz", submission_id=te.submission_id.values.astype(str), logit=lt, y=te.y_tle.values, T=T)
log("test logits saved", len(lt))
log("DONE")
