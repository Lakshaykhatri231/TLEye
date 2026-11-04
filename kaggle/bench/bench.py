"""TLEye Phase 3 benchmark (Kaggle GPU): measure real LoRA training throughput, memory and zero-step quality
for Qwen2.5-Coder 0.5B vs 1.5B on the real prompts. Prints a JSON summary; writes bench.json."""
import glob, json, os, subprocess, sys, time

# Kaggle's image ships torchao 0.10.0 but peft needs >= 0.16 when torchao is present; we don't use it -> remove.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
path = glob.glob("/kaggle/input/**/tleye_data.parquet", recursive=True)[0]
df = pd.read_parquet(path)
print("rows", len(df), df.split.value_counts().to_dict())

MAX_LEN, BS, STEPS = 1024, 4, 12
MODELS = ["Qwen/Qwen2.5-Coder-0.5B", "Qwen/Qwen2.5-Coder-1.5B"]


def prompt(r):
    return (f"language: {r.rt}\ntime limit: {r.time_limit_ms / 1000:g}s\nconstraints: {r.constraints}\n"
            f"code:\n{r.code}")


train = df[df.split == "train"].sample(2000, random_state=0)
results = []
for name in MODELS:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lens = [len(tok(prompt(r), truncation=False)["input_ids"]) for r in train.head(300).itertuples()]
    stats = {"model": name, "tok_len_median": int(pd.Series(lens).median()),
             "tok_len_p90": int(pd.Series(lens).quantile(.9)), "share_over_max": float((pd.Series(lens) > MAX_LEN).mean())}
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, torch_dtype=torch.float16)
    model.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg).cuda()
    for p in model.parameters():                       # trainable params in fp32 for the grad scaler
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    stats["trainable_params_M"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.train(); toks = 0; t_total = 0.0; losses = []
    rows = list(train.head(BS * STEPS).itertuples())
    for s in range(STEPS):
        b = rows[s * BS:(s + 1) * BS]
        enc = tok([prompt(r) for r in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        y = torch.tensor([float(r.y_tle) for r in b], device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        torch.cuda.synchronize(); dt = time.time() - t0
        if s >= 2:                                      # skip warmup steps
            t_total += dt; toks += int(enc["attention_mask"].sum())
        losses.append(float(loss))
    n_meas = STEPS - 2
    stats |= {"steps_measured": n_meas, "sec_per_step": round(t_total / n_meas, 3),
              "real_tokens_per_sec": round(toks / t_total), "rows_per_sec": round(n_meas * BS / t_total, 2),
              "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "losses": [round(l, 3) for l in losses]}
    stats["hours_per_40k_rows"] = round(40000 / stats["rows_per_sec"] / 3600, 2)
    stats["hours_per_142k_train_rows"] = round(142400 / stats["rows_per_sec"] / 3600, 2)
    results.append(stats); print(json.dumps(stats))
    del model, opt; torch.cuda.empty_cache()

out = {"gpu": torch.cuda.get_device_name(0), "n_gpu": torch.cuda.device_count(), "max_len": MAX_LEN, "batch": BS,
       "results": results}
json.dump(out, open("/kaggle/working/bench.json", "w"), indent=1)
print("BENCH_DONE", json.dumps(out))
