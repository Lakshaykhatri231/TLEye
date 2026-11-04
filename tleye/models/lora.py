"""Local inference for the confirmed LoRA verdict model (Qwen2.5-Coder-0.5B + adapter in models/lora).

The prompt is byte-for-byte the training prompt (kaggle/confirm/confirm.py, config N: no symbolic prefix):
  language / time limit / constraints / code, truncated to 1024 tokens, then the fixed suffix the head reads from.
Source code is only tokenized here, never executed. Needs the `train` extras (torch, transformers, peft).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tleye.data.kaggle_pack import MAX_CHARS, constraint_text_from_html


def available(path: str | Path = "models/lora") -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (Path(path) / "adapter_model.safetensors").exists() and (Path(path) / "meta.json").exists()


def build_prompt(code: bytes, language: str, runtime: str, statement_html: str, time_limit_ms: float) -> str:
    rt = "PyPy" if runtime.startswith("PyPy") else language          # same mapping as the training pack
    text = code.decode("utf8", "replace")[:MAX_CHARS]
    return (f"language: {rt}\ntime limit: {time_limit_ms / 1000:g}s\nconstraints: {constraint_text_from_html(statement_html)}\n"
            f"code:\n{text}")


class LoraVerdict:
    def __init__(self, path: str | Path = "models/lora", device: str | None = None):
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        transformers.utils.logging.set_verbosity_error()      # the 'score.weight MISSING' note is expected: the head is loaded from the adapter below
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(self.meta["base_model"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.suffix = self.tok(self.meta["suffix"], add_special_tokens=False)["input_ids"]
        base = AutoModelForSequenceClassification.from_pretrained(self.meta["base_model"], num_labels=1, dtype=torch.float32)
        base.config.pad_token_id = self.tok.pad_token_id
        self.model = PeftModel.from_pretrained(base, str(self.path)).eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.torch = torch

    def logit(self, prompt: str) -> float:
        ids = self.tok(prompt, truncation=True, max_length=self.meta["max_len"] - len(self.suffix), add_special_tokens=False)["input_ids"]
        x = self.torch.tensor([ids + self.suffix], device=self.device)
        with self.torch.no_grad():
            return float(self.model(input_ids=x, attention_mask=self.torch.ones_like(x)).logits.squeeze())

    def predict(self, code: bytes, language: str, runtime: str, statement_html: str, time_limit_ms: float) -> dict:
        z = self.logit(build_prompt(code, language, runtime, statement_html, time_limit_ms))
        p = 1 / (1 + math.exp(-z / self.meta["T"]))
        return {"logit": z, "p_tle": p, "threshold": self.meta["thr_f1_val"], "flagged": p >= self.meta["thr_f1_val"]}
