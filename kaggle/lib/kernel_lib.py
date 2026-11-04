"""Shared code for the interpretability / explanation kernels. Uploaded inside the private sidecar dataset so kernels and local tests run the SAME code.
Nothing from the dataset is executed; source code is only tokenized."""
import hashlib
import json
import math
import os

import numpy as np

SUFFIX = "\n### verdict:"
MAX_LEN = 1024


def fmt(v):
    """Same number format as tleye/data/kaggle_pack.py:fmt (the classifier was trained on this text)."""
    if v >= 1e300:
        return "huge"
    e = math.floor(math.log10(v)) if v > 0 else 0
    return str(int(v)) if v < 1000 else f"{v / 10 ** e:g}e{e}"


def parse_items(line):
    if line == "none":
        return []
    return [tuple(x.split("<=", 1)) for x in line.split(", ")]


def scale_var(line, var, factor):
    """Constraints line with ONE variable's bound scaled; None if it is absent or not a finite number."""
    items, hit = parse_items(line), False
    out = []
    for k, v in items:
        if k == var:
            if v == "huge":
                return None
            v = fmt(max(1.0, float(v) * factor))
            hit = True
        out.append(f"{k}<={v}")
    return ", ".join(out) if hit else None


def reorder_items(line):
    """Placebo: same facts, different order. None when there is nothing to reorder."""
    items = parse_items(line)
    return ", ".join(f"{k}<={v}" for k, v in reversed(items)) if len(items) > 1 else None


def tl_text(time_limit_ms, factor=1.0, placebo=False):
    s = time_limit_ms * factor / 1000
    return f"{s:.1f}s" if placebo else f"{s:g}s"


def prompt(rt, tl, constraints, code, analysis=None):
    """Byte-identical to kaggle/seeds/seeds.py:prompt when analysis is None, and to kaggle/confirm config P when it is given."""
    mid = f"analysis: {analysis}\n" if analysis is not None else ""
    return f"language: {rt}\ntime limit: {tl}\nconstraints: {constraints}\n" + mid + f"code:\n{code}"


def encode(tok, texts, suffix_ids, max_len=MAX_LEN):
    out = []
    for i in range(0, len(texts), 2000):
        enc = tok(texts[i:i + 2000], truncation=True, max_length=max_len - len(suffix_ids), add_special_tokens=False)["input_ids"]
        out += [e + suffix_ids for e in enc]
    return out


def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
