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


# ---------------- explanation completion: fixed template, model fills the values ----------------
TEMPLATE = ["\nanalysis: loop_depth=", " loops=", " recursion=", " sorts=", " sort_in_loop=", " listed_calls_in_loop=", " n_var=", " log10_n=", " est_log10_ops=", "\ntle:"]
FIELDS = ["loop_depth", "loops", "recursion", "sorts", "sort_in_loop", "listed_calls_in_loop", "n_var", "log10_n", "est_log10_ops"]
MAX_VALUE_TOKENS = 10


def target_values(target):
    """'loop_depth=2 loops=3 ...' -> ['2', '3', ...] in FIELDS order."""
    kv = dict(x.split("=", 1) for x in target.split())
    return [kv[f] for f in FIELDS]


def values_to_target(values):
    return " ".join(f"{f}={v}" for f, v in zip(FIELDS, values))


class Completion:
    """Tokenized template. Each value ends with the FIRST token of the next piece (its terminator); the rest of that piece is then forced."""

    def __init__(self, tok):
        self.tok = tok
        self.pieces = [tok(p, add_special_tokens=False)["input_ids"] for p in TEMPLATE]
        self.terms = [p[0] for p in self.pieces[1:]]

    def encode(self, values):
        """(ids, is_value) where is_value marks tokens the model must predict: value tokens plus each terminator."""
        ids, mask = list(self.pieces[0]), [False] * len(self.pieces[0])
        for k, v in enumerate(values):
            vt = self.tok(v, add_special_tokens=False)["input_ids"]
            nxt = self.pieces[k + 1]
            ids += vt + nxt
            mask += [True] * len(vt) + [True] + [False] * (len(nxt) - 1)
        return ids, mask

    def machine(self):
        return Machine(self)


class Machine:
    """Row-level state for greedy generation: decides the next token to feed from the model's previous prediction."""

    def __init__(self, c):
        self.c, self.k, self.val, self.values, self.done = c, 0, [], [], False
        self.queue = list(c.pieces[0])

    def next(self, pred):
        """Token to feed now; `pred` is the model's argmax after the previously fed token."""
        if self.queue:
            t = self.queue.pop(0)
        else:
            if pred == self.c.terms[self.k] or len(self.val) >= MAX_VALUE_TOKENS:
                self.values.append(self.c.tok.decode(self.val)); self.val = []; self.k += 1
                t = self.c.pieces[self.k][0]
                self.queue = list(self.c.pieces[self.k][1:])
            else:
                self.val.append(int(pred)); t = int(pred)
        if not self.queue and self.k == len(FIELDS):
            self.done = True
        return t
