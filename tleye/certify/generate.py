"""Build a battery of worst-case-ish inputs from a parsed Input spec + parsed constraints.

Only generates when EVERY variable has a numeric upper bound and (for strings) a known alphabet; otherwise it
returns a reason and the problem counts as not certifiable (never guessed). All values stay inside the parsed
[lower, upper] ranges, sizes use their maximum. It does NOT guarantee semantic constraints (tree, distinct, ...);
those problems are flagged upstream (coverage.py) and excluded from `simple`.
"""
from __future__ import annotations

import re

import numpy as np

from tleye.certify.bounds import CertBounds, upper_for
from tleye.certify.inputspec import Spec, eval_expr

STRATS = ["random", "asc", "desc", "all_max", "all_min", "alt", "min_sizes"]
MAX_TOKENS = 3_000_000          # skip inputs that would be absurdly large for a quick sandbox run


class GenError(Exception):
    pass


def _base(n: str) -> str:
    m = re.match(r"[A-Za-z]+", n)
    return m.group(0) if m else n


def _alphabet(sec: str) -> str:
    s = sec.lower()
    if "lowercase" in s:
        return "abcdefghijklmnopqrstuvwxyz"
    if "uppercase" in s:
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if re.search(r"\b0\b[^.]*\b1\b", s) and "consist" in s:
        return "01"
    raise GenError("unknown_alphabet")


def _is_string(name: str, cons, sec: str) -> bool:
    return f"|{name}|" in cons.upper or bool(
        re.search(rf"\b{re.escape(name)}\b[^.\n]*(?:consists? of|is a string|string of)", sec))


def _ints(strategy: str, lo: int, hi: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if hi < lo:
        raise GenError("empty_range")
    if strategy == "all_max":
        return np.full(n, hi, dtype=np.int64)
    if strategy == "all_min":
        return np.full(n, lo, dtype=np.int64)
    if strategy == "alt":
        return np.where(np.arange(n) % 2 == 0, hi, lo).astype(np.int64)
    v = rng.integers(lo, hi + 1, size=n, dtype=np.int64)
    if strategy == "asc":
        v.sort()
    elif strategy == "desc":
        v = np.sort(v)[::-1]
    return v


def _string(strategy: str, alpha: str, length: int, rng: np.random.Generator) -> str:
    if strategy == "all_max":
        return alpha[0] * length
    if strategy == "all_min":
        return alpha[-1] * length
    if strategy == "alt":
        return (alpha[0] + alpha[min(1, len(alpha) - 1)]) * (length // 2) + alpha[0] * (length % 2)
    idx = rng.integers(0, len(alpha), size=length)
    if strategy in ("asc", "desc"):
        idx.sort()
        idx = idx[::-1] if strategy == "desc" else idx
    return "".join(alpha[i] for i in idx)


def _apply_relations(cb, cols: list[str], data: list[list[str]], e: dict[str, int]) -> None:
    """Row-wise a_i < b_i / a_i <= b_i / a_i != b_i between two columns of the SAME table, repaired in place and kept inside each column's range."""
    idx = {_base(c): j for j, c in enumerate(cols)}
    for a_, op, b_ in cb.relations:
        if a_ not in idx or b_ not in idx:
            continue
        ia, ib = idx[a_], idx[b_]
        lo_a, lo_b = int(cb.lower.get(a_, 1)), int(cb.lower.get(b_, 1))
        hi_a, hi_b = upper_for(cb, a_, e), upper_for(cb, b_, e)
        if hi_a is None or hi_b is None:
            raise GenError("relation_unbounded")
        for r in range(len(data[ia])):
            va, vb = int(data[ia][r]), int(data[ib][r])
            if op in ("<", "<=") and va > vb:
                va, vb = vb, va
            if (op == "<" and va == vb) or (op == "!=" and va == vb):
                if vb + 1 <= hi_b:
                    vb += 1
                elif op == "<" and va - 1 >= lo_a:
                    va -= 1
                elif op == "!=" and vb - 1 >= lo_b:
                    vb -= 1
                else:
                    raise GenError("relation_unsatisfiable")
            if not (lo_a <= va <= hi_a and lo_b <= vb <= hi_b):
                raise GenError("relation_breaks_range")
            data[ia][r], data[ib][r] = str(va), str(vb)


def build_inputs(spec: Spec, cons, sec: str, seed: int = 0) -> dict[str, str]:
    """-> {strategy: input_text}. Raises GenError(reason) when it cannot generate safely.
    `cons` is a CertBounds (preferred) or the frozen reader's Constraints; expression bounds (A_i <= N-1) are evaluated with the scalar values
    of EACH input type, so the minimum-size input never violates them. An input type that cannot be generated validly (empty range) is skipped."""
    if not spec.ok:
        raise GenError(spec.reason or "bad_spec")
    cb = cons if isinstance(cons, CertBounds) else CertBounds(upper=dict(cons.upper), lower=dict(cons.lower), upper_expr=dict(getattr(cons, "refs", {})))
    up, lo_d = cb.upper, cb.lower
    bases = {_base(x) for x in spec.names()}
    strings = {n for n in bases if _is_string(n, cb, sec)}
    alpha = _alphabet(sec) if strings else ""
    scalars = bases - strings
    env: dict[str, int] = {}
    for n in scalars:
        if n not in up:
            raise GenError(f"unbounded:{n}")
        env[n] = int(up[n])
    env_min = {n: int(min(lo_d.get(n, 1 if up[n] >= 1 else up[n]), up[n])) for n in scalars}
    for _ in range(3):                                   # minima must also respect expression bounds evaluated at the minima
        for n in scalars:
            h = upper_for(cb, n, env_min)
            if h is not None and env_min[n] > h:
                env_min[n] = int(h)

    def str_len(b: str, e: dict[str, int], minimal: bool) -> int:
        key = f"|{b}|"
        if key in cb.upper_expr:                         # equality like |S| = A+B+1: exact length for this input type
            try:
                return int(eval_expr(cb.upper_expr[key], e))
            except Exception as ex:                      # noqa: BLE001
                raise GenError(f"bad_length_expr:{cb.upper_expr[key]}") from ex
        if key not in up:
            return 0
        return int(lo_d.get(key, 1)) if minimal else int(up[key])

    def one(strat: str) -> str:
        rng = np.random.default_rng([seed, STRATS.index(strat)])
        minimal = strat == "min_sizes"
        e = env_min if minimal else env
        lines = []
        for ln in spec.lines:
            if ln[0] == "scalars":
                parts = []
                for name in ln[1]:
                    b = _base(name)
                    if b in strings:
                        parts.append(_string("random" if minimal else strat, alpha, str_len(b, e, minimal), rng))
                    else:
                        parts.append(str(e[b]))
                lines.append(" ".join(parts))
                continue
            kind, cols, expr = (ln[0], [ln[1]] if ln[0] == "array" else ln[1], ln[2])
            try:
                n = eval_expr(expr, e)
            except Exception as ex:                      # noqa: BLE001 - any failure means we cannot size this safely
                raise GenError(f"bad_length_expr:{expr}") from ex
            if n < 0 or n * len(cols) > MAX_TOKENS:
                raise GenError("too_large_or_negative")
            data = []
            for c in cols:
                b = _base(c)
                if b in strings:
                    L = str_len(b, e, minimal)
                    if L <= 0 or L * n > 20_000_000:
                        raise GenError("string_too_large")
                    data.append([_string("random" if minimal else strat, alpha, L, rng) for _ in range(n)])
                else:
                    hi = upper_for(cb, b, e)
                    if hi is None:
                        raise GenError(f"unbounded:{b}")
                    lo = int(lo_d.get(b, 1 if hi >= 1 else hi))
                    data.append([str(x) for x in _ints("random" if minimal else strat, lo, hi, n, rng)])
            if kind == "table":
                _apply_relations(cb, cols, data, e)
            if kind == "array":
                lines.append(" ".join(data[0]))
            else:
                lines.extend(" ".join(row) for row in zip(*data))
        return "\n".join(lines) + "\n"

    out: dict[str, str] = {}
    for strat in STRATS:
        try:
            out[strat] = one(strat)
        except GenError:
            if strat != "min_sizes":                     # the edge-case input is optional; the others are required
                raise
    return out
