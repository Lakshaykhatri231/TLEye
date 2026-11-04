"""Certificate-side bounds reader (v2): fills in size limits the frozen model-side reader (tleye.constraints.parser) misses.

The frozen reader is NOT modified, because the LoRA model was trained on its output. This module runs the frozen reader first and only ADDS:
  * expression bounds      `0 <= A_i <= N-1`, `x_i <= W`, `A_i <= 2N`      -> stored as EXPRESSIONS and evaluated per input type
  * absolute-value bounds  `|A_i| <= 10^9`, `|x_i|,|y_i| <= 10^6`          -> [-v, v]
  * string-length equality `|S| = A+B+1`, `S is N characters long`
after normalising `(1 <= i <= N)` index parentheticals and `1 \\ 000` thousands separators.
Why expressions: `A_i <= N` stored as N's maximum is wrong for the minimum-size input (N small), so the generator evaluates the expression with the
values of THAT input type.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tleye.certify.inputspec import eval_expr
from tleye.constraints.parser import LE, REL, Constraints, parse_number, parse_text

_LEQ = r"(?:\\leqq?|\\le(?![a-zA-Z])|≤|≦|<=)"
_INDEX_PAREN = re.compile(r"\(\s*[0-9]*\s*(?:\\leqq?|≤|≦|<=|<)\s*[ijk]\s*(?:\\leqq?|≤|≦|<=|<)\s*[A-Za-z0-9\-+]+\s*(?:,[^)]*)?\)")
_IJ_PAREN = re.compile(r"\(\s*[ijk]\s*(?:\\neq?|≠)\s*[ijk]\s*\)")
_THOUSANDS = re.compile(r"(?<=\d)[ \t]*\\?[ \t]+(?=\d{3}(?!\d))")
_VAR = r"[A-Za-z][A-Za-z0-9]*"
_SUB = r"(?:_\{?[\w,+\-]+\}?)?"
_ABS_LIST = re.compile(rf"^((?:\|\s*{_VAR}{_SUB}\s*\|\s*,?\s*)+){_LEQ}\s*(.+)$")
_EQ_LEN = re.compile(r"^\|\s*([A-Za-z])\s*\|\s*=\s*(.+)$")
_REL_PAIR = re.compile(rf"({_VAR})_\{{?([ij])\}}?\s*(\\leqq?|≤|≦|<=|<|\\neq|\\ne\b|≠)\s*({_VAR})_\{{?\2\}}?(?![\w+\-*])")
_N_LONG = re.compile(r"\b([A-Za-z])\s+is\s+(\w+)\s+characters?\s+long\b")


def normalise(sec: str) -> str:
    s = _INDEX_PAREN.sub(" ", sec)
    s = _IJ_PAREN.sub(" ", s)
    s = re.sub(r"\\,|\\;|\\!", "", s)
    return _THOUSANDS.sub("", s)


def norm_expr(e: str) -> str:
    e = e.strip().strip("$").replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("\\ ", "").replace("{", "(").replace("}", ")")
    e = re.sub(r"(?<=[A-Za-z0-9\)])\s*\(", "*(", e)                       # N(N-1) -> N*(N-1)
    return e                                                              # eval_expr maps ^ to ** itself


def _ids(e: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]+", e))


@dataclass
class CertBounds:
    upper: dict[str, float] = field(default_factory=dict)        # numeric maximum of each variable
    lower: dict[str, float] = field(default_factory=dict)
    upper_expr: dict[str, str] = field(default_factory=dict)     # var -> expression in other variables (per-input-type evaluation)
    relations: list[tuple[str, str, str]] = field(default_factory=list)   # same-row (a, op, b), op in {"<", "<=", "!="}, e.g. a_i < b_i
    found_section: bool = False


def _base(v: str) -> str:
    m = re.match(r"\|?([A-Za-z]+)", v.strip())
    return m.group(1) if m else v


def parse(sec: str) -> CertBounds:
    s = normalise(sec)
    c: Constraints = parse_text(s)
    out = CertBounds(upper=dict(c.upper), lower=dict(c.lower), found_section=True)
    for v, r in c.refs.items():                                   # frozen reader's var-var relations become expressions
        if v != r:                                                # (x_i < x_{i+1} would otherwise make x refer to itself)
            out.upper_expr.setdefault(v, r)
    for line in s.splitlines():
        line = line.strip().strip("$")
        m = _ABS_LIST.match(line)
        if m:                                                     # |x_i|, |y_i| <= 10^6
            rhs = parse_number(m.group(2).split("(")[0])
            if rhs is not None:
                for name in re.findall(rf"\|\s*({_VAR}){_SUB}\s*\|", m.group(1)):
                    out.upper[name] = max(out.upper.get(name, 0), rhs); out.lower[name] = min(out.lower.get(name, 0), -rhs)
            continue
        m = _EQ_LEN.match(line)
        if m:                                                     # |S| = A+B+1
            out.upper_expr[f"|{m.group(1)}|"] = norm_expr(m.group(2))
            continue
        for m in _N_LONG.finditer(line):                          # S is N characters long
            out.upper_expr[f"|{m.group(1)}|"] = m.group(2)
        parts, ops = REL.split(line), [x.group(0) for x in REL.finditer(line)]
        for i, o in enumerate(ops):
            if o not in LE:
                continue
            left, right = parts[i].strip(), parts[i + 1].strip().split("(")[0].strip()
            if parse_number(right) is not None or not right or not re.fullmatch(r"[A-Za-z0-9_{}+\-*/^()\\ ]+", right):
                continue
            if parse_number(left) is None and _ids(left) and not re.fullmatch(rf"{_VAR}{_SUB}(\s*,\s*{_VAR}{_SUB})*", left.replace(" ", "")):
                continue                                          # left side is itself an expression: skip
            e = norm_expr(right)
            if not _ids(e) or not re.fullmatch(r"[A-Za-z0-9+\-*/^() ]+", e):
                continue
            for v in re.split(r",\s*", left):
                v = _base(v) if v.strip() else ""
                if v and re.fullmatch(_VAR, v):
                    out.upper_expr.setdefault(v, e)
    for m in _REL_PAIR.finditer(s):
        a_, o, b_ = m.group(1), m.group(3), m.group(4)
        op = "!=" if o in ("\\neq", "\\ne", "≠") else ("<" if o == "<" else "<=")
        if (a_, op, b_) not in out.relations:
            out.relations.append((a_, op, b_))
    _resolve(out)
    return out


def _resolve(b: CertBounds) -> None:
    """Fixpoint: turn expressions into numeric maxima / minima whenever every identifier has one."""
    for _ in range(6):
        changed = False
        for v, e in list(b.upper_expr.items()):
            ids = _ids(e)
            if ids <= set(b.upper):
                try:
                    val = float(eval_expr(e, {k: b.upper[k] for k in ids}))
                except Exception:                                  # noqa: BLE001 - unevaluable -> leave unresolved
                    continue
                if v not in b.upper or val < b.upper[v] or v.startswith("|"):
                    if b.upper.get(v) != val:
                        b.upper[v] = val; changed = True
            if v.startswith("|") and ids <= set(b.lower):          # string length equality: also give a minimum
                try:
                    lv = float(eval_expr(e, {k: b.lower[k] for k in ids}))
                    if b.lower.get(v) != lv:
                        b.lower[v] = lv; changed = True
                except Exception:                                  # noqa: BLE001
                    pass
        if not changed:
            break


def upper_for(b: CertBounds, var: str, env: dict[str, int]) -> int | None:
    """Maximum of `var` for an input type whose scalar values are `env` (None if unbounded)."""
    cap = b.upper.get(var)
    e = b.upper_expr.get(var)
    if e is not None and _ids(e) <= set(env):
        try:
            v = eval_expr(e, env)
            cap = v if cap is None else (v if var.startswith("|") else min(cap, v))
        except Exception:                                          # noqa: BLE001
            pass
    return None if cap is None else int(cap)
