"""AtCoder `Input` format block -> a small generation spec (scalars / arrays / tables / strings).

Supported (everything else returns a reason string and is counted as NOT certifiable):
  `N M`                      scalar line                      -> ("scalars", [N, M])
  `A_1 A_2 ... A_N`          array line                       -> ("array", A, "N")
  `a_1 b_1 / a_2 b_2 / : / a_N b_N`  table with row ellipsis  -> ("table", [a, b], "N")
  `S`                        a lone variable (string or int, decided by constraints later)
Interactive / special I/O ("Input and output") is rejected. Ragged rows (`k_i s_i1 ... s_ik_i`) are rejected.
"""
from __future__ import annotations

import ast
import html
import operator as op
import re
from dataclasses import dataclass, field

ELLIPSIS = {"...", "\\ldots", "\\dots", "\\cdots", "…", "⋯"}
VDOTS = {":", "\\vdots", "⋮", "...", "\\ldots", "…", "\\dots"}
TOKEN = re.compile(r"^([A-Za-z]+)(?:_\{?([\w+\-*,]+)\}?)?$")
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv, ast.Pow: op.pow}


@dataclass
class Spec:
    lines: list[tuple] = field(default_factory=list)   # see module docstring
    reason: str = ""                                    # non-empty => unsupported

    @property
    def ok(self) -> bool:
        return not self.reason and bool(self.lines)

    def names(self) -> set[str]:
        out: set[str] = set()
        for ln in self.lines:
            if ln[0] == "scalars":
                out |= set(ln[1])
            elif ln[0] == "array":
                out.add(ln[1])
            elif ln[0] == "table":
                out |= set(ln[1])
        return out


def eval_expr(expr: str, env: dict[str, float]) -> int:
    """'N-1', '2N', 'M+1', 'N*M' with variables from env; AST-checked, never `eval` on raw text."""
    e = re.sub(r"(\d)([A-Za-z])", r"\1*\2", expr.replace("{", "").replace("}", "").replace("^", "**"))
    node = ast.parse(e, mode="eval").body

    def go(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = go(n.left), go(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 4:
                raise ValueError("pow")
            return _OPS[type(n.op)](a, b)
        raise ValueError("unsupported expr")

    return int(go(node))


def input_block(html_text: str) -> tuple[list[str] | None, str]:
    """Return (lines of the first <pre> after the 'Input' heading, reason)."""
    m = re.search(r"<h3>\s*Input\s*</h3>(.*?)(?=<h3>|\Z)", html_text, re.S | re.I)
    if not m:
        if re.search(r"<h3>\s*Input and output", html_text, re.I):
            return None, "interactive_or_special_io"
        return None, "no_input_section"
    pres = re.findall(r"<pre>(.*?)</pre>", m.group(1), re.S | re.I)
    if not pres:
        return None, "no_pre_block"
    if len(pres) > 1:        # e.g. 'T' then a separate per-case format: the first block is only the beginning of the real input
        return None, "multiple_format_blocks"
    txt = html.unescape(re.sub(r"<[^>]+>", "", pres[0]))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in txt.splitlines()]
    return [ln for ln in lines if ln], ""


def _tok(t: str):
    m = TOKEN.match(t)
    return (m.group(1), m.group(2)) if m else None


def parse_input_spec(html_text: str) -> Spec:
    lines, why = input_block(html_text)
    if why:
        return Spec(reason=why)
    spec, i = Spec(), 0
    while i < len(lines):
        toks = lines[i].split()
        if any(t in ELLIPSIS for t in toks):                            # array line  A_1 A_2 ... A_N
            body = [t for t in toks if t not in ELLIPSIS]
            parsed = [_tok(t) for t in body]
            if len(toks) < 3 or toks.index(next(t for t in toks if t in ELLIPSIS)) != len(toks) - 2 or any(p is None for p in parsed):
                return Spec(reason="unsupported_ellipsis_line")
            bases = {p[0] for p in parsed}
            if len(bases) != 1 or parsed[-1][1] is None:
                return Spec(reason="ragged_or_mixed_ellipsis_line")
            first, last = parsed[0][1], parsed[-1][1]
            spec.lines.append(("array", parsed[0][0], f"{last}+1" if first == "0" else last))   # A_0 ... A_N has N+1 elements
            i += 1
            continue
        parsed = [_tok(t) for t in toks]
        if any(p is None for p in parsed):
            return Spec(reason="unparseable_token")
        # table: this line + (optional second line) + vdots + last line with same bases
        j = i + 1
        while j < len(lines) and lines[j].split() and lines[j].split()[0] not in VDOTS and len(lines[j].split()) == len(toks) \
                and [(_tok(t) or (None,))[0] for t in lines[j].split()] == [p[0] for p in parsed] and j - i < 3:
            j += 1
        if j < len(lines) and lines[j].split() and lines[j].split()[0] in VDOTS and j + 1 < len(lines):
            last = [_tok(t) for t in lines[j + 1].split()]
            if len(last) == len(parsed) and all(x and x[0] == p[0] for x, p in zip(last, parsed)) and last[0][1]:
                rows = f"{last[0][1]}+1" if parsed[0][1] == "0" else last[0][1]                 # a_0 ... a_N has N+1 rows
                spec.lines.append(("table", [p[0] for p in parsed], rows))
                i = j + 2
                continue
            return Spec(reason="unsupported_table")
        spec.lines.append(("scalars", [t.replace("{", "").replace("}", "") for t in toks]))
        i += 1
    return spec
