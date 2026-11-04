"""AtCoder statement -> input-size constraints (regex first; no LLM fallback yet).

Reads the "Constraints" section (stops at "Input"/"Partial Score"), splits each line into
relation chains (`1 \\leq a,b \\leq 15`, `2 \\leq N \\leq 2 \\times 10^5`) and records the
numeric upper bound of each variable. Bounds that refer to another variable (`A_i \\leq N`)
are resolved once N is known. Numbers are evaluated with a tiny AST evaluator (never `eval`).
"""
from __future__ import annotations

import ast
import html
import operator as op
import re
from dataclasses import dataclass, field

REL = re.compile(r"\\leqq?|\\le(?![a-zA-Z])|\\lt|≤|≦|<=|<|\\geqq?|\\ge(?![a-zA-Z])|\\gt|≥|≧|>=|>")
LE = {"\\leq", "\\leqq", "\\le", "\\lt", "≤", "≦", "<=", "<"}
GE = {"\\geq", "\\geqq", "\\ge", "\\gt", "≥", "≧", ">=", ">"}
HUGE = 1e300
SIZE_VARS = set("NMQKHWTLEV") | set("nmqkhwtlev")
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(a) > 1e6 or abs(b) > 1e7:
                raise ValueError("pow too large")
            if a >= 2 and b > 300:
                return HUGE                      # e.g. 10^{200000}: bigger than any float
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("unsupported")


def parse_number(s: str) -> float | None:
    """'2 \\times 10^5' -> 200000.0, '100,000' -> 100000.0, '10^{9}+7' -> 1000000007.0, else None."""
    t = s.strip().strip("$").strip()
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("\\,", "").replace("\\ ", "")
    t = re.sub(r"\\mathrm\{[^}]*\}|\\text\{[^}]*\}", "", t)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)          # 100,000 -> 100000
    t = t.replace("{", "(").replace("}", ")").replace("^", "**").replace(" ", "")
    t = re.sub(r"\)\(", ")*(", t)
    if not t or not re.fullmatch(r"[0-9+\-*/().]+", t) or not re.search(r"\d", t):
        return None
    try:
        v = float(_eval(ast.parse(t, mode="eval")))
    except Exception:
        return None
    return v


def _norm_var(tok: str) -> str:
    """A_i -> A, c_{i1} -> c, |s| -> |s|, x_i,y_i handled by caller splitting on commas."""
    tok = tok.strip().strip("$")
    if tok.startswith("|") and tok.endswith("|"):
        return tok
    return re.sub(r"_\{?[^,\s]*\}?$", "", tok).strip()


@dataclass
class Constraints:
    upper: dict[str, float] = field(default_factory=dict)   # var -> numeric upper bound
    lower: dict[str, float] = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)      # var -> var it is bounded by
    found_section: bool = False

    @property
    def max_bound(self) -> float | None:
        return max(self.upper.values()) if self.upper else None

    @property
    def primary_size_bound(self) -> float | None:
        """Bound of the main input-size variable by AtCoder convention: first of N, |S|, H, R, Q, M."""
        for v in ("N", "n", "|S|", "|s|", "|A|", "H", "h", "R", "Q", "M"):
            if v in self.upper:
                return self.upper[v]
        return None

    @property
    def size_bound(self) -> float | None:
        """Largest bound among count/length-style variables (N, M, Q, K, H, W, T, L, |s|)."""
        c = [v for k, v in self.upper.items() if k in SIZE_VARS or k.startswith("|")]
        return max(c) if c else None


def constraints_text(html_text: str) -> str | None:
    h = re.sub(r"(?i)</(?:li|p|pre|div|h\d|ul|ol|section)>|<br\s*/?>", "\n", html_text)
    t = html.unescape(re.sub(r"<[^>]+>", " ", h))
    t = re.sub(r"[ \t]+", " ", t)
    m = re.search(r"Constraints(.*?)(?:\s+Input\s|Partial Score|\Z)", t, re.DOTALL)
    return m.group(1) if m else None


_NUM = r"(?P<num>[\d][\d,\s\\a-z{}^+\-*()]*?)"
_NL = [
    # "K is an integer between 1 and 10^6", "N is a positive integer not exceeding 999"
    (re.compile(r"\b(?P<v>[A-Za-z](?:_\{?\w+\}?)?)\s+is\s+(?:an?\s+)?(?:positive\s+|non-negative\s+)?(?:integer\s+)?"
                r"(?:between\s+[\d,^{}\\ a-z]+?\s+and|not\s+exceeding|at\s+most|no\s+more\s+than)\s+(?P<ub>\d[\d,]*(?:\s*\\times\s*10\^\{?\d+\}?|\^\{?\d+\}?)?)"), False),
    # "lengths of S and T are each at least 1 and at most 1000" / "S has length between 1 and 100"
    (re.compile(r"(?:length|lengths)\s+(?:of\s+)?(?P<v>[A-Za-z])(?:\s+and\s+(?P<v2>[A-Za-z]))?\s+(?:is|are)\s+(?:each\s+)?"
                r"(?:between\s+[\d,^{}\\ a-z]+?\s+and|at\s+least\s+\S+\s+and\s+at\s+most|at\s+most)\s+(?P<ub>\d[\d,]*(?:\s*\\times\s*10\^\{?\d+\}?|\^\{?\d+\}?)?)"), True),
    (re.compile(r"(?P<v>[A-Za-z])(?:\s+and\s+(?P<v2>[A-Za-z]))?\s+have\s+lengths?\s+between\s+[\d,^{}\\ a-z]+?\s+and\s+"
                r"(?P<ub>\d[\d,]*(?:\s*\\times\s*10\^\{?\d+\}?|\^\{?\d+\}?)?)"), True),
    (re.compile(r"length\s+of\s+(?P<v>[A-Za-z])\s+(?:is|are)\s+between\s+[\d,^{}\\ a-z]+?\s+and\s+"
                r"(?P<ub>\d[\d,]*(?:\s*\\times\s*10\^\{?\d+\}?|\^\{?\d+\}?)?)"), True),
]
_EQ = re.compile(r"^(?P<v>\|?[A-Za-z]\w*\|?)\s*=\s*(?P<ub>[\d][\d,^{}\\ a-z+*]*)$")


def _nl(out: Constraints, line: str) -> None:
    for rx, is_len in _NL:
        for m in rx.finditer(line):
            ub = parse_number(m.group("ub"))
            if ub is None:
                continue
            for g in ("v", "v2"):
                v = m.groupdict().get(g)
                if v:
                    v = _norm_var(v)
                    key = f"|{v}|" if is_len else v
                    out.upper[key] = max(out.upper.get(key, 0), ub)
    m = _EQ.match(line)
    if m:
        ub = parse_number(m.group("ub"))
        if ub is not None:
            out.upper[_norm_var(m.group("v"))] = max(out.upper.get(_norm_var(m.group("v")), 0), ub)


def parse_text(sec: str) -> Constraints:
    out = Constraints(found_section=True)
    for line in sec.splitlines():
        _nl(out, line.strip().strip("$"))
        line = line.strip().strip("$")
        if not REL.search(line):
            continue
        parts = REL.split(line)
        ops = [m.group(0) for m in REL.finditer(line)]
        for i, o in enumerate(ops):
            left, right = parts[i].strip(), parts[i + 1].strip()
            # a relation chain `a <= b <= c`: use only the adjacent operands
            if o in GE:
                left, right, o = right, left, "\\leq"      # normalise to left <= right
            elif o not in LE:
                continue
            lnum, rnum = parse_number(left), parse_number(right)
            lvars = [_norm_var(x) for x in re.split(r",\s*(?![^{]*\})", left)] if lnum is None else []
            rvars = [_norm_var(x) for x in re.split(r",\s*(?![^{]*\})", right)] if rnum is None else []
            if rnum is not None:
                for v in lvars:
                    if v and len(v) <= 12 and re.fullmatch(r"[A-Za-z|][A-Za-z0-9|]*", v):
                        out.upper[v] = max(out.upper.get(v, 0), rnum)
            if lnum is not None:
                for v in rvars:
                    if v and len(v) <= 12 and re.fullmatch(r"[A-Za-z|][A-Za-z0-9|]*", v):
                        out.lower[v] = min(out.lower.get(v, lnum), lnum)
            if rnum is None and lnum is None and len(rvars) == 1:
                for v in lvars:
                    if v and re.fullmatch(r"[A-Za-z|][A-Za-z0-9|]*", v):
                        out.refs[v] = rvars[0]
    for v, r in out.refs.items():                           # resolve A_i <= N
        if v not in out.upper and r in out.upper:
            out.upper[v] = out.upper[r]
    return out


def parse_statement(html_text: str) -> Constraints:
    sec = constraints_text(html_text)
    return Constraints(found_section=False) if sec is None else parse_text(sec)
