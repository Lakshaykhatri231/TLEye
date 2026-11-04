"""Certificate coverage spike: for each AtCoder problem, can we GENERATE a valid-looking max-size input?

Categories (first match wins): special_io / no_input_section / unsupported_format / unbounded_variable /
semantic_risk (format ok + bounded, but constraints text contains structure we cannot yet guarantee) / simple.
    python -I -m tleye.certify.coverage
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tleye.certify.bounds import parse as cert_parse
from tleye.certify.inputspec import parse_input_spec
from tleye.constraints.parser import REL, constraints_text

R = Path("data/raw/extracted/Project_CodeNet")
FLAGS = {
    "tree": r"\btrees?\b", "permutation": r"permutation", "distinct": r"distinct|pairwise|different|unique|no two|are all different",
    "connected": r"connected", "sorted_chain": r"sorted|increasing|decreasing|\w_\{?\w+\}?\s*<\s*\w_\{?\w+\}?\s*<",
    "sum_limit": r"sum of|in total|total length|\\sum", "graph": r"\bgraph\b", "guarantee": r"guarantee|it is possible|always possible",
    "scalar_strict": r"(?<![_\w\\])[a-z]\s*<\s*[a-z](?![_\w])",
    "var_order": r"[A-Za-z]_\{?\w+\}?\s*(?:<|\\leq|≤|≦)\s*[A-Za-z]_\{?\w+\}?",
    "prime": r"\bprime", "other_relation": r"\\neq|\\ne\b|≠|divisible|multiple of",
}


def base(n: str) -> str:
    m = re.match(r"[A-Za-z]+", n)
    return m.group(0) if m else n


BENIGN_PROSE = [   # constraint sentences that need no special generation (or are handled elsewhere)
    r"all (?:the )?(?:input )?(?:values|numbers|given numbers|elements)(?: in (?:the )?input)? are integers?", r"[\w, _{}\\]+ (?:is|are) (?:an? )?integers?(?: number)?s?\b(?!.*(?:even|odd))",
    r"consists? (?:only )?of (?:the )?(?:lowercase|uppercase)? ?english letters?", r"(?:lowercase|uppercase) english letters?", r"\(\s*a\s*-\s*z\s*\)",
    r"\bis an? (?:non-?empty )?string\b", r"^\W*$", r"^\s*\|\s*[a-z]\s*\|\s*=", r"\bis \w+ characters? long\b",
    r"\blengths? of [a-z](?: and [a-z])? (?:is|are)(?: each)? (?:between|at least|at most)\b.*", r"\b[a-z](?: and [a-z])? have lengths? between\b",
    r"partial|subtask|subscore|points?\b.*awarded|^\s*score", r"^\W*all values in\W*$",
]


def residual_prose(sec: str) -> list[str]:
    """Constraint sentences that no check explains. If any is left, we cannot promise a generated input respects the problem's rules."""
    out = []
    for ln in sec.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln or REL.search(ln):
            continue
        low = ln.lower()
        if any(re.search(b, low) for b in BENIGN_PROSE):
            continue
        out.append(ln[:80])
    return out


def _drop_handled_relation_flags(flags: list[str], spec, c, sec: str) -> list[str]:
    """`a_i < b_i` / `a_i != b_i` are generated exactly when both are columns of one table and every occurrence in the text was captured."""
    if c is None or not c.relations:
        return flags
    tables = [{base(x) for x in ln[1]} for ln in spec.lines if ln[0] == "table"]
    if not all(any({a_, b_} <= t for t in tables) for a_, _, b_ in c.relations):
        return flags
    low = sec.lower()
    n_order = len(re.findall(r"[a-za-z]_\{?\w+\}?\s*(?:<|\\leq|≤|≦)\s*[a-za-z]_\{?\w+\}?", low))
    n_neq = len(re.findall(r"\\neq|\\ne\b|≠", low))
    got_order = sum(1 for _, o, _ in c.relations if o in ("<", "<="))
    got_neq = sum(1 for _, o, _ in c.relations if o == "!=")
    out = list(flags)
    if "var_order" in out and n_order == got_order:
        out.remove("var_order")
    if "other_relation" in out and n_neq == got_neq and not re.search(r"divisible|multiple of", low):
        out.remove("other_relation")
    return out


def classify(html_text: str) -> tuple[str, list[str]]:
    spec = parse_input_spec(html_text)
    if spec.reason == "interactive_or_special_io":
        return "special_io", []
    if spec.reason == "no_input_section":
        return "no_input_section", []
    if not spec.ok:
        return "unsupported_format", [spec.reason or "empty"]
    sec = constraints_text(html_text) or ""
    c = cert_parse(sec) if sec else None
    upper = c.upper if c else {}
    need = {base(n) for n in spec.names()}
    for ln in spec.lines:                                # variables used inside length expressions
        if ln[0] in ("array", "table"):
            need |= {base(x) for x in re.findall(r"[A-Za-z]+", ln[-1])}
    # a scalar like `S` may be a string bounded as |S|; accept either form
    missing = sorted(n for n in need if n not in upper and f"|{n}|" not in upper)
    if missing:
        return "unbounded_variable", missing
    low = sec.lower()
    flags = [k for k, rx in FLAGS.items() if re.search(rx, low)]
    flags = _drop_handled_relation_flags(flags, spec, c, sec)
    resid = residual_prose(sec)
    if resid and not flags:
        return "unmodeled_text", resid[:2]
    return ("semantic_risk", flags) if flags else ("simple", [])


def main() -> None:
    pl = pd.read_csv(R / "metadata" / "problem_list.csv")
    sp = pd.read_parquet("data/processed/splits.parquet").set_index("problem_id")["split"]
    rows = []
    for pid in pl[pl.dataset == "AtCoder"].id:
        cat, why = classify((R / "problem_descriptions" / f"{pid}.html").read_text(errors="replace"))
        rows.append(dict(problem_id=pid, split=sp.get(pid, "none"), category=cat, detail=";".join(why)))
    d = pd.DataFrame(rows)
    d.to_parquet("data/processed/cert_coverage.parquet")
    tab = d.groupby(["category"]).size().rename("all").to_frame().join(
        d[d.split == "test"].groupby("category").size().rename("test")).fillna(0).astype(int)
    out = ["# Certificate coverage spike (script-generated)", "",
           f"AtCoder problems: {len(d)} (test split: {(d.split == 'test').sum()})", "", tab.to_markdown(), "",
           "## semantic_risk breakdown (a problem can have several flags)", ""]
    fl = d[d.category == "semantic_risk"].detail.str.split(";").explode().value_counts()
    out += [fl.to_markdown(), "", "## unsupported_format reasons", "",
            d[d.category == "unsupported_format"].detail.value_counts().to_markdown(), ""]
    Path("reports/cert_coverage.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
