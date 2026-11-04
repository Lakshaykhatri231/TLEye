"""Symbolic features from tree-sitter ASTs: SAME columns for C++, Java and Python.

All features are cheap structural signals (loop nesting, recursion, sort / linear-op calls inside
loops, I/O style). Source is parsed as bytes and never executed.
"""
from __future__ import annotations

import re

import tree_sitter as ts
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_python

_LANGS = {
    "C++": ts.Language(tree_sitter_cpp.language()),
    "Java": ts.Language(tree_sitter_java.language()),
    "Python": ts.Language(tree_sitter_python.language()),
}
_PARSERS: dict[str, ts.Parser] = {}

LOOPS = {
    "C++": {"for_statement", "while_statement", "do_statement", "for_range_loop"},
    "Java": {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"},
    "Python": {"for_statement", "while_statement", "for_in_clause"},      # for_in_clause = comprehension
}
FUNCS = {"C++": {"function_definition"}, "Java": {"method_declaration"}, "Python": {"function_definition"}}
CALLS = {"C++": {"call_expression"}, "Java": {"method_invocation"}, "Python": {"call"}}
SORTS = {"sort", "sorted", "stable_sort", "partial_sort", "nth_element"}
LINEAR_OPS = {  # O(n) (or worse) library calls that hurt when they sit inside a loop
    "find", "index", "count", "remove", "insert", "erase", "contains", "indexof", "lastindexof",
    "substr", "substring", "reverse", "copy", "pop", "strcat", "strlen", "memset", "fill", "sum",
    "min", "max", "join", "concat", "copyof", "copyofrange", "tolist", "list", "set", "dict", "deepcopy",
    "to_string", "append_left", "insert_at",
}
_LAST_IDENT = re.compile(r"([A-Za-z_]\w*)\s*(?:<[^<>]*>)?\s*$")
_FUNC_NAME = re.compile(r"([A-Za-z_]\w*)\s*\(")

FEATURE_NAMES = [
    "n_lines", "n_chars", "n_nodes", "n_loops", "max_loop_depth", "n_loops_ge2", "n_loops_ge3",
    "n_funcs", "recursive", "n_rec_calls", "n_sort", "sort_in_loop", "linear_in_loop", "n_calls_in_loop",
    "memo_hint", "io_fast", "io_slow_calls", "has_numpy", "has_lru_cache", "has_while_true",
]


def _parser(lang: str) -> ts.Parser:
    if lang not in _PARSERS:
        _PARSERS[lang] = ts.Parser(_LANGS[lang])
    return _PARSERS[lang]


def _text(node: ts.Node) -> str:
    return node.text.decode("utf8", "ignore") if node.text else ""


def _callee_name(node: ts.Node, lang: str) -> str:
    f = node.child_by_field_name("name" if lang == "Java" else "function")
    if f is None:
        return ""
    m = _LAST_IDENT.search(_text(f))
    return m.group(1) if m else ""


def _func_name(node: ts.Node, lang: str) -> str:
    if lang == "C++":
        d = node.child_by_field_name("declarator")
        m = _FUNC_NAME.search(_text(d)) if d is not None else None
        return m.group(1) if m else ""
    n = node.child_by_field_name("name")
    return _text(n) if n is not None else ""


def extract(code: bytes, lang: str) -> dict[str, float]:
    tree = _parser(lang).parse(code)
    loops, funcs, calls = LOOPS[lang], FUNCS[lang], CALLS[lang]
    f = dict.fromkeys(FEATURE_NAMES, 0)
    f["n_chars"], f["n_lines"] = len(code), code.count(b"\n") + 1
    # stack: (node, loop_depth, enclosing function name)
    stack = [(tree.root_node, 0, "")]
    while stack:
        node, depth, fname = stack.pop()
        f["n_nodes"] += 1
        t = node.type
        nd, nf = depth, fname
        if t in funcs:
            nf = _func_name(node, lang)
            f["n_funcs"] += 1
        if t in loops:
            f["n_loops"] += 1
            nd = depth + 1
            f["max_loop_depth"] = max(f["max_loop_depth"], nd)
            f["n_loops_ge2"] += nd >= 2
            f["n_loops_ge3"] += nd >= 3
        if t in calls:
            name = _callee_name(node, lang)
            low = name.lower()
            if depth > 0:
                f["n_calls_in_loop"] += 1
            if low in SORTS:
                f["n_sort"] += 1
                f["sort_in_loop"] += depth > 0
            if depth > 0 and low in LINEAR_OPS:
                f["linear_in_loop"] += 1
            if fname and name == fname:
                f["n_rec_calls"] += 1
        for ch in node.children:
            stack.append((ch, nd, nf))
    f["recursive"] = int(f["n_rec_calls"] > 0)
    s = code.decode("utf8", "ignore")
    f["memo_hint"] = len(re.findall(r"\b(?:memo|dp|cache|visited|seen)\w*", s, re.IGNORECASE))
    f["io_fast"] = int(bool(re.search(
        r"sync_with_stdio|scanf|getchar|fread|BufferedReader|StreamTokenizer|DataInputStream|sys\.stdin|readline|read\(\)", s)))
    f["io_slow_calls"] = len(re.findall(r"\bcin\s*>>|\bendl\b|new\s+Scanner|\binput\s*\(", s))
    f["has_numpy"] = int(bool(re.search(r"\bnumpy\b|\bnp\.", s)))
    f["has_lru_cache"] = int(bool(re.search(r"lru_cache|functools\.cache", s)))
    f["has_while_true"] = int(bool(re.search(r"while\s*\(?\s*(?:true|1|True)\s*\)?\s*[:{]", s)))
    return f
