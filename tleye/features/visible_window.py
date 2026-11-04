"""Explanation target computed on the code the model actually SEES (the decoded 1024-token prompt window).

Why not the old `sym` string: it used a fill value when N was unknown, a fallback to the largest bound, depth clipped to at least 1 even
with no loops, and it described code beyond the window on truncated prompts. Here unknown stays `unknown`, and every field can be checked.
Written to a sidecar parquet joined on submission_id; the data pack is never rewritten (row order feeds df.sample in the Kaggle scripts).

    python -I -m tleye.features.visible_window      -> data/processed/kaggle_sidecar/sft_targets.parquet
"""
from __future__ import annotations

import math
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from tleye.constraints.parser import parse_statement
from tleye.features.ast_features import extract

ROOT = Path("data/raw/extracted/Project_CodeNet/problem_descriptions")
PRIMARY_ORDER = ("N", "n", "|S|", "|s|", "|A|", "H", "h", "R", "Q", "M")   # same order as Constraints.primary_size_bound
MAX_LOG10 = 30.0                                                           # beyond this the bound is a value, not a loop size
MAX_LEN = 1024
SUFFIX = "\n### verdict:"


def header(rt: str, time_limit_ms: float, constraints: str) -> str:
    """Everything before the code; byte-identical to the classifier's prompt (no prefix)."""
    return f"language: {rt}\ntime limit: {time_limit_ms / 1000:g}s\nconstraints: {constraints}\ncode:\n"


def primary_var(upper: dict, constraints_line: str) -> tuple[str, float | None]:
    """(variable, log10 bound) for the main size variable, or ('unknown', None) if absent, not shown to the model, or absurdly large."""
    for v in PRIMARY_ORDER:
        if v in upper and f"{v}<=" in constraints_line:
            if upper[v] <= 0:
                return "unknown", None
            lg = math.log10(min(upper[v], 1e300))
            return (v, lg) if lg <= MAX_LOG10 else ("unknown", None)
    return "unknown", None


def target_string(f: dict, n_var: str, log10_n: float | None) -> str:
    """The analysis text. est_log10_ops = max(depth, 1) x log10_n, stated only when N is known."""
    depth = int(f["max_loop_depth"])
    if log10_n is None:
        ln, est = "unknown", "unknown"
    else:
        ln, est = f"{log10_n:.1f}", f"{max(depth, 1) * log10_n:.1f}"
    return (f"loop_depth={depth} loops={int(f['n_loops'])} recursion={'yes' if f['recursive'] else 'no'} sorts={int(f['n_sort'])} "
            f"sort_in_loop={int(f['sort_in_loop'])} listed_calls_in_loop={int(f['linear_in_loop'])} n_var={n_var} log10_n={ln} est_log10_ops={est}")


FIELD_RE = re.compile(r"loop_depth=(\d+) loops=(\d+) recursion=(yes|no) sorts=(\d+) sort_in_loop=(\d+) listed_calls_in_loop=(\d+) "
                      r"n_var=(\S+) log10_n=(unknown|[\d.]+) est_log10_ops=(unknown|[\d.]+)")


def parse_target(s: str) -> dict | None:
    """Back from text to fields; None if malformed. Used to score generated analyses."""
    m = FIELD_RE.search(s)
    if not m:
        return None
    g = m.groups()
    fl = lambda x: None if x == "unknown" else float(x)  # noqa: E731
    return {"loop_depth": int(g[0]), "loops": int(g[1]), "recursion": g[2] == "yes", "sorts": int(g[3]), "sort_in_loop": int(g[4]),
            "listed_calls_in_loop": int(g[5]), "n_var": g[6], "log10_n": fl(g[7]), "est_log10_ops": fl(g[8])}


def visible_code(tok, head: str, code: str, n_suffix: int, max_len: int = MAX_LEN) -> tuple[str, bool, int]:
    """(code visible to the model, was it truncated, untruncated prompt token count)."""
    ids = tok(head + code, add_special_tokens=False, truncation=False)["input_ids"]
    keep = max_len - n_suffix
    text = tok.decode(ids[:keep], skip_special_tokens=False)
    assert text.startswith(head[:20]), "prompt header lost in decoding"
    return (text[len(head):] if text.startswith(head) else text.split("code:\n", 1)[-1]), len(ids) > keep, len(ids)


def _work(a: tuple[str, str, bool]) -> dict:
    return extract(a[0].encode("utf8", "replace"), a[1])


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
    n_suffix = len(tok(SUFFIX, add_special_tokens=False)["input_ids"])
    d = pd.read_parquet("data/processed/kaggle_pack/tleye_data.parquet", columns=["submission_id", "problem_id", "split", "language", "rt", "time_limit_ms", "constraints", "code"])
    prim = {}
    for pid, line in d.groupby("problem_id").constraints.first().items():
        prim[pid] = primary_var(parse_statement((ROOT / f"{pid}.html").read_text(errors="replace")).upper, line)
    vis, trunc, ntok = [], [], []
    for r in d.itertuples():
        v, t, n = visible_code(tok, header(r.rt, r.time_limit_ms, r.constraints), r.code, n_suffix)
        vis.append(v); trunc.append(t); ntok.append(n)
    with Pool(6) as pool:
        feats = pool.map(_work, [(v, l, False) for v, l in zip(vis, d.language)], chunksize=2000)
    full = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "max_loop_depth", "n_loops", "recursive", "n_sort", "sort_in_loop", "linear_in_loop"]).set_index("submission_id")
    rows = []
    for sid, pid, f, t, n in zip(d.submission_id, d.problem_id, feats, trunc, ntok):
        nv, ln = prim[pid]
        fu = full.loc[sid]
        rows.append({"submission_id": sid, "truncated": t, "n_prompt_tokens": n, "n_var": nv, "log10_n": ln, "target": target_string(f, nv, ln),
                     "vis_loop_depth": int(f["max_loop_depth"]), "vis_loops": int(f["n_loops"]), "vis_recursive": int(f["recursive"]), "vis_sorts": int(f["n_sort"]),
                     "vis_sort_in_loop": int(f["sort_in_loop"]), "vis_listed": int(f["linear_in_loop"]),
                     "full_loop_depth": int(fu.max_loop_depth), "full_recursive": int(fu.recursive)})
    o = pd.DataFrame(rows)
    out = Path("data/processed/kaggle_sidecar"); out.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out / "sft_targets.parquet", index=False)
    tr = o[o.truncated]
    print(f"rows {len(o)} | truncated {o.truncated.mean():.1%} | N unknown {o.n_var.eq('unknown').mean():.1%}")
    print(f"truncated rows where visible depth differs from full-code depth: {(tr.vis_loop_depth != tr.full_loop_depth).mean():.1%} (n={len(tr)})")
    print(f"untruncated rows where they differ (should be near 0): {(o[~o.truncated].vis_loop_depth != o[~o.truncated].full_loop_depth).mean():.2%}")
    print("target tokens: median", int(o.target.sample(500, random_state=0).map(lambda s: len(tok(s)["input_ids"])).median()), "| example:", o.target.iloc[1000])


if __name__ == "__main__":
    main()
