"""tleye check: verdict (P(TLE)) + sandbox certificate for ONE solution against ONE problem.

    python -I -m tleye.cli check sol.py --problem p03459 [--lang Python] [--runtime PyPy] [--certify] [--prior 0.1] [--json]
    python -I -m tleye.cli check sol.cpp --statement stmt.html --time-limit-ms 2000 --certify

Three-way outcome (never claims safety):
  CERTIFIED TLE  - a generated input made the program use >= the time limit in the sandbox (OUR hardware, not the judge)
  SUSPECTED TLE  - model flags it but no certificate (no input failed, or the problem/format is not supported yet)
  LIKELY AC      - model does not flag it; an estimate, NOT a proof
The solution is only parsed as text here; it executes ONLY inside the Docker sandbox, and only with --certify.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

R = Path("data/raw/extracted/Project_CodeNet")
EXT = {".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".java": "Java", ".py": "Python"}


def _shift(p: float, train_prev: float, deploy_prev: float) -> float:
    """Move a probability from the capped training prevalence to an assumed deployment prevalence (logit shift)."""
    import math
    p = min(max(p, 1e-9), 1 - 1e-9)
    z = math.log(p / (1 - p)) + math.log(deploy_prev / (1 - deploy_prev)) - math.log(train_prev / (1 - train_prev))
    return 1 / (1 + math.exp(-z))


def _cached(cache: dict | None, key: str, make):
    """Keep expensive model objects alive between calls (the web demo passes a dict; the CLI passes None)."""
    if cache is None:
        return make()
    if key not in cache:
        cache[key] = make()
    return cache[key]


def check(sol: Path, language: str, runtime: str, html: str, time_limit_ms: float, do_certify: bool, prior: float | None,
          model: str = "auto", device: str = "cpu", cache: dict | None = None) -> dict:
    from tleye.constraints.parser import parse_statement
    from tleye.models import lora

    code = sol.read_bytes()
    use_lora = lora.available() if model == "auto" else model == "lora"
    if use_lora:
        if not lora.available():
            raise SystemExit("LoRA model unavailable: install the `train` extras (torch, transformers, peft) and keep models/lora/")
        m = _cached(cache, "lora", lambda: lora.LoraVerdict(device=device))
        r = m.predict(code, language, runtime, html, time_limit_ms)
        p, thr, train_prev, name = r["p_tle"], r["threshold"], m.meta["train_prevalence"], "LoRA Qwen2.5-Coder-0.5B (confirmed model)"
    else:
        from tleye.models.b2 import B2
        b2 = _cached(cache, "b2m", lambda: B2(variant="monotone"))
        p = float(b2.predict_proba(b2.featurize(code, language, runtime, html, time_limit_ms))[0])
        thr, train_prev, name = float(b2.meta["thr_f1_raw_val"]), b2.meta["train_prevalence"], "B2m monotone LightGBM (CPU)"
    c = parse_statement(html)
    out = {"model": name, "language": language, "runtime": runtime, "time_limit_ms": time_limit_ms,
           "parsed": {"primary_size_bound": c.primary_size_bound, "max_bound": c.max_bound, "n_vars_bounded": len(c.upper)},
           "p_tle_model": round(p, 4), "flag_threshold": round(thr, 4), "flagged": p >= thr,
           "p_tle_at_assumed_prevalence": None if prior is None else round(_shift(p, train_prev, prior), 4),
           "note": "model probability is calibrated to the capped training prevalence (~0.39), not to real-world TLE rates"}
    if not out["flagged"]:
        out["outcome"] = "LIKELY AC"; out["detail"] = "model does not flag this solution (an estimate, not a proof)"
        return out
    out["outcome"] = "SUSPECTED TLE"
    if not do_certify:
        out["detail"] = "flagged by the model; run with --certify to try to confirm with a generated input"
        return out
    from tleye.certify.bounds import parse as cert_parse
    from tleye.certify.coverage import classify
    from tleye.certify.generate import GenError, build_inputs
    from tleye.certify.inputspec import parse_input_spec
    from tleye.certify.runner import certify_submission
    from tleye.constraints.parser import constraints_text

    cat, why = classify(html)
    out["certificate_support"] = {"category": cat, "detail": why}
    if cat != "simple":
        out["detail"] = f"flagged; no certificate attempted: problem not supported yet ({cat}: {', '.join(why) or 'n/a'})"
        return out
    sec = constraints_text(html) or ""
    try:
        battery = build_inputs(parse_input_spec(html), cert_parse(sec), sec)
    except GenError as e:
        out["detail"] = f"flagged; could not generate inputs: {e}"
        return out
    rt = "PyPy" if runtime.startswith("PyPy") else language
    cert = certify_submission(sol, rt, battery, time_limit_ms)
    out["certificate"] = {"status": cert.status, "runs": cert.runs, "max_cpu_s": cert.max_cpu_s}
    if cert.status == "certified":
        worst = max(cert.runs, key=lambda r: r["cpu_s"])
        out["outcome"] = "CERTIFIED TLE"
        out["detail"] = (f"input '{worst['input']}' used {worst['cpu_s']:.2f}s CPU (limit {time_limit_ms / 1000:g}s) in the sandbox; "
                         "measured on THIS machine's arm64 container, not the contest judge")
    elif cert.status == "not_certified":
        out["detail"] = "flagged by the model, but none of the generated inputs reached the limit. NOT reported as safe"
    else:
        out["detail"] = f"flagged by the model; certificate inconclusive ({cert.status}). NOT reported as safe"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tleye")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("solution", type=Path)
    c.add_argument("--problem", help="CodeNet AtCoder problem id, e.g. p03459")
    c.add_argument("--statement", type=Path, help="statement HTML (AtCoder format) for problems outside CodeNet")
    c.add_argument("--time-limit-ms", type=float)
    c.add_argument("--lang", choices=["C++", "Java", "Python"])
    c.add_argument("--runtime", default=None, help="Python only: 'PyPy' for PyPy (default CPython)")
    c.add_argument("--certify", action="store_true", help="run the sandbox certificate for flagged solutions (needs Docker/Colima)")
    c.add_argument("--prior", type=float, help="assumed real TLE prevalence, to report a shifted probability")
    c.add_argument("--model", choices=["auto", "lora", "b2m"], default="auto",
                   help="auto = the confirmed LoRA model if torch/transformers/peft are installed, else the CPU LightGBM B2m")
    c.add_argument("--device", default="cpu", help="LoRA only: cpu (default, verified) or mps")
    c.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    sol = a.solution.resolve()
    lang = a.lang or EXT.get(sol.suffix.lower())
    if not lang:
        print("cannot infer language from extension; pass --lang", file=sys.stderr); return 2
    runtime = (a.runtime or lang) if lang == "Python" else lang
    if a.problem:
        f = R / "problem_descriptions" / f"{a.problem}.html"
        pl = pd.read_csv(R / "metadata" / "problem_list.csv").set_index("id")
        if not f.exists() or a.problem not in pl.index:
            print(f"unknown problem {a.problem}", file=sys.stderr); return 2
        html, tl = f.read_text(errors="replace"), float(a.time_limit_ms or pl.loc[a.problem, "time_limit"])
    elif a.statement and a.time_limit_ms:
        html, tl = a.statement.read_text(errors="replace"), a.time_limit_ms
    else:
        print("give --problem, or --statement together with --time-limit-ms", file=sys.stderr); return 2
    res = check(sol, lang, runtime, html, tl, a.certify, a.prior, a.model, a.device)
    if a.json:
        print(json.dumps(res, indent=1, default=float))
    else:
        print(f"parsed constraints : primary size <= {res['parsed']['primary_size_bound']}, max bound <= {res['parsed']['max_bound']}")
        print(f"model              : {res['model']}")
        print(f"P(TLE) model       : {res['p_tle_model']}  (flag threshold {res['flag_threshold']})")
        if res["p_tle_at_assumed_prevalence"] is not None:
            print(f"P(TLE) @ prevalence: {res['p_tle_at_assumed_prevalence']}")
        print(f"OUTCOME            : {res['outcome']}\n{res['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
