"""Checkable code summary: the explanation model writes field values, and a real AST check on the same visible code is shown beside them.

The summary is NOT the reason for the verdict: replacing it barely moves the explanation model's own verdict (reports/sft_test.md, labelled "decorative").
The verdict always comes from the verdict model; nothing here can change it. Source code is only tokenized and parsed, never executed.

Loaded in its OWN PROCESS (macOS: LightGBM and these models must not share one):
    python -I -m tleye.models.explain   < request.json   (the CLI and the demo call this through `explain()`)
Adapter: models/explain, seed 0 of the Stage 5 run, the same seed as the deployed verdict model; it was not picked using test results.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "models" / "explain"
FAITHFULNESS_NOTE = ("A checkable code summary, not the reason for the verdict: on the test split, replacing it barely moved the explanation model's own verdict "
                     "(reports/sft_test.md). The AST column is the real check, on the part of the code the model can see.")
LABELS = {"loop_depth": "deepest loop nesting", "loops": "number of loops", "recursion": "recursion", "sorts": "sort calls", "sort_in_loop": "sorts inside a loop",
          "listed_calls_in_loop": "listed calls inside a loop (min, max, set and similar)", "n_var": "main size variable", "log10_n": "log10 of its bound", "est_log10_ops": "rough log10 operations, depth x log10 N"}
TOLERANCE = 0.05          # numeric fields are written with one decimal


def _kernel_lib():
    """The same template code the Kaggle run used, not a copy of it."""
    spec = importlib.util.spec_from_file_location("kernel_lib", ROOT / "kaggle" / "lib" / "kernel_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def available() -> bool:
    """Looks for the packages WITHOUT importing them: this runs in the parent, which may already hold LightGBM (importing torch there segfaults on macOS)."""
    return all(importlib.util.find_spec(m) for m in ("peft", "torch", "transformers")) and (ADAPTER / "adapter_model.safetensors").exists()


def same_value(a: str, b: str) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return math.isfinite(x) and math.isfinite(y) and abs(x - y) <= TOLERANCE


def compare(fields: list[str], model_values: list[str], ast_values: list[str]) -> list[dict]:
    return [{"field": f, "label": LABELS[f], "model": m, "ast": a, "match": same_value(m, a)} for f, m, a in zip(fields, model_values, ast_values)]


def ast_summary(tok, code: bytes, language: str, rt: str, html: str, time_limit_ms: float, n_suffix: int) -> tuple[str, bool]:
    """The same tree-sitter target the model was trained on, on the same visible window. (target text, was the code cut off)."""
    from tleye.constraints.parser import parse_statement
    from tleye.data.kaggle_pack import MAX_CHARS, constraint_text_from_html
    from tleye.features.ast_features import extract
    from tleye.features.visible_window import header, primary_var, target_string, visible_code

    line = constraint_text_from_html(html)
    text = code.decode("utf8", "replace")[:MAX_CHARS]
    vis, truncated, _ = visible_code(tok, header(rt, time_limit_ms, line), text, n_suffix)
    nv, ln = primary_var(parse_statement(html).upper, line)
    return target_string(extract(vis.encode("utf8", "replace"), language), nv, ln), truncated


class Explainer:
    def __init__(self, device: str = "cpu"):
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        transformers.utils.logging.set_verbosity_error()
        self.kl, self.torch, self.device = _kernel_lib(), torch, device
        base_name = json.loads((ADAPTER / "adapter_config.json").read_text())["base_model_name_or_path"]
        self.tok = AutoTokenizer.from_pretrained(base_name)
        self.suffix = self.tok(self.kl.SUFFIX, add_special_tokens=False)["input_ids"]
        self.comp = self.kl.Completion(self.tok)
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, str(ADAPTER)).eval().to(device)

    def generate(self, prompt: str) -> list[str]:
        """Forced template, model-filled values: the same procedure as the Kaggle run, one row at a time."""
        ids = self.tok(prompt, truncation=True, max_length=self.kl.MAX_LEN - len(self.suffix), add_special_tokens=False)["input_ids"] + self.suffix
        t = self.torch
        machine = self.comp.machine()
        with t.no_grad():
            o = self.model(input_ids=t.tensor([ids], device=self.device), use_cache=True)
            for _ in range(200):
                if machine.done:
                    break
                fed = machine.next(int(o.logits[0, -1].argmax()))
                o = self.model(input_ids=t.tensor([[fed]], device=self.device), past_key_values=o.past_key_values, use_cache=True)
        assert machine.done, "generation did not finish"
        return [v.strip() for v in machine.values]

    def explain(self, code: bytes, language: str, runtime: str, html: str, time_limit_ms: float) -> dict:
        from tleye.models.lora import build_prompt
        rt = "PyPy" if runtime.startswith("PyPy") else language
        vals = self.generate(build_prompt(code, language, runtime, html, time_limit_ms))
        target, truncated = ast_summary(self.tok, code, language, rt, html, time_limit_ms, len(self.suffix))
        rows = compare(self.kl.FIELDS, vals, self.kl.target_values(target))
        return {"fields": rows, "mismatches": sum(not r["match"] for r in rows), "code_cut_off": truncated, "note": FAITHFULNESS_NOTE}


def explain(code: bytes, language: str, runtime: str, html: str, time_limit_ms: float, device: str = "cpu", timeout: int = 600) -> dict:
    """Run in a child process. Failures come back as {"error": ...}: a summary problem must never change or hide the verdict."""
    if not available():
        return {"error": "explanation model unavailable: needs the `train` extras and models/explain/"}
    req = {"code": base64.b64encode(code).decode(), "language": language, "runtime": runtime, "html": html, "time_limit_ms": time_limit_ms, "device": device}
    try:
        r = subprocess.run([sys.executable, "-I", "-m", "tleye.models.explain"], input=json.dumps(req), capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"error": f"explanation timed out after {timeout} s"}
    if r.returncode != 0:
        return {"error": f"explanation process failed: {r.stderr[-300:]}"}
    return json.loads(r.stdout.split("EXPLAIN_JSON", 1)[1])


def render_markdown(ex: dict) -> str:
    if "error" in ex:
        return f"*No code summary: {ex['error']}*"
    rows = "\n".join(f"| {r['label']} | {r['model']} | {r['ast']} | {'yes' if r['match'] else '**NO**'} |" for r in ex["fields"])
    cut = " The code was cut off at the model's 1,024-token window, so both columns describe only the visible part." if ex["code_cut_off"] else ""
    return f"### Code summary ({ex['mismatches']} of {len(ex['fields'])} fields disagree with the AST check)\n\n| field | model says | AST check | match |\n|---|---|---|---|\n{rows}\n\n*{ex['note']}{cut}*"


def render_text(ex: dict) -> str:
    if "error" in ex:
        return f"code summary       : unavailable ({ex['error']})"
    out = [f"code summary       : {ex['mismatches']} of {len(ex['fields'])} fields disagree with the AST check"]
    out += [f"  {'' if r['match'] else '! '}{r['field']}: model {r['model']} | AST {r['ast']}" for r in ex["fields"]]
    if ex["code_cut_off"]:
        out.append("  code was cut off at the model's window: both columns describe only the visible part")
    return "\n".join(out + [ex["note"]])


def _child() -> None:
    q = json.loads(sys.stdin.read())
    res = Explainer(device=q["device"]).explain(base64.b64decode(q["code"]), q["language"], q["runtime"], q["html"], q["time_limit_ms"])
    print("EXPLAIN_JSON" + json.dumps(res))


if __name__ == "__main__":
    _child()
