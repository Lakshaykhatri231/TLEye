"""TLEye web demo (Gradio). Paste a solution, pick a CodeNet AtCoder problem (or paste a statement), get the three-way outcome.

    python -I -m tleye.app [--model auto|lora|b2m] [--port 7860]

ONE model per process (macOS: LightGBM and the LoRA model must not be loaded in the same process). Binds to 127.0.0.1 only.
The solution is only parsed as text unless 'Try to certify' is ticked, in which case it runs ONLY inside the Docker sandbox (needs Docker/Colima).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from tleye.cli import R, check

EXT = {"C++": ".cpp", "Java": ".java", "Python": ".py"}
CACHE: dict = {}
MODEL_CHOICE = "auto"
HOSTED = bool(os.environ.get("TLEYE_HOSTED"))          # e.g. a Hugging Face Space: no Docker, no CodeNet files
EXAMPLES = Path(__file__).parent / "examples"
# Constraints of the example problem (p02779), written out by hand, so hosted demos do not need CodeNet's statement files.
EXAMPLE_STATEMENT = "<h3>Constraints</h3><ul><li>2 \\leq N \\leq 200000</li><li>1 \\leq A_i \\leq 10^9</li><li>All values in input are integers.</li></ul>"


def can_certify() -> bool:
    return not HOSTED and shutil.which("docker") is not None


def has_codenet() -> bool:
    return (R / "metadata" / "problem_list.csv").exists()


def run_check(code: str, language: str, python_runtime: str, problem_id: str, statement_html: str, time_limit_s: float, certify: bool, summarize: bool = False) -> tuple[str, dict]:
    """Return (markdown summary, raw result dict). Pure function so it can be tested without a browser."""
    if not code.strip():
        return "Paste a solution first.", {}
    if certify and not can_certify():
        return "Certificates need a local Docker sandbox and are not available here; untick 'Try to certify'.", {}
    problem_id = (problem_id or "").strip()
    if statement_html.strip():
        html, tl = statement_html, float(time_limit_s) * 1000.0
        if not time_limit_s or time_limit_s <= 0:
            return "A pasted statement needs a time limit in seconds.", {}
    elif problem_id:
        f = R / "problem_descriptions" / f"{problem_id}.html"
        pl = pd.read_csv(R / "metadata" / "problem_list.csv").set_index("id")
        if not f.exists() or problem_id not in pl.index:
            return f"Unknown problem id `{problem_id}` (use a CodeNet AtCoder id such as `p02779`, or paste a statement).", {}
        html, tl = f.read_text(errors="replace"), float(pl.loc[problem_id, "time_limit"])
    else:
        return "Enter a problem id (e.g. `p02779`) or paste a statement with a time limit.", {}
    runtime = python_runtime if language == "Python" else language
    with tempfile.TemporaryDirectory(dir=Path("data/scratch").resolve() if Path("data/scratch").exists() else None) as td:
        src = Path(td) / f"sol{EXT[language]}"
        src.write_text(code)
        res = check(src, language, runtime, html, tl, certify, None, model=MODEL_CHOICE, cache=CACHE)
        if summarize:
            from tleye.models import explain  # its own process; cannot change the verdict
            res["explanation"] = explain.explain(src.read_bytes(), language, runtime, html, tl)
    icon = {"CERTIFIED TLE": "🔴", "SUSPECTED TLE": "🟠", "LIKELY AC": "🟢"}[res["outcome"]]
    md = [f"## {icon} {res['outcome']}", "", res["detail"], "",
          f"- **Model:** {res['model']}",
          f"- **P(TLE):** {res['p_tle_model']} (flag threshold {res['flag_threshold']}). Calibrated to a ~40%-TLE training sample, not to real-world rates.",
          f"- **Parsed size bounds:** main size ≤ {res['parsed']['primary_size_bound']}, largest bound ≤ {res['parsed']['max_bound']}",
          f"- **Time limit used:** {tl / 1000:g}s"]
    cert = res.get("certificate")
    if cert:
        md += ["", f"**Sandbox certificate:** {cert['status']} (measured on this machine's container, not the contest judge)", "",
               "| generated input | exit code | CPU seconds |", "|---|---|---|"] + [f"| {r['input']} | {r['rc']} | {r['cpu_s']:.2f} |" for r in cert["runs"]]
    elif res.get("certificate_support"):
        cs = res["certificate_support"]
        md += ["", f"*No certificate attempted: problem category `{cs['category']}` {cs['detail'] or ''}.*"]
    if "explanation" in res:
        from tleye.models import explain
        md += ["", explain.render_markdown(res["explanation"])]
    md += ["", "*Never reported as safe: LIKELY AC is an estimate, SUSPECTED TLE means no input was confirmed.*"]
    return "\n".join(md), res


def build_ui():
    import gradio as gr

    from tleye.models import explain
    codenet, certify_ok, explain_ok = has_codenet(), can_certify(), explain.available()
    examples = [[p.read_text(), "C++", "CPython", "", EXAMPLE_STATEMENT, 2.0, False, False] for p in
                (EXAMPLES / "p02779_slow.cpp", EXAMPLES / "p02779_fast.cpp") if p.exists()]
    with gr.Blocks(title="TLEye") as demo:
        gr.Markdown("# TLEye\nWill this competitive-programming solution hit **Time Limit Exceeded**? Paste the code and the problem's constraints.\n\n"
                    + ("*Code is only read as text here, never run.*" if not certify_ok else "*Code only executes inside the Docker sandbox, and only if you tick the box.*")
                    + " Source: [GitHub](https://github.com/Lakshaykhatri231/TLEye) · model: [Hugging Face](https://huggingface.co/LakshayKhatri231/tleye-lora-0.5b)")
        with gr.Row():
            with gr.Column(scale=3):
                code = gr.Code(label="Solution", language="cpp", lines=22)
                with gr.Row():
                    lang = gr.Dropdown(["C++", "Java", "Python"], value="C++", label="Language")
                    rt = gr.Dropdown(["CPython", "PyPy"], value="CPython", label="Python runtime (Python only)")
                with gr.Row():
                    pid = gr.Textbox(label="CodeNet AtCoder problem id", placeholder="p02779", visible=codenet)
                    tls = gr.Number(label="Time limit (s)", value=0)
                stmt = gr.Textbox(label="Problem statement or just its constraints (AtCoder HTML)" if not codenet else "...or paste an AtCoder statement (HTML)", lines=3)
                cert = gr.Checkbox(label="Try to certify in the sandbox (needs Docker; can take up to a minute)", value=False, visible=certify_ok)
                summ = gr.Checkbox(label="Add a code summary checked against the AST (slower; does not change the verdict)", value=False, visible=explain_ok)
                btn = gr.Button("Check", variant="primary")
            with gr.Column(scale=2):
                out = gr.Markdown()
                raw = gr.JSON(label="Raw result", open=False)
        btn.click(run_check, [code, lang, rt, pid, stmt, tls, cert, summ], [out, raw])
        lang.change(lambda l: gr.update(language={"C++": "cpp", "Java": "java", "Python": "python"}[l]), lang, code)
        if examples:
            gr.Examples(examples, [code, lang, rt, pid, stmt, tls, cert, summ], label="Examples: real AtCoder submissions (p02779: are all N numbers distinct?)")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["auto", "lora", "b2m"], default="auto"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(); MODEL_CHOICE = a.model
    build_ui().launch(server_name="127.0.0.1", server_port=a.port)
