"""TLEye web demo (Gradio). Paste a solution, pick a CodeNet AtCoder problem (or paste a statement), get the three-way outcome.

    python -I -m tleye.app [--model auto|lora|b2m] [--port 7860]

ONE model per process (macOS: LightGBM and the LoRA model must not be loaded in the same process). Binds to 127.0.0.1 only.
The solution is only parsed as text unless 'Try to certify' is ticked, in which case it runs ONLY inside the Docker sandbox (needs Docker/Colima).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from tleye.cli import R, check

EXT = {"C++": ".cpp", "Java": ".java", "Python": ".py"}
CACHE: dict = {}
MODEL_CHOICE = "auto"


def run_check(code: str, language: str, python_runtime: str, problem_id: str, statement_html: str, time_limit_s: float, certify: bool) -> tuple[str, dict]:
    """Return (markdown summary, raw result dict). Pure function so it can be tested without a browser."""
    if not code.strip():
        return "Paste a solution first.", {}
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
    md += ["", "*Never reported as safe: LIKELY AC is an estimate, SUSPECTED TLE means no input was confirmed.*"]
    return "\n".join(md), res


def build_ui():
    import gradio as gr

    ex_dir = Path("tleye/examples")
    examples = [[p.read_text(), "C++", "CPython", pid, "", 0.0, False] for pid, p in
                [("p02779", ex_dir / "p02779_slow.cpp"), ("p02779", ex_dir / "p02779_fast.cpp")] if p.exists()]
    with gr.Blocks(title="TLEye") as demo:
        gr.Markdown("# TLEye\nWill this solution hit **Time Limit Exceeded**? Paste code, pick a problem, optionally try to confirm with a sandboxed run.\n\n"
                    "*Runs locally. Code only executes inside the Docker sandbox, and only if you tick the box.*")
        with gr.Row():
            with gr.Column(scale=3):
                code = gr.Code(label="Solution", language="cpp", lines=22)
                with gr.Row():
                    lang = gr.Dropdown(["C++", "Java", "Python"], value="C++", label="Language")
                    rt = gr.Dropdown(["CPython", "PyPy"], value="CPython", label="Python runtime (Python only)")
                with gr.Row():
                    pid = gr.Textbox(label="CodeNet AtCoder problem id", placeholder="p02779")
                    tls = gr.Number(label="Time limit (s), only for a pasted statement", value=0)
                stmt = gr.Textbox(label="...or paste an AtCoder statement (HTML)", lines=3)
                cert = gr.Checkbox(label="Try to certify in the sandbox (needs Docker/Colima; can take up to a minute)", value=False)
                btn = gr.Button("Check", variant="primary")
            with gr.Column(scale=2):
                out = gr.Markdown()
                raw = gr.JSON(label="Raw result", open=False)
        btn.click(run_check, [code, lang, rt, pid, stmt, tls, cert], [out, raw])
        lang.change(lambda l: gr.update(language={"C++": "cpp", "Java": "java", "Python": "python"}[l]), lang, code)
        if examples:
            gr.Examples(examples, [code, lang, rt, pid, stmt, tls, cert], label="Examples (real AtCoder submissions on p02779)")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["auto", "lora", "b2m"], default="auto"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(); MODEL_CHOICE = a.model
    build_ui().launch(server_name="127.0.0.1", server_port=a.port)
