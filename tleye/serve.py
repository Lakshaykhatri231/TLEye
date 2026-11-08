"""HTTP inference service:  uvicorn tleye.serve:app --host 0.0.0.0 --port 8000

    GET  /health   -> model name and readiness
    POST /check    -> verdict for one solution (same result as `tleye check`), optional code summary

One verdict model per process (TLEYE_MODEL = auto | lora | b2m; auto = LoRA if its weights and packages are present, else LightGBM).
Certificates are off unless TLEYE_ALLOW_CERTIFY=1: they need a Docker daemon, which a service container normally does not have.
Submitted code is only parsed and tokenized here; it is never executed by this process.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tleye.cli import EXT, R, check

MODEL = os.environ.get("TLEYE_MODEL", "auto")
ALLOW_CERTIFY = os.environ.get("TLEYE_ALLOW_CERTIFY") == "1"
MAX_CODE_BYTES = 200_000
CACHE: dict = {}
LOCK = threading.Lock()          # the models are not shared across threads; one request at a time per process
SUFFIX = {v: k for k, v in EXT.items() if k in (".cpp", ".java", ".py")}


class CheckRequest(BaseModel):
    code: str = Field(..., description="solution source code")
    language: Literal["C++", "Java", "Python"]
    runtime: Literal["CPython", "PyPy"] | None = Field(None, description="Python only")
    problem_id: str | None = Field(None, description="CodeNet AtCoder id, e.g. p02779 (needs the CodeNet files mounted)")
    statement_html: str | None = Field(None, description="AtCoder statement HTML, for problems outside CodeNet")
    time_limit_ms: float | None = Field(None, gt=0, description="required with statement_html")
    certify: bool = False
    explain: bool = False
    prior: float | None = Field(None, gt=0, lt=1)


def _problem(req: CheckRequest) -> tuple[str, float]:
    if req.statement_html:
        if not req.time_limit_ms:
            raise HTTPException(422, "statement_html needs time_limit_ms")
        return req.statement_html, req.time_limit_ms
    if req.problem_id:
        f = R / "problem_descriptions" / f"{req.problem_id}.html"
        meta = R / "metadata" / "problem_list.csv"
        if not (f.exists() and meta.exists()):
            raise HTTPException(404, f"problem {req.problem_id} not available here: send statement_html and time_limit_ms instead")
        pl = pd.read_csv(meta).set_index("id")
        if req.problem_id not in pl.index:
            raise HTTPException(404, f"unknown problem {req.problem_id}")
        return f.read_text(errors="replace"), float(req.time_limit_ms or pl.loc[req.problem_id, "time_limit"])
    raise HTTPException(422, "give problem_id, or statement_html with time_limit_ms")


def run(req: CheckRequest) -> dict:
    if len(req.code.encode()) > MAX_CODE_BYTES:
        raise HTTPException(413, f"code larger than {MAX_CODE_BYTES} bytes")
    if req.certify and not ALLOW_CERTIFY:
        raise HTTPException(403, "certificates are disabled in this service (set TLEYE_ALLOW_CERTIFY=1 on a host with Docker)")
    html, tl = _problem(req)
    runtime = (req.runtime or "CPython") if req.language == "Python" else req.language
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td, LOCK:
        src = Path(td) / f"sol{SUFFIX[req.language]}"
        src.write_text(req.code)
        res = check(src, req.language, runtime, html, tl, req.certify, req.prior, model=MODEL, cache=CACHE)
        if req.explain:
            from tleye.models import explain
            res["explanation"] = explain.explain(src.read_bytes(), req.language, runtime, html, tl)
    res["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


WARMUP = CheckRequest(code="int main(){}", language="C++", statement_html="<h3>Constraints</h3><ul><li>1 \\leq N \\leq 10</li></ul>", time_limit_ms=2000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    run(WARMUP)                  # load the model once at start-up, so /health reports a ready model
    yield


app = FastAPI(title="TLEye", description="Predicts Time Limit Exceeded for competitive-programming solutions.", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": next(iter(CACHE), None), "certify_enabled": ALLOW_CERTIFY}


@app.post("/check")
def check_endpoint(req: CheckRequest) -> dict:
    return run(req)


if os.environ.get("TLEYE_DEMO") == "1":          # also serve the web demo at /, sharing this process's model (one copy in memory)
    import gradio as gr

    import tleye.app as demo
    demo.CACHE, demo.MODEL_CHOICE = CACHE, MODEL
    app = gr.mount_gradio_app(app, demo.build_ui(), path="/")
