"""The summary comparison, plus parity with the Kaggle run in a subprocess (macOS: this model must not share a process with LightGBM)."""
import base64
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tleye.cli as cli
from tleye.models import explain as ex

R = Path("data/raw/extracted/Project_CodeNet")
SHARD = "reports/kaggle_sft_full/cfg_s0/test_gen/*.npz"
EXAMPLE = {"fields": ex.compare(["loop_depth", "n_var"], ["2", "N"], ["3", "N"]), "mismatches": 1, "code_cut_off": True, "note": ex.FAITHFULNESS_NOTE}


def test_numbers_compare_with_a_tolerance_and_text_exactly():
    assert ex.same_value("5.30", "5.3") and not ex.same_value("5.4", "5.3")
    assert ex.same_value("unknown", "unknown") and not ex.same_value("unknown", "5.3") and not ex.same_value("nan", "nan")


def test_compare_marks_each_disagreement():
    assert [r["match"] for r in EXAMPLE["fields"]] == [False, True]


def test_renderings_show_mismatches_and_never_call_it_the_reason():
    md, txt = ex.render_markdown(EXAMPLE), ex.render_text(EXAMPLE)
    assert "**NO**" in md and "1 of 2" in md and "cut off" in md and "! loop_depth" in txt
    assert "not the reason for the verdict" in md and "not the reason for the verdict" in txt


def test_a_summary_failure_is_shown_not_raised():
    assert "No code summary" in ex.render_markdown({"error": "x"}) and "unavailable" in ex.render_text({"error": "x"})


def test_explain_reports_a_missing_model_instead_of_raising(monkeypatch):
    monkeypatch.setattr(ex, "available", lambda: False)
    assert "unavailable" in ex.explain(b"", "C++", "C++", "", 2000)["error"]


def test_cli_explain_adds_the_summary_and_leaves_the_verdict_alone(monkeypatch, capsys, tmp_path):
    sol, st = tmp_path / "a.cpp", tmp_path / "s.html"
    sol.write_text("int main(){}"); st.write_text("<h3>Constraints</h3><ul><li>1 \\leq N \\leq 10^5</li></ul>")
    base = {"model": "m", "parsed": {"primary_size_bound": 1, "max_bound": 1}, "p_tle_model": .2, "flag_threshold": .4, "p_tle_at_assumed_prevalence": None, "outcome": "LIKELY AC", "detail": "d"}
    monkeypatch.setattr(cli, "check", lambda *a, **k: dict(base))
    monkeypatch.setattr(ex, "explain", lambda *a, **k: EXAMPLE)
    assert cli.main(["check", str(sol), "--statement", str(st), "--time-limit-ms", "2000", "--explain", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["outcome"] == "LIKELY AC" and res["p_tle_model"] == .2 and res["explanation"]["mismatches"] == 1


def _ready():
    return ex.available() and glob.glob(SHARD) and (R / "metadata").exists()


@pytest.mark.skipif(not _ready(), reason="explanation adapter/deps/data not available")
def test_local_generation_matches_the_kaggle_run():
    """Kaggle: fp16 and padded batches; here: fp32, one row. Most rows should give the same text, and the AST column must equal the stored target."""
    z = np.load(sorted(glob.glob(SHARD))[0])
    want = pd.DataFrame({"submission_id": z["submission_id"].astype(str), "text": z["text"]}).head(4)
    sel = pd.read_parquet("data/processed/selected.parquet", columns=["submission_id", "path", "language", "runtime", "problem_id", "time_limit_ms"]).merge(want, on="submission_id")
    target = pd.read_parquet("data/processed/kaggle_sidecar/sft_targets.parquet", columns=["submission_id", "target"]).set_index("submission_id").target
    same, ast_ok = [], []
    for r in sel.itertuples():
        html = (R / "problem_descriptions" / f"{r.problem_id}.html").read_text(errors="replace")
        req = {"code": base64.b64encode(Path(r.path).read_bytes()).decode(), "language": r.language, "runtime": r.runtime, "html": html, "time_limit_ms": r.time_limit_ms, "device": "cpu"}
        p = subprocess.run([sys.executable, "-I", "-m", "tleye.models.explain"], input=json.dumps(req), capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr[-800:]
        res = json.loads(p.stdout.split("EXPLAIN_JSON", 1)[1])
        same.append(" ".join(f"{f['field']}={f['model']}" for f in res["fields"]) == r.text)
        ast_ok.append(" ".join(f"{f['field']}={f['ast']}" for f in res["fields"]) == target[r.submission_id])
    assert all(ast_ok), ast_ok
    assert np.mean(same) >= 0.75, same
