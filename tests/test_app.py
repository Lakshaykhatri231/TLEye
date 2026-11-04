import os
import shutil
import subprocess
from pathlib import Path

import pytest

import tleye.app as app

pytestmark = pytest.mark.skipif(not (Path("models/b2/model_monotone.txt").exists() and Path("tleye/examples/p02779_slow.cpp").exists()
                                     and (app.R / "metadata").exists()), reason="model/examples/data not available")
SLOW = Path("tleye/examples/p02779_slow.cpp")
FAST = Path("tleye/examples/p02779_fast.cpp")


@pytest.fixture(autouse=True)
def lightgbm_only(monkeypatch):
    monkeypatch.setattr(app, "MODEL_CHOICE", "b2m")      # never load the LoRA model in the pytest process (macOS OpenMP clash)
    monkeypatch.setattr(app, "CACHE", {})


def test_input_validation_messages():
    assert "Paste a solution" in app.run_check("", "C++", "CPython", "p02779", "", 0, False)[0]
    assert "Unknown problem id" in app.run_check("int main(){}", "C++", "CPython", "pXXXXX", "", 0, False)[0]
    assert "Enter a problem id" in app.run_check("int main(){}", "C++", "CPython", "", "", 0, False)[0]
    assert "time limit" in app.run_check("int main(){}", "C++", "CPython", "", "<h3>Constraints</h3>", 0, False)[0]


def test_verdict_without_certify_never_runs_code_and_never_claims_certified():
    md, res = app.run_check(SLOW.read_text(), "C++", "CPython", "p02779", "", 0, False)
    assert res["outcome"] in {"LIKELY AC", "SUSPECTED TLE"} and "certificate" not in res
    assert 0 <= res["p_tle_model"] <= 1 and "P(TLE)" in md and "Never reported as safe" in md


@pytest.mark.skipif(bool(os.environ.get("TLEYE_SKIP_DOCKER")) or shutil.which("docker") is None
                    or subprocess.run(["docker", "info"], capture_output=True).returncode != 0, reason="docker not available")
def test_certify_confirms_the_known_slow_example_and_never_certifies_the_fast_one():
    _, slow = app.run_check(SLOW.read_text(), "C++", "CPython", "p02779", "", 0, True)
    _, fast = app.run_check(FAST.read_text(), "C++", "CPython", "p02779", "", 0, True)
    assert slow["outcome"] == "CERTIFIED TLE"
    assert fast["outcome"] != "CERTIFIED TLE"
