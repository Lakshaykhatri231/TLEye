"""The HTTP service, with the CPU LightGBM model (the LoRA model must not load in the pytest process on macOS)."""
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(not Path("models/b2/model_monotone.txt").exists(), reason="LightGBM model not available")

STATEMENT = "<h3>Constraints</h3><ul><li>2 \\leq N \\leq 2 \\times 10^5</li><li>1 \\leq A_i \\leq 10^9</li></ul>"
SLOW = Path("tleye/examples/p02779_slow.cpp")


@pytest.fixture(scope="module")
def client():
    import tleye.serve as serve
    serve.MODEL = "b2m"
    from fastapi.testclient import TestClient
    with TestClient(serve.app) as c:
        yield c


def test_health_reports_the_loaded_model(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model"] == "b2m" and r["certify_enabled"] is False


def test_check_returns_a_verdict_with_timing(client):
    r = client.post("/check", json={"code": SLOW.read_text(), "language": "C++", "statement_html": STATEMENT, "time_limit_ms": 2000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] in ("SUSPECTED TLE", "LIKELY AC") and 0 <= body["p_tle_model"] <= 1 and body["elapsed_ms"] >= 0
    assert "certificate" not in body


@pytest.mark.parametrize("payload,code", [
    ({"code": "x", "language": "C++", "statement_html": STATEMENT}, 422),                          # statement without a time limit
    ({"code": "x", "language": "C++"}, 422),                                                     # no problem at all
    ({"code": "x", "language": "Rust", "statement_html": STATEMENT, "time_limit_ms": 2000}, 422),  # unsupported language
    ({"code": "x", "language": "C++", "statement_html": STATEMENT, "time_limit_ms": 2000, "certify": True}, 403),
    ({"code": "x" * 200_001, "language": "C++", "statement_html": STATEMENT, "time_limit_ms": 2000}, 413),
])
def test_bad_requests_are_rejected_with_a_clear_status(client, payload, code):
    assert client.post("/check", json=payload).status_code == code
