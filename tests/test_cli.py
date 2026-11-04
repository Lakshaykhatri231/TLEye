import json
import subprocess
import sys
from pathlib import Path

import pytest

SEL = Path("data/processed/selected.parquet")


@pytest.mark.skipif(not (SEL.exists() and Path("models/b2/model_monotone.txt").exists()), reason="processed data/model not built")
def test_check_without_certify_returns_a_valid_outcome():
    import pandas as pd
    row = pd.read_parquet(SEL, columns=["path", "problem_id", "language"]).query("language == 'C++'").iloc[0]
    out = subprocess.run([sys.executable, "-I", "-m", "tleye.cli", "check", row.path, "--problem", row.problem_id, "--json"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res["outcome"] in {"LIKELY AC", "SUSPECTED TLE"}          # never CERTIFIED without --certify
    assert 0.0 <= res["p_tle_model"] <= 1.0 and "certificate" not in res


@pytest.mark.skipif(not (SEL.exists() and Path("reports/kaggle_confirm/cfgN/test_logits.npz").exists()), reason="processed data / Kaggle outputs not present")
def test_check_with_lora_matches_the_kaggle_probability():
    import numpy as np
    import pandas as pd

    from tleye.models.lora import available
    if not available():
        pytest.skip("torch/transformers/peft or models/lora missing")
    z = np.load("reports/kaggle_confirm/cfgN/test_logits.npz"); T = float(z["T"])
    sel = pd.read_parquet(SEL, columns=["submission_id", "path", "problem_id", "language"]).set_index("submission_id")
    sid = next(s for s in z["submission_id"] if sel.loc[s, "language"] == "C++")
    kaggle_p = 1 / (1 + np.exp(-float(z["logit"][list(z["submission_id"]).index(sid)]) / T))
    out = subprocess.run([sys.executable, "-I", "-m", "tleye.cli", "check", sel.loc[sid, "path"], "--problem", sel.loc[sid, "problem_id"],
                          "--model", "lora", "--json"], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-500:]
    res = json.loads(out.stdout)
    assert "LoRA" in res["model"] and abs(res["p_tle_model"] - kaggle_p) < 0.03, (res["p_tle_model"], kaggle_p)
