"""Local LoRA inference must reproduce the logits the Kaggle run wrote (same prompt, tokenization, adapter, head).
Kaggle ran fp16 with padded batches; here fp32 single examples, so small numeric differences are expected.
Runs in a SUBPROCESS: on macOS, loading the LoRA model in a process where LightGBM has already predicted segfaults (OpenMP clash)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R = Path("data/raw/extracted/Project_CodeNet")
LOGITS = Path("reports/kaggle_confirm/cfgN/test_logits.npz")

CHILD = """
import json, sys
from pathlib import Path
from tleye.models.lora import LoraVerdict
R = Path("data/raw/extracted/Project_CodeNet")
rows = json.loads(sys.stdin.read())
m = LoraVerdict(device="cpu")
out = []
for r in rows:
    html = (R / "problem_descriptions" / f"{r['problem_id']}.html").read_text(errors="replace")
    out.append(m.predict(Path(r["path"]).read_bytes(), r["language"], r["runtime"], html, r["time_limit_ms"])["logit"])
print("LOGITS_JSON" + json.dumps(out))
"""


def _ready():
    try:
        from tleye.models.lora import available
    except Exception:  # noqa: BLE001
        return False
    return available() and LOGITS.exists() and (R / "metadata").exists()


@pytest.mark.skipif(not _ready(), reason="LoRA deps/adapter/data not available")
def test_local_logits_match_kaggle_logits():
    z = np.load(LOGITS)
    kag = pd.Series(z["logit"], index=z["submission_id"])
    sel = pd.read_parquet("data/processed/selected.parquet", columns=["submission_id", "path", "language", "runtime", "problem_id", "time_limit_ms"])
    rows = sel[sel.submission_id.isin(kag.index)].sample(16, random_state=0)
    res = subprocess.run([sys.executable, "-I", "-c", CHILD], input=rows.drop(columns="submission_id").to_json(orient="records"),
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-800:]
    local = np.array(json.loads(res.stdout.split("LOGITS_JSON")[1]))
    ref = np.array([float(kag[s]) for s in rows.submission_id])
    assert np.corrcoef(local, ref)[0, 1] > 0.99, np.c_[local, ref]
    assert np.median(np.abs(local - ref)) < 0.1 and np.abs(local - ref).max() < 0.5, np.c_[local, ref]
