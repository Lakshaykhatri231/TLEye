from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tleye.models.b2 import B2

R = Path("data/raw/extracted/Project_CodeNet")
needs_data = pytest.mark.skipif(not (Path("models/b2/model.txt").exists() and (R / "metadata").exists()
                                     and Path("data/processed/b2_test_preds.parquet").exists()),
                                reason="processed data / model not built")


@needs_data
def test_inference_features_reproduce_stored_test_predictions():
    """Re-featurize real test submissions from raw code + statement and match the stored B2 predictions."""
    b2 = B2()
    preds = pd.read_parquet("data/processed/b2_test_preds.parquet").set_index("submission_id")["p_raw"]
    sel = pd.read_parquet("data/processed/selected.parquet", columns=["submission_id", "path", "language", "runtime", "problem_id", "time_limit_ms", "code_size"])
    rows = sel[sel.submission_id.isin(preds.index)].sample(60, random_state=0)
    diffs = []
    for r in rows.itertuples():
        html = (R / "problem_descriptions" / f"{r.problem_id}.html").read_text(errors="replace")
        X = b2.featurize(Path(r.path).read_bytes(), r.language, r.runtime, html, r.time_limit_ms)
        diffs.append(abs(float(b2.predict_proba(X)[0]) - float(preds[r.submission_id])))
    diffs = np.array(diffs)
    # code_size in the metadata can differ from len(bytes) (e.g. line endings), so allow a tiny tolerance on log_code
    assert np.median(diffs) < 1e-9 and diffs.max() < 0.02, (np.median(diffs), diffs.max())


def test_prior_shift_is_identity_at_training_prevalence_and_monotone():
    if not Path("models/b2/meta.json").exists():
        pytest.skip("model not built")
    b2 = B2()
    p = np.array([0.1, 0.4, 0.9])
    assert np.allclose(b2.adjust_prior(p, b2.meta["train_prevalence"]), p, atol=1e-9)
    low = b2.adjust_prior(p, 0.05)
    assert (low < p).all() and (np.diff(low) > 0).all()
