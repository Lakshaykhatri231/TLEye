"""CI regression gate: the committed B2m model must (a) reproduce the stored predictions on a frozen fixture and
(b) not lose more than `max_allowed_drop` AUROC versus ci/baseline_eval.json. Fixture holds derived numeric features only."""
import json

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


def test_model_matches_frozen_predictions_and_auroc_gate():
    meta = json.load(open("models/b2/meta_monotone.json"))
    base = json.load(open("ci/baseline_eval.json"))
    fx = pd.read_parquet("tests/fixtures/ci_subset.parquet")
    booster = lgb.Booster(model_file="models/b2/model_monotone.txt")
    p = booster.predict(fx[meta["cols"]])
    assert abs(p - fx["p_expected"].values).max() < 1e-6, "model or feature order changed vs frozen predictions"
    auc = roc_auc_score(fx["y_tle"], p)
    assert auc >= base["auroc"] - base["max_allowed_drop"], f"AUROC regression: {auc:.4f} vs baseline {base['auroc']}"


def test_fixture_has_no_source_code_and_covers_all_languages():
    fx = pd.read_parquet("tests/fixtures/ci_subset.parquet")
    assert not any(c in fx.columns for c in ("code", "path")), "fixture must not contain source code"
    assert set(fx.language) == {"C++", "Java", "Python"} and fx.problem_id.nunique() > 100
