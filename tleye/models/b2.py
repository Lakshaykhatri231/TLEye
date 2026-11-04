"""Load the persisted B2 LightGBM verdict model and build its feature row for a NEW submission.

Mirrors the training pipeline (tleye.features.build + tleye.models.baselines): AST features, parsed constraints,
log10 time limit, language one-hots. Source code is only parsed as bytes, never executed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tleye.constraints.parser import parse_statement
from tleye.features.ast_features import extract
from tleye.models.baselines import add_est

MAX_BYTES = 400_000   # same cap as the feature build


def _l10(v: float | None) -> float:
    return math.log10(min(v, 1e300)) if v is not None and v > 0 else float("nan")


class B2:
    def __init__(self, path: str | Path = "models/b2", variant: str = "plain"):
        """variant: 'plain' (B2) or 'monotone' (B2m: size/estimate up -> P(TLE) up, time limit up -> down; adopted for the CLI)."""
        p = Path(path)
        model, meta = ("model_monotone.txt", "meta_monotone.json") if variant == "monotone" else ("model.txt", "meta.json")
        self.variant = variant
        self.booster = lgb.Booster(model_file=str(p / model))
        self.meta = json.loads((p / meta).read_text())
        self.cols: list[str] = self.meta["cols"]

    def featurize(self, code: bytes, language: str, runtime: str, statement_html: str, time_limit_ms: float) -> pd.DataFrame:
        code = code[:MAX_BYTES]
        f = extract(code, language)
        c = parse_statement(statement_html)
        f.update(c_section=int(c.found_section), c_n_upper=len(c.upper), log10_max_bound=_l10(c.max_bound),
                 log10_primary=_l10(c.primary_size_bound), log10_size=_l10(c.size_bound))
        d = pd.DataFrame([f])
        d["log_tl"] = math.log10(time_limit_ms / 1000.0)
        d["log_code"] = math.log10(max(len(code), 1))
        d["lang_cpp"], d["lang_java"], d["lang_py"] = int(language == "C++"), int(language == "Java"), int(language == "Python")
        d["is_pypy"] = int(runtime.startswith("PyPy"))
        d = add_est(d, self.meta["fill_log10_primary"])
        return d[self.cols]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.cols])

    def adjust_prior(self, p: np.ndarray | float, deployment_prevalence: float) -> np.ndarray | float:
        """Shift probabilities from the capped training prevalence to an assumed deployment prevalence (logit shift)."""
        pi_t, pi_d = self.meta["train_prevalence"], deployment_prevalence
        p = np.clip(p, 1e-9, 1 - 1e-9)
        z = np.log(p / (1 - p)) + math.log(pi_d / (1 - pi_d)) - math.log(pi_t / (1 - pi_t))
        return 1 / (1 + np.exp(-z))
