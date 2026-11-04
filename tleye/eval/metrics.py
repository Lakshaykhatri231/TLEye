"""Evaluation metrics with problem-clustered bootstrap CIs (resample PROBLEMS, not rows)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on [0,1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold on score s maximising F1 (tune on VAL only)."""
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    f = [f1_score(y, s >= t, zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f))])


def point_metrics(y, s, thr: float | None = None, prob: bool = False) -> dict[str, float]:
    y, s = np.asarray(y), np.asarray(s, float)
    out = {"n": len(y), "prevalence": float(y.mean())}
    if len(np.unique(y)) < 2:
        return out | {"auroc": float("nan"), "auprc": float("nan")}
    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    if thr is not None:
        out["f1"] = float(f1_score(y, s >= thr, zero_division=0))
    if prob:
        out["ece"] = ece(y, s)
        out["brier"] = float(np.mean((s - y) ** 2))
    return out


def cluster_bootstrap(groups: np.ndarray, fn: Callable[[np.ndarray], float],
                      n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of fn(row_indices) with problems resampled with replacement."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[i] for i in pick])
        try:
            v = fn(idx)
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics_with_ci(y, s, groups, thr: float | None = None, prob: bool = False,
                    n_boot: int = 500, seed: int = 0) -> dict[str, tuple[float, float, float] | float]:
    """Return {metric: (point, lo, hi)} plus n / n_problems / prevalence."""
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    pm = point_metrics(y, s, thr, prob)
    res: dict = {"n": pm["n"], "n_problems": len(np.unique(groups)), "prevalence": pm["prevalence"]}
    fns = {
        "auroc": lambda i: roc_auc_score(y[i], s[i]),
        "auprc": lambda i: average_precision_score(y[i], s[i]),
    }
    if thr is not None:
        fns["f1"] = lambda i: f1_score(y[i], s[i] >= thr, zero_division=0)
    if prob:
        fns["ece"] = lambda i: ece(y[i], s[i])
    for name, fn in fns.items():
        lo, hi = cluster_bootstrap(groups, fn, n_boot, seed)
        res[name] = (pm.get(name, float("nan")), lo, hi)
    return res
