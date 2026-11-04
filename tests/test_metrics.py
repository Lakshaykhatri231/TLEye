import numpy as np
import pytest

from tleye.eval.metrics import (
    best_f1_threshold,
    cluster_bootstrap,
    ece,
    metrics_with_ci,
    point_metrics,
)


def test_ece_perfectly_calibrated_is_zero_and_overconfident_is_large():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert ece(y, np.full(8, 0.5)) == pytest.approx(0.0)
    # always predicting 0.9 when truth rate is 0.5 -> |0.5-0.9| = 0.4
    assert ece(y, np.full(8, 0.9)) == pytest.approx(0.4)


def test_ece_two_bins_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.9, 0.9])
    # bin(0.1): mean y 0 vs p .1 -> .1 * 0.5 ; bin(0.9): mean y 1 vs .9 -> .1 * 0.5
    assert ece(y, p) == pytest.approx(0.1)


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert point_metrics(y, [0.1, 0.2, 0.8, 0.9])["auroc"] == 1.0
    assert point_metrics(y, [0.9, 0.8, 0.2, 0.1])["auroc"] == 0.0


def test_best_threshold_separates_classes():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    t = best_f1_threshold(y, s)
    assert 0.4 <= t <= 0.6


def test_cluster_bootstrap_resamples_whole_problems_and_ci_contains_point():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 10)
    y = rng.integers(0, 2, 600)
    s = y * 0.5 + rng.random(600)
    res = metrics_with_ci(y, s, groups, n_boot=200)
    point, lo, hi = res["auroc"]
    assert lo <= point <= hi and res["n_problems"] == 60 and res["n"] == 600


def test_bootstrap_indices_come_in_whole_groups():
    groups = np.repeat(np.arange(5), 4)
    seen = []
    cluster_bootstrap(groups, lambda i: (seen.append(len(i)), 0.0)[1], n_boot=20)
    assert all(n % 4 == 0 for n in seen)
