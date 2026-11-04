import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location("probe_lib", "kaggle/lib/probe_lib.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


def test_folds_never_split_a_problem():
    g = np.repeat(np.arange(40), 5); f = pl.group_folds(g, 5)
    for x in np.unique(g):
        assert len(set(f[g == x])) == 1
    assert len(set(f)) == 5


def test_probe_recovers_a_real_signal_and_finds_nothing_in_noise():
    rng = np.random.default_rng(0); n, g = 1200, np.repeat(np.arange(60), 20)
    X = rng.normal(size=(n, 40)); y = (X[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(float)
    _, sc, _ = pl.cv_select(X, y, g, "binary")
    assert sc > 0.9
    _, sc0, _ = pl.cv_select(rng.normal(size=(n, 40)), y, g, "binary")
    assert 0.4 < sc0 < 0.6


def test_auroc_matches_hand_values_with_ties():
    assert pl.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert pl.auroc([0, 1], [0.5, 0.5]) == 0.5


def test_demeaning_removes_problem_identity():
    g = np.repeat(np.arange(30), 10); rng = np.random.default_rng(1)
    level = np.repeat(rng.normal(size=30) * 10, 10)
    y = level + rng.normal(size=300)                    # target dominated by the problem
    p = level + rng.normal(size=300)                    # predictor only knows the problem
    assert np.corrcoef(y, p)[0, 1] > 0.9 and abs(pl.demeaned_corr(y, p, g)) < 0.2


def test_controls_break_the_right_structure():
    g = np.repeat(np.arange(20), 5); y = np.arange(100.0)
    s = pl.shuffle_within_groups(y, g)
    for x in np.unique(g):
        assert sorted(s[g == x]) == sorted(y[g == x])
    lv = np.repeat(np.arange(20.0), 5); pm = pl.permute_across_groups(lv, g, seed=3)
    assert sorted(set(pm)) == sorted(set(lv)) and (pm != lv).any()
    assert all(len(set(pm[g == x])) == 1 for x in np.unique(g))


def test_holm_on_known_values():
    assert np.allclose(pl.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
