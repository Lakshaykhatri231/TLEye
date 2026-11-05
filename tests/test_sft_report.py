import numpy as np
import pandas as pd

from tleye.eval.sft_report import balanced_accuracy, depth_class, gate_decision, majority_by_language, values


def test_gate_prefers_dual_and_needs_both_conditions():
    good, low_auc, no_depth = {"best_auroc": .82, "depth_diff_lo": .1}, {"best_auroc": .79, "depth_diff_lo": .1}, {"best_auroc": .9, "depth_diff_lo": 0.0}
    assert gate_decision({"dual": good, "v1only": good})["stage5"] == "dual"
    assert gate_decision({"dual": low_auc, "v1only": good})["stage5"] == "v1only"
    assert gate_decision({"dual": no_depth, "v1only": low_auc})["stage5"] is None


def test_exactly_the_threshold_passes():
    assert gate_decision({"dual": {"best_auroc": .80, "depth_diff_lo": .001}})["passed"]["dual"]


def test_depth_class_caps_and_marks_garbage_wrong():
    assert depth_class(["0", "3", "4", "9", "x", "", "²"]).tolist() == [0, 3, 4, 4, -1, -1, -1]


def test_balanced_accuracy_weights_classes_equally_and_constant_guess_scores_one_over_k():
    y = np.array([0] * 90 + [1] * 10)
    assert balanced_accuracy(y, np.zeros(100, int)) == 0.5
    assert balanced_accuracy(y, y) == 1.0


def test_majority_is_per_language():
    d = pd.DataFrame({"language": ["a", "a", "a", "b", "b"], "x": ["1", "1", "2", "3", "3"]})
    assert majority_by_language(d, "x") == {"a": "1", "b": "3"}


def test_values_survive_odd_generated_values():
    kv = values("loop_depth=2 loops=abc n_var=|S| log10_n=unknown")
    assert kv == {"loop_depth": "2", "loops": "abc", "n_var": "|S|", "log10_n": "unknown"}
