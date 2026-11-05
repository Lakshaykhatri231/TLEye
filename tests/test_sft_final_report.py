import pytest

from tleye.eval.sft_final_report import is_num, label, lower_label, naming, regex_baseline, verdict_claims


@pytest.mark.parametrize("ma,flip,lo,want", [(0.19, 0.5, None, "decorative"), (0.2, 0.0, None, "weak"), (0.5, 0.05, None, "decision-relevant"),
                                             (0.49, 0.5, None, "weak"), (0.9, 0.04, None, "weak"), (0.9, 0.5, -0.1, "weak"), (0.9, 0.5, 0.1, "decision-relevant"), (0.9, 0.5, 0.0, "weak")])
def test_labels_follow_the_predeclared_thresholds(ma, flip, lo, want):
    assert label(ma, flip, lo) == want


def test_seed_disagreement_uses_the_lower_label():
    assert lower_label(["decision-relevant", "weak"]) == "weak"
    assert lower_label(["decorative", "decision-relevant"]) == "decorative"


def test_reason_needs_both_donor_and_neutral():
    assert naming("decision-relevant", "decision-relevant") == "the reason for the verdict"
    assert naming("decision-relevant", "weak") == naming("weak", "decision-relevant") == "a checkable code summary"


def test_verdict_claims_need_both_seeds():
    assert verdict_claims([-0.01, -0.019]) == {"no_meaningful_loss": True, "beats_classifier": False}
    assert verdict_claims([-0.01, -0.02])["no_meaningful_loss"] is False
    assert verdict_claims([0.01, 0.02])["beats_classifier"] is True


def test_regex_baseline_counts_text_only_and_reads_the_first_constraint():
    r = regex_baseline("for(;;){while(1){sort(a);}} // forest", "N<=2e5, M<=100")
    assert r == {"loops": "2", "sorts": "1", "n_var": "N", "log10_n": "5.3"}
    assert regex_baseline("", "none")["n_var"] == "unknown"
    assert regex_baseline("", "N<=huge")["log10_n"] == "unknown"


def test_is_num_rejects_non_finite():
    assert is_num("5.3") and not is_num("nan") and not is_num("inf") and not is_num("unknown")
