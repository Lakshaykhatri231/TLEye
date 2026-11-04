from tleye.eval.metamorphic_lora import claims


def test_claim_rules_on_synthetic_numbers():
    ok = claims(n_lo=0.2, n_mean=0.4, placebo_abs=0.05, tl_mean=-0.3, tl_hi=-0.1, gate_n=0.9, gate_tl=-0.6)
    assert all(ok.values())
    weak = claims(n_lo=0.02, n_mean=0.1, placebo_abs=0.05, tl_mean=-0.1, tl_hi=0.02, gate_n=0.3, gate_tl=-0.1)
    assert weak["C1_responds_to_N"] and not weak["C2_beyond_noise"] and not weak["C3_responds_to_time_limit"]
    assert not weak["patch_gate_N"] and not weak["patch_gate_time_limit"]
    neg = claims(n_lo=-0.3, n_mean=-0.1, placebo_abs=0.0, tl_mean=0.2, tl_hi=0.4, gate_n=-1.0, gate_tl=0.7)
    assert not neg["C1_responds_to_N"] and not neg["C3_responds_to_time_limit"] and not neg["patch_gate_N"] and neg["patch_gate_time_limit"]
