import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("kernel_lib", "kaggle/lib/kernel_lib.py")
kl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kl)


@pytest.mark.parametrize("v", [5, 999, 1000, 2e5, 1e9, 1e18, 3e5, 1e300])
def test_number_format_matches_the_training_pack(v):
    pytest.importorskip("pandas")
    from tleye.data.kaggle_pack import fmt as pack_fmt
    assert kl.fmt(v) == pack_fmt(v)


def test_prompt_is_identical_to_what_the_classifier_trained_on():
    src = Path("kaggle/seeds/seeds.py").read_text()
    assert 'f"language: {r.rt}\\ntime limit: {r.time_limit_ms / 1000:g}s\\nconstraints: {r.constraints}\\n"' in src
    assert kl.prompt("C++", kl.tl_text(2000), "N<=1e5", "int main(){}") == "language: C++\ntime limit: 2s\nconstraints: N<=1e5\ncode:\nint main(){}"
    assert kl.prompt("C++", "2s", "N<=1e5", "x", analysis="a=1") == "language: C++\ntime limit: 2s\nconstraints: N<=1e5\nanalysis: a=1\ncode:\nx"


def test_scaling_touches_only_the_named_variable():
    assert kl.scale_var("N<=2e5, A<=1e9", "N", 10) == "N<=2e6, A<=1e9"
    assert kl.scale_var("A<=1e9, N<=100", "N", 1 / 100) == "A<=1e9, N<=1"
    assert kl.scale_var("A<=1e9", "N", 10) is None
    assert kl.scale_var("N<=huge", "N", 10) is None
    assert kl.scale_var("N<=5", "N", 1 / 1000) == "N<=1"          # never below 1


def test_placebos_keep_the_facts():
    assert kl.reorder_items("N<=2e5, A<=1e9") == "A<=1e9, N<=2e5"
    assert kl.reorder_items("N<=2e5") is None
    assert kl.tl_text(2000) == "2s" and kl.tl_text(2000, placebo=True) == "2.0s" and kl.tl_text(2000, factor=2) == "4s" and kl.tl_text(1000, factor=0.5) == "0.5s"
