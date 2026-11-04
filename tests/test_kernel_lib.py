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


def _tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")


TARGET = "loop_depth=2 loops=13 recursion=no sorts=1 sort_in_loop=0 listed_calls_in_loop=4 n_var=|S| log10_n=5.3 est_log10_ops=10.6"


def test_target_values_round_trip():
    assert kl.values_to_target(kl.target_values(TARGET)) == TARGET
    u = "loop_depth=0 loops=0 recursion=yes sorts=0 sort_in_loop=0 listed_calls_in_loop=0 n_var=unknown log10_n=unknown est_log10_ops=unknown"
    assert kl.values_to_target(kl.target_values(u)) == u


def test_terminators_cannot_be_confused_with_value_tokens():
    tok = _tok(); c = kl.Completion(tok)
    assert len(set(c.terms)) == len(c.terms)
    vals = set()
    for v in ["0", "1", "2", "9", "13", "100", "yes", "no", "N", "n", "H", "R", "Q", "M", "|S|", "|s|", "|A|", "unknown", "5.3", "10.6", "18.0"]:
        vals |= set(tok(v, add_special_tokens=False)["input_ids"])
    assert not vals & set(c.terms)


def test_machine_driven_by_a_perfect_model_reproduces_the_encoding():
    tok = _tok(); c = kl.Completion(tok); values = kl.target_values(TARGET)
    ids, mask = c.encode(values); m = c.machine(); fed = []; pred = None
    for i in range(len(ids)):
        pred = ids[i] if mask[i] else None          # a perfect model predicts exactly the masked (value / terminator) tokens
        fed.append(m.next(pred))
        if m.done:
            break
    assert fed == ids and m.done and [v.strip() for v in m.values] == values


def test_machine_stops_a_runaway_value():
    tok = _tok(); c = kl.Completion(tok); m = c.machine(); digit = tok("7", add_special_tokens=False)["input_ids"][0]
    for _ in range(400):
        m.next(digit)
        if m.done:
            break
    assert m.done and all(len(tok(v, add_special_tokens=False)["input_ids"]) <= kl.MAX_VALUE_TOKENS for v in m.values)
