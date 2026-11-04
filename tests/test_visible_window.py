import pytest

from tleye.features.ast_features import extract
from tleye.features.visible_window import header, parse_target, primary_var, target_string

CASES = [
    ("C++", b"int main(){int n;for(int i=0;i<n;i++)for(int j=0;j<n;j++){}}", 2, 2),
    ("Java", b"class Main{public static void main(String[] a){for(int i=0;i<3;i++){}}}", 1, 1),
    ("Python", b"n=int(input())\nfor i in range(n):\n  for j in range(n):\n    for k in range(n):\n      pass\n", 3, 3),
]


@pytest.mark.parametrize("lang,code,depth,loops", CASES)
def test_target_round_trips_for_every_language(lang, code, depth, loops):
    f = extract(code, lang)
    p = parse_target(target_string(f, "N", 5.3))
    assert p["loop_depth"] == depth and p["loops"] == loops and p["log10_n"] == 5.3
    assert p["est_log10_ops"] == pytest.approx(round(max(depth, 1) * 5.3, 1))


def test_no_loops_still_counts_one_pass_and_says_so():
    p = parse_target(target_string(extract(b"print(1)\n", "Python"), "N", 5.0))
    assert p["loop_depth"] == 0 and p["est_log10_ops"] == 5.0


def test_unknown_n_is_stated_never_filled():
    p = parse_target(target_string(extract(b"print(1)\n", "Python"), "unknown", None))
    assert p["n_var"] == "unknown" and p["log10_n"] is None and p["est_log10_ops"] is None


def test_primary_var_requires_the_model_to_see_it_and_rejects_huge_bounds():
    assert primary_var({"N": 1e5}, "N<=1e5") == ("N", 5.0)
    assert primary_var({"N": 1e5}, "A<=1e9") == ("unknown", None)        # bound not in the prompt's constraints line
    assert primary_var({"N": 1e300}, "N<=huge")[0] == "unknown"
    assert primary_var({"M": 10.0, "N": 100.0}, "N<=100, M<=10")[0] == "N"


def test_malformed_text_is_rejected():
    assert parse_target("loop_depth=two nonsense") is None


def test_header_matches_the_classifier_prompt_format():
    assert header("PyPy", 2000, "N<=1e5") == "language: PyPy\ntime limit: 2s\nconstraints: N<=1e5\ncode:\n"


def test_visible_window_cuts_the_code_and_flags_truncation():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from tleye.features.visible_window import visible_code
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")
    h = header("C++", 2000, "N<=1e5")
    short, t1, _ = visible_code(tok, h, "int main(){}", 5)
    assert short == "int main(){}" and t1 is False
    long_code = "int x;\n" * 2000
    vis, t2, n = visible_code(tok, h, long_code, 5)
    assert t2 is True and n > 1024 and 0 < len(vis) < len(long_code) and long_code.startswith(vis[:50])
