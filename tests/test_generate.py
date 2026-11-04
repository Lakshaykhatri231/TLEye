import pytest

from tleye.certify.generate import STRATS, GenError, build_inputs
from tleye.certify.inputspec import parse_input_spec
from tleye.constraints.parser import constraints_text, parse_text


def problem(pre: str, cons: str):
    html = (f"<h3>Constraints</h3><ul><li>{cons}</li></ul><h3>Input</h3><p>fmt</p><pre>{pre}</pre><h3>Output</h3>")
    sec = constraints_text(html) or ""
    return parse_input_spec(html), parse_text(sec), sec


def test_array_sizes_ranges_and_strategies():
    spec, cons, sec = problem("N\nA_1 A_2 ... A_N", "1 \\leq N \\leq 1000</li><li>1 \\leq A_i \\leq 10^9")
    out = build_inputs(spec, cons, sec)
    assert set(out) == set(STRATS)
    for s, txt in out.items():
        lines = txt.strip().split("\n")
        if s == "min_sizes":                       # edge-case input: scalars at their lower bound
            assert lines[0] == "1" and len(lines[1].split()) == 1
            continue
        assert lines[0] == "1000" and len(lines[1].split()) == 1000
        vals = list(map(int, lines[1].split()))
        assert min(vals) >= 1 and max(vals) <= 10 ** 9
    asc = list(map(int, out["asc"].split("\n")[1].split())); assert asc == sorted(asc)
    desc = list(map(int, out["desc"].split("\n")[1].split())); assert desc == sorted(desc, reverse=True)
    assert set(out["all_max"].split("\n")[1].split()) == {str(10 ** 9)}


def test_table_rows_use_length_expression():
    spec, cons, sec = problem("N M\na_1 b_1\n:\na_M b_M", "1 \\leq N \\leq 50</li><li>1 \\leq M \\leq 20</li><li>1 \\leq a_i,b_i \\leq N")
    txt = build_inputs(spec, cons, sec)["random"].strip().split("\n")
    assert txt[0] == "50 20" and len(txt) == 1 + 20
    assert all(1 <= int(x) <= 50 for row in txt[1:] for x in row.split())


def test_string_with_alphabet_and_deterministic_seed():
    spec, cons, sec = problem("N\nS", "1 \\leq N \\leq 10</li><li>1 \\leq |S| \\leq 8</li><li>S consists of lowercase English letters.")
    a, b = build_inputs(spec, cons, sec, seed=1), build_inputs(spec, cons, sec, seed=1)
    assert a == b
    s = a["random"].strip().split("\n")[1]
    assert len(s) == 8 and s.isalpha() and s.islower()


def test_refuses_instead_of_guessing():
    spec, cons, sec = problem("N\nA_1 A_2 ... A_N", "1 \\leq N \\leq 10")           # A unbounded
    with pytest.raises(GenError, match="unbounded"):
        build_inputs(spec, cons, sec)
    spec, cons, sec = problem("N\nS", "1 \\leq N \\leq 10</li><li>|S| = 5</li><li>S consists of ( and ).")
    with pytest.raises(GenError, match="alphabet"):
        build_inputs(spec, cons, sec)


def _cert_problem(pre: str, cons_lines: str):
    from tleye.certify.bounds import parse as cert_parse
    html = f"<h3>Constraints</h3><ul><li>{cons_lines}</li></ul><h3>Input</h3><p>fmt</p><pre>{pre}</pre><h3>Output</h3>"
    sec = constraints_text(html) or ""
    return parse_input_spec(html), cert_parse(sec), sec


def test_every_input_type_respects_expression_bounds():
    spec, cb, sec = _cert_problem("N\nA_1 A_2 ... A_N", "2 \\leq N \\leq 50</li><li>0 \\leq A_i \\leq N-1")
    out = build_inputs(spec, cb, sec)
    for strat, txt in out.items():
        lines = txt.strip().split("\n"); n = int(lines[0]); vals = list(map(int, lines[1].split()))
        assert len(vals) == n and all(0 <= v <= n - 1 for v in vals), (strat, n, max(vals))
    assert out["min_sizes"].split("\n")[0] == "2"                  # the edge-case input really uses the minimum N


def test_absolute_value_bounds_allow_negative_numbers():
    spec, cb, sec = _cert_problem("N\nA_1 A_2 ... A_N", "1 \\leq N \\leq 200</li><li>|A_i| \\leq 10^9")
    vals = list(map(int, build_inputs(spec, cb, sec)["random"].split("\n")[1].split()))
    assert min(vals) < 0 < max(vals) and max(abs(v) for v in vals) <= 10 ** 9


def test_string_length_equality_is_exact_for_each_input_type():
    spec, cb, sec = _cert_problem("A B\nS", "1 \\leq A,B \\leq 5</li><li>|S| = A+B+1</li><li>S consists of lowercase English letters.")
    out = build_inputs(spec, cb, sec)
    for strat, txt in out.items():
        lines = txt.strip().split("\n"); a_, b_ = map(int, lines[0].split())
        assert len(lines[1]) == a_ + b_ + 1, (strat, lines)
    assert out["min_sizes"].split("\n")[0] == "1 1" and len(out["min_sizes"].split("\n")[1]) == 3


def test_min_size_input_is_skipped_not_generated_invalid():
    spec, cb, sec = _cert_problem("N\nA_1 A_2 ... A_N", "1 \\leq N \\leq 50</li><li>1 \\leq A_i \\leq N-1")   # N=1 -> A_i in [1,0]: impossible
    out = build_inputs(spec, cb, sec)
    assert "min_sizes" not in out and "all_max" in out


def test_table_rows_obey_ordering_and_inequality_in_every_input_type():
    spec, cb, sec = _cert_problem("N M\na_1 b_1\n:\na_M b_M", "2 \\leq N \\leq 40</li><li>1 \\leq M \\leq 30</li><li>1 \\leq a_i &lt; b_i \\leq N")
    out = build_inputs(spec, cb, sec)
    for strat, txt in out.items():
        lines = txt.strip().split("\n"); n, m = map(int, lines[0].split()); assert len(lines) == 1 + m
        for row in lines[1:]:
            a_, b_ = map(int, row.split()); assert 1 <= a_ < b_ <= n, (strat, n, row)
    spec, cb, sec = _cert_problem("N\nc_1 d_1\n:\nc_N d_N", "2 \\leq N \\leq 40</li><li>1 \\leq c_i,d_i \\leq 5</li><li>c_i \\neq d_i")
    for strat, txt in build_inputs(spec, cb, sec).items():
        assert all(len(set(r.split())) == 2 for r in txt.strip().split("\n")[1:]), strat


def test_unsatisfiable_relation_is_refused_not_generated_invalid():
    spec, cb, sec = _cert_problem("N\na_1 b_1\n:\na_N b_N", "2 \\leq N \\leq 10</li><li>1 \\leq a_i,b_i \\leq 1</li><li>a_i &lt; b_i")
    with pytest.raises(GenError, match="relation"):
        build_inputs(spec, cb, sec)


def test_coverage_accepts_table_relations_but_not_scalar_ones():
    from tleye.certify.coverage import classify
    pre_t = "<h3>Constraints</h3><ul><li>2 \\leq N \\leq 100</li><li>1 \\leq M \\leq 100</li><li>1 \\leq a_i &lt; b_i \\leq N</li></ul><h3>Input</h3><pre>N M\na_1 b_1\n:\na_M b_M</pre><h3>Output</h3>"
    assert classify(pre_t)[0] == "simple"
    pre_s = "<h3>Constraints</h3><ul><li>2 \\leq N \\leq 100</li><li>1 \\leq A &lt; B \\leq N</li></ul><h3>Input</h3><pre>N A B</pre><h3>Output</h3>"
    assert classify(pre_s)[0] == "semantic_risk"      # scalar A < B is not handled yet
