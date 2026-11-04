import pytest

from tleye.constraints.parser import parse_number, parse_text


@pytest.mark.parametrize("s,v", [
    ("10^5", 1e5), ("2 \\times 10^5", 2e5), ("100,000", 1e5), ("10^{9}+7", 1e9 + 7),
    ("2\\times10^{5}", 2e5), ("5000", 5000), ("10^18", 1e18), ("2^{31}", 2 ** 31),
])
def test_numbers(s, v):
    assert parse_number(s) == v


@pytest.mark.parametrize("s", ["N", "A_i", "", "10^", "__import__('os')", "9**9**9"])
def test_non_numbers_and_unsafe(s):
    assert parse_number(s) is None


def test_chain_and_refs():
    c = parse_text("1 \\leq N \\leq 10^5\n1 \\leq A_i \\leq N\n1 \\leq a,b \\leq 15\n1 \\leq B_i \\leq 10^4")
    assert c.upper["N"] == 1e5
    assert c.upper["A"] == 1e5          # resolved from A_i <= N
    assert c.upper["a"] == 15 and c.upper["b"] == 15
    assert c.upper["B"] == 1e4
    assert c.size_bound == 1e5 and c.max_bound == 1e5


def test_unicode_relations_and_greater_than():
    c = parse_text("1 ≦ N ≦ 5000\n2 ≤ W ≤ 300\n10^5 \\geq M")
    assert c.upper["N"] == 5000 and c.upper["W"] == 300 and c.upper["M"] == 1e5


def test_no_relation_lines_ignored():
    c = parse_text("All values in input are integers.\ns consists of the letters 0 and 1 .")
    assert c.upper == {} and c.max_bound is None


def test_natural_language_patterns():
    from tleye.constraints.parser import HUGE
    c = parse_text("K is an integer between 1 and 10^6 (inclusive).\nN is a positive integer not exceeding 999 .")
    assert c.upper["K"] == 1e6 and c.upper["N"] == 999
    c = parse_text("S and T have lengths between 1 and 2\\times 10^5 (inclusive).")
    assert c.upper["|S|"] == 2e5 and c.upper["|T|"] == 2e5
    c = parse_text("The lengths of S and T are each at least 1 and at most 1000 .")
    assert c.upper["|S|"] == 1000 and c.upper["|T|"] == 1000
    assert parse_text("|S| = 3").upper["|S|"] == 3
    assert parse_text("0 \\leq N < 10^{200000}").upper["N"] == HUGE


def test_section_stops_at_input_even_without_newline():
    from tleye.constraints.parser import parse_statement
    html = "<h3>Constraints</h3><ul><li>1 \\leq N \\leq 200,000</li><li>All values are integers.</li></ul><h3>Input</h3><pre>N M</pre>"
    c = parse_statement(html)
    assert c.upper == {"N": 200000.0}
