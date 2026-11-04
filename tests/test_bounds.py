from tleye.certify.bounds import normalise, parse, upper_for
from tleye.constraints.parser import parse_text


def test_expression_bound_is_evaluated_per_input_type():
    b = parse("2≦N≦200,000\n0≦A_i,B_i≦N-1")
    assert b.upper["N"] == 200000 and b.upper["A"] == 199999 and b.upper["B"] == 199999
    assert upper_for(b, "A", {"N": 200000}) == 199999
    assert upper_for(b, "A", {"N": 2}) == 1            # minimum-size input: A_i must not exceed N-1 for THAT N


def test_var_reference_bound_follows_the_input_type():
    b = parse("1 \\leq N \\leq 100\n1 \\leq A_i \\leq N")
    assert upper_for(b, "A", {"N": 100}) == 100 and upper_for(b, "A", {"N": 3}) == 3


def test_absolute_value_bounds_give_symmetric_ranges():
    b = parse("1 \\leq N \\leq 100\n|A_i| \\leq 10^9\n|x_i|,|y_i| \\leq 10^6(1\\leq i\\leq N)")
    assert b.upper["A"] == 1e9 and b.lower["A"] == -1e9
    assert b.upper["x"] == b.upper["y"] == 1e6 and b.lower["y"] == -1e6


def test_trailing_index_parenthetical_no_longer_blocks_the_bound():
    assert parse("1 \\leq Q \\leq 100\n1\\leq A_i,B_i\\leq 10^9(1\\leq i\\leq Q)").upper["A"] == 1e9
    assert normalise("1 \\leq N \\leq 1 \\ 000 (inclusive)").startswith("1 \\leq N \\leq 1000")


def test_thousands_fix_does_not_join_numbers_across_lines():
    assert "100\n200" in normalise("1 \\leq A \\leq 100\n200 \\leq B")


def test_string_length_equalities():
    b = parse("1≤A,B≤5\n|S|=A+B+1")
    assert b.upper["|S|"] == 11 and b.lower["|S|"] == 3
    assert upper_for(b, "|S|", {"A": 1, "B": 1}) == 3 and upper_for(b, "|S|", {"A": 5, "B": 5}) == 11
    assert parse("1 \\leq N \\leq 10\nS is N characters long").upper["|S|"] == 10


def test_chained_order_resolves_to_numbers():
    b = parse("2 \\leq N \\leq 100\n1 \\leq A < B \\leq N")
    assert b.upper["B"] == 100 and b.upper["A"] == 100


def test_never_removes_or_changes_what_the_frozen_reader_found():
    sec = "1 \\leq N \\leq 10^5\n1 \\leq A_i \\leq 10^9\n2 \\leq K \\leq N\nK is an integer."
    v1, v2 = parse_text(sec), parse(sec)
    for k, v in v1.upper.items():
        assert v2.upper[k] == v


def test_same_row_relations_are_extracted_and_other_patterns_ignored():
    b = parse("2 \\leq N \\leq 100\n1 \\leq a_i < b_i \\leq N\nc_i \\neq d_i\nx_i \\leq y_i\nx_i < x_{i+1}\ni \\neq j")
    assert ("a", "<", "b") in b.relations and ("c", "!=", "d") in b.relations and ("x", "<=", "y") in b.relations
    assert len(b.relations) == 3            # x_i < x_{i+1} (different subscripts) and 'i != j' are NOT same-row relations
