import pytest

from tleye.certify.inputspec import eval_expr, parse_input_spec


def page(pre: str, heading: str = "Input") -> str:
    return f"<section><h3>{heading}</h3><p>Input is given from Standard Input in the following format:</p><pre>{pre}</pre></section><h3>Output</h3>"


def test_scalars_and_array():
    s = parse_input_spec(page("<var>N</var> <var>M</var>\n<var>A_1</var> <var>A_2</var> <var>...</var> <var>A_M</var>\n"))
    assert s.ok and s.lines == [("scalars", ["N", "M"]), ("array", "A", "M")]


def test_table_with_colon_and_vdots():
    s = parse_input_spec(page("<var>N</var>\n<var>t_1</var> <var>x_1</var> <var>y_1</var>\n<var>t_2</var> <var>x_2</var> <var>y_2</var>\n<var>:</var>\n<var>t_N</var> <var>x_N</var> <var>y_N</var>\n"))
    assert s.ok and s.lines == [("scalars", ["N"]), ("table", ["t", "x", "y"], "N")]
    s = parse_input_spec(page("N\na_1 b_1\n\\vdots\na_{N-1} b_{N-1}\n"))
    assert s.ok and s.lines[1] == ("table", ["a", "b"], "N-1")


def test_string_line():
    s = parse_input_spec(page("N\nS\n"))
    assert s.ok and s.lines == [("scalars", ["N"]), ("scalars", ["S"])]


@pytest.mark.parametrize("pre", ["N\nc_1 w_{1,1} w_{1,2} ... w_{1,c_1}\n:\nc_{N-1} w_{N-1,1} ... w_{N-1,c_{N-1}}\n",
                                 "N M\nk_1 s_{11} s_{12} ... s_{1k_1}\n:\nk_M s_{M1} ... s_{Mk_M}\n"])
def test_ragged_rows_are_rejected_not_guessed(pre):
    assert not parse_input_spec(page(pre)).ok


def test_interactive_and_missing_blocks_rejected():
    assert parse_input_spec(page("x", heading="Input and output")).reason == "interactive_or_special_io"
    assert parse_input_spec("<h3>Constraints</h3>").reason == "no_input_section"


def test_eval_expr_is_safe_and_correct():
    assert eval_expr("N-1", {"N": 10}) == 9 and eval_expr("2N", {"N": 5}) == 10 and eval_expr("N*M", {"N": 3, "M": 4}) == 12
    for bad in ["__import__('os')", "N**99", "open('x')"]:
        with pytest.raises(Exception):
            eval_expr(bad, {"N": 3})


def test_zero_indexed_arrays_and_tables_have_one_more_element():
    s = parse_input_spec(page("N\nA_0 A_1 A_2 \\cdots A_N\n"))
    assert s.lines == [("scalars", ["N"]), ("array", "A", "N+1")]
    s = parse_input_spec(page("N\nA_0 B_0\nA_1 B_1\n:\nA_N B_N\n"))
    assert s.lines[1] == ("table", ["A", "B"], "N+1")
    assert parse_input_spec(page("N\nA_1 A_2 ... A_N\n")).lines[1] == ("array", "A", "N")      # 1-indexed unchanged


def test_multiple_format_blocks_are_rejected_not_truncated_to_the_first():
    two = "<h3>Input</h3><p>fmt</p><pre>T\n</pre><p>Each case is given in the following format:</p><pre>N A B</pre><h3>Output</h3>"
    assert parse_input_spec(two).reason == "multiple_format_blocks"
