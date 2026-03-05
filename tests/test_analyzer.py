"""Tests for the BASIC analyzer."""

import pytest
from basic_tile.lexer import lex
from basic_tile.parser import parse
from basic_tile.analyzer import analyze, AnalyzeError, BasicType


def analyze_src(src: str):
    return analyze(parse(lex(src)))


def test_let_type_inference_float():
    result = analyze_src("LET X = 42.0")
    assert result.symbols["X"].type == BasicType.F32


def test_let_type_inference_int():
    result = analyze_src("LET X = 42")
    assert result.symbols["X"].type == BasicType.I32


def test_integer_suffix():
    result = analyze_src("LET X% = 42.0")
    assert result.symbols["X%"].type == BasicType.I32


def test_expression_type_promotion():
    result = analyze_src("LET X = 1\nLET Y = X * 2.0")
    assert result.symbols["Y"].type == BasicType.F32


def test_comparison_yields_bool():
    result = analyze_src("IF X > 0 THEN END")
    # X should be inferred
    assert "X" in result.symbols


def test_for_loop_int():
    result = analyze_src("FOR I = 1 TO 10\nNEXT I")
    assert result.symbols["I"].type == BasicType.I32


def test_for_loop_float():
    result = analyze_src("FOR X = 0.0 TO 1.0 STEP 0.1\nNEXT X")
    assert result.symbols["X"].type == BasicType.F32


def test_dim_array():
    result = analyze_src("DIM A(10)")
    assert result.symbols["A"].is_array is True
    assert result.symbols["A"].array_size == 10


def test_data_values():
    result = analyze_src("DATA 1, 2, 3.5")
    assert result.data_values == [1, 2, 3.5]


def test_data_string_rejected():
    with pytest.raises(AnalyzeError):
        analyze_src('DATA 1, "hello", 3')


def test_input_vars():
    result = analyze_src('INPUT "X: "; X')
    assert "X" in result.input_vars


def test_goto_detected():
    result = analyze_src("10 GOTO 20\n20 END")
    assert result.has_goto is True


def test_no_goto():
    result = analyze_src("LET X = 1\nEND")
    assert result.has_goto is False


def test_function_type():
    result = analyze_src("LET X = SQR(16)")
    assert result.symbols["X"].type == BasicType.F32


def test_int_function_type():
    result = analyze_src("LET X = INT(3.7)")
    assert result.symbols["X"].type == BasicType.I32


def test_nested_if():
    src = "IF X > 0 THEN\nIF Y > 0 THEN\nPRINT X\nENDIF\nENDIF"
    result = analyze_src(src)
    assert "X" in result.symbols
    assert "Y" in result.symbols


def test_while_loop():
    src = "LET X = 10\nWHILE X > 0\nX = X - 1\nWEND"
    result = analyze_src(src)
    assert "X" in result.symbols


def test_read_vars():
    src = "DATA 1, 2\nREAD X, Y"
    result = analyze_src(src)
    assert "X" in result.symbols
    assert "Y" in result.symbols


def test_division_promotes_to_float():
    result = analyze_src("LET X = 10 / 3")
    assert result.symbols["X"].type == BasicType.F32


def test_complete_program():
    src = """10 REM Test program
20 LET X = 42.0
30 LET Y = X * 2.0
40 IF Y > 80 THEN
50   PRINT "big"
60 ELSE
70   PRINT "small"
80 ENDIF
90 FOR I = 1 TO 5
100   PRINT I
110 NEXT I
120 END"""
    result = analyze_src(src)
    assert result.symbols["X"].type == BasicType.F32
    assert result.symbols["Y"].type == BasicType.F32
    assert result.symbols["I"].type == BasicType.I32
