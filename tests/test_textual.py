"""Tests for the textual backend."""

from pathlib import Path

import pytest
from cutile_basic.lexer import lex
from cutile_basic.parser import parse
from cutile_basic.analyzer import analyze
from cutile_basic.textual import TextualBackend


def compile_src(src: str) -> str:
    tokens = lex(src)
    prog = parse(tokens)
    analyzed = analyze(prog)
    return TextualBackend(analyzed).generate()


def test_module_structure():
    text = compile_src("END")
    assert "cuda_tile.module @basic_program {" in text
    assert "entry @main()" in text
    assert "  return" in text


def test_constant_float():
    text = compile_src("LET X = 42.0")
    assert "constant <f32:" in text
    assert "tile<f32>" in text


def test_constant_int():
    text = compile_src("LET X = 42")
    assert "constant <i32: 42>" in text
    assert "tile<i32>" in text


def test_arithmetic():
    text = compile_src("LET X = 3.0\nLET Y = X * 2.0")
    assert "mulf" in text
    assert "rounding<nearest_even>" in text


def test_integer_arithmetic():
    text = compile_src("LET X = 3\nLET Y = X + 2")
    assert "addi" in text


def test_print_string():
    text = compile_src('PRINT "Hello"')
    assert 'print "Hello\\n"' in text


def test_print_variable():
    text = compile_src("LET X = 42.0\nPRINT X")
    assert 'print "%f\\n"' in text


def test_print_mixed():
    text = compile_src('LET X = 42.0\nPRINT "X = "; X')
    assert 'print "X = %f\\n"' in text


def test_if_else():
    src = 'LET X = 1.0\nIF X > 0 THEN\nPRINT "pos"\nELSE\nPRINT "neg"\nENDIF'
    text = compile_src(src)
    assert "cmpf greater_than ordered" in text
    assert "if %" in text
    assert "} else {" in text


def test_if_no_else():
    src = 'LET X = 1.0\nIF X > 0 THEN\nPRINT "pos"\nENDIF'
    text = compile_src(src)
    assert "if %" in text
    assert "else" not in text


def test_for_loop():
    src = "FOR I = 1 TO 5\nPRINT I\nNEXT I"
    text = compile_src(src)
    assert "for %" in text
    assert "step" in text
    assert "continue" in text
    # Should have half-open bound computation
    assert "addi" in text


def test_for_loop_with_step():
    src = "FOR I = 0 TO 10 STEP 2\nNEXT I"
    text = compile_src(src)
    assert "for %" in text
    assert "step" in text


def test_comparison_ops():
    for op, expected in [
        (">", "greater_than"),
        ("<", "less_than"),
        (">=", "greater_than_or_equal"),
        ("<=", "less_than_or_equal"),
        ("=", "equal"),
        ("<>", "not_equal"),
    ]:
        text = compile_src(f"LET X = 1.0\nIF X {op} 0 THEN END")
        assert expected in text, f"Expected '{expected}' for op '{op}'"


def test_integer_comparison():
    text = compile_src("LET X = 1\nIF X > 0 THEN END")
    assert "cmpi greater_than" in text


def test_function_sqr():
    text = compile_src("LET X = SQR(16.0)")
    assert "sqrt" in text


def test_function_abs():
    text = compile_src("LET X = ABS(-5.0)")
    assert "absf" in text


def test_function_int():
    text = compile_src("LET X = INT(3.7)")
    assert "ftoi" in text


def test_function_sin():
    text = compile_src("LET X = SIN(1.0)")
    assert "= sin " in text


def test_function_cos():
    text = compile_src("LET X = COS(1.0)")
    assert "= cos " in text


def test_function_tan():
    text = compile_src("LET X = TAN(1.0)")
    assert "= tan " in text


def test_function_exp():
    text = compile_src("LET X = EXP(1.0)")
    assert "= exp " in text


def test_function_log():
    text = compile_src("LET X = LOG(1.0)")
    assert "= log " in text


def test_function_sgn():
    text = compile_src("LET X = SGN(5.0)")
    assert "cmpf" in text
    assert "select" in text


def test_rem_comment():
    text = compile_src("REM This is a comment")
    assert "// This is a comment" in text


def test_data_read():
    src = "DATA 1, 2, 3\nREAD X, Y, Z"
    text = compile_src(src)
    # Should generate constants for the data values
    assert "constant" in text


def test_dim_array():
    text = compile_src("DIM A(10)")
    assert "tile<10x" in text


def test_while_loop():
    src = "LET X = 10\nWHILE X > 0\nX = X - 1\nWEND"
    text = compile_src(src)
    assert "for %" in text  # WHILE compiled as bounded for


def test_unary_minus():
    text = compile_src("LET X = 5.0\nLET Y = -X")
    assert "negf" in text


def test_logical_and():
    src = "LET X = 1.0\nIF X > 0 AND X < 10 THEN END"
    text = compile_src(src)
    assert "andi" in text


def test_logical_or():
    src = "LET X = 1.0\nIF X < 0 OR X > 10 THEN END"
    text = compile_src(src)
    assert "ori" in text


def test_mixed_type_arithmetic():
    src = "LET X = 3\nLET Y = X * 2.0"
    text = compile_src(src)
    assert "itof" in text
    assert "mulf" in text


def test_power():
    src = "LET X = 2.0\nLET Y = X ^ 3.0"
    text = compile_src(src)
    assert "= pow " in text


def test_modulo():
    src = "LET X = 10\nLET Y = X MOD 3"
    text = compile_src(src)
    assert "remi" in text


def test_end_generates_return():
    text = compile_src("LET X = 1\nEND")
    # Should have two returns: one from END and one from kernel exit
    lines = [l.strip() for l in text.splitlines()]
    assert lines.count("return") >= 2


def test_ssa_numbering():
    text = compile_src("LET X = 1\nLET Y = 2\nLET Z = X + Y")
    # Check that SSA values are properly numbered
    assert "%0" in text
    assert "%1" in text
    assert "%2" in text


def test_hello_world_program():
    src = """10 REM Hello World in BASIC
20 PRINT "Hello, World!"
30 LET X = 42.0
40 LET Y = X * 2.0
50 PRINT "X = "; X
60 PRINT "Y = "; Y
70 IF Y > 80 THEN
80   PRINT "Y is large"
90 ELSE
100  PRINT "Y is small"
110 ENDIF
120 FOR I = 1 TO 5
130   PRINT "I = "; I
140 NEXT I
150 END"""
    text = compile_src(src)
    assert "cuda_tile.module @basic_program" in text
    assert "entry @main()" in text
    assert "Hello, World!" in text
    assert "mulf" in text
    assert "cmpf" in text
    assert "if %" in text
    assert "for %" in text


# --- Example .bas files ---

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _read_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text()


class TestExampleHelloCodegen:
    """Generate textual output from examples/hello.bas and verify output."""

    def test_generates_without_error(self):
        text = compile_src(_read_example("hello.bas"))
        assert text is not None
        assert len(text) > 0

    def test_module_and_entry(self):
        text = compile_src(_read_example("hello.bas"))
        assert "cuda_tile.module @basic_program" in text
        assert "entry @main()" in text

    def test_matches_inline_hello_world(self):
        """The example file should produce the same output as the inline version."""
        file_text = compile_src(_read_example("hello.bas"))
        inline_src = """10 REM Hello World in BASIC
20 PRINT "Hello, World!"
30 LET X = 42.0
40 LET Y = X * 2.0
50 PRINT "X = "; X
60 PRINT "Y = "; Y
70 IF Y > 80 THEN
80   PRINT "Y is large"
90 ELSE
100  PRINT "Y is small"
110 ENDIF
120 FOR I = 1 TO 5
130   PRINT "I = "; I
140 NEXT I
150 END"""
        inline_text = compile_src(inline_src)
        assert file_text == inline_text

    def test_comment_preserved(self):
        text = compile_src(_read_example("hello.bas"))
        assert "// Hello World in BASIC" in text

    def test_print_hello_world(self):
        text = compile_src(_read_example("hello.bas"))
        assert "Hello, World!" in text

    def test_arithmetic_and_control_flow(self):
        text = compile_src(_read_example("hello.bas"))
        assert "mulf" in text
        assert "cmpf" in text
        assert "if %" in text
        assert "} else {" in text
        assert "for %" in text


class TestExampleVectorAddCodegen:
    """Generate textual output from examples/vector_add.bas and verify output."""

    def test_generates_without_error(self):
        text = compile_src(_read_example("vector_add.bas"))
        assert text is not None
        assert len(text) > 0

    def test_module_structure(self):
        text = compile_src(_read_example("vector_add.bas"))
        assert "cuda_tile.module @basic_program" in text

    def test_has_kernel_params(self):
        """INPUT A(), B() should generate kernel parameters."""
        text = compile_src(_read_example("vector_add.bas"))
        assert "entry @main(" in text
        assert "tile<f32>" in text


class TestExampleGemmCodegen:
    """Generate textual output from examples/gemm.bas and verify output."""

    def test_generates_without_error(self):
        text = compile_src(_read_example("gemm.bas"))
        assert text is not None
        assert len(text) > 0

    def test_module_structure(self):
        text = compile_src(_read_example("gemm.bas"))
        assert "cuda_tile.module @basic_program" in text

    def test_has_kernel_params(self):
        text = compile_src(_read_example("gemm.bas"))
        assert "entry @main(" in text

    def test_for_loop_generated(self):
        text = compile_src(_read_example("gemm.bas"))
        assert "for %" in text

    def test_comment_preserved(self):
        text = compile_src(_read_example("gemm.bas"))
        assert "// GEMM" in text
