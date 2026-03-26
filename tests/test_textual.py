"""Tests for the textual backend."""

from pathlib import Path

import pytest
from cutile_basic._lexer import lex
from cutile_basic._parser import parse
from cutile_basic._analyzer import analyze
from cutile_basic._textual import TextualBackend


def compile_src(src: str) -> str:
    tokens = lex(src)
    prog = parse(tokens)
    analyzed = analyze(prog)
    return TextualBackend(analyzed).generate()


def test_module_structure():
    mlir = compile_src("END")
    assert "cuda_tile.module @basic_program {" in mlir
    assert "entry @main()" in mlir
    assert "  return" in mlir


def test_constant_float():
    mlir = compile_src("LET X = 42.0")
    assert "constant <f32:" in mlir
    assert "tile<f32>" in mlir


def test_constant_int():
    mlir = compile_src("LET X = 42")
    assert "constant <i32: 42>" in mlir
    assert "tile<i32>" in mlir


def test_arithmetic():
    mlir = compile_src("LET X = 3.0\nLET Y = X * 2.0")
    assert "mulf" in mlir
    assert "rounding<nearest_even>" in mlir


def test_integer_arithmetic():
    mlir = compile_src("LET X = 3\nLET Y = X + 2")
    assert "addi" in mlir


def test_print_string():
    mlir = compile_src('PRINT "Hello"')
    assert 'print "Hello\\n"' in mlir


def test_print_variable():
    mlir = compile_src("LET X = 42.0\nPRINT X")
    assert 'print "%f\\n"' in mlir


def test_print_mixed():
    mlir = compile_src('LET X = 42.0\nPRINT "X = "; X')
    assert 'print "X = %f\\n"' in mlir


def test_if_else():
    src = 'LET X = 1.0\nIF X > 0 THEN\nPRINT "pos"\nELSE\nPRINT "neg"\nENDIF'
    mlir = compile_src(src)
    assert "cmpf greater_than ordered" in mlir
    assert "if %" in mlir
    assert "} else {" in mlir


def test_if_no_else():
    src = 'LET X = 1.0\nIF X > 0 THEN\nPRINT "pos"\nENDIF'
    mlir = compile_src(src)
    assert "if %" in mlir
    assert "else" not in mlir


def test_for_loop():
    src = "FOR I = 1 TO 5\nPRINT I\nNEXT I"
    mlir = compile_src(src)
    assert "for %" in mlir
    assert "step" in mlir
    assert "continue" in mlir
    # Should have half-open bound computation
    assert "addi" in mlir


def test_for_loop_with_step():
    src = "FOR I = 0 TO 10 STEP 2\nNEXT I"
    mlir = compile_src(src)
    assert "for %" in mlir
    assert "step" in mlir


def test_comparison_ops():
    for op, expected in [
        (">", "greater_than"),
        ("<", "less_than"),
        (">=", "greater_than_or_equal"),
        ("<=", "less_than_or_equal"),
        ("=", "equal"),
        ("<>", "not_equal"),
    ]:
        mlir = compile_src(f"LET X = 1.0\nIF X {op} 0 THEN END")
        assert expected in mlir, f"Expected '{expected}' for op '{op}'"


def test_integer_comparison():
    mlir = compile_src("LET X = 1\nIF X > 0 THEN END")
    assert "cmpi greater_than" in mlir


def test_function_sqr():
    mlir = compile_src("LET X = SQR(16.0)")
    assert "sqrt" in mlir


def test_function_abs():
    mlir = compile_src("LET X = ABS(-5.0)")
    assert "absf" in mlir


def test_function_int():
    mlir = compile_src("LET X = INT(3.7)")
    assert "ftoi" in mlir


def test_function_sin():
    mlir = compile_src("LET X = SIN(1.0)")
    assert "= sin " in mlir


def test_rem_comment():
    mlir = compile_src("REM This is a comment")
    assert "// This is a comment" in mlir


def test_data_read():
    src = "DATA 1, 2, 3\nREAD X, Y, Z"
    mlir = compile_src(src)
    # Should generate constants for the data values
    assert "constant" in mlir


def test_dim_array():
    mlir = compile_src("DIM A(10)")
    assert "tile<10x" in mlir


def test_while_loop():
    src = "LET X = 10\nWHILE X > 0\nX = X - 1\nWEND"
    mlir = compile_src(src)
    assert "for %" in mlir  # WHILE compiled as bounded for


def test_unary_minus():
    mlir = compile_src("LET X = 5.0\nLET Y = -X")
    assert "negf" in mlir


def test_logical_and():
    src = "LET X = 1.0\nIF X > 0 AND X < 10 THEN END"
    mlir = compile_src(src)
    assert "andi" in mlir


def test_logical_or():
    src = "LET X = 1.0\nIF X < 0 OR X > 10 THEN END"
    mlir = compile_src(src)
    assert "ori" in mlir


def test_mixed_type_arithmetic():
    src = "LET X = 3\nLET Y = X * 2.0"
    mlir = compile_src(src)
    assert "itof" in mlir
    assert "mulf" in mlir


def test_power():
    src = "LET X = 2.0\nLET Y = X ^ 3.0"
    mlir = compile_src(src)
    assert "= pow " in mlir


def test_modulo():
    src = "LET X = 10\nLET Y = X MOD 3"
    mlir = compile_src(src)
    assert "remi" in mlir


def test_end_generates_return():
    mlir = compile_src("LET X = 1\nEND")
    # Should have two returns: one from END and one from kernel exit
    lines = [l.strip() for l in mlir.splitlines()]
    assert lines.count("return") >= 2


def test_ssa_numbering():
    mlir = compile_src("LET X = 1\nLET Y = 2\nLET Z = X + Y")
    # Check that SSA values are properly numbered
    assert "%0" in mlir
    assert "%1" in mlir
    assert "%2" in mlir


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
    mlir = compile_src(src)
    assert "cuda_tile.module @basic_program" in mlir
    assert "entry @main()" in mlir
    assert "Hello, World!" in mlir
    assert "mulf" in mlir
    assert "cmpf" in mlir
    assert "if %" in mlir
    assert "for %" in mlir


# --- Example .bas files ---

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _read_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text()


class TestExampleHelloCodegen:
    """Generate MLIR from examples/hello.bas and verify output."""

    def test_generates_without_error(self):
        mlir = compile_src(_read_example("hello.bas"))
        assert mlir is not None
        assert len(mlir) > 0

    def test_module_and_entry(self):
        mlir = compile_src(_read_example("hello.bas"))
        assert "cuda_tile.module @basic_program" in mlir
        assert "entry @main()" in mlir

    def test_matches_inline_hello_world(self):
        """The example file should produce the same output as the inline version."""
        file_mlir = compile_src(_read_example("hello.bas"))
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
        inline_mlir = compile_src(inline_src)
        assert file_mlir == inline_mlir

    def test_comment_preserved(self):
        mlir = compile_src(_read_example("hello.bas"))
        assert "// Hello World in BASIC" in mlir

    def test_print_hello_world(self):
        mlir = compile_src(_read_example("hello.bas"))
        assert "Hello, World!" in mlir

    def test_arithmetic_and_control_flow(self):
        mlir = compile_src(_read_example("hello.bas"))
        assert "mulf" in mlir
        assert "cmpf" in mlir
        assert "if %" in mlir
        assert "} else {" in mlir
        assert "for %" in mlir


class TestExampleVectorAddCodegen:
    """Generate MLIR from examples/vector_add.bas and verify output."""

    def test_generates_without_error(self):
        mlir = compile_src(_read_example("vector_add.bas"))
        assert mlir is not None
        assert len(mlir) > 0

    def test_module_structure(self):
        mlir = compile_src(_read_example("vector_add.bas"))
        assert "cuda_tile.module @basic_program" in mlir

    def test_has_kernel_params(self):
        """INPUT A(), B() should generate kernel parameters."""
        mlir = compile_src(_read_example("vector_add.bas"))
        assert "entry @main(" in mlir
        assert "tile<f32>" in mlir


class TestExampleGemmCodegen:
    """Generate MLIR from examples/gemm.bas and verify output."""

    def test_generates_without_error(self):
        mlir = compile_src(_read_example("gemm.bas"))
        assert mlir is not None
        assert len(mlir) > 0

    def test_module_structure(self):
        mlir = compile_src(_read_example("gemm.bas"))
        assert "cuda_tile.module @basic_program" in mlir

    def test_has_kernel_params(self):
        mlir = compile_src(_read_example("gemm.bas"))
        assert "entry @main(" in mlir

    def test_for_loop_generated(self):
        mlir = compile_src(_read_example("gemm.bas"))
        assert "for %" in mlir

    def test_comment_preserved(self):
        mlir = compile_src(_read_example("gemm.bas"))
        assert "// GEMM" in mlir
