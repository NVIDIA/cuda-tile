"""Tests for the BASIC parser."""

import pytest
from cutile_basic.lexer import lex
from cutile_basic.parser import parse, ParseError
from cutile_basic import ast_nodes as ast


def parse_src(src: str) -> ast.Program:
    return parse(lex(src))


def test_let_statement():
    prog = parse_src("LET X = 42")
    assert len(prog.statements) == 1
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.LetStatement)
    assert isinstance(stmt.target, ast.Variable)
    assert stmt.target.name == "X"
    assert isinstance(stmt.value, ast.NumberLiteral)
    assert stmt.value.value == 42


def test_implicit_let():
    prog = parse_src("X = 42")
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.LetStatement)
    assert stmt.target.name == "X"


def test_let_with_expression():
    prog = parse_src("LET Y = X * 2.0 + 1")
    stmt = prog.statements[0]
    assert isinstance(stmt.value, ast.BinaryOp)
    assert stmt.value.op == "+"


def test_print_string():
    prog = parse_src('PRINT "Hello"')
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.PrintStatement)
    assert len(stmt.items) == 1
    assert isinstance(stmt.items[0], ast.StringLiteral)


def test_print_multiple():
    prog = parse_src('PRINT "X = "; X')
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.PrintStatement)
    assert len(stmt.items) == 2


def test_print_trailing_semicolon():
    prog = parse_src('PRINT "no newline";')
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.PrintStatement)
    assert stmt.newline is False


def test_if_single_line():
    prog = parse_src("IF X > 0 THEN PRINT X")
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.IfStatement)
    assert len(stmt.then_body) == 1
    assert len(stmt.else_body) == 0


def test_if_block():
    src = 'IF X > 0 THEN\nPRINT "pos"\nELSE\nPRINT "neg"\nENDIF'
    prog = parse_src(src)
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.IfStatement)
    assert len(stmt.then_body) == 1
    assert len(stmt.else_body) == 1


def test_for_loop():
    src = "FOR I = 1 TO 10\nPRINT I\nNEXT I"
    prog = parse_src(src)
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.ForStatement)
    assert stmt.var.name == "I"
    assert len(stmt.body) == 1
    assert stmt.step is None


def test_for_with_step():
    src = "FOR I = 0 TO 100 STEP 5\nPRINT I\nNEXT I"
    prog = parse_src(src)
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.ForStatement)
    assert isinstance(stmt.step, ast.NumberLiteral)
    assert stmt.step.value == 5


def test_while_loop():
    src = "WHILE X > 0\nX = X - 1\nWEND"
    prog = parse_src(src)
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.WhileStatement)
    assert len(stmt.body) == 1


def test_goto():
    prog = parse_src("GOTO 100")
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.GotoStatement)
    assert stmt.target == 100


def test_gosub_return():
    src = "GOSUB 100\nRETURN"
    prog = parse_src(src)
    assert isinstance(prog.statements[0], ast.GosubStatement)
    assert isinstance(prog.statements[1], ast.ReturnStatement)


def test_dim():
    prog = parse_src("DIM A(10)")
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.DimStatement)
    assert stmt.name == "A"
    assert len(stmt.sizes) == 1


def test_rem():
    prog = parse_src("REM This is a comment")
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.RemStatement)


def test_data_read():
    src = "DATA 1, 2, 3\nREAD X, Y, Z"
    prog = parse_src(src)
    assert isinstance(prog.statements[0], ast.DataStatement)
    assert prog.statements[0].values == [1, 2, 3]
    assert isinstance(prog.statements[1], ast.ReadStatement)
    assert len(prog.statements[1].variables) == 3


def test_end():
    prog = parse_src("END")
    assert isinstance(prog.statements[0], ast.EndStatement)


def test_input():
    prog = parse_src('INPUT "Enter X: "; X')
    stmt = prog.statements[0]
    assert isinstance(stmt, ast.InputStatement)
    assert stmt.prompt == "Enter X: "
    assert stmt.variables[0].name == "X"


def test_nested_expression():
    prog = parse_src("LET X = (1 + 2) * 3")
    stmt = prog.statements[0]
    assert isinstance(stmt.value, ast.BinaryOp)
    assert stmt.value.op == "*"
    assert isinstance(stmt.value.left, ast.BinaryOp)
    assert stmt.value.left.op == "+"


def test_function_call():
    prog = parse_src("LET X = SQR(16)")
    stmt = prog.statements[0]
    assert isinstance(stmt.value, ast.FunctionCall)
    assert stmt.value.name == "SQR"


def test_unary_minus():
    prog = parse_src("LET X = -5")
    stmt = prog.statements[0]
    assert isinstance(stmt.value, ast.UnaryOp)
    assert stmt.value.op == "-"


def test_power_right_associative():
    prog = parse_src("LET X = 2 ^ 3 ^ 4")
    stmt = prog.statements[0]
    # Should be 2 ^ (3 ^ 4)
    assert isinstance(stmt.value, ast.BinaryOp)
    assert stmt.value.op == "^"
    assert isinstance(stmt.value.right, ast.BinaryOp)
    assert stmt.value.right.op == "^"


def test_line_numbers_in_line_map():
    src = "10 LET X = 1\n20 LET Y = 2\n30 END"
    prog = parse_src(src)
    assert 10 in prog.line_map
    assert 20 in prog.line_map
    assert 30 in prog.line_map


def test_comparison_operators():
    for op_str in ["=", "<>", "<", ">", "<=", ">="]:
        src = f"IF X {op_str} 0 THEN END"
        prog = parse_src(src)
        stmt = prog.statements[0]
        assert isinstance(stmt, ast.IfStatement)
        assert isinstance(stmt.condition, ast.BinaryOp)
        assert stmt.condition.op == op_str


def test_array_access_in_expression():
    prog = parse_src("LET X = A(5)")
    stmt = prog.statements[0]
    assert isinstance(stmt.value, ast.ArrayAccess)
    assert stmt.value.name == "A"


def test_array_assignment():
    prog = parse_src("LET A(5) = 42")
    stmt = prog.statements[0]
    assert isinstance(stmt.target, ast.ArrayAccess)
