"""Tests for the BASIC lexer."""

import pytest
from basic_tile.lexer import lex, LexError
from basic_tile.tokens import TokenType


def tok_types(source: str) -> list[TokenType]:
    return [t.type for t in lex(source) if t.type not in (TokenType.NEWLINE, TokenType.EOF)]


def test_number_literals():
    tokens = lex("42 3.14 1E5")
    types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
    assert types == [TokenType.INTEGER, TokenType.FLOAT, TokenType.FLOAT]
    assert tokens[0].value == "42"
    assert tokens[1].value == "3.14"
    assert tokens[2].value == "1E5"


def test_string_literal():
    tokens = lex('"Hello, World!"')
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "Hello, World!"


def test_unterminated_string():
    with pytest.raises(LexError):
        lex('"oops')


def test_keywords():
    assert tok_types("LET PRINT IF THEN ELSE ENDIF") == [
        TokenType.LET, TokenType.PRINT, TokenType.IF,
        TokenType.THEN, TokenType.ELSE, TokenType.ENDIF,
    ]


def test_keywords_case_insensitive():
    assert tok_types("let print") == [TokenType.LET, TokenType.PRINT]


def test_identifiers():
    tokens = lex("X Y% myVar")
    types = tok_types("X Y% myVar")
    assert types == [TokenType.IDENTIFIER, TokenType.IDENTIFIER, TokenType.IDENTIFIER]
    assert tokens[0].value == "X"
    assert tokens[1].value == "Y%"


def test_operators():
    assert tok_types("+ - * / ^ = <> < > <= >=") == [
        TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
        TokenType.CARET, TokenType.EQ, TokenType.NEQ, TokenType.LT,
        TokenType.GT, TokenType.LE, TokenType.GE,
    ]


def test_delimiters():
    assert tok_types("( ) , ; :") == [
        TokenType.LPAREN, TokenType.RPAREN, TokenType.COMMA,
        TokenType.SEMICOLON, TokenType.COLON,
    ]


def test_rem_comment():
    tokens = lex("REM This is a comment")
    assert tokens[0].type == TokenType.REM
    assert tokens[0].value == "This is a comment"


def test_line_numbers():
    tokens = lex("10 LET X = 5")
    assert tokens[0].type == TokenType.INTEGER
    assert tokens[0].value == "10"
    assert tokens[1].type == TokenType.LET


def test_multiline():
    src = "LET X = 1\nLET Y = 2"
    tokens = lex(src)
    types = [t.type for t in tokens]
    assert TokenType.NEWLINE in types


def test_for_to_step():
    assert tok_types("FOR I = 1 TO 10 STEP 2") == [
        TokenType.FOR, TokenType.IDENTIFIER, TokenType.EQ,
        TokenType.INTEGER, TokenType.TO, TokenType.INTEGER,
        TokenType.STEP, TokenType.INTEGER,
    ]


def test_logical_operators():
    assert tok_types("AND OR NOT MOD") == [
        TokenType.AND, TokenType.OR, TokenType.NOT, TokenType.MOD,
    ]


def test_unexpected_char():
    with pytest.raises(LexError):
        lex("LET X = @")


def test_complete_program():
    src = '10 PRINT "Hello"\n20 LET X = 42.0\n30 END'
    tokens = lex(src)
    # Should not raise and should end with EOF
    assert tokens[-1].type == TokenType.EOF
