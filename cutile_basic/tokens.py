"""Token types and Token dataclass for the BASIC lexer."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Keywords
    LET = auto()
    PRINT = auto()
    INPUT = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    ENDIF = auto()
    FOR = auto()
    TO = auto()
    STEP = auto()
    NEXT = auto()
    WHILE = auto()
    WEND = auto()
    GOTO = auto()
    GOSUB = auto()
    RETURN = auto()
    DIM = auto()
    REM = auto()
    DATA = auto()
    READ = auto()
    END = auto()
    STOP = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    MOD = auto()
    OUTPUT = auto()
    TILE = auto()
    BID = auto()

    # Operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    CARET = auto()      # ^
    EQ = auto()         # =
    NEQ = auto()        # <>
    LT = auto()         # <
    GT = auto()         # >
    LE = auto()         # <=
    GE = auto()         # >=

    # Delimiters
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    COMMA = auto()      # ,
    SEMICOLON = auto()  # ;

    # Special
    NEWLINE = auto()
    EOF = auto()
    COLON = auto()      # : (statement separator)


KEYWORDS = {
    "LET": TokenType.LET,
    "PRINT": TokenType.PRINT,
    "INPUT": TokenType.INPUT,
    "IF": TokenType.IF,
    "THEN": TokenType.THEN,
    "ELSE": TokenType.ELSE,
    "ENDIF": TokenType.ENDIF,
    "FOR": TokenType.FOR,
    "TO": TokenType.TO,
    "STEP": TokenType.STEP,
    "NEXT": TokenType.NEXT,
    "WHILE": TokenType.WHILE,
    "WEND": TokenType.WEND,
    "GOTO": TokenType.GOTO,
    "GOSUB": TokenType.GOSUB,
    "RETURN": TokenType.RETURN,
    "DIM": TokenType.DIM,
    "REM": TokenType.REM,
    "DATA": TokenType.DATA,
    "READ": TokenType.READ,
    "END": TokenType.END,
    "STOP": TokenType.STOP,
    "AND": TokenType.AND,
    "OR": TokenType.OR,
    "NOT": TokenType.NOT,
    "MOD": TokenType.MOD,
    "OUTPUT": TokenType.OUTPUT,
    "TILE": TokenType.TILE,
    "BID": TokenType.BID,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"
