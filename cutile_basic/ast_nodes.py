"""AST node classes for the BASIC parser."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# --- Expressions ---

@dataclass
class NumberLiteral:
    value: float | int
    line: int = 0

@dataclass
class StringLiteral:
    value: str
    line: int = 0

@dataclass
class Variable:
    name: str
    line: int = 0

@dataclass
class ArrayAccess:
    name: str
    index: Expression
    index2: Expression | None = None  # second index for 2D access
    line: int = 0

@dataclass
class UnaryOp:
    op: str          # "-", "NOT"
    operand: Expression
    line: int = 0

@dataclass
class BinaryOp:
    op: str          # "+", "-", "*", "/", "^", "MOD", "AND", "OR", comparisons
    left: Expression
    right: Expression
    line: int = 0

@dataclass
class FunctionCall:
    name: str        # ABS, SQR, INT, SIN, COS, TAN, EXP, LOG, SGN
    arg: Expression
    line: int = 0

Expression = NumberLiteral | StringLiteral | Variable | ArrayAccess | UnaryOp | BinaryOp | FunctionCall


# --- Statements ---

@dataclass
class LetStatement:
    target: Variable | ArrayAccess
    value: Expression
    line: int = 0

@dataclass
class PrintStatement:
    items: list[Expression]
    newline: bool = True      # False if trailing semicolon
    line: int = 0

@dataclass
class InputStatement:
    prompt: Optional[str]
    variables: list[Variable]
    is_array: list[bool] = field(default_factory=list)  # per-variable array flag
    line: int = 0

@dataclass
class IfStatement:
    condition: Expression
    then_body: list[Statement]
    else_body: list[Statement]
    line: int = 0

@dataclass
class ForStatement:
    var: Variable
    start: Expression
    end: Expression
    step: Optional[Expression]
    body: list[Statement]
    line: int = 0

@dataclass
class WhileStatement:
    condition: Expression
    body: list[Statement]
    line: int = 0

@dataclass
class GotoStatement:
    target: int              # line number
    line: int = 0

@dataclass
class GosubStatement:
    target: int              # line number
    line: int = 0

@dataclass
class ReturnStatement:
    line: int = 0

@dataclass
class DimStatement:
    name: str
    sizes: list[Expression]  # DIM A(512, 512) → sizes=[512, 512]
    line: int = 0

@dataclass
class RemStatement:
    comment: str
    line: int = 0

@dataclass
class DataStatement:
    values: list[float | int | str]
    line: int = 0

@dataclass
class ReadStatement:
    variables: list[Variable]
    line: int = 0

@dataclass
class EndStatement:
    line: int = 0

@dataclass
class StopStatement:
    line: int = 0

@dataclass
class TileStatement:
    name: str
    sizes: list[Expression]  # TILE A(128, 32) → sizes=[128, 32]
    line: int = 0

@dataclass
class MmaStatement:
    acc_var: str            # accumulator variable name
    a_access: ArrayAccess   # 2D tile load from A
    b_access: ArrayAccess   # 2D tile load from B
    line: int = 0

@dataclass
class OutputStatement:
    variables: list[Variable]
    line: int = 0

Statement = (
    LetStatement | PrintStatement | InputStatement | IfStatement |
    ForStatement | WhileStatement | GotoStatement | GosubStatement |
    ReturnStatement | DimStatement | RemStatement | DataStatement |
    ReadStatement | EndStatement | StopStatement |
    TileStatement | MmaStatement | OutputStatement
)


@dataclass
class Program:
    statements: list[Statement]
    line_map: dict[int, int] = field(default_factory=dict)  # BASIC line# → index
