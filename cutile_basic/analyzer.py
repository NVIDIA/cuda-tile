"""Semantic analyzer: type inference, GOTO elimination, DATA/READ resolution."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from . import ast_nodes as ast


class AnalyzeError(Exception):
    pass


class BasicType(Enum):
    F32 = auto()
    I32 = auto()
    I1 = auto()   # boolean
    STRING = auto()

    def tile_type(self) -> str:
        return {
            BasicType.F32: "tile<f32>",
            BasicType.I32: "tile<i32>",
            BasicType.I1: "tile<i1>",
        }[self]

    def scalar_type(self) -> str:
        return {
            BasicType.F32: "f32",
            BasicType.I32: "i32",
            BasicType.I1: "i1",
        }[self]


@dataclass
class SymbolInfo:
    name: str
    type: BasicType
    is_array: bool = False
    array_size: int | None = None


@dataclass
class AnalyzedProgram:
    statements: list[ast.Statement]
    symbols: dict[str, SymbolInfo]
    data_values: list[float | int]
    input_vars: list[str]       # Variables used in INPUT (become kernel params)
    output_vars: list[str] = None  # Variables used in OUTPUT (array results)
    has_goto: bool = False


class Analyzer:
    def __init__(self):
        self.symbols: dict[str, SymbolInfo] = {}
        self.data_values: list[float | int] = []
        self.input_vars: list[str] = []
        self.output_vars: list[str] = []
        self.has_goto = False
        self.goto_targets: set[int] = set()

    def analyze(self, program: ast.Program) -> AnalyzedProgram:
        # First pass: collect DATA values and detect GOTOs
        self._collect_metadata(program.statements)

        # Second pass: type inference on all statements
        for stmt in program.statements:
            self._analyze_stmt(stmt)

        # Try simple GOTO elimination
        new_stmts = program.statements
        if self.has_goto:
            new_stmts = self._eliminate_gotos(program)

        return AnalyzedProgram(
            statements=new_stmts,
            symbols=self.symbols,
            data_values=self.data_values,
            input_vars=self.input_vars,
            output_vars=self.output_vars,
            has_goto=self.has_goto,
        )

    def _collect_metadata(self, stmts: list[ast.Statement]):
        for stmt in stmts:
            if isinstance(stmt, ast.DataStatement):
                for v in stmt.values:
                    if isinstance(v, str):
                        raise AnalyzeError("String values in DATA not supported for Tile IR")
                    self.data_values.append(v)
            elif isinstance(stmt, ast.GotoStatement):
                self.has_goto = True
                self.goto_targets.add(stmt.target)
            elif isinstance(stmt, ast.GosubStatement):
                self.has_goto = True
                self.goto_targets.add(stmt.target)
            elif isinstance(stmt, ast.IfStatement):
                self._collect_metadata(stmt.then_body)
                self._collect_metadata(stmt.else_body)
            elif isinstance(stmt, ast.ForStatement):
                self._collect_metadata(stmt.body)
            elif isinstance(stmt, ast.WhileStatement):
                self._collect_metadata(stmt.body)

    def _analyze_stmt(self, stmt: ast.Statement):
        if isinstance(stmt, ast.LetStatement):
            val_type = self._infer_type(stmt.value)
            name = stmt.target.name if isinstance(stmt.target, ast.Variable) else stmt.target.name
            if name.endswith("%"):
                val_type = BasicType.I32
            if name not in self.symbols:
                self.symbols[name] = SymbolInfo(name=name, type=val_type)
            else:
                # Existing variable — may widen type
                existing = self.symbols[name]
                if existing.type == BasicType.I32 and val_type == BasicType.F32:
                    existing.type = BasicType.F32

        elif isinstance(stmt, ast.PrintStatement):
            for item in stmt.items:
                self._infer_type(item)

        elif isinstance(stmt, ast.InputStatement):
            for i, var in enumerate(stmt.variables):
                name = var.name
                is_arr = stmt.is_array[i] if i < len(stmt.is_array) else False
                if name not in self.symbols:
                    typ = BasicType.I32 if name.endswith("%") else BasicType.F32
                    self.symbols[name] = SymbolInfo(name=name, type=typ, is_array=is_arr)
                elif is_arr:
                    self.symbols[name].is_array = True
                self.input_vars.append(name)

        elif isinstance(stmt, ast.IfStatement):
            self._infer_type(stmt.condition)
            for s in stmt.then_body:
                self._analyze_stmt(s)
            for s in stmt.else_body:
                self._analyze_stmt(s)

        elif isinstance(stmt, ast.ForStatement):
            name = stmt.var.name
            start_type = self._infer_type(stmt.start)
            end_type = self._infer_type(stmt.end)
            if stmt.step:
                self._infer_type(stmt.step)
            # For loop var type: i32 if start and end are both int, else f32
            var_type = BasicType.I32 if (start_type == BasicType.I32 and end_type == BasicType.I32) else BasicType.F32
            if name.endswith("%"):
                var_type = BasicType.I32
            self.symbols[name] = SymbolInfo(name=name, type=var_type)
            for s in stmt.body:
                self._analyze_stmt(s)

        elif isinstance(stmt, ast.WhileStatement):
            self._infer_type(stmt.condition)
            for s in stmt.body:
                self._analyze_stmt(s)

        elif isinstance(stmt, ast.DimStatement):
            name = stmt.name
            typ = BasicType.I32 if name.endswith("%") else BasicType.F32
            size = None
            if stmt.sizes and isinstance(stmt.sizes[0], ast.NumberLiteral):
                size = int(stmt.sizes[0].value)
            self.symbols[name] = SymbolInfo(name=name, type=typ, is_array=True, array_size=size)

        elif isinstance(stmt, ast.TileStatement):
            pass  # Tile sizes stored in AST, extracted by backend

        elif isinstance(stmt, ast.MmaStatement):
            # Register accumulator variable
            if stmt.acc_var not in self.symbols:
                self.symbols[stmt.acc_var] = SymbolInfo(
                    name=stmt.acc_var, type=BasicType.F32, is_array=True)
            # Register matrix names
            for name in (stmt.a_access.name, stmt.b_access.name):
                if name not in self.symbols:
                    self.symbols[name] = SymbolInfo(
                        name=name, type=BasicType.F32, is_array=True)

        elif isinstance(stmt, ast.TileStoreStatement):
            name = stmt.target.name
            if name not in self.symbols:
                self.symbols[name] = SymbolInfo(
                    name=name, type=BasicType.F32, is_array=True)

        elif isinstance(stmt, ast.OutputStatement):
            for var in stmt.variables:
                name = var.name
                self.output_vars.append(name)

        elif isinstance(stmt, ast.ReadStatement):
            for var in stmt.variables:
                name = var.name
                if name not in self.symbols:
                    typ = BasicType.I32 if name.endswith("%") else BasicType.F32
                    self.symbols[name] = SymbolInfo(name=name, type=typ)

    def _infer_type(self, expr: ast.Expression) -> BasicType:
        if isinstance(expr, ast.NumberLiteral):
            return BasicType.I32 if isinstance(expr.value, int) else BasicType.F32

        if isinstance(expr, ast.StringLiteral):
            return BasicType.STRING

        if isinstance(expr, ast.Variable):
            name = expr.name
            if name == "BID":
                return BasicType.I32
            if name in self.symbols:
                return self.symbols[name].type
            # Infer from suffix
            typ = BasicType.I32 if name.endswith("%") else BasicType.F32
            self.symbols[name] = SymbolInfo(name=name, type=typ)
            return typ

        if isinstance(expr, ast.ArrayAccess):
            self._infer_type(expr.index)
            name = expr.name
            if name in self.symbols:
                return self.symbols[name].type
            return BasicType.F32

        if isinstance(expr, ast.UnaryOp):
            if expr.op == "NOT":
                self._infer_type(expr.operand)
                return BasicType.I1
            return self._infer_type(expr.operand)

        if isinstance(expr, ast.BinaryOp):
            lt = self._infer_type(expr.left)
            rt = self._infer_type(expr.right)
            if expr.op in ("=", "<>", "<", ">", "<=", ">="):
                return BasicType.I1
            if expr.op in ("AND", "OR"):
                return BasicType.I1
            # Arithmetic: promote to f32 if either operand is f32
            if lt == BasicType.F32 or rt == BasicType.F32:
                return BasicType.F32
            if expr.op in ("/", "^"):
                return BasicType.F32
            return BasicType.I32

        if isinstance(expr, ast.FunctionCall):
            self._infer_type(expr.arg)
            if expr.name == "INT":
                return BasicType.I32
            if expr.name == "SGN":
                return BasicType.I32
            return BasicType.F32

        return BasicType.F32

    def _eliminate_gotos(self, program: ast.Program) -> list[ast.Statement]:
        """Simple GOTO elimination: convert forward GOTOs to if/skip patterns.
        For complex cases, leave them as-is (codegen will use a state machine)."""
        # For now, just pass through — a full implementation would restructure
        # the control flow. The codegen handles GOTOs via a state-machine pattern.
        return program.statements


def analyze(program: ast.Program) -> AnalyzedProgram:
    """Analyze a parsed BASIC program."""
    return Analyzer().analyze(program)
