"""Bytecode backend: AST → cuTile bytecode."""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path

from .lexer import lex
from .parser import parse
from .analyzer import analyze

from cuda.tile._bytecode import (
    write_bytecode,
    BytecodeVersion,
    EntryHints,
    encode_AbsFOp,
    encode_AddFOp,
    encode_AddIOp,
    encode_AndIOp,
    encode_CmpFOp,
    encode_CmpIOp,
    encode_ConstantOp,
    encode_ContinueOp,
    encode_CosOp,
    encode_DivFOp,
    encode_DivIOp,
    encode_ExpOp,
    encode_ForOp,
    encode_TanOp,
    encode_FToIOp,
    encode_GetTileBlockIdOp,
    encode_IfOp,
    encode_IToFOp,
    encode_LoadViewTkoOp,
    encode_LogOp,
    encode_MakePartitionViewOp,
    encode_MakeTokenOp,
    encode_MakeTensorViewOp,
    encode_MmaFOp,
    encode_MulFOp,
    encode_MulIOp,
    encode_NegFOp,
    encode_OrIOp,
    encode_PowOp,
    encode_PrintTkoOp,
    encode_RemFOp,
    encode_RemIOp,
    encode_ReturnOp,
    encode_SelectOp,
    encode_SinOp,
    encode_SqrtOp,
    encode_StoreViewTkoOp,
    encode_SubFOp,
    encode_SubIOp,
    encode_XOrIOp,
    encode_YieldOp,
)
from cuda.tile._bytecode.code_builder import CodeBuilder, Value
from cuda.tile._bytecode.debug_info import DebugAttrId
from cuda.tile._bytecode.encodings import (
    ComparisonOrdering,
    ComparisonPredicate,
    IntegerOverflow,
    MemoryOrderingSemantics,
    RoundingMode,
    Signedness,
)
from cuda.tile._bytecode.type import PaddingValue, SimpleType, TypeTable

from . import ast_nodes as ast
from .analyzer import AnalyzedProgram, BasicType


class BytecodeBackendError(Exception):
    pass


class BytecodeBackend:
    """Compile an AnalyzedProgram directly to cuTile bytecode."""

    def __init__(self, analyzed: AnalyzedProgram, gpu_arch: str = "sm_120",
                 array_size: int | None = None, num_ctas: int | None = None):
        self.analyzed = analyzed
        self.symbols = analyzed.symbols
        self.gpu_arch = gpu_arch
        self.array_size = array_size if array_size is not None else self._infer_array_size(analyzed)
        self.num_ctas = num_ctas
        self.data_index = 0
        self._returned = False
        self._array_kernel_meta: dict | None = None

        # Populated during generate()
        self.tt: TypeTable | None = None
        self.builder: CodeBuilder | None = None
        self.var_map: dict[str, Value] = {}

        # Type IDs (set in _init_types)
        self.i32_t = None
        self.f32_t = None
        self.i1_t = None

        # Compositional codegen state (built up as statements are lowered)
        self._views: dict[str, Value] = {}
        self._tensor_views: dict[str, Value] = {}
        self._array_dims: dict[str, list[int]] = {}
        self._tile_types: dict[str, object] = {}
        self._token: Value | None = None
        self._token_type = None
        self._var_ir_types: dict[str, object] = {}
        self._param_arrays: list[str] = []

    @staticmethod
    def _infer_array_size(analyzed: AnalyzedProgram) -> int | None:
        """Infer array_size from DIM-declared 1-D arrays in the analyzed program."""
        for info in analyzed.symbols.values():
            if info.is_array and info.array_size is not None:
                return info.array_size
        return None

    def _entry_hints(self) -> dict:
        """Build hints dict for writer.function based on num_ctas."""
        if self.num_ctas is None:
            return {}
        return {self.gpu_arch: EntryHints(num_cta_in_cga=self.num_ctas)}

    def _init_types(self, tt: TypeTable):
        """Create the tile type IDs we use."""
        self.tt = tt
        i32_s = tt.simple(SimpleType.I32)
        f32_s = tt.simple(SimpleType.F32)
        i1_s = tt.simple(SimpleType.I1)
        self.i32_t = tt.tile(i32_s, [])
        self.f32_t = tt.tile(f32_s, [])
        self.i1_t = tt.tile(i1_s, [])
        self.token_t = tt.simple(SimpleType.Token)

    def _type_id(self, typ: BasicType):
        """Map BasicType to a TypeId."""
        return {
            BasicType.I32: self.i32_t,
            BasicType.F32: self.f32_t,
            BasicType.I1: self.i1_t,
        }[typ]

    def _type_of_expr(self, expr: ast.Expression) -> BasicType:
        """Infer the type of an expression (mirrors TextualBackend._type_of_expr)."""
        if isinstance(expr, ast.NumberLiteral):
            return BasicType.I32 if isinstance(expr.value, int) else BasicType.F32
        if isinstance(expr, ast.StringLiteral):
            return BasicType.STRING
        if isinstance(expr, ast.Variable):
            if expr.name == "BID":
                return BasicType.I32
            info = self.symbols.get(expr.name)
            return info.type if info else BasicType.F32
        if isinstance(expr, ast.ArrayAccess):
            info = self.symbols.get(expr.name)
            return info.type if info else BasicType.F32
        if isinstance(expr, ast.UnaryOp):
            if expr.op == "NOT":
                return BasicType.I1
            return self._type_of_expr(expr.operand)
        if isinstance(expr, ast.BinaryOp):
            if expr.op in ("=", "<>", "<", ">", "<=", ">=", "AND", "OR"):
                return BasicType.I1
            lt = self._type_of_expr(expr.left)
            rt = self._type_of_expr(expr.right)
            if lt == BasicType.F32 or rt == BasicType.F32:
                return BasicType.F32
            if expr.op in ("/", "^"):
                return BasicType.F32
            return BasicType.I32
        if isinstance(expr, ast.FunctionCall):
            if expr.name in ("INT", "SGN"):
                return BasicType.I32
            return BasicType.F32
        return BasicType.F32

    def _find_modified_vars(self, stmts: list[ast.Statement]) -> list[str]:
        """Find BASIC variable names assigned in a block of statements."""
        modified: list[str] = []
        seen: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, ast.LetStatement):
                name = stmt.target.name if isinstance(stmt.target, ast.Variable) else None
                if name and name not in seen:
                    seen.add(name)
                    modified.append(name)
            elif isinstance(stmt, ast.ReadStatement):
                for var in stmt.variables:
                    if var.name not in seen:
                        seen.add(var.name)
                        modified.append(var.name)
            elif isinstance(stmt, ast.MmaStatement):
                if stmt.acc_var not in seen:
                    seen.add(stmt.acc_var)
                    modified.append(stmt.acc_var)
            elif isinstance(stmt, ast.IfStatement):
                for name in self._find_modified_vars(stmt.then_body):
                    if name not in seen:
                        seen.add(name)
                        modified.append(name)
                for name in self._find_modified_vars(stmt.else_body):
                    if name not in seen:
                        seen.add(name)
                        modified.append(name)
            elif isinstance(stmt, ast.ForStatement):
                for name in self._find_modified_vars(stmt.body):
                    if name not in seen:
                        seen.add(name)
                        modified.append(name)
            elif isinstance(stmt, ast.WhileStatement):
                for name in self._find_modified_vars(stmt.body):
                    if name not in seen:
                        seen.add(name)
                        modified.append(name)
        return modified

    # ---- Constants ----

    def _const_i32(self, val: int) -> Value:
        return encode_ConstantOp(self.builder, self.i32_t, struct.pack("<i", val))

    def _const_f32(self, val: float) -> Value:
        return encode_ConstantOp(self.builder, self.f32_t, struct.pack("<f", val))

    def _const_i1(self, val: bool) -> Value:
        return encode_ConstantOp(self.builder, self.i1_t, struct.pack("<?", val))

    def _const(self, value, typ: BasicType) -> Value:
        if typ == BasicType.F32:
            return self._const_f32(float(value))
        elif typ == BasicType.I32:
            return self._const_i32(int(value))
        elif typ == BasicType.I1:
            return self._const_i1(bool(value))
        raise BytecodeBackendError(f"Cannot create constant of type {typ}")

    # ---- Cast helpers ----

    def _cast_to_f32(self, val: Value) -> Value:
        return encode_IToFOp(
            self.builder, self.f32_t, val,
            Signedness.Signed, RoundingMode.NEAREST_EVEN,
        )

    def _cast_to_i32(self, val: Value) -> Value:
        return encode_FToIOp(
            self.builder, self.i32_t, val,
            Signedness.Signed, RoundingMode.NEAREST_INT_TO_ZERO,
        )

    # ---- Arithmetic helpers ----

    def _addi(self, lhs: Value, rhs: Value) -> Value:
        return encode_AddIOp(self.builder, self.i32_t, lhs, rhs, IntegerOverflow.NONE)

    def _addf(self, lhs: Value, rhs: Value) -> Value:
        return encode_AddFOp(self.builder, self.f32_t, lhs, rhs, RoundingMode.NEAREST_EVEN, False)

    def _subi(self, lhs: Value, rhs: Value) -> Value:
        return encode_SubIOp(self.builder, self.i32_t, lhs, rhs, IntegerOverflow.NONE)

    def _subf(self, lhs: Value, rhs: Value) -> Value:
        return encode_SubFOp(self.builder, self.f32_t, lhs, rhs, RoundingMode.NEAREST_EVEN, False)

    def _muli(self, lhs: Value, rhs: Value) -> Value:
        return encode_MulIOp(self.builder, self.i32_t, lhs, rhs, IntegerOverflow.NONE)

    def _mulf(self, lhs: Value, rhs: Value) -> Value:
        return encode_MulFOp(self.builder, self.f32_t, lhs, rhs, RoundingMode.NEAREST_EVEN, False)

    def _divi(self, lhs: Value, rhs: Value) -> Value:
        return encode_DivIOp(self.builder, self.i32_t, lhs, rhs, Signedness.Signed, RoundingMode.ZERO)

    def _divf(self, lhs: Value, rhs: Value) -> Value:
        return encode_DivFOp(self.builder, self.f32_t, lhs, rhs, RoundingMode.NEAREST_EVEN, False)

    # ---- Expression codegen ----

    def _gen_expr(self, expr: ast.Expression) -> Value:
        if isinstance(expr, ast.NumberLiteral):
            if isinstance(expr.value, int):
                return self._const_i32(expr.value)
            else:
                return self._const_f32(expr.value)

        if isinstance(expr, ast.StringLiteral):
            raise BytecodeBackendError("String expressions not supported")

        if isinstance(expr, ast.Variable):
            if expr.name in self.var_map:
                return self.var_map[expr.name]
            info = self.symbols.get(expr.name)
            typ = info.type if info else BasicType.F32
            val = self._const(0, typ)
            self.var_map[expr.name] = val
            return val

        if isinstance(expr, ast.ArrayAccess):
            name = expr.name
            if name in self._views:
                indices = [self._ensure_i32(self._gen_expr(expr.index), expr.index)]
                if expr.index2 is not None:
                    indices.append(self._ensure_i32(self._gen_expr(expr.index2), expr.index2))
                tile_type = self._tile_types[name]
                tile_val, new_tok = encode_LoadViewTkoOp(
                    self.builder, tile_type, self._token_type, self._views[name],
                    indices, self._token,
                    MemoryOrderingSemantics.WEAK, None, None,
                )
                self._token = new_tok
                return tile_val
            raise BytecodeBackendError(f"Array {name} has no view")

        if isinstance(expr, ast.UnaryOp):
            return self._gen_unary(expr)

        if isinstance(expr, ast.BinaryOp):
            return self._gen_binop(expr)

        if isinstance(expr, ast.FunctionCall):
            return self._gen_function(expr)

        raise BytecodeBackendError(f"Unknown expression type: {type(expr).__name__}")

    def _gen_unary(self, expr: ast.UnaryOp) -> Value:
        operand = self._gen_expr(expr.operand)
        if expr.op == "-":
            typ = self._type_of_expr(expr.operand)
            if typ == BasicType.F32:
                return encode_NegFOp(self.builder, self.f32_t, operand)
            else:
                zero = self._const_i32(0)
                return self._subi(zero, operand)
        elif expr.op == "NOT":
            ones = self._const_i1(True)
            return encode_XOrIOp(self.builder, self.i1_t, operand, ones)
        raise BytecodeBackendError(f"Unknown unary op: {expr.op}")

    def _expr_tile_shape(self, expr: ast.Expression) -> list[int] | None:
        """Return the tile shape if the expression produces a tile-valued result."""
        if isinstance(expr, ast.ArrayAccess):
            name = expr.name
            info = self.symbols.get(name)
            if info and info.tile_shape:
                return info.tile_shape
            return None
        if isinstance(expr, ast.BinaryOp):
            ls = self._expr_tile_shape(expr.left)
            if ls is not None:
                return ls
            return self._expr_tile_shape(expr.right)
        return None

    def _gen_binop(self, expr: ast.BinaryOp) -> Value:
        left = self._gen_expr(expr.left)
        right = self._gen_expr(expr.right)
        lt = self._type_of_expr(expr.left)
        rt = self._type_of_expr(expr.right)

        tile_shape = self._expr_tile_shape(expr)

        # Comparisons
        if expr.op in ("=", "<>", "<", ">", "<=", ">="):
            return self._gen_comparison(expr.op, left, right, lt, rt)

        # Logical
        if expr.op == "AND":
            return encode_AndIOp(self.builder, self.i1_t, left, right)
        if expr.op == "OR":
            return encode_OrIOp(self.builder, self.i1_t, left, right)

        # Type promotion for arithmetic
        is_float = (lt == BasicType.F32 or rt == BasicType.F32)
        if is_float and lt == BasicType.I32:
            left = self._cast_to_f32(left)
        if is_float and rt == BasicType.I32:
            right = self._cast_to_f32(right)

        # Division and power are always float
        if expr.op in ("/", "^") and not is_float:
            left = self._cast_to_f32(left)
            right = self._cast_to_f32(right)
            is_float = True

        if tile_shape is not None:
            f32_s = self.tt.simple(SimpleType.F32)
            i32_s = self.tt.simple(SimpleType.I32)
            result_type = self.tt.tile(f32_s, tile_shape) if is_float else self.tt.tile(i32_s, tile_shape)
        else:
            result_type = self.f32_t if is_float else self.i32_t

        if expr.op == "+":
            if is_float:
                return encode_AddFOp(self.builder, result_type, left, right, RoundingMode.NEAREST_EVEN, False)
            else:
                return encode_AddIOp(self.builder, result_type, left, right, IntegerOverflow.NONE)
        elif expr.op == "-":
            if is_float:
                return encode_SubFOp(self.builder, result_type, left, right, RoundingMode.NEAREST_EVEN, False)
            else:
                return encode_SubIOp(self.builder, result_type, left, right, IntegerOverflow.NONE)
        elif expr.op == "*":
            if is_float:
                return encode_MulFOp(self.builder, result_type, left, right, RoundingMode.NEAREST_EVEN, False)
            else:
                return encode_MulIOp(self.builder, result_type, left, right, IntegerOverflow.NONE)
        elif expr.op == "/":
            if is_float:
                return encode_DivFOp(self.builder, result_type, left, right, RoundingMode.NEAREST_EVEN, False)
            else:
                return encode_DivIOp(self.builder, result_type, left, right, Signedness.Signed, RoundingMode.ZERO)
        elif expr.op == "MOD":
            if is_float:
                return encode_RemFOp(self.builder, result_type, left, right)
            else:
                return encode_RemIOp(self.builder, result_type, left, right, Signedness.Signed)
        elif expr.op == "^":
            return encode_PowOp(self.builder, result_type, left, right)

        raise BytecodeBackendError(f"Unknown binary op: {expr.op}")

    def _gen_comparison(self, op: str, left: Value, right: Value,
                        lt: BasicType, rt: BasicType) -> Value:
        is_float = (lt == BasicType.F32 or rt == BasicType.F32)
        if is_float and lt == BasicType.I32:
            left = self._cast_to_f32(left)
        if is_float and rt == BasicType.I32:
            right = self._cast_to_f32(right)

        pred_map = {
            "=": ComparisonPredicate.EQUAL,
            "<>": ComparisonPredicate.NOT_EQUAL,
            "<": ComparisonPredicate.LESS_THAN,
            ">": ComparisonPredicate.GREATER_THAN,
            "<=": ComparisonPredicate.LESS_THAN_OR_EQUAL,
            ">=": ComparisonPredicate.GREATER_THAN_OR_EQUAL,
        }
        pred = pred_map[op]

        if is_float:
            return encode_CmpFOp(
                self.builder, self.i1_t, left, right,
                pred, ComparisonOrdering.ORDERED,
            )
        else:
            return encode_CmpIOp(
                self.builder, self.i1_t, left, right,
                pred, Signedness.Signed,
            )

    def _gen_function(self, expr: ast.FunctionCall) -> Value:
        arg = self._gen_expr(expr.arg)
        arg_type = self._type_of_expr(expr.arg)

        # Cast to f32 for math functions (except SGN)
        if arg_type == BasicType.I32 and expr.name not in ("SGN",):
            arg = self._cast_to_f32(arg)

        if expr.name == "ABS":
            return encode_AbsFOp(self.builder, self.f32_t, arg)
        elif expr.name == "SQR":
            return encode_SqrtOp(self.builder, self.f32_t, arg, RoundingMode.NEAREST_EVEN, False)
        elif expr.name == "SIN":
            return encode_SinOp(self.builder, self.f32_t, arg)
        elif expr.name == "COS":
            return encode_CosOp(self.builder, self.f32_t, arg)
        elif expr.name == "TAN":
            return encode_TanOp(self.builder, self.f32_t, arg)
        elif expr.name == "EXP":
            return encode_ExpOp(self.builder, self.f32_t, arg, RoundingMode.FULL)
        elif expr.name == "LOG":
            return encode_LogOp(self.builder, self.f32_t, arg)
        elif expr.name == "INT":
            if arg_type == BasicType.I32:
                return arg
            return self._cast_to_i32(arg)
        elif expr.name == "SGN":
            zero = self._const(0, arg_type)
            neg_one = self._const_i32(-1)
            one = self._const_i32(1)
            zero_i = self._const_i32(0)
            if arg_type == BasicType.F32:
                lt_val = encode_CmpFOp(
                    self.builder, self.i1_t, arg, zero,
                    ComparisonPredicate.LESS_THAN, ComparisonOrdering.ORDERED,
                )
                gt_val = encode_CmpFOp(
                    self.builder, self.i1_t, arg, zero,
                    ComparisonPredicate.GREATER_THAN, ComparisonOrdering.ORDERED,
                )
            else:
                lt_val = encode_CmpIOp(
                    self.builder, self.i1_t, arg, zero,
                    ComparisonPredicate.LESS_THAN, Signedness.Signed,
                )
                gt_val = encode_CmpIOp(
                    self.builder, self.i1_t, arg, zero,
                    ComparisonPredicate.GREATER_THAN, Signedness.Signed,
                )
            sel1 = encode_SelectOp(self.builder, self.i32_t, gt_val, one, zero_i)
            return encode_SelectOp(self.builder, self.i32_t, lt_val, neg_one, sel1)
        raise BytecodeBackendError(f"Unknown function: {expr.name}")

    # ---- Statement codegen ----

    def _gen_stmt(self, stmt: ast.Statement):
        if isinstance(stmt, ast.RemStatement):
            return
        if isinstance(stmt, (ast.EndStatement, ast.StopStatement)):
            encode_ReturnOp(self.builder, operands=[])
            self._returned = True
            return
        if isinstance(stmt, ast.LetStatement):
            self._gen_let(stmt)
        elif isinstance(stmt, ast.PrintStatement):
            self._gen_print(stmt)
        elif isinstance(stmt, ast.IfStatement):
            self._gen_if(stmt)
        elif isinstance(stmt, ast.ForStatement):
            self._gen_for(stmt)
        elif isinstance(stmt, ast.WhileStatement):
            self._gen_while(stmt)
        elif isinstance(stmt, ast.ReadStatement):
            self._gen_read(stmt)
        elif isinstance(stmt, ast.DimStatement):
            self._gen_dim(stmt)
        elif isinstance(stmt, ast.TileStatement):
            self._gen_tile(stmt)
        elif isinstance(stmt, ast.MmaStatement):
            self._gen_mma(stmt)
        elif isinstance(stmt, ast.TileStoreStatement):
            self._gen_tile_store(stmt)
        elif isinstance(stmt, ast.InputStatement):
            pass
        elif isinstance(stmt, ast.DataStatement):
            pass
        elif isinstance(stmt, ast.OutputStatement):
            pass
        elif isinstance(stmt, (ast.GotoStatement, ast.GosubStatement, ast.ReturnStatement)):
            pass

    def _gen_let(self, stmt: ast.LetStatement):
        if isinstance(stmt.target, ast.ArrayAccess) and stmt.target.name in self._views:
            self._gen_let_array(stmt)
            return

        if isinstance(stmt.target, ast.Variable):
            name = stmt.target.name
            info = self.symbols.get(name)

            if (info and info.tile_shape
                    and name not in self._param_values
                    and isinstance(stmt.value, ast.NumberLiteral)):
                f32_s = self.tt.simple(SimpleType.F32)
                tile_t = self.tt.tile(f32_s, info.tile_shape)
                val = encode_ConstantOp(
                    self.builder, tile_t,
                    struct.pack("<f", float(stmt.value.value)),
                )
                self.var_map[name] = val
                self._var_ir_types[name] = tile_t
                return

            val = self._gen_expr(stmt.value)
            expr_type = self._type_of_expr(stmt.value)
            if info and info.type == BasicType.F32 and expr_type == BasicType.I32:
                val = self._cast_to_f32(val)
            elif info and info.type == BasicType.I32 and expr_type == BasicType.F32:
                val = self._cast_to_i32(val)
            self.var_map[name] = val

    def _gen_let_array(self, stmt: ast.LetStatement):
        """Lower LET C(...) = expr as a tiled store."""
        target = stmt.target
        self._ensure_partition_view(target.name)

        indices = [self._ensure_i32(self._gen_expr(target.index), target.index)]
        if target.index2 is not None:
            indices.append(self._ensure_i32(self._gen_expr(target.index2), target.index2))

        result_tile = self._gen_expr(stmt.value)

        encode_StoreViewTkoOp(
            self.builder, self._token_type, result_tile,
            self._views[target.name], indices, self._token,
            MemoryOrderingSemantics.WEAK, None, None,
        )

    def _gen_print(self, stmt: ast.PrintStatement):
        if not stmt.items:
            encode_PrintTkoOp(self.builder, self.token_t, args=[], token=None, str="\n")
            return

        fmt_parts: list[str] = []
        operands: list[Value] = []

        for item in stmt.items:
            if isinstance(item, ast.StringLiteral):
                fmt_parts.append(item.value)
            else:
                val = self._gen_expr(item)
                expr_type = self._type_of_expr(item)
                if expr_type in (BasicType.I32, BasicType.I1):
                    fmt_parts.append("%d")
                else:
                    fmt_parts.append("%f")
                operands.append(val)

        fmt = "".join(fmt_parts)
        if stmt.newline:
            fmt += "\n"

        encode_PrintTkoOp(self.builder, self.token_t, args=operands, token=None, str=fmt)

    def _gen_if(self, stmt: ast.IfStatement):
        cond = self._gen_expr(stmt.condition)

        # Find variables modified inside the if/else
        then_modified = self._find_modified_vars(stmt.then_body)
        else_modified = self._find_modified_vars(stmt.else_body)
        all_modified_set: set[str] = set()
        all_modified: list[str] = []
        for name in then_modified + else_modified:
            if name not in all_modified_set and name in self.var_map:
                all_modified_set.add(name)
                all_modified.append(name)

        result_types = [self._type_id(self.symbols[n].type) for n in all_modified] if all_modified else []

        nbb = encode_IfOp(self.builder, result_types=result_types, condition=cond)

        # Then block
        saved = dict(self.var_map)
        with nbb.new_block([]) as _then_args:
            for s in stmt.then_body:
                self._gen_stmt(s)
            yield_vals = [self.var_map.get(n, saved.get(n)) for n in all_modified]
            encode_YieldOp(self.builder, operands=yield_vals)
        then_map = dict(self.var_map)

        # Else block
        self.var_map = dict(saved)
        with nbb.new_block([]) as _else_args:
            if stmt.else_body:
                for s in stmt.else_body:
                    self._gen_stmt(s)
            yield_vals = [self.var_map.get(n, saved.get(n)) for n in all_modified]
            encode_YieldOp(self.builder, operands=yield_vals)

        # Restore and update with results
        self.var_map = dict(saved)
        results = nbb.done()
        for i, name in enumerate(all_modified):
            self.var_map[name] = results[i]

    def _body_has_token_ops(self, stmts: list[ast.Statement]) -> bool:
        """Check if a body contains operations that thread the token."""
        for stmt in stmts:
            if isinstance(stmt, (ast.MmaStatement, ast.TileStoreStatement)):
                return True
            if isinstance(stmt, ast.LetStatement):
                if isinstance(stmt.target, ast.ArrayAccess) and stmt.target.name in self._views:
                    return True
            if isinstance(stmt, ast.ForStatement):
                if self._body_has_token_ops(stmt.body):
                    return True
            if isinstance(stmt, ast.IfStatement):
                if self._body_has_token_ops(stmt.then_body) or self._body_has_token_ops(stmt.else_body):
                    return True
        return False

    def _var_type_id(self, name: str):
        """Get the actual IR type ID for a variable, preferring _var_ir_types."""
        if name in self._var_ir_types:
            return self._var_ir_types[name]
        info = self.symbols.get(name)
        if info:
            return self._type_id(info.type)
        return self.f32_t

    def _gen_for(self, stmt: ast.ForStatement):
        lb = self._gen_expr(stmt.start)
        end_val = self._gen_expr(stmt.end)
        var_type = self._type_of_expr(stmt.start)
        type_id = self._type_id(var_type)

        if stmt.step:
            step = self._gen_expr(stmt.step)
        else:
            step = self._const(1, var_type)

        if var_type == BasicType.F32:
            ub = self._addf(end_val, step)
        else:
            ub = self._addi(end_val, step)

        modified = self._find_modified_vars(stmt.body)
        iter_vars = [(name, self.var_map[name]) for name in modified if name in self.var_map]

        iter_type_ids = [self._var_type_id(n) for n, _ in iter_vars]
        init_values = [v for _, v in iter_vars]

        carry_token = (self._token is not None
                       and self._body_has_token_ops(stmt.body))

        if carry_token:
            iter_type_ids.append(self._token_type)
            init_values.append(self._token)

        nbb = encode_ForOp(
            self.builder,
            result_types=iter_type_ids,
            lowerBound=lb,
            upperBound=ub,
            step=step,
            initValues=init_values,
            unsignedCmp=False,
        )

        block_arg_types = [type_id] + iter_type_ids

        saved = dict(self.var_map)
        saved_token = self._token
        with nbb.new_block(block_arg_types) as body_args:
            self.var_map[stmt.var.name] = body_args[0]
            for i, (name, _) in enumerate(iter_vars):
                self.var_map[name] = body_args[1 + i]
            if carry_token:
                self._token = body_args[1 + len(iter_vars)]

            for s in stmt.body:
                self._gen_stmt(s)

            yield_vals = [self.var_map[name] for name, _ in iter_vars]
            if carry_token:
                yield_vals.append(self._token)
            encode_ContinueOp(self.builder, operands=yield_vals)

        results = nbb.done()
        self.var_map = dict(saved)
        for i, (name, _) in enumerate(iter_vars):
            self.var_map[name] = results[i]
        if carry_token:
            self._token = results[len(iter_vars)]
        else:
            self._token = saved_token

    def _gen_while(self, stmt: ast.WhileStatement):
        lb = self._const_i32(0)
        ub = self._const_i32(1000000)
        step = self._const_i32(1)

        modified = self._find_modified_vars(stmt.body)
        iter_vars = [(name, self.var_map[name]) for name in modified if name in self.var_map]
        iter_type_ids = [self._type_id(self.symbols[n].type) for n, _ in iter_vars]
        init_values = [v for _, v in iter_vars]

        nbb = encode_ForOp(
            self.builder,
            result_types=iter_type_ids,
            lowerBound=lb,
            upperBound=ub,
            step=step,
            initValues=init_values,
            unsignedCmp=False,
        )

        block_arg_types = [self.i32_t] + iter_type_ids

        saved = dict(self.var_map)
        with nbb.new_block(block_arg_types) as body_args:
            for i, (name, _) in enumerate(iter_vars):
                self.var_map[name] = body_args[1 + i]

            cond = self._gen_expr(stmt.condition)

            if_nbb = encode_IfOp(self.builder, result_types=list(iter_type_ids), condition=cond)

            pre_body = dict(self.var_map)
            with if_nbb.new_block([]) as _then:
                for s in stmt.body:
                    self._gen_stmt(s)
                yield_vals = [self.var_map[name] for name, _ in iter_vars]
                encode_YieldOp(self.builder, operands=yield_vals)

            self.var_map = dict(pre_body)
            with if_nbb.new_block([]) as _else:
                yield_vals = [self.var_map[name] for name, _ in iter_vars]
                encode_YieldOp(self.builder, operands=yield_vals)

            if_results = if_nbb.done()
            self.var_map = dict(pre_body)
            for i, (name, _) in enumerate(iter_vars):
                self.var_map[name] = if_results[i]

            encode_ContinueOp(self.builder, operands=[self.var_map[name] for name, _ in iter_vars])

        results = nbb.done()
        self.var_map = dict(saved)
        for i, (name, _) in enumerate(iter_vars):
            self.var_map[name] = results[i]

    def _gen_read(self, stmt: ast.ReadStatement):
        for var in stmt.variables:
            if self.data_index < len(self.analyzed.data_values):
                val = self.analyzed.data_values[self.data_index]
                self.data_index += 1
                info = self.symbols.get(var.name)
                typ = info.type if info else BasicType.F32
                self.var_map[var.name] = self._const(val, typ)

    # ---- DIM / TILE / MMA / STORE lowering ----

    def _gen_dim(self, stmt: ast.DimStatement):
        """Lower a DIM statement: record sizes and create tensor views for parameter arrays."""
        name = stmt.name
        sizes = [int(s.value) for s in stmt.sizes if isinstance(s, ast.NumberLiteral)]
        self._array_dims[name] = sizes

        if name not in self._param_values:
            return

        tt = self.tt
        f32_s = tt.simple(SimpleType.F32)

        if len(sizes) == 1:
            total = sizes[0]
            tv_t = tt.tensor_view(f32_s, [total], [1])
        elif len(sizes) == 2:
            tv_t = tt.tensor_view(f32_s, sizes, [sizes[1], 1])
        else:
            return

        tv_val = encode_MakeTensorViewOp(
            self.builder, tv_t, self._param_values[name], [], []
        )
        self._tensor_views[name] = tv_val
        self._tile_types[f"__tv_type_{name}__"] = tv_t

    def _gen_tile(self, stmt: ast.TileStatement):
        """Lower a TILE statement: create partition view for the named variable."""
        name = stmt.name
        if name in self._views:
            raise BytecodeBackendError(
                f"Tile shape for '{name}' already declared. "
                f"Cannot redeclare with a second TILE statement."
            )

        if name in self._param_values:
            self._ensure_partition_view(name)

    def _ensure_partition_view(self, name: str):
        """Create a partition view for an array if one doesn't exist yet.
        Uses the tile_shape declared in the symbol table."""
        if name in self._views:
            return
        info = self.symbols.get(name)
        if not info or not info.tile_shape:
            raise BytecodeBackendError(
                f"Array '{name}' has no declared tile shape. "
                f"Use 'TILE {name}(...)' to declare it."
            )
        part_shape = info.tile_shape

        tv_val = self._tensor_views.get(name)
        if tv_val is None:
            raise BytecodeBackendError(f"No tensor view for array '{name}'")

        tt = self.tt
        f32_s = tt.simple(SimpleType.F32)

        tv_t = self._tile_types.get(f"__tv_type_{name}__")
        if tv_t is None:
            dims = self._array_dims.get(name, part_shape)
            tv_t = tt.tensor_view(f32_s, dims, [dims[-1], 1] if len(dims) == 2 else [1])

        pv_t = tt.partition_view(
            part_shape, tv_t, list(range(len(part_shape))), PaddingValue.Zero
        )
        pv_val = encode_MakePartitionViewOp(self.builder, pv_t, tv_val)
        self._views[name] = pv_val

        tile_t = tt.tile(f32_s, part_shape)
        self._tile_types[name] = tile_t

    def _ensure_i32(self, val: Value, expr: ast.Expression) -> Value:
        """Cast to i32 if expression type is f32 (needed for tile indices)."""
        typ = self._type_of_expr(expr)
        if typ == BasicType.F32:
            return self._cast_to_i32(val)
        return val

    def _gen_mma(self, stmt: ast.MmaStatement):
        """Generate MmaFOp. Partition shapes come from each array's DIM TILE declaration."""
        if stmt.acc_var not in self.var_map:
            raise BytecodeBackendError(
                f"Accumulator '{stmt.acc_var}' used in MMA but not initialized. "
                f"Add 'LET {stmt.acc_var} = 0' before the loop."
            )

        self._ensure_partition_view(stmt.a_access.name)
        self._ensure_partition_view(stmt.b_access.name)

        tile_a_val = self._gen_expr(stmt.a_access)
        tile_b_val = self._gen_expr(stmt.b_access)
        acc = self.var_map[stmt.acc_var]

        acc_type = self._var_ir_types.get(stmt.acc_var)
        new_acc = encode_MmaFOp(
            self.builder, acc_type, tile_a_val, tile_b_val, acc,
        )
        self.var_map[stmt.acc_var] = new_acc

    def _gen_tile_store(self, stmt: ast.TileStoreStatement):
        """Generate StoreViewTkoOp. Partition shape from DIM TILE declaration."""
        self._ensure_partition_view(stmt.target.name)

        idx0 = self._ensure_i32(self._gen_expr(stmt.target.index), stmt.target.index)
        idx1 = self._ensure_i32(self._gen_expr(stmt.target.index2), stmt.target.index2)

        pv = self._views[stmt.target.name]
        tile_val = self.var_map[stmt.value_var]

        encode_StoreViewTkoOp(
            self.builder, self._token_type, tile_val,
            pv, [idx0, idx1], self._token,
            MemoryOrderingSemantics.WEAK, None, None,
        )

    # ---- Main entry points ----

    def _derive_param_arrays(self) -> list[str]:
        """Build deduplicated list of arrays that become function parameters."""
        param_arrays: list[str] = []
        seen: set[str] = set()
        for name in (self.analyzed.input_vars or []) + (self.analyzed.output_vars or []):
            info = self.symbols.get(name)
            if info and info.is_array and name not in seen:
                seen.add(name)
                param_arrays.append(name)
        return param_arrays

    def _compute_metadata(self):
        """Compute _array_kernel_meta from accumulated codegen state."""
        if not self._param_arrays:
            return

        input_arrays = [n for n in self.analyzed.input_vars
                        if self.symbols.get(n) and self.symbols[n].is_array]
        output_arrays = [n for n in (self.analyzed.output_vars or [])
                         if self.symbols.get(n) and self.symbols[n].is_array]

        meta: dict = {
            "all_arrays": self._param_arrays,
            "input_arrays": input_arrays,
            "output_arrays": output_arrays,
            "dims": {},
            "tile_shapes": {},
        }

        for name, dims in self._array_dims.items():
            meta["dims"][name] = dims
            info = self.symbols.get(name)
            if info and info.tile_shape:
                meta["tile_shapes"][name] = info.tile_shape

        # Compute grid_size from the first output array's dims and tile shape.
        for name in output_arrays:
            dims = self._array_dims.get(name)
            info = self.symbols.get(name)
            if not dims or not info or not info.tile_shape:
                continue
            tiles_per_dim = [
                (d + t - 1) // t
                for d, t in zip(dims, info.tile_shape)
            ]
            grid_size = 1
            for n in tiles_per_dim:
                grid_size *= n
            meta["grid_size"] = grid_size
            break

        self._array_kernel_meta = meta

    def generate(self, array_size: int | None = None) -> bytes:
        """Generate cuTile bytecode from the analyzed program."""
        if array_size is not None:
            self.array_size = array_size

        self._param_arrays = self._derive_param_arrays()

        buf = bytearray()

        with write_bytecode(1, buf, BytecodeVersion.V_13_2) as writer:
            tt = writer.type_table
            self._init_types(tt)

            f32_s = tt.simple(SimpleType.F32)
            ptr_f32 = tt.pointer(f32_s)
            tile_ptr_f32 = tt.tile(ptr_f32, [])
            self._token_type = tt.simple(SimpleType.Token)

            param_types = [tile_ptr_f32] * len(self._param_arrays)

            with writer.function(
                "main", param_types, [], True, self._entry_hints(), DebugAttrId(0)
            ) as fb:
                self.builder = fb.code_builder
                self.var_map = {}
                self.data_index = 0
                self._returned = False
                self._views = {}
                self._tensor_views = {}
                self._array_dims = {}
                self._tile_types = {}
                self._token = None
                self._var_ir_types = {}

                # Map function parameters to array names
                self._param_values = {
                    name: fb.parameters[i]
                    for i, name in enumerate(self._param_arrays)
                }

                if self._param_arrays:
                    bid_x, _, _ = encode_GetTileBlockIdOp(
                        self.builder, self.i32_t, self.i32_t, self.i32_t
                    )
                    self.var_map["BID"] = bid_x
                    self._token = encode_MakeTokenOp(
                        self.builder, self._token_type
                    )

                for stmt in self.analyzed.statements:
                    self._gen_stmt(stmt)

                if not self._returned:
                    encode_ReturnOp(self.builder, operands=[])

        self._compute_metadata()
        return bytes(buf)

    def compile_to_cubin(self, output_dir: str | None = None, array_size: int | None = None) -> str:
        """Generate bytecode, run tileiras, return path to .cubin."""
        bytecode = self.generate(array_size=array_size)

        output_dir = tempfile.mkdtemp(prefix="cutile_basic_bc_", dir=output_dir)
        bc_path = os.path.join(output_dir, "program.tilebc")
        cubin_path = os.path.join(output_dir, "program.cubin")

        with open(bc_path, "wb") as f:
            f.write(bytecode)

        tileiras = _find_tileiras()
        result = subprocess.run(
            [str(tileiras), f"--gpu-name={self.gpu_arch}", bc_path, "-o", cubin_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise BytecodeBackendError(
                f"tileiras failed (exit {result.returncode}):\n{result.stderr}"
            )

        return cubin_path


def _find_tileiras() -> Path:
    """Locate the tileiras binary."""
    import shutil

    found = shutil.which("tileiras")
    if found:
        return Path(found)
    p = Path("/usr/local/cuda/bin/tileiras")
    if p.is_file() and os.access(p, os.X_OK):
        return p
    try:
        import nvidia.cu13.bin as _nbin
        for pkg_dir in _nbin.__path__:
            p = Path(pkg_dir) / "tileiras"
            if p.is_file() and os.access(p, os.X_OK):
                return p
    except ImportError:
        pass
    raise BytecodeBackendError(
        "tileiras not found. Ensure CUDA toolkit is installed."
    )


class CompilationResult:
    """Result of compiling BASIC source to a .cubin file."""

    def __init__(self, cubin_path: str, meta: dict):
        self.cubin_path = cubin_path
        self.meta = meta

    def __repr__(self) -> str:
        return f"CompilationResult(cubin_path={self.cubin_path!r}, meta={self.meta!r})"


def compile_basic_to_cubin(
    source: str,
    *,
    gpu_arch: str | None = None,
    array_size: int | None = None,
    num_ctas: int | None = None,
) -> CompilationResult:
    """Compile BASIC source to a .cubin via the bytecode backend.

    Args:
        source: BASIC source code.
        gpu_arch: Target GPU architecture (e.g. ``"sm_120"``).
            ``None`` auto-detects from the current device.
        array_size: Total elements per array; ``None`` infers from DIM.
        num_ctas: CTAs-per-CGA optimisation hint; ``None`` disables.

    Returns:
        A :class:`CompilationResult` with ``cubin_path`` and kernel ``meta``.
    """
    from .gpu import detect_gpu_arch

    if gpu_arch is None:
        gpu_arch = detect_gpu_arch()

    tokens = lex(source)
    program = parse(tokens)
    analyzed = analyze(program)
    backend = BytecodeBackend(
        analyzed, gpu_arch=gpu_arch, array_size=array_size, num_ctas=num_ctas,
    )
    cubin_path = backend.compile_to_cubin()
    return CompilationResult(
        cubin_path=cubin_path, meta=backend._array_kernel_meta or {},
    )
