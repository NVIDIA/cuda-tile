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
    encode_FToIOp,
    encode_GetTileBlockIdOp,
    encode_IfOp,
    encode_IToFOp,
    encode_JoinTokensOp,
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
        self.array_size = array_size  # total elements; None = infer from DIM
        self.num_ctas = num_ctas      # CTAs per CGA optimization hint; None = no hint
        self.data_index = 0
        self._returned = False  # track if last stmt was a return
        self._array_kernel_meta: dict | None = None  # set by _generate_array_kernel

        # Populated during generate()
        self.tt: TypeTable | None = None
        self.builder: CodeBuilder | None = None
        self.var_map: dict[str, Value] = {}

        # Type IDs (set in _init_types)
        self.i32_t = None
        self.f32_t = None
        self.i1_t = None

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
            # Arrays not directly supported; DATA/READ inlines constants
            raise BytecodeBackendError("Array access not supported in bytecode backend")

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

    def _gen_binop(self, expr: ast.BinaryOp) -> Value:
        left = self._gen_expr(expr.left)
        right = self._gen_expr(expr.right)
        lt = self._type_of_expr(expr.left)
        rt = self._type_of_expr(expr.right)

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

        if expr.op == "+":
            return self._addf(left, right) if is_float else self._addi(left, right)
        elif expr.op == "-":
            return self._subf(left, right) if is_float else self._subi(left, right)
        elif expr.op == "*":
            return self._mulf(left, right) if is_float else self._muli(left, right)
        elif expr.op == "/":
            return self._divf(left, right) if is_float else self._divi(left, right)
        elif expr.op == "MOD":
            if is_float:
                return encode_RemFOp(self.builder, self.f32_t, left, right)
            else:
                return encode_RemIOp(self.builder, self.i32_t, left, right, Signedness.Signed)
        elif expr.op == "^":
            return encode_PowOp(self.builder, self.f32_t, left, right)

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
            s = encode_SinOp(self.builder, self.f32_t, arg)
            c = encode_CosOp(self.builder, self.f32_t, arg)
            return self._divf(s, c)
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
            pass  # Arrays not needed for scalar patterns
        elif isinstance(stmt, ast.InputStatement):
            pass  # Handled as kernel params
        elif isinstance(stmt, ast.DataStatement):
            pass  # Already collected by analyzer
        elif isinstance(stmt, ast.OutputStatement):
            pass  # Handled at a higher level
        elif isinstance(stmt, (ast.TileStatement, ast.MmaStatement, ast.TileStoreStatement)):
            pass  # Handled by GEMM kernel path
        elif isinstance(stmt, (ast.GotoStatement, ast.GosubStatement, ast.ReturnStatement)):
            pass  # Not supported

    def _gen_let(self, stmt: ast.LetStatement):
        val = self._gen_expr(stmt.value)
        if isinstance(stmt.target, ast.Variable):
            # Cast if value type doesn't match variable's declared type
            expr_type = self._type_of_expr(stmt.value)
            info = self.symbols.get(stmt.target.name)
            if info and info.type == BasicType.F32 and expr_type == BasicType.I32:
                val = self._cast_to_f32(val)
            elif info and info.type == BasicType.I32 and expr_type == BasicType.F32:
                val = self._cast_to_i32(val)
            self.var_map[stmt.target.name] = val

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

    def _gen_for(self, stmt: ast.ForStatement):
        lb = self._gen_expr(stmt.start)
        end_val = self._gen_expr(stmt.end)
        var_type = self._type_of_expr(stmt.start)
        type_id = self._type_id(var_type)

        if stmt.step:
            step = self._gen_expr(stmt.step)
        else:
            step = self._const(1, var_type)

        # Half-open upper bound: end + step
        if var_type == BasicType.F32:
            ub = self._addf(end_val, step)
        else:
            ub = self._addi(end_val, step)

        # Find iter_values (variables modified in loop body that already exist)
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

        # Block args: (iv, *iter_values)
        block_arg_types = [type_id] + iter_type_ids

        saved = dict(self.var_map)
        with nbb.new_block(block_arg_types) as body_args:
            iv = body_args[0]
            self.var_map[stmt.var.name] = iv
            for i, (name, _) in enumerate(iter_vars):
                self.var_map[name] = body_args[1 + i]

            for s in stmt.body:
                self._gen_stmt(s)

            # Yield current values of iter vars
            yield_vals = [self.var_map[name] for name, _ in iter_vars]
            encode_ContinueOp(self.builder, operands=yield_vals)

        results = nbb.done()
        self.var_map = dict(saved)
        for i, (name, _) in enumerate(iter_vars):
            self.var_map[name] = results[i]

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

    # ---- Array kernel support ----

    TILE_SIZE = 128  # default tile size (elements per CTA)

    def _is_array_kernel(self) -> bool:
        """Detect the array kernel pattern: DIM arrays + INPUT arrays + FOR + OUTPUT."""
        has_input_arrays = bool(
            self.analyzed.input_vars
            and any(
                self.symbols.get(v) and self.symbols[v].is_array
                for v in self.analyzed.input_vars
            )
        )
        has_output_arrays = bool(
            self.analyzed.output_vars
            and any(
                self.symbols.get(v) and self.symbols[v].is_array
                for v in self.analyzed.output_vars
            )
        )
        return has_input_arrays and has_output_arrays

    def _get_array_info(self) -> dict:
        """Extract array kernel metadata: input/output arrays, sizes, array LET stmts."""
        input_arrays = []
        for name in self.analyzed.input_vars:
            info = self.symbols.get(name)
            if info and info.is_array:
                input_arrays.append(name)
        output_arrays = []
        for name in (self.analyzed.output_vars or []):
            info = self.symbols.get(name)
            if info and info.is_array:
                output_arrays.append(name)

        # Collect array LET statements: LET C(BID) = A(BID) + B(BID)
        array_let_stmts = []
        for stmt in self.analyzed.statements:
            if isinstance(stmt, ast.LetStatement) and isinstance(stmt.target, ast.ArrayAccess):
                target_name = stmt.target.name
                if self.symbols.get(target_name) and self.symbols[target_name].is_array:
                    array_let_stmts.append(stmt)

        # Get tile size from the first DIM'd array
        tile_size = None
        for name in input_arrays + output_arrays:
            info = self.symbols.get(name)
            if info and info.array_size:
                tile_size = info.array_size
                break

        return {
            "input_arrays": input_arrays,
            "output_arrays": output_arrays,
            "array_let_stmts": array_let_stmts,
            "tile_size": tile_size,
        }

    def _load_array_tile(self, name: str, index: Value,
                         views: dict[str, Value], tokens: list[Value]) -> Value:
        """Load a tile from an array's partition view at the given index."""
        token_t = self.var_map["__token_type__"]
        pv = views[name]
        tok = tokens[-1] if tokens else self.var_map["__init_token__"]
        tile_type = self.var_map[f"__tile_type_{name}__"]
        tile_val, new_tok = encode_LoadViewTkoOp(
            self.builder, tile_type, token_t, pv,
            [index], tok,
            MemoryOrderingSemantics.WEAK, None, None,
        )
        tokens.append(new_tok)
        return tile_val

    def _gen_array_expr(self, expr: ast.Expression,
                        views: dict[str, Value], tokens: list[Value]) -> Value:
        """Generate bytecode for an expression in the array kernel context.

        Array accesses A(BID) become tile loads from partition views.
        Scalar expressions use the normal _gen_expr path.
        """
        if isinstance(expr, ast.ArrayAccess):
            name = expr.name
            if name in views:
                index = self._gen_expr(expr.index)
                return self._load_array_tile(name, index, views, tokens)
            raise BytecodeBackendError(f"Array {name} not in views")

        if isinstance(expr, ast.BinaryOp):
            left = self._gen_array_expr(expr.left, views, tokens)
            right = self._gen_array_expr(expr.right, views, tokens)
            lt = self._type_of_expr(expr.left)
            rt = self._type_of_expr(expr.right)

            # Get the tile-sized type
            tile_size = self._active_tile_size
            is_float = (lt == BasicType.F32 or rt == BasicType.F32)
            if expr.op in ("/", "^"):
                is_float = True

            f32_s = self.tt.simple(SimpleType.F32)
            i32_s = self.tt.simple(SimpleType.I32)
            result_type = self.tt.tile(f32_s, [tile_size]) if is_float else self.tt.tile(i32_s, [tile_size])

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
                return encode_DivFOp(self.builder, result_type, left, right, RoundingMode.NEAREST_EVEN, False)
            raise BytecodeBackendError(f"Unsupported array op: {expr.op}")

        # Scalar constant — broadcast to tile size? Not needed if both sides are arrays
        return self._gen_expr(expr)

    def _generate_array_kernel(self, array_size: int) -> bytes:
        """Generate a tiled array kernel."""
        info = self._get_array_info()
        input_arrays = info["input_arrays"]
        output_arrays = info["output_arrays"]
        array_let_stmts = info["array_let_stmts"]
        tile_size = info["tile_size"] or self.TILE_SIZE
        self._active_tile_size = tile_size

        if not array_let_stmts:
            raise BytecodeBackendError("Cannot detect array kernel pattern")

        grid_size = (array_size + tile_size - 1) // tile_size

        # All arrays that need pointer parameters (inputs + outputs, deduplicated)
        all_arrays: list[str] = []
        seen = set()
        for name in input_arrays + output_arrays:
            if name not in seen:
                seen.add(name)
                all_arrays.append(name)

        buf = bytearray()
        with write_bytecode(1, buf, BytecodeVersion.V_13_2) as writer:
            tt = writer.type_table
            self._init_types(tt)

            f32_s = tt.simple(SimpleType.F32)
            i32_s = tt.simple(SimpleType.I32)
            token_s = tt.simple(SimpleType.Token)

            ptr_f32 = tt.pointer(f32_s)
            tile_ptr_f32 = tt.tile(ptr_f32, [])
            tv_f32 = tt.tensor_view(f32_s, [array_size], [1])
            pv_f32 = tt.partition_view([tile_size], tv_f32, [0], PaddingValue.Zero)
            tile_f32 = tt.tile(f32_s, [tile_size])

            param_types = [tile_ptr_f32] * len(all_arrays)

            with writer.function(
                "main", param_types, [], True, self._entry_hints(), DebugAttrId(0)
            ) as fb:
                self.builder = fb.code_builder
                self.var_map = {}

                # Map parameters to array names
                param_vals = {name: fb.parameters[i] for i, name in enumerate(all_arrays)}

                # Get tile block ID — exposed as BID in BASIC
                bid_x, _, _ = encode_GetTileBlockIdOp(
                    self.builder, self.i32_t, self.i32_t, self.i32_t
                )
                self.var_map["BID"] = bid_x
                self.var_map["__token_type__"] = token_s

                # Create tensor views and partition views
                views: dict[str, Value] = {}
                for name in all_arrays:
                    tv = encode_MakeTensorViewOp(
                        self.builder, tv_f32, param_vals[name], [], []
                    )
                    pv = encode_MakePartitionViewOp(self.builder, pv_f32, tv)
                    views[name] = pv

                # Store tile type for loads
                for name in all_arrays:
                    self.var_map[f"__tile_type_{name}__"] = tile_f32

                # Create initial token
                tok = encode_MakeTokenOp(self.builder, token_s)
                self.var_map["__init_token__"] = tok

                # Process array LET statements as tiled operations
                load_tokens: list[Value] = []
                for body_stmt in array_let_stmts:
                    target_name = body_stmt.target.name
                    if target_name in views:
                        # Evaluate the store index (e.g. BID)
                        store_index = self._gen_expr(body_stmt.target.index)

                        result_tile = self._gen_array_expr(
                            body_stmt.value, views, load_tokens
                        )
                        # Join all load tokens
                        if len(load_tokens) > 1:
                            tok_joined = encode_JoinTokensOp(
                                self.builder, token_s, load_tokens
                            )
                        elif load_tokens:
                            tok_joined = load_tokens[0]
                        else:
                            tok_joined = tok

                        encode_StoreViewTkoOp(
                            self.builder, token_s, result_tile,
                            views[target_name], [store_index], tok_joined,
                            MemoryOrderingSemantics.WEAK, None, None,
                        )

                encode_ReturnOp(self.builder, operands=[])

        self._array_kernel_meta = {
            "all_arrays": all_arrays,
            "input_arrays": input_arrays,
            "output_arrays": output_arrays,
            "array_size": array_size,
            "tile_size": tile_size,
            "grid_size": grid_size,
        }
        return bytes(buf)

    # ---- GEMM kernel support ----

    def _find_stmt_recursive(self, stmt_type, stmts=None):
        """Find first statement of given type, recursing into FOR/IF bodies."""
        if stmts is None:
            stmts = self.analyzed.statements
        for stmt in stmts:
            if isinstance(stmt, stmt_type):
                return stmt
            if isinstance(stmt, ast.ForStatement):
                result = self._find_stmt_recursive(stmt_type, stmt.body)
                if result:
                    return result
            if isinstance(stmt, ast.IfStatement):
                result = self._find_stmt_recursive(stmt_type, stmt.then_body)
                if result:
                    return result
                result = self._find_stmt_recursive(stmt_type, stmt.else_body)
                if result:
                    return result
        return None

    def _is_gemm_kernel(self) -> bool:
        has_tile = any(isinstance(s, ast.TileStatement)
                       for s in self.analyzed.statements)
        has_mma = self._find_stmt_recursive(ast.MmaStatement) is not None
        return has_tile and has_mma

    def _get_dim_sizes(self) -> dict[str, list[int]]:
        """Extract dimension sizes from DIM statements."""
        dims = {}
        for stmt in self.analyzed.statements:
            if isinstance(stmt, ast.DimStatement):
                sizes = [int(s.value) for s in stmt.sizes
                         if isinstance(s, ast.NumberLiteral)]
                dims[stmt.name] = sizes
        return dims

    def _gen_gemm_stmt(self, stmt: ast.Statement):
        """Generate bytecode for a statement in the GEMM kernel context."""
        if isinstance(stmt, (ast.RemStatement, ast.DimStatement,
                             ast.TileStatement, ast.InputStatement,
                             ast.OutputStatement)):
            return
        if isinstance(stmt, (ast.EndStatement, ast.StopStatement)):
            return  # ReturnOp emitted after all stmts
        if isinstance(stmt, ast.LetStatement):
            self._gen_let(stmt)
        elif isinstance(stmt, ast.ForStatement):
            self._gen_gemm_for(stmt)
        elif isinstance(stmt, ast.MmaStatement):
            self._gen_mma(stmt)
        elif isinstance(stmt, ast.TileStoreStatement):
            self._gen_tile_store(stmt)

    def _gen_gemm_for(self, stmt: ast.ForStatement):
        """Generate a FOR loop that carries the MMA accumulator and token."""
        lb = self._gen_expr(stmt.start)
        end_val = self._gen_expr(stmt.end)
        step = self._gen_expr(stmt.step) if stmt.step else self._const_i32(1)

        var_type = self._type_of_expr(stmt.start)
        iv_type_id = self._type_id(var_type)

        # Half-open upper bound
        if var_type == BasicType.F32:
            ub = self._addf(end_val, step)
        else:
            ub = self._addi(end_val, step)

        tile_c_t = self._gemm_types['tile_c']
        token_s = self._gemm_types['token_s']
        acc_var = self._gemm_acc_var

        # Find scalar vars modified in body (exclude tile-valued ACC)
        modified = self._find_modified_vars(stmt.body)
        scalar_iter = [(n, self.var_map[n]) for n in modified
                       if n in self.var_map and n != acc_var]
        scalar_types = [self._type_id(self.symbols[n].type) for n, _ in scalar_iter]

        # Loop carries: ACC tile, token, then scalar vars
        result_types = [tile_c_t, token_s] + scalar_types
        init_values = [self.var_map[acc_var], self._gemm_token] + [v for _, v in scalar_iter]

        nbb = encode_ForOp(
            self.builder,
            result_types=result_types,
            lowerBound=lb,
            upperBound=ub,
            step=step,
            initValues=init_values,
            unsignedCmp=False,
        )

        block_arg_types = [iv_type_id, tile_c_t, token_s] + scalar_types

        saved = dict(self.var_map)
        saved_token = self._gemm_token

        with nbb.new_block(block_arg_types) as body_args:
            self.var_map[stmt.var.name] = body_args[0]  # induction var
            self.var_map[acc_var] = body_args[1]
            self._gemm_token = body_args[2]
            for i, (name, _) in enumerate(scalar_iter):
                self.var_map[name] = body_args[3 + i]

            for s in stmt.body:
                self._gen_gemm_stmt(s)

            yield_vals = [self.var_map[acc_var], self._gemm_token]
            yield_vals += [self.var_map[n] for n, _ in scalar_iter]
            encode_ContinueOp(self.builder, operands=yield_vals)

        results = nbb.done()
        self.var_map = dict(saved)
        self.var_map[acc_var] = results[0]
        self._gemm_token = results[1]
        for i, (name, _) in enumerate(scalar_iter):
            self.var_map[name] = results[2 + i]

    def _ensure_i32(self, val: Value, expr: ast.Expression) -> Value:
        """Cast to i32 if expression type is f32 (needed for tile indices)."""
        typ = self._type_of_expr(expr)
        if typ == BasicType.F32:
            return self._cast_to_i32(val)
        return val

    def _gen_mma(self, stmt: ast.MmaStatement):
        """Generate tile loads + MmaFOp for MMA ACC, A(i,j), B(i,j)."""
        a_idx0 = self._ensure_i32(self._gen_expr(stmt.a_access.index), stmt.a_access.index)
        a_idx1 = self._ensure_i32(self._gen_expr(stmt.a_access.index2), stmt.a_access.index2)
        b_idx0 = self._ensure_i32(self._gen_expr(stmt.b_access.index), stmt.b_access.index)
        b_idx1 = self._ensure_i32(self._gen_expr(stmt.b_access.index2), stmt.b_access.index2)

        tile_a_t = self._gemm_types['tile_a']
        tile_b_t = self._gemm_types['tile_b']
        tile_c_t = self._gemm_types['tile_c']
        token_s = self._gemm_types['token_s']

        pv_a = self._gemm_views[stmt.a_access.name]
        pv_b = self._gemm_views[stmt.b_access.name]

        # Load A tile
        tile_a_val, tok_a = encode_LoadViewTkoOp(
            self.builder, tile_a_t, token_s, pv_a,
            [a_idx0, a_idx1], self._gemm_token,
            MemoryOrderingSemantics.WEAK, None, None,
        )

        # Load B tile
        tile_b_val, tok_b = encode_LoadViewTkoOp(
            self.builder, tile_b_t, token_s, pv_b,
            [b_idx0, b_idx1], tok_a,
            MemoryOrderingSemantics.WEAK, None, None,
        )

        # MMA: acc = A_tile * B_tile + acc
        acc = self.var_map[stmt.acc_var]
        new_acc = encode_MmaFOp(
            self.builder, tile_c_t, tile_a_val, tile_b_val, acc,
        )

        self.var_map[stmt.acc_var] = new_acc
        self._gemm_token = tok_b

    def _gen_tile_store(self, stmt: ast.TileStoreStatement):
        """Generate StoreViewTkoOp for STORE C(i,j), ACC."""
        idx0 = self._ensure_i32(self._gen_expr(stmt.target.index), stmt.target.index)
        idx1 = self._ensure_i32(self._gen_expr(stmt.target.index2), stmt.target.index2)

        token_s = self._gemm_types['token_s']
        pv = self._gemm_views[stmt.target.name]
        tile_val = self.var_map[stmt.value_var]

        encode_StoreViewTkoOp(
            self.builder, token_s, tile_val,
            pv, [idx0, idx1], self._gemm_token,
            MemoryOrderingSemantics.WEAK, None, None,
        )

    def _generate_gemm_kernel(self) -> bytes:
        """Generate a tiled GEMM kernel from TILE/MMA/STORE statements."""
        # Extract tile sizes from TILE statement
        tile_stmt = None
        for stmt in self.analyzed.statements:
            if isinstance(stmt, ast.TileStatement):
                tile_stmt = stmt
                break
        tm, tn, tk = tile_stmt.tm, tile_stmt.tn, tile_stmt.tk

        # Find MMA and STORE to determine matrix names
        mma_stmt = self._find_stmt_recursive(ast.MmaStatement)
        store_stmt = self._find_stmt_recursive(ast.TileStoreStatement)
        a_name = mma_stmt.a_access.name
        b_name = mma_stmt.b_access.name
        c_name = store_stmt.target.name
        self._gemm_acc_var = mma_stmt.acc_var

        # Get matrix dimensions from DIM statements
        dims = self._get_dim_sizes()
        M, K = dims[a_name]
        K2, N = dims[b_name]
        assert K == K2, f"K dimension mismatch: {a_name} has K={K}, {b_name} has K={K2}"

        num_m_tiles = (M + tm - 1) // tm
        num_n_tiles = (N + tn - 1) // tn
        grid_size = num_m_tiles * num_n_tiles

        all_arrays = [a_name, b_name, c_name]

        buf = bytearray()
        with write_bytecode(1, buf, BytecodeVersion.V_13_2) as writer:
            tt = writer.type_table
            self._init_types(tt)

            f32_s = tt.simple(SimpleType.F32)
            token_s = tt.simple(SimpleType.Token)
            ptr_f32 = tt.pointer(f32_s)
            tile_ptr_f32 = tt.tile(ptr_f32, [])

            # 2D tensor views
            tv_a_t = tt.tensor_view(f32_s, [M, K], [K, 1])
            tv_b_t = tt.tensor_view(f32_s, [K, N], [N, 1])
            tv_c_t = tt.tensor_view(f32_s, [M, N], [N, 1])

            # 2D partition views
            pv_a_t = tt.partition_view([tm, tk], tv_a_t, [0, 1], PaddingValue.Zero)
            pv_b_t = tt.partition_view([tk, tn], tv_b_t, [0, 1], PaddingValue.Zero)
            pv_c_t = tt.partition_view([tm, tn], tv_c_t, [0, 1], PaddingValue.Zero)

            # 2D tile types
            tile_a_t = tt.tile(f32_s, [tm, tk])
            tile_b_t = tt.tile(f32_s, [tk, tn])
            tile_c_t = tt.tile(f32_s, [tm, tn])

            self._gemm_types = {
                'tile_a': tile_a_t, 'tile_b': tile_b_t, 'tile_c': tile_c_t,
                'token_s': token_s,
            }

            param_types = [tile_ptr_f32] * 3

            with writer.function(
                "main", param_types, [], True, self._entry_hints(), DebugAttrId(0)
            ) as fb:
                self.builder = fb.code_builder
                self.var_map = {}

                # BID
                bid_x, _, _ = encode_GetTileBlockIdOp(
                    self.builder, self.i32_t, self.i32_t, self.i32_t
                )
                self.var_map["BID"] = bid_x

                # Create tensor views and partition views
                tv_vals = {}
                for i, (name, tv_t) in enumerate(zip(all_arrays, [tv_a_t, tv_b_t, tv_c_t])):
                    tv_vals[name] = encode_MakeTensorViewOp(
                        self.builder, tv_t, fb.parameters[i], [], []
                    )

                self._gemm_views = {}
                for name, pv_t in zip(all_arrays, [pv_a_t, pv_b_t, pv_c_t]):
                    self._gemm_views[name] = encode_MakePartitionViewOp(
                        self.builder, pv_t, tv_vals[name]
                    )

                # Initial token
                self._gemm_token = encode_MakeTokenOp(self.builder, token_s)

                # Initialize accumulator to zeros
                acc_init = encode_ConstantOp(
                    self.builder, tile_c_t, struct.pack("<f", 0.0)
                )
                self.var_map[self._gemm_acc_var] = acc_init

                # Walk all statements
                for stmt in self.analyzed.statements:
                    self._gen_gemm_stmt(stmt)

                encode_ReturnOp(self.builder, operands=[])

        self._array_kernel_meta = {
            "all_arrays": all_arrays,
            "input_arrays": [a_name, b_name],
            "output_arrays": [c_name],
            "array_size": M * N,
            "tile_size": tm,
            "grid_size": grid_size,
            "M": M, "N": N, "K": K,
            "tm": tm, "tn": tn, "tk": tk,
        }
        return bytes(buf)

    # ---- Main entry points ----

    def generate(self, array_size: int | None = None) -> bytes:
        """Generate cuTile bytecode from the analyzed program."""
        if self._is_gemm_kernel():
            return self._generate_gemm_kernel()

        if self._is_array_kernel():
            size = array_size or self.array_size
            if size is None:
                raise BytecodeBackendError(
                    "array_size must be provided for array kernels"
                )
            return self._generate_array_kernel(size)

        buf = bytearray()

        with write_bytecode(1, buf, BytecodeVersion.V_13_2) as writer:
            self._init_types(writer.type_table)

            with writer.function(
                "main", [], [], True, self._entry_hints(), DebugAttrId(0)
            ) as fb:
                self.builder = fb.code_builder
                self.var_map = {}
                self.data_index = 0
                self._returned = False

                for stmt in self.analyzed.statements:
                    self._gen_stmt(stmt)

                if not self._returned:
                    encode_ReturnOp(self.builder, operands=[])

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
        gpu_arch: Target GPU architecture (e.g. ``"sm_90"``).
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
