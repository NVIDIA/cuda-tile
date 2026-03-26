"""Textual backend: AST → CUDA Tile IR MLIR text."""

from __future__ import annotations
from . import _ast_nodes as ast
from ._analyzer import AnalyzedProgram, BasicType, SymbolInfo
from ._lexer import lex
from ._parser import parse
from ._analyzer import analyze


class TextualBackendError(Exception):
    pass


class TextualBackend:
    def __init__(self, analyzed: AnalyzedProgram):
        self.analyzed = analyzed
        self.symbols = analyzed.symbols
        self.lines: list[str] = []
        self.indent = 0
        self.ssa_counter = 0
        self.var_ssa: dict[str, str] = {}  # BASIC var name → current SSA name
        self.data_index = 0

    def next_ssa(self) -> str:
        name = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def emit(self, line: str):
        self.lines.append("  " * self.indent + line)

    def emit_blank(self):
        self.lines.append("")

    def generate(self) -> str:
        self.lines = []
        self.emit('cuda_tile.module @basic_program {')
        self.indent += 1

        # Emit kernel entry function
        params = self._build_params()
        if params:
            param_str = ", ".join(params)
            self.emit(f'entry @main({param_str}) {{')
        else:
            self.emit('entry @main() {')
        self.indent += 1

        # Generate code for all statements
        for stmt in self.analyzed.statements:
            self._gen_stmt(stmt)

        # Emit return
        self.emit("return")
        self.indent -= 1
        self.emit("}")

        self.indent -= 1
        self.emit("}")
        self.emit_blank()
        return "\n".join(self.lines)

    def _build_params(self) -> list[str]:
        """Build kernel parameter list from INPUT variables."""
        params = []
        seen = set()
        for name in self.analyzed.input_vars:
            if name in seen:
                continue
            seen.add(name)
            info = self.symbols.get(name)
            if info:
                tile_type = info.type.tile_type()
                safe = self._safe_name(name)
                ssa = self.next_ssa()
                self.var_ssa[name] = ssa
                params.append(f"{ssa}: {tile_type}")
        return params

    def _safe_name(self, name: str) -> str:
        """Convert BASIC variable name to valid MLIR identifier."""
        return name.replace("%", "_int").replace("$", "_str")

    def _tile_type(self, name: str) -> str:
        """Get Tile IR type for a BASIC variable."""
        info = self.symbols.get(name)
        if info:
            return info.type.tile_type()
        return "tile<f32>"

    def _scalar_type(self, name: str) -> str:
        info = self.symbols.get(name)
        if info:
            return info.type.scalar_type()
        return "f32"

    def _type_of_expr(self, expr: ast.Expression) -> BasicType:
        """Infer the type of an expression."""
        if isinstance(expr, ast.NumberLiteral):
            return BasicType.I32 if isinstance(expr.value, int) else BasicType.F32
        if isinstance(expr, ast.StringLiteral):
            return BasicType.STRING
        if isinstance(expr, ast.Variable):
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

    def _gen_stmt(self, stmt: ast.Statement):
        if isinstance(stmt, ast.RemStatement):
            self.emit(f"// {stmt.comment}")
            return

        if isinstance(stmt, ast.EndStatement) or isinstance(stmt, ast.StopStatement):
            self.emit("return")
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
        elif isinstance(stmt, ast.DimStatement):
            self._gen_dim(stmt)
        elif isinstance(stmt, ast.ReadStatement):
            self._gen_read(stmt)
        elif isinstance(stmt, ast.InputStatement):
            pass  # Already handled as kernel params
        elif isinstance(stmt, ast.DataStatement):
            pass  # Already collected by analyzer
        elif isinstance(stmt, ast.OutputStatement):
            pass  # Handled at a higher level
        elif isinstance(stmt, ast.GotoStatement):
            self.emit(f"// GOTO {stmt.target} (not supported in Tile IR)")
        elif isinstance(stmt, ast.GosubStatement):
            self.emit(f"// GOSUB {stmt.target} (not supported in Tile IR)")
        elif isinstance(stmt, ast.ReturnStatement):
            self.emit("// RETURN (GOSUB not supported in Tile IR)")

    def _gen_let(self, stmt: ast.LetStatement):
        ssa = self._gen_expr(stmt.value)
        if isinstance(stmt.target, ast.Variable):
            self.var_ssa[stmt.target.name] = ssa
        elif isinstance(stmt.target, ast.ArrayAccess):
            idx_ssa = self._gen_expr(stmt.target.index)
            arr_ssa = self.var_ssa.get(stmt.target.name)
            if arr_ssa:
                tile_type = self._tile_type(stmt.target.name)
                self.emit(f"// store {ssa} into {stmt.target.name}[{idx_ssa}]")

    def _gen_print(self, stmt: ast.PrintStatement):
        if not stmt.items:
            self.emit('print "\\n"')
            return

        # Build format string and operands
        fmt_parts: list[str] = []
        operands: list[str] = []
        operand_types: list[str] = []

        for item in stmt.items:
            if isinstance(item, ast.StringLiteral):
                fmt_parts.append(item.value)
            else:
                ssa = self._gen_expr(item)
                expr_type = self._type_of_expr(item)
                if expr_type == BasicType.I32:
                    fmt_parts.append("%d")
                elif expr_type == BasicType.I1:
                    fmt_parts.append("%d")
                else:
                    fmt_parts.append("%f")
                operands.append(ssa)
                operand_types.append(expr_type.tile_type())

        fmt = "".join(fmt_parts)
        if stmt.newline:
            fmt += "\\n"

        if operands:
            ops_str = ", ".join(f"{o} : {t}" for o, t in zip(operands, operand_types))
            self.emit(f'print "{fmt}", {ops_str}')
        else:
            self.emit(f'print "{fmt}"')

    def _gen_if(self, stmt: ast.IfStatement):
        cond_ssa = self._gen_expr(stmt.condition)

        if stmt.else_body:
            self.emit(f"if {cond_ssa} {{")
            self.indent += 1
            for s in stmt.then_body:
                self._gen_stmt(s)
            self.indent -= 1
            self.emit("} else {")
            self.indent += 1
            for s in stmt.else_body:
                self._gen_stmt(s)
            self.indent -= 1
            self.emit("}")
        else:
            self.emit(f"if {cond_ssa} {{")
            self.indent += 1
            for s in stmt.then_body:
                self._gen_stmt(s)
            self.indent -= 1
            self.emit("}")

    def _gen_for(self, stmt: ast.ForStatement):
        lb_ssa = self._gen_expr(stmt.start)
        # Tile IR for loops use half-open intervals: [lb, ub)
        # So we need ub = end + step
        end_ssa = self._gen_expr(stmt.end)
        var_type = self._type_of_expr(stmt.start)
        tile_type = var_type.tile_type()

        if stmt.step:
            step_ssa = self._gen_expr(stmt.step)
        else:
            step_ssa = self._gen_const(1, var_type)

        # Compute half-open upper bound: end + step
        ub_ssa = self.next_ssa()
        if var_type == BasicType.F32:
            self.emit(f"{ub_ssa} = addf {end_ssa}, {step_ssa} rounding<nearest_even> : {tile_type}")
        else:
            self.emit(f"{ub_ssa} = addi {end_ssa}, {step_ssa} : {tile_type}")

        # Find variables modified in the loop body (for iter_values)
        modified = self._find_modified_vars(stmt.body)
        # Only include vars that already have SSA values (defined before loop)
        iter_vars = [(name, self.var_ssa[name]) for name in modified if name in self.var_ssa]

        # Build iter_values clause
        iter_var = self.next_ssa()
        if iter_vars:
            iter_params = []
            iter_inner_ssas = {}
            iter_types = []
            for name, init_ssa in iter_vars:
                inner_ssa = self.next_ssa()
                vtype = self._tile_type(name)
                iter_params.append(f"{inner_ssa} = {init_ssa}")
                iter_inner_ssas[name] = inner_ssa
                iter_types.append(vtype)
            iter_str = ", ".join(iter_params)
            types_str = ", ".join(iter_types)
            # Result SSAs for after the loop
            result_ssas = []
            for name, _ in iter_vars:
                result_ssas.append(self.next_ssa())
            results_str = ", ".join(result_ssas)
            self.emit(f"{results_str} = for {iter_var} in ({lb_ssa} to {ub_ssa}, step {step_ssa}) : {tile_type}")
            self.emit(f"                            iter_values({iter_str}) -> ({types_str}) {{")
        else:
            self.emit(f"for {iter_var} in ({lb_ssa} to {ub_ssa}, step {step_ssa}) : {tile_type} {{")
        self.indent += 1

        # Save old SSA mappings
        old_ssas = {name: self.var_ssa.get(name) for name in [stmt.var.name] + [n for n, _ in iter_vars]}

        # Map loop variable and iter_values
        self.var_ssa[stmt.var.name] = iter_var
        for name, inner_ssa in (iter_inner_ssas if iter_vars else {}).items():
            self.var_ssa[name] = inner_ssa

        for s in stmt.body:
            self._gen_stmt(s)

        # Yield modified values
        if iter_vars:
            yield_vals = ", ".join(f"{self.var_ssa[name]} : {self._tile_type(name)}" for name, _ in iter_vars)
            self.emit(f"continue {yield_vals}")
        else:
            self.emit("continue")
        self.indent -= 1
        self.emit("}")

        # After the loop, map variables to the result SSAs
        if iter_vars:
            for i, (name, _) in enumerate(iter_vars):
                self.var_ssa[name] = result_ssas[i]

    def _gen_while(self, stmt: ast.WhileStatement):
        # Emit as a bounded for loop with a conditional break
        # Or use Tile IR's while-like construct
        # For simplicity, use a for loop with large bound and break
        max_iters = self._gen_const(1000000, BasicType.I32)
        zero = self._gen_const(0, BasicType.I32)
        one = self._gen_const(1, BasicType.I32)
        iter_var = self.next_ssa()

        self.emit(f"for {iter_var} in ({zero} to {max_iters}, step {one}) : tile<i32> {{")
        self.indent += 1

        cond_ssa = self._gen_expr(stmt.condition)
        # Negate condition for break
        ones = self._gen_const(True, BasicType.I1)
        neg_ssa = self.next_ssa()
        self.emit(f"{neg_ssa} = xori {cond_ssa}, {ones} : tile<i1>")
        self.emit(f"if {neg_ssa} {{")
        self.indent += 1
        self.emit("break")
        self.indent -= 1
        self.emit("}")

        for s in stmt.body:
            self._gen_stmt(s)

        self.emit("continue")
        self.indent -= 1
        self.emit("}")

    def _gen_dim(self, stmt: ast.DimStatement):
        info = self.symbols.get(stmt.name)
        if info and info.array_size:
            scalar_type = info.type.scalar_type()
            arr_type = f"tile<{info.array_size}x{scalar_type}>"
            ssa = self.next_ssa()
            self.emit(f"{ssa} = constant <{scalar_type}: 0.0> : {arr_type}")
            self.var_ssa[stmt.name] = ssa

    def _gen_read(self, stmt: ast.ReadStatement):
        for var in stmt.variables:
            if self.data_index < len(self.analyzed.data_values):
                val = self.analyzed.data_values[self.data_index]
                self.data_index += 1
                info = self.symbols.get(var.name)
                typ = info.type if info else BasicType.F32
                ssa = self._gen_const(val, typ)
                self.var_ssa[var.name] = ssa
            else:
                self.emit(f"// READ {var.name}: no more DATA values")

    def _gen_expr(self, expr: ast.Expression) -> str:
        if isinstance(expr, ast.NumberLiteral):
            if isinstance(expr.value, int):
                return self._gen_const(expr.value, BasicType.I32)
            else:
                return self._gen_const(expr.value, BasicType.F32)

        if isinstance(expr, ast.StringLiteral):
            # Strings only used in PRINT format strings — shouldn't reach here
            raise TextualBackendError("String expressions not supported in Tile IR")

        if isinstance(expr, ast.Variable):
            if expr.name in self.var_ssa:
                return self.var_ssa[expr.name]
            # Undefined variable — emit zero
            info = self.symbols.get(expr.name)
            typ = info.type if info else BasicType.F32
            ssa = self._gen_const(0, typ)
            self.var_ssa[expr.name] = ssa
            return ssa

        if isinstance(expr, ast.ArrayAccess):
            idx_ssa = self._gen_expr(expr.index)
            arr_ssa = self.var_ssa.get(expr.name, "%undef")
            result = self.next_ssa()
            scalar_type = self._scalar_type(expr.name)
            self.emit(f"{result} = extract {arr_ssa}[{idx_ssa}] : tile<{scalar_type}>")
            return result

        if isinstance(expr, ast.UnaryOp):
            operand_ssa = self._gen_expr(expr.operand)
            result = self.next_ssa()
            if expr.op == "-":
                typ = self._type_of_expr(expr.operand)
                tile_type = typ.tile_type()
                if typ == BasicType.F32:
                    self.emit(f"{result} = negf {operand_ssa} : {tile_type}")
                else:
                    zero = self._gen_const(0, typ)
                    self.emit(f"{result} = subi {zero}, {operand_ssa} : {tile_type}")
            elif expr.op == "NOT":
                ones = self._gen_const(True, BasicType.I1)
                self.emit(f"{result} = xori {operand_ssa}, {ones} : tile<i1>")
            return result

        if isinstance(expr, ast.BinaryOp):
            return self._gen_binop(expr)

        if isinstance(expr, ast.FunctionCall):
            return self._gen_function(expr)

        raise TextualBackendError(f"Unknown expression type: {type(expr).__name__}")

    def _gen_binop(self, expr: ast.BinaryOp) -> str:
        left_ssa = self._gen_expr(expr.left)
        right_ssa = self._gen_expr(expr.right)
        result = self.next_ssa()

        lt = self._type_of_expr(expr.left)
        rt = self._type_of_expr(expr.right)

        # Insert casts if needed
        if expr.op not in ("=", "<>", "<", ">", "<=", ">=", "AND", "OR"):
            if lt == BasicType.I32 and rt == BasicType.F32:
                cast = self.next_ssa()
                self.emit(f"{cast} = itof {left_ssa} : tile<i32> -> tile<f32>")
                left_ssa = cast
                lt = BasicType.F32
            elif lt == BasicType.F32 and rt == BasicType.I32:
                cast = self.next_ssa()
                self.emit(f"{cast} = itof {right_ssa} : tile<i32> -> tile<f32>")
                right_ssa = cast
                rt = BasicType.F32

        result_type = self._type_of_expr(expr)
        tile_type = result_type.tile_type()

        # Comparisons
        if expr.op in ("=", "<>", "<", ">", "<=", ">="):
            return self._gen_comparison(expr.op, left_ssa, right_ssa, lt, rt, result)

        # Logical ops
        if expr.op == "AND":
            self.emit(f"{result} = andi {left_ssa}, {right_ssa} : tile<i1>")
            return result
        if expr.op == "OR":
            self.emit(f"{result} = ori {left_ssa}, {right_ssa} : tile<i1>")
            return result

        # Arithmetic
        is_float = (lt == BasicType.F32 or rt == BasicType.F32)
        op_type = BasicType.F32 if is_float else BasicType.I32
        op_tile = op_type.tile_type()

        if expr.op == "+":
            op = "addf" if is_float else "addi"
            rounding = " rounding<nearest_even>" if is_float else ""
            self.emit(f"{result} = {op} {left_ssa}, {right_ssa}{rounding} : {op_tile}")
        elif expr.op == "-":
            op = "subf" if is_float else "subi"
            rounding = " rounding<nearest_even>" if is_float else ""
            self.emit(f"{result} = {op} {left_ssa}, {right_ssa}{rounding} : {op_tile}")
        elif expr.op == "*":
            op = "mulf" if is_float else "muli"
            rounding = " rounding<nearest_even>" if is_float else ""
            self.emit(f"{result} = {op} {left_ssa}, {right_ssa}{rounding} : {op_tile}")
        elif expr.op == "/":
            op = "divf" if is_float else "divi"
            rounding = " rounding<nearest_even>" if is_float else ""
            self.emit(f"{result} = {op} {left_ssa}, {right_ssa}{rounding} : tile<f32>")
        elif expr.op == "MOD":
            if is_float:
                self.emit(f"{result} = remf {left_ssa}, {right_ssa} : {op_tile}")
            else:
                self.emit(f"{result} = remi {left_ssa}, {right_ssa} : {op_tile}")
        elif expr.op == "^":
            self.emit(f"{result} = pow {left_ssa}, {right_ssa} : tile<f32>")

        return result

    def _gen_comparison(self, op: str, left: str, right: str,
                        lt: BasicType, rt: BasicType, result: str) -> str:
        # Determine if float or int comparison
        is_float = (lt == BasicType.F32 or rt == BasicType.F32)

        # Cast if mismatched
        if is_float and lt == BasicType.I32:
            cast = self.next_ssa()
            self.emit(f"{cast} = itof {left} : tile<i32> -> tile<f32>")
            left = cast
        if is_float and rt == BasicType.I32:
            cast = self.next_ssa()
            self.emit(f"{cast} = itof {right} : tile<i32> -> tile<f32>")
            right = cast

        cmp_map_f = {
            "=": "equal ordered",
            "<>": "not_equal ordered",
            "<": "less_than ordered",
            ">": "greater_than ordered",
            "<=": "less_than_or_equal ordered",
            ">=": "greater_than_or_equal ordered",
        }
        cmp_map_i = {
            "=": "equal",
            "<>": "not_equal",
            "<": "less_than",
            ">": "greater_than",
            "<=": "less_than_or_equal",
            ">=": "greater_than_or_equal",
        }

        if is_float:
            cmp_pred = cmp_map_f[op]
            src_type = "tile<f32>"
            self.emit(f"{result} = cmpf {cmp_pred} {left}, {right} : {src_type}")
        else:
            cmp_pred = cmp_map_i[op]
            src_type = "tile<i32>"
            self.emit(f"{result} = cmpi {cmp_pred} {left}, {right}, signed : {src_type}")

        return result

    def _gen_function(self, expr: ast.FunctionCall) -> str:
        arg_ssa = self._gen_expr(expr.arg)
        result = self.next_ssa()
        arg_type = self._type_of_expr(expr.arg)

        # Cast to f32 if needed for math functions
        if arg_type == BasicType.I32 and expr.name not in ("SGN",):
            cast = self.next_ssa()
            self.emit(f"{cast} = itof {arg_ssa} : tile<i32> -> tile<f32>")
            arg_ssa = cast

        func_map = {
            "ABS": "absf",
            "SQR": "sqrt",
            "SIN": "sin",
            "COS": "cos",
            "TAN": "tan",
            "EXP": "exp",
            "LOG": "log",
        }

        if expr.name in func_map:
            op = func_map[expr.name]
            self.emit(f"{result} = {op} {arg_ssa} : tile<f32>")
        elif expr.name == "INT":
            if arg_type == BasicType.I32:
                return arg_ssa  # already int
            self.emit(f"{result} = ftoi {arg_ssa} : tile<f32> -> tile<i32>")
        elif expr.name == "SGN":
            # sgn(x): -1 if x<0, 0 if x==0, 1 if x>0
            zero = self._gen_const(0, arg_type)
            neg_one = self._gen_const(-1, BasicType.I32)
            one = self._gen_const(1, BasicType.I32)
            zero_i = self._gen_const(0, BasicType.I32)

            lt_ssa = self.next_ssa()
            gt_ssa = self.next_ssa()
            if arg_type == BasicType.F32:
                self.emit(f"{lt_ssa} = cmpf less_than ordered {arg_ssa}, {zero} : tile<f32>")
                self.emit(f"{gt_ssa} = cmpf greater_than ordered {arg_ssa}, {zero} : tile<f32>")
            else:
                self.emit(f"{lt_ssa} = cmpi less_than {arg_ssa}, {zero}, signed : tile<i32>")
                self.emit(f"{gt_ssa} = cmpi greater_than {arg_ssa}, {zero}, signed : tile<i32>")

            sel1 = self.next_ssa()
            self.emit(f"{sel1} = select {gt_ssa}, {one}, {zero_i} : tile<i32>")
            self.emit(f"{result} = select {lt_ssa}, {neg_one}, {sel1} : tile<i32>")
        else:
            raise TextualBackendError(f"Unknown function: {expr.name}")

        return result

    def _gen_const(self, value: int | float, typ: BasicType) -> str:
        ssa = self.next_ssa()
        tile_type = typ.tile_type()
        scalar_type = typ.scalar_type()

        if typ == BasicType.F32:
            fval = float(value)
            self.emit(f"{ssa} = constant <{scalar_type}: {fval:e}> : {tile_type}")
        elif typ == BasicType.I32:
            ival = int(value)
            self.emit(f"{ssa} = constant <{scalar_type}: {ival}> : {tile_type}")
        elif typ == BasicType.I1:
            bval = "true" if value else "false"
            self.emit(f"{ssa} = constant <{scalar_type}: {bval}> : {tile_type}")

        return ssa


def compile_basic_to_textual(source: str) -> str:
    """Compile BASIC source code to CUDA Tile IR MLIR text."""
    tokens = lex(source)
    program = parse(tokens)
    analyzed = analyze(program)
    return TextualBackend(analyzed).generate()
