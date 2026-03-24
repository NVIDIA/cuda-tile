"""Recursive descent parser for BASIC."""

from __future__ import annotations
from .tokens import Token, TokenType, KEYWORDS
from .lexer import BUILTIN_FUNCTIONS
from . import ast_nodes as ast


class ParseError(Exception):
    def __init__(self, msg: str, token: Token):
        super().__init__(f"Parse error at L{token.line}:{token.col}: {msg}")
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tt: TokenType) -> Token:
        tok = self.peek()
        if tok.type != tt:
            raise ParseError(f"Expected {tt.name}, got {tok.type.name}", tok)
        return self.advance()

    def at(self, *types: TokenType) -> bool:
        return self.peek().type in types

    def match(self, *types: TokenType) -> Token | None:
        if self.at(*types):
            return self.advance()
        return None

    def skip_newlines(self):
        while self.at(TokenType.NEWLINE):
            self.advance()

    def _at_block_end(self, *types: TokenType) -> bool:
        """Check if current position is at a block-ending keyword,
        accounting for optional line numbers before the keyword."""
        if self.at(TokenType.EOF):
            return True
        if self.at(*types):
            return True
        # Check if line_number followed by block-end keyword
        if self.at(TokenType.INTEGER) and self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1].type in types
        return False

    def parse(self) -> ast.Program:
        stmts: list[ast.Statement] = []
        line_map: dict[int, int] = {}
        self.skip_newlines()
        while not self.at(TokenType.EOF):
            line_num = None
            if self.at(TokenType.INTEGER):
                tok = self.peek()
                # Check if this is a line number (first token on line or after newline)
                line_num = int(tok.value)
                self.advance()

            result = self.parse_statement()
            if line_num is not None:
                line_map[line_num] = len(stmts)
            if isinstance(result, list):
                stmts.extend(result)
            else:
                stmts.append(result)

            # Consume statement separator (newline, colon, or EOF)
            if self.at(TokenType.COLON):
                self.advance()
            else:
                self.skip_newlines()

        return ast.Program(statements=stmts, line_map=line_map)

    def parse_statement(self) -> ast.Statement:
        tok = self.peek()

        if tok.type == TokenType.LET:
            return self.parse_let()
        elif tok.type == TokenType.PRINT:
            return self.parse_print()
        elif tok.type == TokenType.INPUT:
            return self.parse_input()
        elif tok.type == TokenType.IF:
            return self.parse_if()
        elif tok.type == TokenType.FOR:
            return self.parse_for()
        elif tok.type == TokenType.WHILE:
            return self.parse_while()
        elif tok.type == TokenType.GOTO:
            return self.parse_goto()
        elif tok.type == TokenType.GOSUB:
            return self.parse_gosub()
        elif tok.type == TokenType.RETURN:
            self.advance()
            return ast.ReturnStatement(line=tok.line)
        elif tok.type == TokenType.DIM:
            return self.parse_dim()  # may return list; caller handles
        elif tok.type == TokenType.REM:
            self.advance()
            return ast.RemStatement(comment=tok.value, line=tok.line)
        elif tok.type == TokenType.DATA:
            return self.parse_data()
        elif tok.type == TokenType.READ:
            return self.parse_read()
        elif tok.type == TokenType.END:
            self.advance()
            return ast.EndStatement(line=tok.line)
        elif tok.type == TokenType.STOP:
            self.advance()
            return ast.StopStatement(line=tok.line)
        elif tok.type == TokenType.TILE:
            return self.parse_tile()
        elif tok.type == TokenType.MMA:
            return self.parse_mma()
        elif tok.type == TokenType.STORE:
            return self.parse_tile_store()
        elif tok.type == TokenType.OUTPUT:
            return self.parse_output()
        elif tok.type == TokenType.IDENTIFIER:
            # Implicit LET: X = expr
            return self.parse_implicit_let()
        elif tok.type == TokenType.NEXT:
            # NEXT consumed by for loop, but handle standalone
            raise ParseError("NEXT without FOR", tok)
        else:
            raise ParseError(f"Unexpected token: {tok.type.name}", tok)

    def parse_let(self) -> ast.LetStatement:
        tok = self.expect(TokenType.LET)
        target = self.parse_lvalue()
        self.expect(TokenType.EQ)
        value = self.parse_expression()
        return ast.LetStatement(target=target, value=value, line=tok.line)

    def parse_implicit_let(self) -> ast.LetStatement:
        tok = self.peek()
        target = self.parse_lvalue()
        self.expect(TokenType.EQ)
        value = self.parse_expression()
        return ast.LetStatement(target=target, value=value, line=tok.line)

    def parse_lvalue(self) -> ast.Variable | ast.ArrayAccess:
        tok = self.expect(TokenType.IDENTIFIER)
        if self.match(TokenType.LPAREN):
            idx = self.parse_expression()
            idx2 = None
            if self.match(TokenType.COMMA):
                idx2 = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return ast.ArrayAccess(name=tok.value, index=idx, index2=idx2, line=tok.line)
        return ast.Variable(name=tok.value, line=tok.line)

    def parse_print(self) -> ast.PrintStatement:
        tok = self.expect(TokenType.PRINT)
        items: list[ast.Expression] = []
        newline = True

        if not self.at(TokenType.NEWLINE, TokenType.EOF, TokenType.COLON):
            items.append(self.parse_expression())
            while self.match(TokenType.SEMICOLON, TokenType.COMMA):
                if self.at(TokenType.NEWLINE, TokenType.EOF, TokenType.COLON):
                    newline = False
                    break
                items.append(self.parse_expression())

        return ast.PrintStatement(items=items, newline=newline, line=tok.line)

    def parse_input(self) -> ast.InputStatement:
        tok = self.expect(TokenType.INPUT)
        prompt = None
        if self.at(TokenType.STRING):
            prompt = self.advance().value
            self.expect(TokenType.SEMICOLON)
        variables: list[ast.Variable] = []
        is_array: list[bool] = []
        var_tok = self.expect(TokenType.IDENTIFIER)
        variables.append(ast.Variable(name=var_tok.value, line=var_tok.line))
        is_array.append(bool(self.match(TokenType.LPAREN) and self.expect(TokenType.RPAREN)))
        while self.match(TokenType.COMMA):
            var_tok = self.expect(TokenType.IDENTIFIER)
            variables.append(ast.Variable(name=var_tok.value, line=var_tok.line))
            is_array.append(bool(self.match(TokenType.LPAREN) and self.expect(TokenType.RPAREN)))
        return ast.InputStatement(prompt=prompt, variables=variables, is_array=is_array, line=tok.line)

    def parse_if(self) -> ast.IfStatement:
        tok = self.expect(TokenType.IF)
        cond = self.parse_expression()
        self.expect(TokenType.THEN)

        # Check for single-line IF: IF cond THEN statement
        if not self.at(TokenType.NEWLINE, TokenType.EOF):
            # Single-line IF
            then_body = [self.parse_statement()]
            else_body: list[ast.Statement] = []
            if self.match(TokenType.ELSE):
                else_body = [self.parse_statement()]
            return ast.IfStatement(condition=cond, then_body=then_body, else_body=else_body, line=tok.line)

        # Multi-line IF block
        self.skip_newlines()
        then_body = []
        while not self._at_block_end(TokenType.ELSE, TokenType.ENDIF):
            # Skip line numbers in block
            if self.at(TokenType.INTEGER):
                self.advance()
            then_body.append(self.parse_statement())
            if self.at(TokenType.COLON):
                self.advance()
            else:
                self.skip_newlines()

        else_body = []
        # Consume line number before ELSE if present
        if self.at(TokenType.INTEGER) and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.ELSE:
            self.advance()
        if self.match(TokenType.ELSE):
            self.skip_newlines()
            while not self._at_block_end(TokenType.ENDIF):
                if self.at(TokenType.INTEGER):
                    self.advance()
                else_body.append(self.parse_statement())
                if self.at(TokenType.COLON):
                    self.advance()
                else:
                    self.skip_newlines()

        # Consume line number before ENDIF if present
        if self.at(TokenType.INTEGER):
            self.advance()
        self.expect(TokenType.ENDIF)
        return ast.IfStatement(condition=cond, then_body=then_body, else_body=else_body, line=tok.line)

    def parse_for(self) -> ast.ForStatement:
        tok = self.expect(TokenType.FOR)
        var_tok = self.expect(TokenType.IDENTIFIER)
        var = ast.Variable(name=var_tok.value, line=var_tok.line)
        self.expect(TokenType.EQ)
        start = self.parse_expression()
        self.expect(TokenType.TO)
        end = self.parse_expression()
        step = None
        if self.match(TokenType.STEP):
            step = self.parse_expression()
        self.skip_newlines()

        body: list[ast.Statement] = []
        while not self._at_block_end(TokenType.NEXT):
            if self.at(TokenType.INTEGER):
                self.advance()
            body.append(self.parse_statement())
            if self.at(TokenType.COLON):
                self.advance()
            else:
                self.skip_newlines()

        # Consume line number before NEXT if present
        if self.at(TokenType.INTEGER):
            self.advance()
        if self.match(TokenType.NEXT):
            # Optional variable name after NEXT
            if self.at(TokenType.IDENTIFIER):
                self.advance()

        return ast.ForStatement(var=var, start=start, end=end, step=step, body=body, line=tok.line)

    def parse_while(self) -> ast.WhileStatement:
        tok = self.expect(TokenType.WHILE)
        cond = self.parse_expression()
        self.skip_newlines()

        body: list[ast.Statement] = []
        while not self._at_block_end(TokenType.WEND):
            if self.at(TokenType.INTEGER):
                self.advance()
            body.append(self.parse_statement())
            if self.at(TokenType.COLON):
                self.advance()
            else:
                self.skip_newlines()

        if self.at(TokenType.INTEGER):
            self.advance()
        self.expect(TokenType.WEND)
        return ast.WhileStatement(condition=cond, body=body, line=tok.line)

    def parse_goto(self) -> ast.GotoStatement:
        tok = self.expect(TokenType.GOTO)
        target_tok = self.expect(TokenType.INTEGER)
        return ast.GotoStatement(target=int(target_tok.value), line=tok.line)

    def parse_gosub(self) -> ast.GosubStatement:
        tok = self.expect(TokenType.GOSUB)
        target_tok = self.expect(TokenType.INTEGER)
        return ast.GosubStatement(target=int(target_tok.value), line=tok.line)

    def parse_dim(self) -> ast.DimStatement | list[ast.DimStatement]:
        tok = self.expect(TokenType.DIM)
        dims = [self._parse_one_dim(tok.line)]
        while self.match(TokenType.COMMA):
            dims.append(self._parse_one_dim(tok.line))
        return dims if len(dims) > 1 else dims[0]

    def _parse_one_dim(self, line: int) -> ast.DimStatement:
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)
        sizes: list[ast.Expression] = [self.parse_expression()]
        while self.match(TokenType.COMMA):
            sizes.append(self.parse_expression())
        self.expect(TokenType.RPAREN)
        return ast.DimStatement(name=name_tok.value, sizes=sizes, line=line)

    def parse_data(self) -> ast.DataStatement:
        tok = self.expect(TokenType.DATA)
        values: list[float | int | str] = []
        values.append(self._parse_data_value())
        while self.match(TokenType.COMMA):
            values.append(self._parse_data_value())
        return ast.DataStatement(values=values, line=tok.line)

    def _parse_data_value(self) -> float | int | str:
        if self.at(TokenType.STRING):
            return self.advance().value
        if self.at(TokenType.INTEGER):
            return int(self.advance().value)
        if self.at(TokenType.FLOAT):
            return float(self.advance().value)
        if self.at(TokenType.MINUS):
            self.advance()
            if self.at(TokenType.INTEGER):
                return -int(self.advance().value)
            if self.at(TokenType.FLOAT):
                return -float(self.advance().value)
        raise ParseError("Expected data value", self.peek())

    def parse_output(self) -> ast.OutputStatement:
        tok = self.expect(TokenType.OUTPUT)
        variables: list[ast.Variable] = []
        var_tok = self.expect(TokenType.IDENTIFIER)
        variables.append(ast.Variable(name=var_tok.value, line=var_tok.line))
        while self.match(TokenType.COMMA):
            var_tok = self.expect(TokenType.IDENTIFIER)
            variables.append(ast.Variable(name=var_tok.value, line=var_tok.line))
        return ast.OutputStatement(variables=variables, line=tok.line)

    def parse_tile(self) -> ast.TileStatement:
        tok = self.expect(TokenType.TILE)
        tm = int(self.expect(TokenType.INTEGER).value)
        self.expect(TokenType.COMMA)
        tn = int(self.expect(TokenType.INTEGER).value)
        self.expect(TokenType.COMMA)
        tk = int(self.expect(TokenType.INTEGER).value)
        return ast.TileStatement(tm=tm, tn=tn, tk=tk, line=tok.line)

    def parse_mma(self) -> ast.MmaStatement:
        """MMA ACC, A(row, col), B(row, col)"""
        tok = self.expect(TokenType.MMA)
        acc_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.COMMA)
        a_name = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)
        a_idx = self.parse_expression()
        self.expect(TokenType.COMMA)
        a_idx2 = self.parse_expression()
        self.expect(TokenType.RPAREN)
        a_access = ast.ArrayAccess(name=a_name.value, index=a_idx, index2=a_idx2, line=a_name.line)
        self.expect(TokenType.COMMA)
        b_name = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)
        b_idx = self.parse_expression()
        self.expect(TokenType.COMMA)
        b_idx2 = self.parse_expression()
        self.expect(TokenType.RPAREN)
        b_access = ast.ArrayAccess(name=b_name.value, index=b_idx, index2=b_idx2, line=b_name.line)
        return ast.MmaStatement(acc_var=acc_tok.value, a_access=a_access, b_access=b_access, line=tok.line)

    def parse_tile_store(self) -> ast.TileStoreStatement:
        """STORE C(row, col), ACC"""
        tok = self.expect(TokenType.STORE)
        name = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)
        idx = self.parse_expression()
        self.expect(TokenType.COMMA)
        idx2 = self.parse_expression()
        self.expect(TokenType.RPAREN)
        target = ast.ArrayAccess(name=name.value, index=idx, index2=idx2, line=name.line)
        self.expect(TokenType.COMMA)
        value_tok = self.expect(TokenType.IDENTIFIER)
        return ast.TileStoreStatement(target=target, value_var=value_tok.value, line=tok.line)

    def parse_read(self) -> ast.ReadStatement:
        tok = self.expect(TokenType.READ)
        variables: list[ast.Variable] = []
        var_tok = self.expect(TokenType.IDENTIFIER)
        variables.append(ast.Variable(name=var_tok.value, line=var_tok.line))
        while self.match(TokenType.COMMA):
            var_tok = self.expect(TokenType.IDENTIFIER)
            variables.append(ast.Variable(name=var_tok.value, line=var_tok.line))
        return ast.ReadStatement(variables=variables, line=tok.line)

    # --- Expression parsing (precedence climbing) ---

    def parse_expression(self) -> ast.Expression:
        return self.parse_or()

    def parse_or(self) -> ast.Expression:
        left = self.parse_and()
        while self.at(TokenType.OR):
            op = self.advance()
            right = self.parse_and()
            left = ast.BinaryOp(op="OR", left=left, right=right, line=op.line)
        return left

    def parse_and(self) -> ast.Expression:
        left = self.parse_not()
        while self.at(TokenType.AND):
            op = self.advance()
            right = self.parse_not()
            left = ast.BinaryOp(op="AND", left=left, right=right, line=op.line)
        return left

    def parse_not(self) -> ast.Expression:
        if self.at(TokenType.NOT):
            op = self.advance()
            operand = self.parse_not()
            return ast.UnaryOp(op="NOT", operand=operand, line=op.line)
        return self.parse_comparison()

    def parse_comparison(self) -> ast.Expression:
        left = self.parse_addition()
        cmp_types = (TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE)
        if self.at(*cmp_types):
            op = self.advance()
            right = self.parse_addition()
            left = ast.BinaryOp(op=op.value, left=left, right=right, line=op.line)
        return left

    def parse_addition(self) -> ast.Expression:
        left = self.parse_multiplication()
        while self.at(TokenType.PLUS, TokenType.MINUS):
            op = self.advance()
            right = self.parse_multiplication()
            left = ast.BinaryOp(op=op.value, left=left, right=right, line=op.line)
        return left

    def parse_multiplication(self) -> ast.Expression:
        left = self.parse_power()
        while self.at(TokenType.STAR, TokenType.SLASH, TokenType.MOD):
            op = self.advance()
            op_str = op.value if op.type != TokenType.MOD else "MOD"
            right = self.parse_power()
            left = ast.BinaryOp(op=op_str, left=left, right=right, line=op.line)
        return left

    def parse_power(self) -> ast.Expression:
        left = self.parse_unary()
        if self.at(TokenType.CARET):
            op = self.advance()
            right = self.parse_power()  # Right-associative
            left = ast.BinaryOp(op="^", left=left, right=right, line=op.line)
        return left

    def parse_unary(self) -> ast.Expression:
        if self.at(TokenType.MINUS):
            op = self.advance()
            operand = self.parse_unary()
            return ast.UnaryOp(op="-", operand=operand, line=op.line)
        return self.parse_primary()

    def parse_primary(self) -> ast.Expression:
        tok = self.peek()

        if tok.type == TokenType.INTEGER:
            self.advance()
            return ast.NumberLiteral(value=int(tok.value), line=tok.line)

        if tok.type == TokenType.FLOAT:
            self.advance()
            return ast.NumberLiteral(value=float(tok.value), line=tok.line)

        if tok.type == TokenType.STRING:
            self.advance()
            return ast.StringLiteral(value=tok.value, line=tok.line)

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        if tok.type == TokenType.BID:
            self.advance()
            return ast.Variable(name="BID", line=tok.line)

        if tok.type == TokenType.IDENTIFIER:
            name = tok.value
            upper = name.upper()
            if upper in BUILTIN_FUNCTIONS:
                self.advance()
                self.expect(TokenType.LPAREN)
                arg = self.parse_expression()
                self.expect(TokenType.RPAREN)
                return ast.FunctionCall(name=upper, arg=arg, line=tok.line)
            self.advance()
            if self.at(TokenType.LPAREN):
                self.advance()
                idx = self.parse_expression()
                idx2 = None
                if self.match(TokenType.COMMA):
                    idx2 = self.parse_expression()
                self.expect(TokenType.RPAREN)
                return ast.ArrayAccess(name=name, index=idx, index2=idx2, line=tok.line)
            return ast.Variable(name=name, line=tok.line)

        raise ParseError(f"Expected expression, got {tok.type.name}", tok)


def parse(tokens: list[Token]) -> ast.Program:
    """Parse a list of tokens into an AST."""
    return Parser(tokens).parse()
