"""Lexer for BASIC source code."""

from .tokens import Token, TokenType, KEYWORDS

BUILTIN_FUNCTIONS = {"ABS", "SQR", "INT", "SIN", "COS", "TAN", "EXP", "LOG", "SGN"}


class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"Lex error at L{line}:{col}: {msg}")
        self.line = line
        self.col = col


def lex(source: str) -> list[Token]:
    """Tokenize BASIC source code into a list of tokens."""
    tokens: list[Token] = []
    lines = source.split("\n")

    for line_num, line_text in enumerate(lines, start=1):
        i = 0
        while i < len(line_text):
            ch = line_text[i]
            col = i + 1

            # Whitespace
            if ch in (" ", "\t"):
                i += 1
                continue

            # REM comment — rest of line
            if line_text[i:].upper().startswith("REM") and (
                i + 3 >= len(line_text) or not line_text[i + 3].isalnum()
            ):
                comment = line_text[i + 3:].strip()
                tokens.append(Token(TokenType.REM, comment, line_num, col))
                break

            # String literal
            if ch == '"':
                j = i + 1
                while j < len(line_text) and line_text[j] != '"':
                    j += 1
                if j >= len(line_text):
                    raise LexError("Unterminated string", line_num, col)
                tokens.append(Token(TokenType.STRING, line_text[i + 1 : j], line_num, col))
                i = j + 1
                continue

            # Number (integer or float)
            if ch.isdigit() or (ch == "." and i + 1 < len(line_text) and line_text[i + 1].isdigit()):
                j = i
                has_dot = False
                while j < len(line_text) and (line_text[j].isdigit() or line_text[j] == "."):
                    if line_text[j] == ".":
                        if has_dot:
                            break
                        has_dot = True
                    j += 1
                # Handle scientific notation
                if j < len(line_text) and line_text[j].upper() == "E":
                    j += 1
                    if j < len(line_text) and line_text[j] in "+-":
                        j += 1
                    while j < len(line_text) and line_text[j].isdigit():
                        j += 1
                    has_dot = True  # treat scientific notation as float
                num_str = line_text[i:j]
                if has_dot:
                    tokens.append(Token(TokenType.FLOAT, num_str, line_num, col))
                else:
                    tokens.append(Token(TokenType.INTEGER, num_str, line_num, col))
                i = j
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == "_":
                j = i
                while j < len(line_text) and (line_text[j].isalnum() or line_text[j] == "_"):
                    j += 1
                # Allow trailing $ or % for type suffixes
                if j < len(line_text) and line_text[j] in ("%", "$"):
                    j += 1
                word = line_text[i:j]
                upper = word.upper().rstrip("%$")
                if upper in KEYWORDS:
                    tokens.append(Token(KEYWORDS[upper], word, line_num, col))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, word, line_num, col))
                i = j
                continue

            # Two-character operators
            two = line_text[i : i + 2]
            if two == "<>":
                tokens.append(Token(TokenType.NEQ, "<>", line_num, col))
                i += 2
                continue
            if two == "<=":
                tokens.append(Token(TokenType.LE, "<=", line_num, col))
                i += 2
                continue
            if two == ">=":
                tokens.append(Token(TokenType.GE, ">=", line_num, col))
                i += 2
                continue

            # Single-character operators and delimiters
            single_map = {
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.STAR,
                "/": TokenType.SLASH,
                "^": TokenType.CARET,
                "=": TokenType.EQ,
                "<": TokenType.LT,
                ">": TokenType.GT,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                ",": TokenType.COMMA,
                ";": TokenType.SEMICOLON,
                ":": TokenType.COLON,
            }
            if ch in single_map:
                tokens.append(Token(single_map[ch], ch, line_num, col))
                i += 1
                continue

            raise LexError(f"Unexpected character: {ch!r}", line_num, col)

        # End of line
        tokens.append(Token(TokenType.NEWLINE, "\\n", line_num, len(line_text) + 1))

    # Replace trailing newline with EOF
    if tokens and tokens[-1].type == TokenType.NEWLINE:
        tokens[-1] = Token(TokenType.EOF, "", tokens[-1].line, tokens[-1].col)
    else:
        tokens.append(Token(TokenType.EOF, "", len(lines), 1))

    return tokens
