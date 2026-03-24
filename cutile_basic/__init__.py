"""BASIC to CUDA Tile IR MLIR transpiler."""

from .lexer import lex
from .parser import parse
from .analyzer import analyze
from .codegen import generate


def compile_basic_to_mlir(source: str) -> str:
    """Compile BASIC source code to CUDA Tile IR MLIR text."""
    tokens = lex(source)
    program = parse(tokens)
    analyzed = analyze(program)
    return generate(analyzed)
