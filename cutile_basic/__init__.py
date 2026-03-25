"""BASIC to CUDA Tile IR MLIR transpiler."""

from ._lexer import lex
from ._parser import parse
from ._analyzer import analyze
from ._codegen import generate

__all__ = ["compile_basic_to_mlir"]


def compile_basic_to_mlir(source: str) -> str:
    """Compile BASIC source code to CUDA Tile IR MLIR text."""
    tokens = lex(source)
    program = parse(tokens)
    analyzed = analyze(program)
    return generate(analyzed)
