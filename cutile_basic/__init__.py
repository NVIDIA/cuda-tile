"""BASIC to CUDA Tile IR MLIR transpiler."""

from __future__ import annotations

from ._lexer import lex
from ._parser import parse
from ._analyzer import analyze
from ._codegen import generate

__all__ = ["compile_basic_to_mlir", "compile_basic_to_cubin", "CompilationResult"]


def compile_basic_to_mlir(source: str) -> str:
    """Compile BASIC source code to CUDA Tile IR MLIR text."""
    tokens = lex(source)
    program = parse(tokens)
    analyzed = analyze(program)
    return generate(analyzed)


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
    gpu_arch: str = "sm_120",
    array_size: int | None = None,
    num_ctas: int | None = None,
) -> CompilationResult:
    """Compile BASIC source to a .cubin via the bytecode backend.

    Args:
        source: BASIC source code.
        gpu_arch: Target GPU architecture (default: ``"sm_120"``).
        array_size: Total elements per array; ``None`` infers from DIM.
        num_ctas: CTAs-per-CGA optimisation hint; ``None`` disables.

    Returns:
        A :class:`CompilationResult` with ``cubin_path`` and kernel ``meta``.
    """
    from .bytecode_backend import BytecodeBackend

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
