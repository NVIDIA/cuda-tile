#!/usr/bin/env python3
"""End-to-end demo: compile a BASIC program and run it on the GPU."""

import sys
from pathlib import Path

# Add project root to path so we can import basic_tile
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basic_tile.lexer import lex
from basic_tile.parser import parse
from basic_tile.analyzer import analyze
from basic_tile.bytecode_backend import BytecodeBackend
from basic_tile.gpu_runner import launch_kernel


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <program.bas>")
        sys.exit(1)

    source_path = sys.argv[1]
    source = Path(source_path).read_text()

    print(f"[1/4] Lexing & parsing {source_path} ...", flush=True)
    tokens = lex(source)
    program = parse(tokens)

    print("[2/4] Analyzing ...", flush=True)
    analyzed = analyze(program)

    print("[3/4] Compiling to cubin (bytecode backend) ...", flush=True)
    backend = BytecodeBackend(analyzed)
    cubin_path = backend.compile_to_cubin()

    print(f"[4/4] Launching kernel on GPU ...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    launch_kernel(cubin_path)
    print("[done] Kernel execution complete.")


if __name__ == "__main__":
    main()
