#!/usr/bin/env python3
"""End-to-end vector add: compile BASIC → cubin, launch on GPU, verify results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutile_basic._lexer import lex
from cutile_basic._parser import parse
from cutile_basic._analyzer import analyze
from cutile_basic.bytecode_backend import BytecodeBackend
from cutile_basic.gpu_runner import launch_kernel

N = 1024


def main():
    source_path = Path(__file__).parent / "vector_add.bas"
    source = source_path.read_text()

    print(f"[1/4] Lexing & parsing {source_path.name} ...", flush=True)
    tokens = lex(source)
    program = parse(tokens)

    print("[2/4] Analyzing ...", flush=True)
    analyzed = analyze(program)

    print("[3/4] Compiling to cubin (bytecode backend) ...", flush=True)
    backend = BytecodeBackend(analyzed, array_size=N)

    if "--dump-ir" in sys.argv:
        print("\n" + backend.dump_tileir() + "\n")

    cubin_path = backend.compile_to_cubin()
    meta = backend._array_kernel_meta
    print(f"       Arrays: {meta['all_arrays']}, "
          f"tile_size={meta['tile_size']}, "
          f"grid_size={meta['grid_size']}")

    print("[4/4] Launching kernel on GPU ...", flush=True)
    # Prepare input data: A = [0, 1, 2, ..., 1023], B = [0, 2, 4, ..., 2046]
    h_a = [float(i) for i in range(N)]
    h_b = [float(i) * 2.0 for i in range(N)]

    results = launch_kernel(
        cubin_path=cubin_path,
        inputs={"A": h_a, "B": h_b},
        outputs=["C"],
        param_order=meta["all_arrays"],
        sizes={name: meta["array_size"] for name in meta["all_arrays"]},
        grid_size=meta["grid_size"],
    )

    h_c = results["C"]

    # Verify
    expected = [a + b for a, b in zip(h_a, h_b)]
    max_diff = max(abs(h_c[i] - expected[i]) for i in range(N))

    print(f"\nResults (showing 5 samples of {N}):")
    for i in [0, 1, 511, 512, 1023]:
        print(f"  C[{i:4d}] = {h_c[i]:10.1f}  (expected {expected[i]:.1f})")

    if max_diff < 0.01:
        print(f"\nVERIFICATION PASSED  (max_diff={max_diff:.6f}, {N} elements)")
    else:
        print(f"\nVERIFICATION FAILED  (max_diff={max_diff:.6f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
