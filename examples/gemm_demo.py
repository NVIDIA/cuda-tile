#!/usr/bin/env python3
"""End-to-end GEMM: compile BASIC → cubin, launch on GPU, verify results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basic_tile.lexer import lex
from basic_tile.parser import parse
from basic_tile.analyzer import analyze
from basic_tile.bytecode_backend import BytecodeBackend
from basic_tile.gpu_runner import launch_kernel


def main():
    source_path = Path(__file__).parent / "gemm.bas"
    source = source_path.read_text()

    print(f"[1/4] Lexing & parsing {source_path.name} ...", flush=True)
    tokens = lex(source)
    program = parse(tokens)

    print("[2/4] Analyzing ...", flush=True)
    analyzed = analyze(program)

    print("[3/4] Compiling to cubin (bytecode backend) ...", flush=True)
    backend = BytecodeBackend(analyzed, num_ctas=2)
    cubin_path = backend.compile_to_cubin()
    meta = backend._array_kernel_meta
    M, N, K = meta["M"], meta["N"], meta["K"]
    tm, tn, tk = meta["tm"], meta["tn"], meta["tk"]
    print(f"       M={M}, N={N}, K={K}, "
          f"tiles=({tm}x{tn}x{tk}), "
          f"grid_size={meta['grid_size']}")

    print("[4/4] Launching kernel on GPU ...", flush=True)
    import random
    random.seed(42)
    h_a = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    h_b = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    a_name, b_name, c_name = meta["all_arrays"]
    results = launch_kernel(
        cubin_path=cubin_path,
        inputs={a_name: h_a, b_name: h_b},
        outputs=[c_name],
        param_order=meta["all_arrays"],
        sizes={a_name: M * K, b_name: K * N, c_name: M * N},
        grid_size=meta["grid_size"],
    )
    h_c = results[c_name]

    # Verify against naive CPU matmul
    expected = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += h_a[i * K + k] * h_b[k * N + j]
            expected[i * N + j] = s

    max_diff = max(abs(h_c[i] - expected[i]) for i in range(M * N))

    print(f"\nResults (showing 5 samples of {M}x{N} = {M*N} elements):")
    samples = [0, 1, M * N // 2, M * N // 2 + 1, M * N - 1]
    for idx in samples:
        row, col = divmod(idx, N)
        print(f"  C[{row},{col}] = {h_c[idx]:10.4f}  (expected {expected[idx]:.4f})")

    tol = K * 1e-5
    if max_diff < tol:
        print(f"\nVERIFICATION PASSED  (max_diff={max_diff:.6f}, tol={tol:.6f})")
    else:
        print(f"\nVERIFICATION FAILED  (max_diff={max_diff:.6f}, tol={tol:.6f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
