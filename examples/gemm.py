#!/usr/bin/env python3
"""End-to-end GEMM: compile BASIC → cubin, launch on GPU, verify results."""

import sys
from pathlib import Path

import cupy as cp
from cuda.core import Device, LaunchConfig, ObjectCode, launch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutile_basic import compile_basic_to_cubin, detect_gpu_arch


def main():
    source = (Path(__file__).parent / "gemm.bas").read_text()

    arch = detect_gpu_arch()
    sm_version = int(arch.removeprefix("sm_"))
    num_ctas = 2 if sm_version >= 100 else None

    print("[1/2] Compiling to cubin ...", flush=True)
    result = compile_basic_to_cubin(source, num_ctas=num_ctas)
    meta = result.meta
    M, N, K = meta["M"], meta["N"], meta["K"]
    tm, tn, tk = meta["tm"], meta["tn"], meta["tk"]
    print(f"      M={M}, N={N}, K={K}, "
          f"tiles=({tm}x{tn}x{tk}), "
          f"grid_size={meta['grid_size']}")

    print("[2/2] Launching kernel on GPU ...", flush=True)
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    kernel = ObjectCode.from_cubin(result.cubin_path).get_kernel("main")

    cp.random.seed(42)
    d_a = cp.random.uniform(-1.0, 1.0, (M, K)).astype(cp.float32)
    d_b = cp.random.uniform(-1.0, 1.0, (K, N)).astype(cp.float32)
    d_c = cp.zeros((M, N), dtype=cp.float32)

    config = LaunchConfig(grid=(meta["grid_size"], 1, 1), block=(1, 1, 1))
    launch(stream, config, kernel, d_a.data.ptr, d_b.data.ptr, d_c.data.ptr)
    stream.sync()

    expected = d_a @ d_b
    max_diff = float(cp.max(cp.abs(d_c - expected)))

    print(f"\nResults (showing 5 samples of {M}x{N} = {M*N} elements):")
    for idx in [0, 1, M * N // 2, M * N // 2 + 1, M * N - 1]:
        row, col = divmod(idx, N)
        print(f"  C[{row},{col}] = {float(d_c[row, col]):10.4f}  "
              f"(expected {float(expected[row, col]):.4f})")

    tol = K * 1e-5
    if max_diff < tol:
        print(f"\nVERIFICATION PASSED  (max_diff={max_diff:.6f}, tol={tol:.6f})")
    else:
        print(f"\nVERIFICATION FAILED  (max_diff={max_diff:.6f}, tol={tol:.6f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
