#!/usr/bin/env python3
"""End-to-end vector add: compile BASIC → cubin, launch on GPU, verify results."""

import sys
from pathlib import Path

import numpy as np
import cupy as cp
from cuda.core import Device, LaunchConfig, ObjectCode, launch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutile_basic import compile_basic_to_cubin

N = 1024


def main():
    source = (Path(__file__).parent / "vector_add.bas").read_text()

    print("[1/2] Compiling to cubin ...", flush=True)
    result = compile_basic_to_cubin(source)
    meta = result.meta
    tile_shapes = meta.get("tile_shapes", {})
    tile_c = tile_shapes["C"]
    grid_size = N // tile_c[0]
    print(f"      N={N}, "
          f"tile_shapes={tile_shapes}, "
          f"grid_size={grid_size}")

    print("[2/2] Launching kernel on GPU ...", flush=True)
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    kernel = ObjectCode.from_cubin(result.cubin_path).get_kernel("main")

    d_a = cp.arange(N, dtype=cp.float32)
    d_b = cp.arange(N, dtype=cp.float32) * 2.0
    d_c = cp.zeros(N, dtype=cp.float32)

    config = LaunchConfig(grid=(grid_size, 1, 1), block=(1, 1, 1))
    launch(stream, config, kernel,
           np.int32(N), d_a.data.ptr, d_b.data.ptr, d_c.data.ptr)
    stream.sync()

    expected = d_a + d_b
    max_diff = float(cp.max(cp.abs(d_c - expected)))

    print(f"\nResults (showing 5 samples of {N}):")
    for i in [0, 1, 511, 512, 1023]:
        print(f"  C[{i:4d}] = {float(d_c[i]):10.1f}  (expected {float(expected[i]):.1f})")

    if max_diff < 0.01:
        print(f"\nVERIFICATION PASSED  (max_diff={max_diff:.6f}, {N} elements)")
    else:
        print(f"\nVERIFICATION FAILED  (max_diff={max_diff:.6f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
