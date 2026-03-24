# basic_tile

A BASIC to CUDA Tile IR transpiler. Write GPU kernels in BASIC, compile them to
CUDA Tile IR (MLIR), and launch them on NVIDIA GPUs.

**[Documentation](https://basic-tile-a22467.gitlab-master-pages.nvidia.com/)**

## Overview

basic_tile extends classic BASIC with tile-based GPU operations (`TILE`, `MMA`,
`STORE`, `OUTPUT`, `BID`), enabling concise expression of GPU kernels such as
vector addition and matrix multiplication.

Two compilation paths are supported:

- **MLIR path** -- emits CUDA Tile IR MLIR text, compiled via `cuda-tile-translate`
  and `tileiras` to `.cubin`
- **Bytecode path** -- compiles directly to cuTile bytecode via `cuda.tile` Python
  APIs, bypassing MLIR text

## Quick Start

Generate MLIR from a BASIC program:

```bash
python -m basic_tile.cli examples/vector_add.bas
```

Compile and run on a GPU:

```bash
python -m basic_tile.cli examples/vector_add.bas --run
```

Or use the Python API:

```python
from basic_tile import compile_basic_to_mlir

source = """
10 DIM A(128), B(128), C(128)
20 INPUT A(), B()
30 LET C(BID) = A(BID) + B(BID)
40 OUTPUT C
50 END
"""

print(compile_basic_to_mlir(source))
```

## Examples

| Program | Description |
|---|---|
| `examples/hello.bas` | Variables, arithmetic, conditionals, loops |
| `examples/fibonacci.bas` | Fibonacci sequence |
| `examples/array_sum.bas` | DATA/READ array summation |
| `examples/vector_add.bas` | GPU vector addition using BID |
| `examples/gemm.bas` | Tiled GPU matrix multiply (TILE, MMA, STORE) |

End-to-end GPU demos:

```bash
python examples/vector_add_demo.py
python examples/gemm_demo.py
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (for kernel execution)
- `cuda-tile-translate` and `tileiras` (MLIR path)
- `cuda.tile` Python package (bytecode path)

## License

Copyright NVIDIA Corporation. All rights reserved.
