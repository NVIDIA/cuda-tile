# cutile-basic

![cutile-basic](graphic.jpg)

A BASIC to CUDA Tile IR compiler. Write GPU kernels in BASIC, compile them to
CUDA Tile IR (MLIR), and launch them on NVIDIA GPUs.

**[Documentation](https://basic-tile-a22467.gitlab-master-pages.nvidia.com/)**

## Overview

cutile-basic extends classic BASIC with tile-based GPU operations (`TILE`, `MMA`,
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
python -m cutile_basic.cli examples/vector_add.bas
```

Compile and run on a GPU:

```bash
python -m cutile_basic.cli examples/vector_add.bas --run
```

Or use the Python API to compile straight to a `.cubin`:

```python
from cutile_basic import compile_basic_to_cubin

source = """
10 DIM A(128), B(128), C(128)
20 INPUT A(), B()
30 LET C(BID) = A(BID) + B(BID)
40 OUTPUT C
50 END
"""

result = compile_basic_to_cubin(source)
print(result.cubin_path)   # path to the compiled .cubin
print(result.meta)          # kernel metadata (arrays, grid size, etc.)
```

To emit MLIR text instead:

```python
from cutile_basic import compile_basic_to_mlir

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
python examples/gpu_demo.py examples/vector_add.bas  # general-purpose launcher
```

## Installation

```bash
git clone <repo-url> cutile-basic
cd cutile-basic
pip install -r requirements.txt
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with compute capability 10.x or 12.x (Blackwell architecture
  or later) and driver r580+
- CUDA Toolkit 13.1 or later
- `cuda-tile[tileiras]` (bytecode backend + `tileiras` assembler)
- `cuda-python`, `cuda-core`, `cupy-cuda13x` (GPU launch and memory management)
- `cuda-tile-translate` (MLIR compilation path only)

## License

Copyright NVIDIA Corporation. All rights reserved.
