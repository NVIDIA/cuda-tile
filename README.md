# cutile-basic

![cutile-basic](graphic.jpg)

A BASIC to CUDA Tile IR compiler. Write GPU kernels in BASIC, compile them to
CUDA Tile IR (MLIR), and launch them on NVIDIA GPUs.

**[Documentation](https://basic-tile-a22467.gitlab-master-pages.nvidia.com/)**

## Overview

cutile-basic extends classic BASIC with tile-based GPU operations (`TILE`, `MMA`,
`STORE`, `OUTPUT`, `BID`), enabling concise expression of GPU kernels such as
vector addition and matrix multiplication.

Two output modes are supported:

- **MLIR text** -- emits human-readable CUDA Tile IR MLIR
- **Cubin** -- compiles directly to `.cubin` via cuTile bytecode and `tileiras`

## Quick Start

Generate MLIR from a BASIC program:

```bash
python -m cutile_basic.cli examples/vector_add.bas
```

Compile to a `.cubin`:

```bash
python -m cutile_basic.cli examples/vector_add.bas --compile-cubin -o vector_add.cubin
```

Run an end-to-end GPU demo:

```bash
python examples/vector_add.py
```

Or use the Python API:

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
from cutile_basic import compile_basic_to_textual

print(compile_basic_to_textual(source))
```

## Examples

| Program | Description |
|---|---|
| `examples/hello.bas` | Variables, arithmetic, conditionals, loops |
| `examples/vector_add.bas` | GPU vector addition using BID |
| `examples/gemm.bas` | Tiled GPU matrix multiply (TILE, MMA, STORE) |

End-to-end GPU demos:

```bash
python examples/vector_add.py
python examples/gemm.py
python examples/hello.py
```

## Installation

```bash
git clone <repo-url> cutile-basic
cd cutile-basic
pip install -r requirements.txt
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with compute capability 8.x, 10.x, 11.x, or 12.x
- NVIDIA Driver r580 or later
- CUDA Toolkit 13.1 or later
- `cuda-tile[tileiras]` (bytecode backend + `tileiras` assembler)
- `cuda-python`, `cuda-core`, `cupy-cuda13x` (GPU launch and memory management)

## License

Copyright NVIDIA Corporation. All rights reserved.
