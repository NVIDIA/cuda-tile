# cutile-basic

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/cutile_basic_icon__padded__white_text.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/cutile_basic_icon__padded__black_text.png">
    <img alt="cutile-basic" src="docs/images/cutile_basic_icon__padded__black_text.png" width="400">
  </picture>
</div>

A BASIC to CUDA Tile IR compiler. Write GPU kernels in BASIC, compile them to
`.cubin` files via cuTile bytecode and `tileiras`, and launch them on NVIDIA GPUs.

**[Documentation](https://basic-tile-a22467.gitlab-master-pages.nvidia.com/)**

## Overview

cutile-basic extends classic BASIC with tile-based GPU operations (`TILE`,
`OUTPUT`, `BID`) and built-in functions like `MMA`, enabling concise expression
of GPU kernels such as vector addition and matrix multiplication.

## Quick Start

Compile a BASIC program to a `.cubin`:

```bash
python -m cutile_basic.cli examples/vector_add.bas -o vector_add.cubin
```

Run an end-to-end GPU demo:

```bash
python examples/vector_add.py
```

Or use the Python API:

```python
from cutile_basic import compile_basic_to_cubin

source = """
10 INPUT N, A(), B()
20 DIM A(N), B(N), C(N)
30 TILE A(128), B(128), C(128)
40 LET C(BID) = A(BID) + B(BID)
50 OUTPUT C
60 END
"""

result = compile_basic_to_cubin(source)
print(result.cubin_path)   # path to the compiled .cubin
print(result.meta)         # kernel metadata (arrays, tile shapes, etc.)
```

## Examples

| Program | Description |
|---|---|
| `examples/hello.bas` | Variables, arithmetic, conditionals, loops |
| `examples/vector_add.bas` | GPU vector addition using BID |
| `examples/gemm.bas` | Tiled GPU matrix multiply (MMA) |

End-to-end GPU demos:

```bash
python examples/vector_add.py
python examples/gemm.py
python examples/hello.py
```

## Installation

```bash
pip install git+https://github.com/nvidia/cuda-tile.git@basic-experimental
```

## Prerequisites

Hardware:
- NVIDIA GPU with Compute Capability 8.x (Ampere), 10.x, 11.x, or 12.x (Blackwell)

Software:
- NVIDIA Driver r580 or later
- Python 3.10 or later
- CUDA Toolkit 13.1 or later
- ``cuda-tile[tileiras]``, ``cuda-python``, ``cuda-core``, ``cupy-cuda13x``

## License

Apache License 2.0 with LLVM Exceptions. See [LICENSE](LICENSE.TXT) for details.
