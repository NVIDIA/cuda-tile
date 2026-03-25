Getting Started
===============

Prerequisites
-------------

**Hardware:**

- NVIDIA GPU with compute capability 8.x, 10.x, 11.x, or 12.x

**Software:**

- NVIDIA Driver r580 or later
- Python 3.10+
- CUDA Toolkit 13.1 or later
- ``cuda-tile[tileiras]``, ``cuda-python``, ``cuda-core``, ``cupy-cuda13x``
  (all installed via ``pip install -r requirements.txt``)
- ``cuda-tile-translate`` (for the MLIR compilation path only)

Installation
------------

Clone the repository and install dependencies:

.. code-block:: bash

   $ git clone <repo-url> cutile-basic
   $ cd cutile-basic
   $ pip install -r requirements.txt

This installs ``cuda-tile[tileiras]``, ``cuda-python``, ``cuda-core``, and
``cupy-cuda13x``. The ``cutile_basic`` package itself is used directly from the
source tree (no separate install step).

Quick Start
-----------

Write a BASIC program (``hello.bas``):

.. code-block:: basic

   10 REM Hello World in BASIC
   20 PRINT "Hello, World!"
   30 LET X = 42.0
   40 LET Y = X * 2.0
   50 PRINT "X = "; X
   60 PRINT "Y = "; Y
   70 END

Compile it to CUDA Tile IR MLIR:

.. code-block:: bash

   $ python -m cutile_basic.cli hello.bas

This prints the generated MLIR to stdout. To write it to a file:

.. code-block:: bash

   $ python -m cutile_basic.cli hello.bas -o hello.mlir

Compile and run on a GPU (requires ``cuda-tile-translate`` and ``tileiras``):

.. code-block:: bash

   $ python -m cutile_basic.cli hello.bas --run

Using the Python API
--------------------

Generate MLIR text:

.. code-block:: python

   from cutile_basic import compile_basic_to_mlir

   source = """
   10 DIM A(128), B(128), C(128)
   20 INPUT A(), B()
   30 LET C(BID) = A(BID) + B(BID)
   40 OUTPUT C
   50 END
   """

   mlir = compile_basic_to_mlir(source)
   print(mlir)

Or compile directly to a ``.cubin`` via the bytecode backend:

.. code-block:: python

   from cutile_basic import compile_basic_to_cubin

   result = compile_basic_to_cubin(source, array_size=1024)
   print(result.cubin_path)   # path to the compiled .cubin
   print(result.meta)          # kernel metadata (arrays, grid size, etc.)

Two Compilation Paths
---------------------

cutile-basic supports two paths from BASIC source to GPU execution:

**MLIR Path**
   Source is compiled to CUDA Tile IR MLIR text, then passed through
   ``cuda-tile-translate`` (MLIR to ``.tilebc``) and ``tileiras``
   (``.tilebc`` to ``.cubin``), and finally launched via the CUDA driver API.

**Bytecode Path**
   Source is compiled directly to cuTile bytecode using the ``cuda.tile._bytecode``
   Python APIs, assembled with ``tileiras``, and launched on the GPU. This path
   bypasses MLIR text entirely and is used by the demo scripts.
