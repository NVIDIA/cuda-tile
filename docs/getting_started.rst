Getting Started
===============

Prerequisites
-------------

- Python 3.10+
- NVIDIA GPU with CUDA support (for running kernels)
- ``cuda-tile-translate`` and ``tileiras`` tools (for the MLIR compilation path)
- ``cuda.tile`` Python package (for the bytecode backend path)

Installation
------------

Clone the repository and ensure the ``basic_tile`` package is on your Python path:

.. code-block:: bash

   $ git clone <repo-url> basic_tile
   $ cd basic_tile

No ``pip install`` step is needed -- the package is used directly from the source
tree.

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

   $ python -m basic_tile.cli hello.bas

This prints the generated MLIR to stdout. To write it to a file:

.. code-block:: bash

   $ python -m basic_tile.cli hello.bas -o hello.mlir

Compile and run on a GPU (requires ``cuda-tile-translate`` and ``tileiras``):

.. code-block:: bash

   $ python -m basic_tile.cli hello.bas --run

Using the Python API
--------------------

.. code-block:: python

   from basic_tile import compile_basic_to_mlir

   source = """
   10 DIM A(128), B(128), C(128)
   20 INPUT A(), B()
   30 LET C(BID) = A(BID) + B(BID)
   40 OUTPUT C
   50 END
   """

   mlir = compile_basic_to_mlir(source)
   print(mlir)

Two Compilation Paths
---------------------

basic_tile supports two paths from BASIC source to GPU execution:

**MLIR Path**
   Source is compiled to CUDA Tile IR MLIR text, then passed through
   ``cuda-tile-translate`` (MLIR to ``.tilebc``) and ``tileiras``
   (``.tilebc`` to ``.cubin``), and finally launched via the CUDA driver API.

**Bytecode Path**
   Source is compiled directly to cuTile bytecode using the ``cuda.tile._bytecode``
   Python APIs, assembled with ``tileiras``, and launched on the GPU. This path
   bypasses MLIR text entirely and is used by the demo scripts.
