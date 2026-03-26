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

Compile it to CUDA Tile IR textual output:

.. code-block:: bash

   $ python -m cutile_basic.cli hello.bas

This prints the generated textual output to stdout. To write it to a file:

.. code-block:: bash

   $ python -m cutile_basic.cli hello.bas -o hello.mlir

Compile to a ``.cubin``:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas --compile-cubin -o vector_add.cubin

Run a GPU demo end-to-end:

.. code-block:: bash

   $ python examples/vector_add.py

Using the Python API
--------------------

Generate textual output:

.. code-block:: python

   from cutile_basic import compile_basic_to_textual

   source = """
   10 DIM A(128), B(128), C(128)
   20 INPUT A(), B()
   30 LET C(BID) = A(BID) + B(BID)
   40 OUTPUT C
   50 END
   """

   text = compile_basic_to_textual(source)
   print(text)

Or compile directly to a ``.cubin`` via the bytecode backend:

.. code-block:: python

   from cutile_basic import compile_basic_to_cubin

   result = compile_basic_to_cubin(source, array_size=1024)
   print(result.cubin_path)   # path to the compiled .cubin
   print(result.meta)          # kernel metadata (arrays, grid size, etc.)

Two Output Modes
----------------

cutile-basic supports two output modes:

**Textual**
   Source is compiled to human-readable CUDA Tile IR text via the textual
   backend. This is the default CLI output and is useful for inspection and
   debugging.

**Cubin (via Bytecode)**
   Source is compiled directly to cuTile bytecode using the ``cuda.tile._bytecode``
   Python APIs, assembled into a ``.cubin`` with ``tileiras``, and can then be
   launched on the GPU from a Python host script. This is used by ``--compile``
   on the CLI and by the demo scripts.
