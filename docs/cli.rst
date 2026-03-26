CLI Reference
=============

cutile-basic provides a command-line interface for compiling BASIC source files.

Usage
-----

.. code-block:: bash

   $ python -m cutile_basic.cli [options] <input.bas>

Arguments
---------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``input``
     - Input ``.bas`` file (required)

Options
-------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``-o, --output FILE``
     - Output file path. Defaults to stdout for MLIR text.
   * - ``--dump-tokens``
     - Dump the token stream and exit.
   * - ``--dump-ast``
     - Dump the parsed AST and exit.
   * - ``--dump-analyzed``
     - Dump the analyzed program (symbols, types, metadata) and exit.
   * - ``--compile-mlir``
     - Compile to CUDA Tile IR MLIR text (default).
   * - ``--compile-cubin``
     - Compile to ``.cubin`` via the bytecode backend.
   * - ``--gpu-arch ARCH``
     - GPU architecture for ``--compile-cubin`` (e.g. ``sm_120``). Default: auto-detect.

Examples
--------

Generate MLIR to stdout:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas

Write MLIR to a file:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas -o vector_add.mlir

Inspect the token stream:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/hello.bas --dump-tokens

Inspect the AST:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/hello.bas --dump-ast

Compile to cubin:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas --compile-cubin -o vector_add.cubin
