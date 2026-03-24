CLI Reference
=============

basic_tile provides a command-line interface for compiling BASIC source files.

Usage
-----

.. code-block:: bash

   $ python -m basic_tile.cli [options] <input.bas>

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
     - Output file path. For MLIR output, writes ``.mlir``. For ``--compile``,
       writes ``.cubin``. Defaults to stdout for MLIR.
   * - ``--dump-tokens``
     - Dump the token stream and exit.
   * - ``--dump-ast``
     - Dump the parsed AST and exit.
   * - ``--dump-analyzed``
     - Dump the analyzed program (symbols, types, metadata) and exit.
   * - ``--compile``
     - Compile through to ``.cubin`` but do not launch on GPU.
   * - ``--run``
     - Compile and launch the kernel on the GPU.
   * - ``--gpu-arch ARCH``
     - GPU architecture override (e.g. ``sm_120``). Default: auto-detect.
   * - ``--cuda-tile-translate PATH``
     - Path to the ``cuda-tile-translate`` binary. Default: searches standard
       locations.

Examples
--------

Generate MLIR to stdout:

.. code-block:: bash

   $ python -m basic_tile.cli examples/vector_add.bas

Write MLIR to a file:

.. code-block:: bash

   $ python -m basic_tile.cli examples/vector_add.bas -o vector_add.mlir

Inspect the token stream:

.. code-block:: bash

   $ python -m basic_tile.cli examples/hello.bas --dump-tokens

Inspect the AST:

.. code-block:: bash

   $ python -m basic_tile.cli examples/hello.bas --dump-ast

Compile to cubin:

.. code-block:: bash

   $ python -m basic_tile.cli examples/vector_add.bas --compile -o vector_add.cubin

Compile and run on GPU:

.. code-block:: bash

   $ python -m basic_tile.cli examples/vector_add.bas --run
