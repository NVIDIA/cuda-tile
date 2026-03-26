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
     - Write the compiled ``.cubin`` to ``FILE``. If omitted, the path to the
       ``.cubin`` is printed to stdout.
   * - ``--dump-tokens``
     - Dump the token stream and exit.
   * - ``--dump-ast``
     - Dump the parsed AST and exit.
   * - ``--dump-analyzed``
     - Dump the analyzed program (symbols, types, metadata) and exit.
   * - ``--gpu-arch ARCH``
     - GPU architecture for compilation (e.g. ``sm_120``). Default: auto-detect.

Examples
--------

Compile and print the path to the ``.cubin``:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas

Write the ``.cubin`` to a file:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas -o vector_add.cubin

Inspect the token stream:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/hello.bas --dump-tokens

Inspect the AST:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/hello.bas --dump-ast

Compile to cubin with an explicit GPU architecture:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas --gpu-arch sm_120 -o vector_add.cubin
