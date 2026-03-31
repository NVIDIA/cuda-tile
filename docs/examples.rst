Examples
========

cutile-basic ships with example programs in the ``examples/`` directory
demonstrating both standard BASIC and GPU tile operations.

Hello World (``examples/hello.bas``)
-------------------------------------

A classic BASIC program showing variables, arithmetic, conditionals, and loops.

.. literalinclude:: ../examples/hello.bas
   :language: basic

Vector Add (``examples/vector_add.bas``)
----------------------------------------

A GPU kernel that computes ``C = A + B`` element-wise using the block ID.

.. literalinclude:: ../examples/vector_add.bas
   :language: basic

The ``INPUT`` statement declares ``A`` and ``B`` as kernel parameters. ``BID``
maps to the CUDA block index, and ``OUTPUT`` marks ``C`` for host readback.

Run it end-to-end with the Python demo script:

.. code-block:: bash

   python examples/vector_add.py

This script lexes, parses, analyzes, compiles to cubin via the bytecode backend,
launches the kernel with test data, and verifies the result.

GEMM (``examples/gemm.bas``)
----------------------------

A tiled matrix multiply: ``C(M,N) = A(M,K) * B(K,N)``.

.. literalinclude:: ../examples/gemm.bas
   :language: basic

``DIM`` declares array dimensions, ``TILE`` declares the tile/partition shape for
each variable. ``LET ACC = 0.0`` initializes the accumulator tile, ``MMA`` performs
matrix multiply-accumulate, and ``LET C(...) = ACC`` writes the result tile.

Run it with:

.. code-block:: bash

   python examples/gemm.py

Python Demo Scripts
-------------------

Three demo scripts in ``examples/`` show end-to-end GPU execution:

``vector_add.py``
   Compiles ``vector_add.bas``, launches with 1024-element arrays, verifies
   ``C[i] = A[i] + B[i]``.

``gemm.py``
   Compiles ``gemm.bas``, launches a 512x512 GEMM, verifies against a CuPy
   reference (``d_a @ d_b``).

``hello.py``
   Compiles ``hello.bas`` to a cubin via the bytecode backend and launches it
   as a single-block kernel. Because ``hello.bas`` has no GPU extensions, this
   serves as a minimal smoke test of the compilation and launch pipeline.

   .. code-block:: bash

      python examples/hello.py
