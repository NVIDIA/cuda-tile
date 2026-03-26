Examples
========

cutile-basic ships with example programs in the ``examples/`` directory
demonstrating both standard BASIC and GPU tile operations.

Hello World (``examples/hello.bas``)
-------------------------------------

A classic BASIC program showing variables, arithmetic, conditionals, and loops.

.. code-block:: basic

   10 REM Hello World in BASIC
   20 PRINT "Hello, World!"
   30 LET X = 42.0
   40 LET Y = X * 2.0
   50 PRINT "X = "; X
   60 PRINT "Y = "; Y
   70 IF Y > 80 THEN
   80   PRINT "Y is large"
   90 ELSE
   100  PRINT "Y is small"
   110 ENDIF
   120 FOR I = 1 TO 5
   130   PRINT "I = "; I
   140 NEXT I
   150 END

Vector Add (``examples/vector_add.bas``)
----------------------------------------

A GPU kernel that computes ``C = A + B`` element-wise using the block ID.

.. code-block:: basic

   10 REM Vector Add: C = A + B
   20 DIM A(128), B(128), C(128)
   30 INPUT A(), B()
   40 LET C(BID) = A(BID) + B(BID)
   50 OUTPUT C
   60 END

The ``INPUT`` statement declares ``A`` and ``B`` as kernel parameters. ``BID``
maps to the CUDA block index, and ``OUTPUT`` marks ``C`` for host readback.

Run it end-to-end with the Python demo script:

.. code-block:: bash

   $ python examples/vector_add.py

This script lexes, parses, analyzes, compiles to cubin via the bytecode backend,
launches the kernel with test data, and verifies the result.

GEMM (``examples/gemm.bas``)
----------------------------

A tiled matrix multiply: ``C(M,N) = A(M,K) * B(K,N)``.

.. code-block:: basic

   10 REM GEMM: C(M,N) = A(M,K) * B(K,N)
   20 DIM A(512, 512), B(512, 512), C(512, 512)
   30 TILE A(128, 32), B(32, 128), C(128, 128), ACC(128, 128)
   40 INPUT A(), B()
   50 LET TILEM = BID / 4
   60 LET TILEN = BID MOD 4
   65 LET ACC = 0.0
   70 FOR K = 0 TO 15
   80   MMA ACC, A(TILEM, K), B(K, TILEN)
   90 NEXT K
   100 STORE C(TILEM, TILEN), ACC
   110 OUTPUT C
   120 END

``DIM`` declares array dimensions, ``TILE`` declares the tile/partition shape for
each variable. ``LET ACC = 0.0`` initializes the accumulator tile, ``MMA`` performs
matrix multiply-accumulate, and ``STORE`` writes the result.

Run it with:

.. code-block:: bash

   $ python examples/gemm.py

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

      $ python examples/hello.py
