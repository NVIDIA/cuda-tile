Examples
========

basic_tile ships with several example programs demonstrating both standard BASIC
and GPU tile operations.

Hello World
-----------

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

Fibonacci
---------

Computes the first 10 Fibonacci numbers using a simple loop.

.. code-block:: basic

   10 REM Fibonacci sequence
   20 LET A = 0
   30 LET B = 1
   40 PRINT "Fibonacci sequence:"
   50 FOR I = 1 TO 10
   60   PRINT A
   70   LET C = A + B
   80   LET A = B
   90   LET B = C
   100 NEXT I
   110 END

Array Sum
---------

Demonstrates ``DATA``/``READ`` statements to sum an inline array.

.. code-block:: basic

   10 REM Array sum example
   20 DIM A(5)
   30 DATA 10, 20, 30, 40, 50
   40 LET SUM = 0
   50 FOR I = 1 TO 5
   60   READ X
   70   LET SUM = SUM + X
   80 NEXT I
   90 PRINT "Sum = "; SUM
   100 END

Vector Add (GPU)
----------------

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

   $ python examples/vector_add_demo.py

This script lexes, parses, analyzes, compiles to cubin via the bytecode backend,
launches the kernel with test data, and verifies the result.

GEMM (GPU)
----------

A tiled matrix multiply: ``C(M,N) = A(M,K) * B(K,N)``.

.. code-block:: basic

   10 REM GEMM: C(M,N) = A(M,K) * B(K,N)
   20 DIM A(512, 512), B(512, 512), C(512, 512)
   30 TILE 128, 128, 32
   40 INPUT A(), B()
   50 LET TILEM = BID / 4
   60 LET TILEN = BID MOD 4
   70 FOR K = 0 TO 15
   80   MMA ACC, A(TILEM, K), B(K, TILEN)
   90 NEXT K
   100 STORE C(TILEM, TILEN), ACC
   110 OUTPUT C
   120 END

This example uses all tile extensions: ``TILE`` sets tile dimensions, ``MMA``
performs matrix multiply-accumulate over tiles, and ``STORE`` writes the
accumulator to the output array.

Run it with:

.. code-block:: bash

   $ python examples/gemm_demo.py

Python Demo Scripts
-------------------

Three demo scripts in ``examples/`` show end-to-end GPU execution:

``vector_add_demo.py``
   Compiles ``vector_add.bas``, launches with 1024-element arrays, verifies
   ``C[i] = A[i] + B[i]``.

``gemm_demo.py``
   Compiles ``gemm.bas``, launches a 512x512 GEMM, verifies against a CPU
   reference implementation.

``gpu_demo.py``
   General-purpose launcher: takes any ``.bas`` file as an argument, compiles
   and runs it on the GPU via the bytecode backend.

   .. code-block:: bash

      $ python examples/gpu_demo.py examples/vector_add.bas
