Language Reference
==================

cutile-basic implements a dialect of BASIC extended with GPU tile operations.
Programs use numbered lines in classic BASIC style.

Program Structure
-----------------

Every statement begins with a line number. Statements execute in line-number order
unless control flow (``GOTO``, ``GOSUB``, ``FOR``, ``WHILE``) redirects execution.

.. code-block:: basic

   10 REM This is a comment
   20 LET X = 42
   30 PRINT X
   40 END

Data Types
----------

============  ===================================
Type          Description
============  ===================================
``F32``       32-bit floating point (default for numeric variables)
``I32``       32-bit integer
``I1``        Boolean (result of comparisons and logical operators)
``STRING``    String literals (quoted with ``""``)
============  ===================================

Variables
---------

Variable names are uppercase identifiers. Arrays are declared with ``DIM`` and
accessed with parenthesized indices.

.. code-block:: basic

   10 LET X = 3.14
   20 DIM A(100)
   30 LET A(0) = X
   40 DIM M(512, 512)
   50 LET M(0, 0) = 1.0

Standard Statements
-------------------

LET
^^^

Assigns a value to a variable or array element.

.. code-block:: basic

   10 LET X = 42
   20 LET A(I) = X + 1

PRINT
^^^^^

Prints values to output. Use ``;`` to separate items or suppress the trailing
newline.

.. code-block:: basic

   10 PRINT "Hello, World!"
   20 PRINT "X = "; X

INPUT
^^^^^

Reads values into variables. For GPU kernels, ``INPUT`` declares kernel parameters.
Variables followed by ``()`` become array (pointer) parameters; plain variables
become scalar integer parameters that can be used to specify array dimensions
and control loop bounds.

.. code-block:: basic

   10 INPUT M, N, K, A(), B()

DIM
^^^

Declares arrays with specified sizes. Supports 1D and 2D arrays. Sizes may be
literal numbers or variables declared via ``INPUT``, enabling dynamically sized
arrays whose dimensions are passed in at launch time.

.. code-block:: basic

   10 DIM A(128)
   20 DIM M(512, 512)
   30 DIM V(N)
   40 DIM X(M, K)

IF / THEN / ELSE / ENDIF
^^^^^^^^^^^^^^^^^^^^^^^^^

Conditional execution.

.. code-block:: basic

   10 IF X > 10 THEN
   20   PRINT "Large"
   30 ELSE
   40   PRINT "Small"
   50 ENDIF

FOR / NEXT
^^^^^^^^^^

Counted loop with optional ``STEP``.

.. code-block:: basic

   10 FOR I = 1 TO 10
   20   PRINT I
   30 NEXT I

   40 FOR J = 0 TO 100 STEP 2
   50   PRINT J
   60 NEXT J

WHILE / WEND
^^^^^^^^^^^^^

Conditional loop.

.. code-block:: basic

   10 LET X = 1
   20 WHILE X < 100
   30   LET X = X * 2
   40 WEND

GOTO
^^^^

Unconditional jump to a line number.

.. note::

   ``GOTO`` is parsed and analyzed, but is not fully supported by the backend;
   it is skipped during code generation.

.. code-block:: basic

   10 GOTO 50

GOSUB / RETURN
^^^^^^^^^^^^^^

Subroutine call and return.

.. note::

   ``GOSUB``/``RETURN`` are parsed and analyzed, but are not fully supported
   by the backend; they are skipped during code generation.

.. code-block:: basic

   10 GOSUB 100
   20 END
   100 PRINT "In subroutine"
   110 RETURN

DATA / READ
^^^^^^^^^^^^

Inline data values read sequentially.

.. code-block:: basic

   10 DATA 10, 20, 30, 40, 50
   20 READ X
   30 PRINT X

REM
^^^

Comment. The rest of the line is ignored.

.. code-block:: basic

   10 REM This is a comment

END / STOP
^^^^^^^^^^

Terminates program execution.

Arrays and Tiles
-----------------

cutile-basic programs work with two kinds of data when targeting the GPU:
**arrays** and **tiles**. Understanding the distinction is key to writing
effective GPU kernels.

**Arrays** are declared with ``DIM`` and live in GPU global memory. They are the
data that the host passes into and receives from the kernel. Arrays can be 1D or
2D, have arbitrary sizes, and are mutable -- the kernel can read from and write to
them.

.. code-block:: basic

   10 DIM A(1024)
   20 DIM M(512, 512)

**Tiles** are fixed-size rectangular chunks of data that the GPU hardware can
process efficiently using tensor cores. Tile dimensions must be powers of two
(e.g. 128 x 128). Unlike arrays, tiles are not directly addressable by the
programmer -- they are implicit in operations like ``MMA``.

The ``TILE`` statement partitions each array into fixed-size tiles and
determines how work is distributed across GPU blocks. Each block uses ``BID``
to select which tile it operates on.

For a 1D element-wise kernel, ``TILE`` partitions vectors into 1D chunks:

.. code-block:: basic

   20 DIM A(1024), B(1024), C(1024)
   30 TILE A(128), B(128), C(128)
   50 LET C(BID) = A(BID) + B(BID)

Here, each of the 8 blocks (1024 / 128) processes a 128-element tile.
``C(BID)`` refers to the ``BID``-th tile of ``C``, not a single element.

For a 2D matrix kernel, ``TILE`` partitions matrices into 2D sub-blocks and
``MMA`` performs tensor-core multiply-accumulate on those tiles:

.. code-block:: basic

   20 DIM A(512, 512), B(512, 512), C(512, 512)
   30 TILE A(128, 32), B(32, 128), C(128, 128), ACC(128, 128)
   80 LET ACC = MMA(A(TILEM, K), B(K, TILEN), ACC)

``A(TILEM, K)`` does not access a single element -- it loads a 128 x 32 tile
from array ``A``. The hardware handles the bulk data movement.

Array dimensions can also be variables declared via ``INPUT``, creating
dynamically sized arrays whose shapes are passed in as scalar kernel parameters:

.. code-block:: basic

   15 INPUT M, N, K, A(), B()
   20 DIM A(M, K), B(K, N), C(M, N)
   30 TILE A(128, 32), B(32, 128), C(128, 128), ACC(128, 128)

``OUTPUT`` marks which arrays should be copied back to the host after execution.
See :doc:`execution_model` for more on how kernels, grids, and data transfer
work.

Tile/GPU Extensions
-------------------

These statements extend BASIC for GPU tile-based computation.

BID
^^^

A built-in variable that evaluates to the current block ID (CTA index) during
kernel execution.

.. code-block:: basic

   40 LET C(BID) = A(BID) + B(BID)

TILE
^^^^

Declares the tile/partition shape for one or more variables.

.. code-block:: basic

   20 TILE A(128), B(128), C(128)
   30 TILE A(128, 32), B(32, 128), C(128, 128), ACC(128, 128)

OUTPUT
^^^^^^

Declares which arrays contain kernel output. Used by the runtime to copy results
back to the host.

.. code-block:: basic

   110 OUTPUT C

INPUT (GPU mode)
^^^^^^^^^^^^^^^^

In GPU kernels, ``INPUT`` declares kernel parameters passed from the host.
Variables with ``()`` become array (pointer) parameters; plain variables become
scalar ``i32`` parameters. Scalar parameters are commonly used to pass array
dimensions so that kernels work with any size.

.. code-block:: basic

   10 INPUT N, A(), B()
   20 INPUT M, N, K, A(), B()

Operators
---------

================  ============
Operator          Description
================  ============
``+``             Addition
``-``             Subtraction / Negation
``*``             Multiplication
``/``             Division
``^``             Exponentiation
``MOD``           Modulo
``=``             Equality
``<>``            Inequality
``<``             Less than
``>``             Greater than
``<=``            Less or equal
``>=``            Greater or equal
``AND``           Logical AND
``OR``            Logical OR
``NOT``           Logical NOT
================  ============

Built-in Functions
------------------

============================  ==========================================
Function                      Description
============================  ==========================================
``ABS(x)``                    Absolute value
``SQR(x)``                    Square root
``INT(x)``                    Integer truncation
``SIN(x)``                    Sine
``COS(x)``                    Cosine
``TAN(x)``                    Tangent
``EXP(x)``                    Exponential (e^x)
``LOG(x)``                    Natural logarithm
``SGN(x)``                    Sign (-1, 0, or 1)
``MMA(A, B, ACC)``            Matrix multiply-accumulate on tiles
============================  ==========================================

``MMA`` loads tiles from arrays ``A`` and ``B``, multiplies them, and
accumulates the result into ``ACC``, returning the updated accumulator.
Use it on the right-hand side of a ``LET`` assignment:

.. code-block:: basic

   80 LET ACC = MMA(A(TILEM, K), B(K, TILEN), ACC)
