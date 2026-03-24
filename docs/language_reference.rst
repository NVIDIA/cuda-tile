Language Reference
==================

basic_tile implements a dialect of BASIC extended with GPU tile operations.
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

.. code-block:: basic

   10 INPUT A(), B()

DIM
^^^

Declares arrays with specified sizes. Supports 1D and 2D arrays.

.. code-block:: basic

   10 DIM A(128)
   20 DIM M(512, 512)

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

.. code-block:: basic

   10 GOTO 50

GOSUB / RETURN
^^^^^^^^^^^^^^

Subroutine call and return.

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

Declares tile dimensions for matrix operations (M, N, K).

.. code-block:: basic

   30 TILE 128, 128, 32

MMA
^^^

Matrix Multiply-Accumulate on tiles. Loads tiles from two arrays and accumulates
into an accumulator variable.

.. code-block:: basic

   80 MMA ACC, A(TILEM, K), B(K, TILEN)

STORE
^^^^^

Stores a tile value (typically an MMA accumulator) into an array.

.. code-block:: basic

   100 STORE C(TILEM, TILEN), ACC

OUTPUT
^^^^^^

Declares which arrays contain kernel output. Used by the runtime to copy results
back to the host.

.. code-block:: basic

   110 OUTPUT C

INPUT (GPU mode)
^^^^^^^^^^^^^^^^

In GPU kernels, ``INPUT`` declares which arrays are kernel parameters passed from
the host.

.. code-block:: basic

   40 INPUT A(), B()

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

===============  ============================
Function         Description
===============  ============================
``ABS(x)``       Absolute value
``SQR(x)``       Square root
``INT(x)``       Integer truncation
``SIN(x)``       Sine
``COS(x)``       Cosine
``TAN(x)``       Tangent
``EXP(x)``       Exponential (e^x)
``LOG(x)``       Natural logarithm
``SGN(x)``       Sign (-1, 0, or 1)
===============  ============================
