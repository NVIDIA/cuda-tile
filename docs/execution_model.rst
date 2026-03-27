Execution Model
===============

cutile-basic compiles BASIC programs into GPU kernels that execute in parallel
on NVIDIA GPUs. This page explains the core concepts needed to understand how
GPU execution works in cutile-basic.

Kernels
-------

A cutile-basic program that uses GPU extensions (``INPUT``, ``OUTPUT``, ``BID``,
``TILE``, ``MMA``) compiles into a **kernel** -- a function that runs
on the GPU rather than the CPU. The host (CPU) is responsible for launching the
kernel, transferring data to/from GPU memory, and collecting results.

.. code-block:: basic

   10 DIM A(128), B(128), C(128)
   20 INPUT A(), B()
   30 LET C(BID) = A(BID) + B(BID)
   40 OUTPUT C
   50 END

This program compiles into a single GPU kernel. The host copies arrays ``A`` and
``B`` to the GPU, launches the kernel, and reads ``C`` back.

Grids and Blocks
----------------

When a kernel launches, it runs across a **grid** of **blocks** (also called
CTAs -- Cooperative Thread Arrays). Each block executes the same program
independently and in parallel. The number of blocks in the grid is determined
by the problem size: for a vector of 128 elements processed one per block, the
grid has 128 blocks.

.. code-block:: text

   Grid
   ┌─────────┬─────────┬─────────┬─────────┬─────────┐
   │ Block 0 │ Block 1 │ Block 2 │  . . .  │Block 127│
   └─────────┴─────────┴─────────┴─────────┴─────────┘

   Each block runs the same BASIC program independently.

Blocks execute concurrently on the GPU's streaming multiprocessors (SMs). The
GPU hardware schedules blocks onto available SMs -- the programmer does not
control which SM runs which block.

BID: Block Identification
-------------------------

Inside a kernel, the built-in variable ``BID`` evaluates to the **block index**
of the currently executing block (0, 1, 2, ...). This is how each block knows
which portion of the data to process.

.. code-block:: basic

   30 LET C(BID) = A(BID) + B(BID)

When Block 0 executes this line, ``BID`` is 0, so it computes ``C(0) = A(0) + B(0)``.
When Block 42 executes the same line, ``BID`` is 42, so it computes
``C(42) = A(42) + B(42)``. All 128 blocks run simultaneously, completing the
entire vector addition in parallel.

Tiled Execution
---------------

For matrix operations, a single element per block is not efficient. Instead,
the data is divided into **tiles** -- rectangular sub-regions that the hardware
can process efficiently using tensor cores.

The ``TILE`` statement declares the tile/partition shape for each variable:

.. code-block:: basic

   30 TILE A(128, 32), B(32, 128), C(128, 128), ACC(128, 128)

Here, ``A`` is partitioned into 128 x 32 tiles, ``B`` into 32 x 128 tiles, and
the output ``C`` and accumulator ``ACC`` are 128 x 128 tiles. Each block
processes one output tile of the result matrix.

.. code-block:: text

   Output matrix C (512 x 512)
   ┌────────┬────────┬────────┬────────┐
   │ Tile   │ Tile   │ Tile   │ Tile   │
   │ (0,0)  │ (0,1)  │ (0,2)  │ (0,3)  │
   │ 128x128│ 128x128│ 128x128│ 128x128│
   ├────────┼────────┼────────┼────────┤
   │ Tile   │ Tile   │ Tile   │ Tile   │
   │ (1,0)  │ (1,1)  │ (1,2)  │ (1,3)  │
   ├────────┼────────┼────────┼────────┤
   │ Tile   │ Tile   │ Tile   │ Tile   │
   │ (2,0)  │ (2,1)  │ (2,2)  │ (2,3)  │
   ├────────┼────────┼────────┼────────┤
   │ Tile   │ Tile   │ Tile   │ Tile   │
   │ (3,0)  │ (3,1)  │ (3,2)  │ (3,3)  │
   └────────┴────────┴────────┴────────┘

   16 blocks total (4 x 4 grid of tiles).
   Each block computes one 128x128 output tile.

Each block uses ``BID`` to determine which tile it is responsible for, then
iterates over the K dimension using ``MMA`` (matrix multiply-accumulate) to
compute the result:

.. code-block:: basic

   50 LET TILEM = BID / 4
   60 LET TILEN = BID MOD 4
   65 LET ACC = 0.0
   70 FOR K = 0 TO 15
   80   LET ACC = MMA(A(TILEM, K), B(K, TILEN), ACC)
   90 NEXT K
   100 LET C(TILEM, TILEN) = ACC

Host-Device Data Flow
---------------------

GPU kernels operate on data in GPU memory, which is separate from CPU memory.
cutile-basic manages this automatically based on the ``INPUT`` and ``OUTPUT``
declarations:

1. **Before launch**: Arrays declared with ``INPUT`` are copied from host
   (CPU) memory to device (GPU) memory.
2. **Kernel execution**: The kernel runs on the GPU, reading from and writing
   to device memory.
3. **After launch**: Arrays declared with ``OUTPUT`` are copied from device
   memory back to host memory.

.. code-block:: text

   Host (CPU)                          Device (GPU)
   ┌──────────┐    INPUT A(), B()     ┌──────────┐
   │ A, B     │ ────────────────────► │ A, B     │
   │          │                       │          │
   │          │                       │ Kernel   │
   │          │                       │ executes │
   │          │                       │          │
   │ C        │ ◄──────────────────── │ C        │
   └──────────┘    OUTPUT C           └──────────┘

For more details on the compilation pipeline that transforms BASIC source into
GPU-executable code, see :doc:`architecture`.
