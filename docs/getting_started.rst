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

The repository ships with example BASIC programs in the ``examples/`` directory.
Compile one to a ``.cubin`` (the compiler prints the path to stdout):

.. code-block:: bash

   $ python -m cutile_basic.cli examples/hello.bas

Write the ``.cubin`` to a specific path:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/hello.bas -o hello.cubin

Another example:

.. code-block:: bash

   $ python -m cutile_basic.cli examples/vector_add.bas -o vector_add.cubin

Run a GPU demo end-to-end:

.. code-block:: bash

   $ python examples/vector_add.py

Using the Python API
--------------------

.. code-block:: python

   from cutile_basic import compile_basic_to_cubin

   source = """
   10 DIM A(1024), B(1024), C(1024)
   20 TILE A(128), B(128), C(128)
   30 INPUT A(), B()
   40 LET C(BID) = A(BID) + B(BID)
   50 OUTPUT C
   60 END
   """

   result = compile_basic_to_cubin(source)
   print(result.cubin_path)   # path to the compiled .cubin
   print(result.meta)          # kernel metadata (arrays, grid size, etc.)
