Architecture
============

cutile-basic compiles BASIC source code through a multi-stage pipeline that
produces a ``.cubin`` for GPU launch.

Pipeline Overview
-----------------

.. code-block:: text

        .bas Source
             │
             ▼
    ___________________
   │       Lexer       │  tokens.py, lexer.py
   |___________________|
             │  list[Token]
             ▼
    ___________________
   │      Parser       │  ast_nodes.py, parser.py
   |___________________|
             │  Program (AST)
             ▼
    ___________________
   │     Analyzer      │  analyzer.py
   |___________________|
             │  AnalyzedProgram
             ▼
    ___________________
   │ Bytecode Backend  │  bytecode.py
   |___________________|
             │
             ▼
         tileiras
              |
              v
          .cubin
              |
              v
        GPU Launch
    (CUDA Driver API)

Stage Details
-------------

Lexer
^^^^^

:Module: ``cutile_basic.lexer``

Tokenizes BASIC source text into a flat list of ``Token`` objects. Each token
carries its type (from ``TokenType`` enum), string value, line number, and column.

The lexer recognizes all standard BASIC keywords plus the tile extensions
(``TILE``, ``OUTPUT``, ``BID``) and built-in function names like ``MMA``.

Parser
^^^^^^

:Module: ``cutile_basic.parser``

Converts the token stream into an AST (``Program`` containing a list of
``Statement`` nodes). Handles BASIC line numbers and builds a ``line_map``
mapping BASIC line numbers to statement indices for ``GOTO``/``GOSUB``.

Supports nested control flow (``IF``/``FOR``/``WHILE``) and expression parsing
with correct operator precedence.

Analyzer
^^^^^^^^

:Module: ``cutile_basic.analyzer``

Performs semantic analysis on the AST:

- Infers types (``F16``, ``F32``, ``I32``, ``I1``, ``STRING``) for all variables
- Tracks array declarations and sizes
- Identifies ``INPUT`` and ``OUTPUT`` variables for GPU kernels
- Collects ``DATA`` values
- Collects ``GOTO``/``GOSUB`` targets

Produces an ``AnalyzedProgram`` with a symbol table and metadata.

Bytecode Backend
^^^^^^^^^^^^^^^^

:Module: ``cutile_basic.bytecode``


Compiles the analyzed program to cuTile bytecode using the
``cuda.tile._bytecode`` Python APIs, assembles a ``.cubin`` with ``tileiras``, and
exposes the result for launch on the GPU. This is the compilation path used by
the CLI and by the demo scripts.

GPU Utilities
^^^^^^^^^^^^^

:Module: ``cutile_basic.gpu``

GPU architecture detection via ``cuda.core``.
