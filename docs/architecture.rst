Architecture
============

cutile-basic compiles BASIC source code through a multi-stage pipeline with two
output modes.

Pipeline Overview
-----------------

.. code-block:: text

   .bas Source
       │
       ▼
   ┌────────┐
   │ Lexer  │  _tokens.py, _lexer.py
   └───┬────┘
       │  list[Token]
       ▼
   ┌────────┐
   │ Parser │  _ast_nodes.py, _parser.py
   └───┬────┘
       │  Program (AST)
       ▼
   ┌──────────┐
   │ Analyzer │  _analyzer.py
   └───┬──────┘
       │  AnalyzedProgram
       ▼
   ┌───────────────────────────────────┐
   │         Two Output Modes          │
   │                                   │
   │  ┌──────────┐    ┌──────────────┐ │
   │  │ Codegen  │    │  Bytecode    │ │
   │  │ (MLIR)   │    │  Backend     │ │
   │  └────┬─────┘    └──────┬───────┘ │
   └───────┼─────────────────┼─────────┘
           │                 │
           ▼                 ▼
       MLIR text          tileiras
     (human-readable)       │
                            ▼
                          .cubin
                            │
                            ▼
                      GPU Launch
                   (CUDA driver API)

Stage Details
-------------

Lexer
^^^^^

:Module: ``cutile_basic._lexer``

Tokenizes BASIC source text into a flat list of ``Token`` objects. Each token
carries its type (from ``TokenType`` enum), string value, line number, and column.

The lexer recognizes all standard BASIC keywords plus the tile extensions
(``TILE``, ``MMA``, ``STORE``, ``OUTPUT``, ``BID``).

Parser
^^^^^^

:Module: ``cutile_basic._parser``

Converts the token stream into an AST (``Program`` containing a list of
``Statement`` nodes). Handles BASIC line numbers and builds a ``line_map``
mapping BASIC line numbers to statement indices for ``GOTO``/``GOSUB``.

Supports nested control flow (``IF``/``FOR``/``WHILE``) and expression parsing
with correct operator precedence.

Analyzer
^^^^^^^^

:Module: ``cutile_basic._analyzer``

Performs semantic analysis on the AST:

- Infers types (``F32``, ``I32``, ``STRING``) for all variables
- Tracks array declarations and sizes
- Identifies ``INPUT`` and ``OUTPUT`` variables for GPU kernels
- Collects ``DATA`` values
- Validates ``GOTO``/``GOSUB`` targets

Produces an ``AnalyzedProgram`` with a symbol table and metadata.

Textual Backend
^^^^^^^^^^^^^^^^

:Module: ``cutile_basic._textual``

Generates CUDA Tile IR MLIR text from the analyzed program. ``INPUT`` variables
become kernel parameters. The output is a ``cuda_tile.module`` with an entry
function. This is the default CLI output and is useful for inspection and
debugging.

Bytecode Backend
^^^^^^^^^^^^^^^^

:Module: ``cutile_basic._bytecode``


Compiles the analyzed program directly to cuTile bytecode using the
``cuda.tile._bytecode`` Python APIs. Bypasses MLIR text entirely and produces
a ``.cubin`` via ``tileiras``. This is the compilation path used by
``--compile-cubin`` on the CLI and by the demo scripts.

GPU Utilities
^^^^^^^^^^^^^

:Module: ``cutile_basic.gpu``

GPU architecture detection via ``cuda.core``.
