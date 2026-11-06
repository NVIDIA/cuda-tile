# RUN: %PYTHON -m pytest %s

"""
Tests for the CAPI CudaTileOptimizer.cpp coverage.
Covers: mlirCudaTileOptFlagsInit, toCpp, mlirCudaTileApplyOptimizations
"""

import pytest

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    applyTileIROptimizations,
    register_dialect,
    TileIROptimizationsOpts,
)
from cuda_tile._mlir.ir import Context, Location, Module


###############################################################################
### applyTileIROptimizations - success path with default opts
### Covers mlirCudaTileOptFlagsInit (non-null config), toCpp default conversion,
### mlirCudaTileApplyOptimizations success path through extractCudaTileModuleOp,
### registerTileIROptPasses, optimizeTileIRModule
###############################################################################


def test_apply_optimizations_default_opts():
    """Exercises the default optimization path: optLevel=3, no FMA fusion."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                    cuda_tile.module @test_module {
                        cuda_tile.entry @test_entry() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )
        opts = TileIROptimizationsOpts()
        assert opts.opt_level == 3
        assert opts.fuse_fma == False
        result = applyTileIROptimizations(m.operation, opts)
        assert result == True


###############################################################################
### applyTileIROptimizations - with FMA fusion enabled
### Covers toCpp with CUDATILE_OPT_FLAG_FUSE_FMA flag set
###############################################################################


def test_apply_optimizations_fuse_fma():
    """Exercises the FMA fusion flag path in toCpp."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                    cuda_tile.module @fma_test {
                        cuda_tile.entry @entry() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )
        opts = TileIROptimizationsOpts()
        opts.fuse_fma = True
        result = applyTileIROptimizations(m.operation, opts)
        assert result == True


###############################################################################
### applyTileIROptimizations - with opt_level 0 (minimal optimization)
### Covers toCpp with optLevel=0, which skips most passes
###############################################################################


def test_apply_optimizations_opt_level_0():
    """Exercises optLevel=0 path (no canonicalize, no CSE, no loop split)."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                    cuda_tile.module @opt0_test {
                        cuda_tile.entry @entry() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )
        opts = TileIROptimizationsOpts()
        opts.opt_level = 0
        result = applyTileIROptimizations(m.operation, opts)
        assert result == True


###############################################################################
### applyTileIROptimizations - with opt_level 1
### Covers optLevel=1 (canonicalize + CSE but no LICM or loop split)
###############################################################################


def test_apply_optimizations_opt_level_1():
    """Exercises optLevel=1 path."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                    cuda_tile.module @opt1_test {
                        cuda_tile.entry @entry() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )
        opts = TileIROptimizationsOpts()
        opts.opt_level = 1
        result = applyTileIROptimizations(m.operation, opts)
        assert result == True


###############################################################################
### applyTileIROptimizations - with opt_level 2
### Covers optLevel=2 (canonicalize + CSE + LICM but no loop split)
###############################################################################


def test_apply_optimizations_opt_level_2():
    """Exercises optLevel=2 path."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                    cuda_tile.module @opt2_test {
                        cuda_tile.entry @entry() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )
        opts = TileIROptimizationsOpts()
        opts.opt_level = 2
        result = applyTileIROptimizations(m.operation, opts)
        assert result == True


###############################################################################
### applyTileIROptimizations - module without cuda_tile.module
### Covers extractCudaTileModuleOp returning null, the failure path at line 58
###############################################################################


def test_apply_optimizations_no_cuda_tile_module():
    """Exercises the extractCudaTileModuleOp failure path (no cuda_tile.module)."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                }
            """
            )
        opts = TileIROptimizationsOpts()
        result = applyTileIROptimizations(m.operation, opts)
        assert result == False


###############################################################################
### applyTileIROptimizations - with FMA and opt_level 0 combined
### Covers toCpp with both flags active simultaneously
###############################################################################


def test_apply_optimizations_fma_opt_level_0():
    """Exercises combined FMA + optLevel=0 in toCpp."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.parse(
                """
                module {
                    cuda_tile.module @combined_test {
                        cuda_tile.entry @entry() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )
        opts = TileIROptimizationsOpts()
        opts.fuse_fma = True
        opts.opt_level = 0
        result = applyTileIROptimizations(m.operation, opts)
        assert result == True
