# RUN: %PYTHON -m pytest %s

"""
Tests for tile_ir/lib/CAPI/Dialect/CudaTileDialect.cpp coverage.

Covers:
  - `if (numCta != 0)`, `if (numWorkerWarps != 0)`, `if (occupancy != 0)`
    branches inside `mlirCudaTileOptimizationHintsAttrGetEntryOpHint`.
  - `if (allowTma != -1)`, `if (latency != 0)` branches inside
    `mlirCudaTileOptimizationHintsAttrGetLoadStoreOpHint` (these are the
    `LoadStoreOpHint` hot paths).
  - The non-empty buffer branch of `mlirCudaTileFreeBuffer` (taken when the
    binding successfully writes bytecode and then frees the returned buffer).
  - The success path of `mlirCudaTileWriteBytecodeToBuffer`.
"""

import io
import pytest

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    OptimizationHintsAttr,
    register_dialect,
    writeBytecode,
)
from cuda_tile._mlir.ir import Context, Location, Module


###############################################################################
### OptimizationHintsAttr.getEntryOpHint — all three numCta/numWorkerWarps/
### occupancy non-zero
### Covers the taken branches of the three independent `if (... != 0)` guards
### in mlirCudaTileOptimizationHintsAttrGetEntryOpHint.
###############################################################################


def test_entry_op_hint_all_non_zero():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getEntryOpHint(
            arch="sm_100", num_cta=2, num_worker_warps=4, occupancy=2
        )
        assert attr is not None


def test_entry_op_hint_only_num_cta_non_zero():
    """Only the numCta branch is taken; the other two are not-taken."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getEntryOpHint(
            arch="sm_100", num_cta=4, num_worker_warps=0, occupancy=0
        )
        assert attr is not None


def test_entry_op_hint_only_num_worker_warps_non_zero():
    """Only the numWorkerWarps branch is taken."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getEntryOpHint(
            arch="sm_90", num_cta=0, num_worker_warps=8, occupancy=0
        )
        assert attr is not None


def test_entry_op_hint_only_occupancy_non_zero():
    """Only the occupancy branch is taken."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getEntryOpHint(
            arch="sm_90", num_cta=0, num_worker_warps=0, occupancy=3
        )
        assert attr is not None


def test_entry_op_hint_all_zero():
    """All three branches are not-taken; the empty inner dict path."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getEntryOpHint(
            arch="sm_100", num_cta=0, num_worker_warps=0, occupancy=0
        )
        assert attr is not None


###############################################################################
### OptimizationHintsAttr.getLoadStoreOpHint — allow_tma and latency
### Covers `if (allowTma != -1)` (both true/false sub-paths) and
### `if (latency != 0)` inside mlirCudaTileOptimizationHintsAttrGetLoadStoreOpHint.
###############################################################################


def test_loadstore_hint_allow_tma_true_and_latency():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getLoadStoreOpHint(
            arch="sm_100", allow_tma=True, latency=12
        )
        assert attr is not None


def test_loadstore_hint_allow_tma_false_only():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getLoadStoreOpHint(
            arch="sm_90", allow_tma=False, latency=0
        )
        assert attr is not None


def test_loadstore_hint_unspecified_allow_tma_with_latency():
    """allow_tma=None encodes the -1 sentinel; latency != 0 still recorded."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = OptimizationHintsAttr.getLoadStoreOpHint(
            arch="sm_90", allow_tma=None, latency=42
        )
        assert attr is not None


###############################################################################
### writeBytecode success path
### Covers the non-empty buffer path through mlirCudaTileWriteBytecodeToBuffer
### and the non-empty branch of mlirCudaTileFreeBuffer
### (`if (buffer.data && buffer.length > 0)` taken).
###############################################################################


def test_write_bytecode_success_calls_free_buffer():
    """Successful bytecode emission populates the buffer and then frees it."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.parse(
                """
                module {
                    cuda_tile.module @bc_test {
                        cuda_tile.entry @entry() {
                            cuda_tile.return
                        }
                    }
                }
                """
            )
        buf = io.BytesIO()
        ok = writeBytecode(buf, module.operation)
        assert ok is True
        # The returned bytecode must be non-empty so the success path through
        # writeBytecode + freeBuffer is genuinely exercised.
        assert len(buf.getvalue()) > 0


def test_write_bytecode_failure_no_cuda_tile_module():
    """A plain builtin.module with no cuda_tile.module triggers the
    extractCudaTileModuleOp failure path (writeBytecode returns false)."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.parse("module {}")
        buf = io.BytesIO()
        ok = writeBytecode(buf, module.operation)
        assert ok is False
