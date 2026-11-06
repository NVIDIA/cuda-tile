# RUN: %PYTHON -m pytest %s

"""
Tests for tile_ir/lib/CAPI/Dialect/CudaTileDialect.cpp coverage.

Covers the `if (!X.has_value()) return {nullptr};` early-return branches in
the CAPI attribute factory functions when an invalid string value is supplied:
  - mlirCudaTileRoundingModeAttrGet
  - mlirCudaTileComparisonOrderingAttrGet
  - mlirCudaTileComparisonPredicateAttrGet
  - mlirCudaTileMemoryOrderingSemanticsAttrGet
  - mlirCudaTileMemoryScopeAttrGet
  - mlirCudaTilePaddingValueAttrGet
  - mlirCudaTileAtomicRMWModeAttrGet
  - mlirCudaTileIntegerOverflowAttrGet
  - mlirCudaTileSignednessAttrGet
  - mlirCudaTileSymbolVisibilityAttrGet
"""

import pytest

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    AtomicRMWModeAttr,
    ComparisonOrderingAttr,
    ComparisonPredicateAttr,
    IntegerOverflowAttr,
    MemoryOrderingSemanticsAttr,
    MemoryScopeAttr,
    PaddingValueAttr,
    RoundingModeAttr,
    SignednessAttr,
    SymbolVisibilityAttr,
    register_dialect,
)
from cuda_tile._mlir.ir import Context


###############################################################################
### RoundingModeAttr.get — invalid string falls into !has_value() branch
### The Python binding then falls back to "nearest_even", but the C function
### returned {nullptr} first which is what we want to cover.
###############################################################################


def test_rounding_mode_attr_invalid_value_triggers_null_branch():
    """Invalid rounding mode hits `if (!roundingMode.has_value()) return {nullptr};`."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        # Binding fallback wraps the null and re-calls with "nearest_even".
        attr = RoundingModeAttr.get("not_a_valid_mode")
        assert attr.value == "nearest_even"


###############################################################################
### ComparisonOrderingAttr.get — invalid value path
### Python binding raises ValueError after the C function returned null.
###############################################################################


def test_comparison_ordering_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid comparison ordering"):
            ComparisonOrderingAttr.get("not_an_ordering")


###############################################################################
### ComparisonPredicateAttr.get — invalid value path
###############################################################################


def test_comparison_predicate_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid comparison predicate"):
            ComparisonPredicateAttr.get("not_a_predicate")


###############################################################################
### MemoryOrderingSemanticsAttr.get — invalid value path
### Binding fallback wraps null and re-calls with "weak".
###############################################################################


def test_memory_ordering_semantics_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        attr = MemoryOrderingSemanticsAttr.get("not_a_semantic")
        assert attr.value == "weak"


###############################################################################
### MemoryScopeAttr.get — invalid value path
###############################################################################


def test_memory_scope_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid memory scope"):
            MemoryScopeAttr.get("not_a_scope")


###############################################################################
### PaddingValueAttr.get — invalid value path
###############################################################################


def test_padding_value_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid padding value"):
            PaddingValueAttr.get("not_a_padding")


###############################################################################
### AtomicRMWModeAttr.get — invalid value path
###############################################################################


def test_atomic_rmw_mode_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid atomic RMW mode"):
            AtomicRMWModeAttr.get("not_a_mode")


###############################################################################
### IntegerOverflowAttr.get — invalid value path
###############################################################################


def test_integer_overflow_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid integer overflow"):
            IntegerOverflowAttr.get("not_overflow")


###############################################################################
### SignednessAttr.get — invalid value path
###############################################################################


def test_signedness_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid signedness"):
            SignednessAttr.get("not_signed")


###############################################################################
### SymbolVisibilityAttr.get — invalid value path
###############################################################################


def test_symbol_visibility_attr_invalid_value_triggers_null_branch():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(ValueError, match="Invalid symbol visibility"):
            SymbolVisibilityAttr.get("not_visible")
