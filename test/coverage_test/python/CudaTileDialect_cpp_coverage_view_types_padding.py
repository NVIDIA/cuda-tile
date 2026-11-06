# RUN: %PYTHON -m pytest %s

"""
Tests for tile_ir/lib/CAPI/Dialect/CudaTileDialect.cpp coverage.

Covers the `if (!mlirAttributeIsNull(paddingValue))` branch (non-null padding
path) in `mlirCudaTilePartitionViewTypeGet`, `mlirCudaTilePartitionViewTypeGetChecked`,
`mlirCudaTileStridedViewTypeGet`, and `mlirCudaTileStridedViewTypeGetChecked`,
as well as the `if (paddingValue)` non-null path in
`mlirCudaTileStridedViewTypeGetPaddingValue` getter.
"""

import pytest

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    PartitionViewType,
    StridedViewType,
    TensorViewType,
    PaddingValueAttr,
    register_dialect,
)
from cuda_tile._mlir.extras import types as T
from cuda_tile._mlir.ir import Context


###############################################################################
### PartitionViewType.get with non-null paddingValue
### Covers the non-null branch of `if (!mlirAttributeIsNull(paddingValue))`
### inside mlirCudaTilePartitionViewTypeGetChecked.
###############################################################################


def test_partition_view_type_get_checked_with_padding_zero():
    """Exercises the non-null paddingValue path of PartitionViewType."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        tv = TensorViewType.get(T.i32(), [64, 32], [32, 1])
        padding = PaddingValueAttr.get("zero")
        pv = PartitionViewType.get(
            tile_shape=[16, 16],
            tensor_view_type=tv,
            dim_map=[0, 1],
            padding_value=padding,
        )
        assert pv is not None
        # The padding_value getter (non-null branch in
        # mlirCudaTilePartitionViewTypeGetPaddingValue) returns the attr we set.
        assert pv.padding_value is not None


def test_partition_view_type_get_checked_with_padding_nan():
    """Another non-null paddingValue case using a different padding kind."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        tv = TensorViewType.get(T.f32(), [128, 64], [64, 1])
        padding = PaddingValueAttr.get("nan")
        pv = PartitionViewType.get(
            tile_shape=[32, 32],
            tensor_view_type=tv,
            dim_map=[0, 1],
            padding_value=padding,
        )
        assert pv is not None
        assert pv.padding_value is not None


###############################################################################
### StridedViewType.get with non-null paddingValue
### Covers the non-null branch in
### mlirCudaTileStridedViewTypeGetChecked and the non-null branch in
### mlirCudaTileStridedViewTypeGetPaddingValue getter.
###############################################################################


def test_strided_view_type_get_checked_with_padding_zero():
    """Exercises the non-null paddingValue path of StridedViewType."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        tv = TensorViewType.get(T.i32(), [64, 32], [32, 1])
        padding = PaddingValueAttr.get("zero")
        sv = StridedViewType.get(
            tile_shape=[16, 16],
            traversal_strides=[1, 1],
            tensor_view_type=tv,
            dim_map=[0, 1],
            padding_value=padding,
        )
        assert sv is not None
        # Hits the non-null branch in mlirCudaTileStridedViewTypeGetPaddingValue
        assert sv.padding_value is not None


def test_strided_view_type_get_checked_with_padding_pos_inf():
    """Another non-null paddingValue case for StridedViewType."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        tv = TensorViewType.get(T.f32(), [256, 128], [128, 1])
        padding = PaddingValueAttr.get("pos_inf")
        sv = StridedViewType.get(
            tile_shape=[64, 32],
            traversal_strides=[2, 1],
            tensor_view_type=tv,
            dim_map=[0, 1],
            padding_value=padding,
        )
        assert sv is not None
        assert sv.padding_value is not None


###############################################################################
### Sanity: null paddingValue path still works (already covered in baseline).
### Included here so the test file is self-consistent — both branches behave
### as expected and the PartitionView/StridedView APIs do not regress.
###############################################################################


def test_partition_view_type_without_padding_still_works():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        tv = TensorViewType.get(T.i32(), [64, 32], [32, 1])
        pv = PartitionViewType.get(
            tile_shape=[16, 16],
            tensor_view_type=tv,
            dim_map=[0, 1],
            padding_value=None,
        )
        assert pv is not None


def test_strided_view_type_without_padding_still_works():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        tv = TensorViewType.get(T.i32(), [64, 32], [32, 1])
        sv = StridedViewType.get(
            tile_shape=[16, 16],
            traversal_strides=[1, 1],
            tensor_view_type=tv,
            dim_map=[0, 1],
            padding_value=None,
        )
        assert sv is not None
