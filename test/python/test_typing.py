# RUN: %PYTHON -m pytest %s

"""
Tests for element type wrappers in cuda_tile.dialects.cuda_tile_ops.

Verifies that the minimal type wrappers for MMA descriptors
work correctly with MLIR types.
"""

import pytest
from cuda_tile._mlir.ir import Context
from cuda_tile._mlir._mlir_libs._cuda_tile import register_dialect


@pytest.fixture(scope="module")
def mlir_context():
    """Create an MLIR context for tests that need types."""
    with Context() as ctx:
        register_dialect(ctx, load=True)
        yield ctx


def test_make_tile_type(mlir_context):
    """Test make_tile_type with both wrappers and raw MLIR types."""
    from cuda_tile._mlir.dialects.cuda_tile_ops import make_tile_type, Int32, Float32
    from cuda_tile._mlir.ir import IntegerType, F32Type

    # With wrappers
    tile_i32 = make_tile_type(Int32, [4, 4])
    assert tile_i32.shape == [4, 4]

    tile_f32 = make_tile_type(Float32, [8])
    assert tile_f32.shape == [8]

    # With raw MLIR types
    tile_raw = make_tile_type(IntegerType.get_signless(32), [2, 2])
    assert tile_raw.shape == [2, 2]


def test_get_mlir_type_helper(mlir_context):
    """Test _get_mlir_type converts wrappers and passes through MLIR types."""
    from cuda_tile._mlir.dialects.cuda_tile_ops import _get_mlir_type, Int32, Float32
    from cuda_tile._mlir.ir import IntegerType, F32Type

    # Wrappers -> MLIR types
    assert _get_mlir_type(Int32) == IntegerType.get_signless(32)
    assert _get_mlir_type(Float32) == F32Type.get()

    # Raw MLIR types pass through
    i32_type = IntegerType.get_signless(32)
    assert _get_mlir_type(i32_type) == i32_type
