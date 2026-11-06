# RUN: %PYTHON -m pytest --capture=sys %s
# Use --capture=sys to avoid nested FD capture on Windows, where stale native
# stderr handles can write diagnostics into a reused temporary-file handle.

"""
Tests direct Python bindings to CudaTile's C API.
"""

import pytest
import tempfile
import os

from cuda_tile._mlir._mlir_libs._cuda_tile import (
    PointerType,
    TileType,
    TensorViewType,
    register_dialect,
    setPreviewFeatures,
    writeBytecode,
)
from cuda_tile._mlir.extras import types as T
from cuda_tile._mlir.ir import Context, Location, Module, Type

###############################################################################
### cuda_tile.PointerType
###############################################################################


def test_pointer_type():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        parsed = Type.parse("!cuda_tile.ptr<i32>")

        assert str(parsed) == "!cuda_tile.ptr<i32>"

        casted = PointerType(parsed)
        assert casted == parsed
        assert casted.pointee_type == T.i32()

        created = PointerType.get(T.i32())
        assert created == casted


###############################################################################
### cuda_tile.TileType
###############################################################################


def test_tile_type():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        parsed = Type.parse("!cuda_tile.tile<64x32xi32>")

        assert str(parsed) == "!cuda_tile.tile<64x32xi32>"

        casted = TileType(parsed)
        assert casted == parsed
        assert casted.shape == [64, 32]
        assert casted.element_type == T.i32()

        created = TileType.get([64, 32], T.i32())
        assert created == casted


###############################################################################
### cuda_tile.TensorViewType
###############################################################################


def test_tensor_view_type():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        parsed = Type.parse("!cuda_tile.tensor_view<64x32xi32, strides=[32,1]>")

        assert str(parsed) == "!cuda_tile.tensor_view<64x32xi32, strides=[32,1]>"

        casted = TensorViewType(parsed)
        assert casted == parsed
        assert casted.element_type == T.i32()
        assert casted.shape == [64, 32]
        assert casted.strides == [32, 1]
        created = TensorViewType.get(T.i32(), [64, 32], [32, 1])
        assert created == casted


def test_dynamic_tensor_view_type_type():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        parsed = Type.parse("!cuda_tile.tensor_view<?x32xi32, strides=[?,1]>")

        assert str(parsed) == "!cuda_tile.tensor_view<?x32xi32, strides=[?,1]>"

        casted = TensorViewType(parsed)
        assert casted == parsed
        assert casted.element_type == T.i32()
        assert casted.shape == [None, 32]
        assert casted.strides == [None, 1]

        created = TensorViewType.get(T.i32(), [None, 32], [None, 1])
        assert created == casted


def test_invalid_tensor_view_type():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        with pytest.raises(
            ValueError,
            match="expected strictly positive value for tensor_view dimension, got -5",
        ):
            TensorViewType.get(T.i32(), [-5, 32], [32, 1])

        with pytest.raises(
            ValueError,
            match="expected strictly positive value for tensor_view dimension, got 0",
        ):
            TensorViewType.get(T.i32(), [32, 32], [32, 0])

        # Ensure kDynamic is not treated as such from Python.
        with pytest.raises(
            ValueError,
            match="expected strictly positive value for tensor_view dimension, got -9223372036854775808",
        ):
            TensorViewType.get(T.i32(), [-9223372036854775808, 32], [32, 1])

        # Ensure kDynamic is not treated as such from Python.
        with pytest.raises(
            ValueError,
            match="expected strictly positive value for tensor_view dimension, got -9223372036854775808",
        ):
            TensorViewType.get(T.i32(), [32, 32], [-9223372036854775808, 1])


###############################################################################
### cuda_tile.PaddingValueAttr
###############################################################################


def test_padding_value_attr():
    from cuda_tile._mlir._mlir_libs._cuda_tile import PaddingValueAttr
    from cuda_tile._mlir.ir import Context, Attribute

    with Context() as ctx:
        register_dialect(ctx, load=True)
        created = PaddingValueAttr.get("zero")
        assert created.value == "zero"

        created = PaddingValueAttr.get("neg_zero")
        assert created.value == "neg_zero"

        created = PaddingValueAttr.get("nan")
        assert created.value == "nan"

        created = PaddingValueAttr.get("pos_inf")
        assert created.value == "pos_inf"

        created = PaddingValueAttr.get("neg_inf")
        assert created.value == "neg_inf"

        with pytest.raises(ValueError, match="Invalid padding value: invalid_value"):
            PaddingValueAttr.get("invalid_value")


###############################################################################
### cuda_tile.RoundingModeAttr
###############################################################################


def test_rounding_mode_attr():
    from cuda_tile._mlir._mlir_libs._cuda_tile import RoundingModeAttr
    from cuda_tile._mlir.ir import Context, Attribute

    with Context() as ctx:
        register_dialect(ctx, load=True)
        # Skip parsing test as the attribute mnemonic isn't registered for parsing
        # directly create the attribute
        created = RoundingModeAttr.get("nearest_even")
        assert created.value == "nearest_even"

        # Test other rounding modes
        rz_mode = RoundingModeAttr.get("zero")
        assert rz_mode.value == "zero"

        rm_mode = RoundingModeAttr.get("negative_inf")
        assert rm_mode.value == "negative_inf"

        rp_mode = RoundingModeAttr.get("positive_inf")
        assert rp_mode.value == "positive_inf"

        full_mode = RoundingModeAttr.get("full")
        assert full_mode.value == "full"

        approx_mode = RoundingModeAttr.get("approx")
        assert approx_mode.value == "approx"


###############################################################################
### cuda_tile.MemoryScopeAttr
###############################################################################


def test_memory_scope_attr():
    from cuda_tile._mlir._mlir_libs._cuda_tile import MemoryScopeAttr
    from cuda_tile._mlir.ir import Context, Attribute

    with Context() as ctx:
        register_dialect(ctx, load=True)
        # Skip parsing test as the attribute mnemonic isn't registered for parsing
        # directly create the attribute
        created = MemoryScopeAttr.get("tl_blk")
        assert created.value == "tl_blk"

        # Test other memory scopes
        device_scope = MemoryScopeAttr.get("device")
        assert device_scope.value == "device"

        sys_scope = MemoryScopeAttr.get("sys")
        assert sys_scope.value == "sys"

        # Test invalid memory scope
        with pytest.raises(ValueError, match="Invalid memory scope: invalid_scope"):
            MemoryScopeAttr.get("invalid_scope")


###############################################################################
### cuda_tile.SymbolVisibilityAttr
###############################################################################


def test_symbol_visibility_attr():
    from cuda_tile._mlir._mlir_libs._cuda_tile import SymbolVisibilityAttr
    from cuda_tile._mlir.ir import Context, Attribute

    with Context() as ctx:
        register_dialect(ctx, load=True)

        # Test public visibility (default)
        public_vis = SymbolVisibilityAttr.get("public")
        assert public_vis.value == "public"

        # Test private visibility
        private_vis = SymbolVisibilityAttr.get("private")
        assert private_vis.value == "private"

        # Test invalid symbol visibility
        with pytest.raises(ValueError, match="Invalid symbol visibility: invalid_vis"):
            SymbolVisibilityAttr.get("invalid_vis")


###############################################################################
### cuda_tile.AtomicRMWModeAttr
###############################################################################


def test_atomic_rmw_mode_attr():
    from cuda_tile._mlir._mlir_libs._cuda_tile import AtomicRMWModeAttr
    from cuda_tile._mlir.ir import Context, Attribute

    with Context() as ctx:
        register_dialect(ctx, load=True)
        # Create and test all atomic RMW modes
        and_mode = AtomicRMWModeAttr.get("and")
        assert and_mode.value == "and"

        or_mode = AtomicRMWModeAttr.get("or")
        assert or_mode.value == "or"

        xor_mode = AtomicRMWModeAttr.get("xor")
        assert xor_mode.value == "xor"

        add_mode = AtomicRMWModeAttr.get("add")
        assert add_mode.value == "add"

        addf_mode = AtomicRMWModeAttr.get("addf")
        assert addf_mode.value == "addf"

        max_mode = AtomicRMWModeAttr.get("max")
        assert max_mode.value == "max"

        min_mode = AtomicRMWModeAttr.get("min")
        assert min_mode.value == "min"

        umax_mode = AtomicRMWModeAttr.get("umax")
        assert umax_mode.value == "umax"

        umin_mode = AtomicRMWModeAttr.get("umin")
        assert umin_mode.value == "umin"

        xchg_mode = AtomicRMWModeAttr.get("xchg")
        assert xchg_mode.value == "xchg"

        # Test invalid atomic RMW mode
        with pytest.raises(ValueError, match="Invalid atomic RMW mode: invalid_mode"):
            AtomicRMWModeAttr.get("invalid_mode")


###############################################################################
### cuda_tile.write_tile_ir_bytecode
###############################################################################


def test_write_tile_ir_bytecode():
    with Context() as ctx:
        register_dialect(ctx, load=True)

        # Create a simple cuda_tile module.
        with Location.unknown(ctx):
            mlir_module = Module.parse(
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

        # Test writing to a temporary file.
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_filename = f.name
            try:
                # This method flushes the file to disk.
                result = writeBytecode(f, mlir_module.operation)
                assert result == True
                assert os.path.getsize(f.name) > 0
            finally:
                f.close()  # Must close before unlink on Windows
                os.unlink(temp_filename)


def test_write_tile_ir_bytecode_with_nested_module():
    with Context() as ctx:
        register_dialect(ctx, load=True)

        # Create a module with nested cuda_tile.module.
        with Location.unknown(ctx):
            mlir_module = Module.parse(
                """
                module {
                    cuda_tile.module @nested_test {
                        cuda_tile.entry @entry_func() {
                            cuda_tile.return
                        }
                    }
                }
            """
            )

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_filename = f.name
            try:
                # This method flushes the file to disk.
                result = writeBytecode(f, mlir_module.operation)
                assert result == True
                assert os.path.getsize(f.name) > 0
            finally:
                f.close()  # Must close before unlink on Windows
                os.unlink(temp_filename)


def test_write_tile_ir_bytecode_invalid_module():
    with Context() as ctx:
        # Create a module without cuda_tile content.
        with Location.unknown(ctx):
            mlir_module = Module.parse(
                """
                builtin.module {
                }
            """
            )

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_filename = f.name
            try:
                # This method flushes the file to disk.
                result = writeBytecode(f, mlir_module.operation)
                assert result == False
            finally:
                f.close()  # Must close before unlink on Windows
                os.unlink(temp_filename)


def test_write_preview_tile_ir_bytecode_requires_explicit_feature():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        assert setPreviewFeatures(ctx, "simt")

        with Location.unknown(ctx):
            mlir_module = Module.parse(
                """
                module {
                    cuda_tile.module @preview_test {
                        cuda_tile.entry @entry_func() {
                            cuda_tile.preview$bytecode_test_op req = 7 opt = 9
                            cuda_tile.return
                        }
                    }
                }
            """
            )

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_filename = f.name
            try:
                assert setPreviewFeatures(ctx, "")
                assert not writeBytecode(f, mlir_module.operation)
                assert os.path.getsize(f.name) == 0
                assert setPreviewFeatures(ctx, "simt")
                assert writeBytecode(f, mlir_module.operation)
                assert os.path.getsize(f.name) > 0
            finally:
                f.close()
                os.unlink(temp_filename)


def test_set_preview_features_rejects_unknown_feature():
    with Context() as ctx:
        register_dialect(ctx, load=True)
        assert not setPreviewFeatures(ctx, "not_a_feature")


def test_public_simt_decl_and_impl_require_preview_bytecode():
    from cuda_tile._mlir import ir
    from cuda_tile._mlir.dialects import cuda_tile_ops

    with Context() as ctx, Location.unknown(ctx):
        register_dialect(ctx, load=True)
        assert setPreviewFeatures(ctx, "simt")
        unknown_loc = Location.unknown(ctx)

        mlir_module = Module.create()
        with ir.InsertionPoint(mlir_module.body):
            cuda_module = cuda_tile_ops.ModuleOp("public_simt", loc=unknown_loc)
            with ir.InsertionPoint(cuda_module.body):
                tile_type = cuda_tile_ops.make_tile_type(cuda_tile_ops.Float32, [32])
                input_access = ir.DictAttr.get(
                    {
                        "cuda_tile.preview$access": ir.Attribute.parse(
                            "#cuda_tile.preview$access<inout>"
                        )
                    }
                )
                fragment_type = cuda_tile_ops.FragmentType.get(
                    [1],
                    cuda_tile_ops.Float32.mlir_type,
                    "local",
                    ir.Attribute.parse("#cuda_tile.preview$layout<(1):(1)>"),
                    ir.Attribute.parse(
                        "#cuda_tile.preview$layout<(1, 32, 1):(32, 1, 0)>"
                    ),
                )
                cuda_tile_ops.extern_func(
                    "public_decl",
                    [tile_type],
                    [tile_type],
                    arg_attrs=[input_access],
                    loc=unknown_loc,
                )
                cuda_tile_ops.extern_impl(
                    "public_impl",
                    "public_decl",
                    [fragment_type],
                    [],
                    arg_attrs=[input_access],
                    synchronization="warp",
                    arch=["sm90"],
                    loc=unknown_loc,
                )

        text = str(mlir_module)
        assert "preview$func" in text
        assert "public_decl" in text
        assert "preview$extern_impl" in text
        assert "public_impl" in text
        assert "synchronization = #cuda_tile<preview$synchronization<warp>>" in text
        assert 'arch = ["sm90"]' in text

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_filename = f.name
            try:
                assert setPreviewFeatures(ctx, "")
                assert not writeBytecode(f, mlir_module.operation)
                assert os.path.getsize(f.name) == 0
                assert setPreviewFeatures(ctx, "simt")
                assert writeBytecode(f, mlir_module.operation)
                assert os.path.getsize(f.name) > 0
            finally:
                f.close()
                os.unlink(temp_filename)


@pytest.mark.parametrize("config", [None, {}])
def test_public_simt_impl_rejects_reserved_config(config):
    from cuda_tile._mlir.dialects import cuda_tile_ops

    with Context() as ctx, Location.unknown(ctx):
        register_dialect(ctx, load=True)
        with pytest.raises(TypeError, match="unexpected keyword argument 'config'"):
            cuda_tile_ops.extern_impl(
                "public_impl", "public_decl", [], [], config=config
            )


def test_public_simt_impl_rejects_unknown_synchronization():
    from cuda_tile._mlir import ir
    from cuda_tile._mlir.dialects import cuda_tile_ops

    with Context() as ctx, Location.unknown(ctx):
        register_dialect(ctx, load=True)
        assert setPreviewFeatures(ctx, "simt")

        mlir_module = Module.create()
        with ir.InsertionPoint(mlir_module.body):
            cuda_module = cuda_tile_ops.ModuleOp("public_simt")
            with ir.InsertionPoint(cuda_module.body):
                fragment_type = cuda_tile_ops.FragmentType.get(
                    [1],
                    cuda_tile_ops.Float32.mlir_type,
                    "local",
                    ir.Attribute.parse("#cuda_tile.preview$layout<(1):(1)>"),
                    ir.Attribute.parse(
                        "#cuda_tile.preview$layout<(1, 32, 1):(32, 1, 0)>"
                    ),
                )
                with pytest.raises(ValueError, match="synchronization must be one of"):
                    cuda_tile_ops.extern_impl(
                        "public_impl",
                        "public_decl",
                        [fragment_type],
                        [],
                        synchronization="cluster",
                    )


def test_public_simt_call_and_tile_library_bridge():
    """Verify public calls and ordered tile-library registration work together."""
    from cuda_tile._mlir import ir
    from cuda_tile._mlir.dialects import cuda_tile_ops

    assert cuda_tile_ops.call.__module__ == cuda_tile_ops.__name__
    assert cuda_tile_ops.use_tile_library.__module__ == cuda_tile_ops.__name__

    with Context() as ctx, Location.unknown(ctx):
        register_dialect(ctx, load=True)
        assert setPreviewFeatures(ctx, "simt")

        mlir_module = Module.parse(
            """
            module {
              cuda_tile.module @public_simt {
                cuda_tile.entry @kernel() {
                  cuda_tile.return
                }
              }
            }
            """
        )
        cuda_module = mlir_module.body.operations[0]
        entry = cuda_module.regions[0].blocks[0].operations[0]
        entry_block = entry.regions[0].blocks[0]

        with ir.InsertionPoint.at_block_begin(entry_block):
            cuda_tile_ops.use_tile_library("first.tilelib")
            cuda_tile_ops.use_tile_library("first.tilelib")
            cuda_tile_ops.use_tile_library("second.tilelib")
            result_type = cuda_tile_ops.make_tile_type(cuda_tile_ops.Float32, [32])
            result = cuda_tile_ops.call(
                "public_decl", [], [result_type], const_args=[7]
            )

        assert isinstance(result, cuda_tile_ops.Tile)
        text = str(mlir_module)
        assert "preview$call @public_decl" in text
        assert "first.tilelib" in text
        assert "second.tilelib" in text

        assert cuda_tile_ops._consume_module_tile_libraries(mlir_module) == [
            "first.tilelib",
            "second.tilelib",
        ]
        assert "cuda_tile.tile_libraries" not in str(mlir_module)
