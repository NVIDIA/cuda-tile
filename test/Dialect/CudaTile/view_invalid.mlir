// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// ****************** cuda_tile.make_tensor_view ******************
// expected-error @below{{strides must not be provided for 0-d tiles}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<f32, strides=[]>

// -----

// expected-error @below{{expected strictly positive integer, got -5}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<?xf32, strides=[-5]>

// -----

// expected-error @below{{expected strictly positive integer, got 0}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<?xf32, strides=[0]>

// -----

// expected-error @below{{expected shape and stride to be of same rank but got shape of rank 1 and stride of rank 2}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<?xf32, strides=[4, 1]>

// -----

// Ensure the explicit value of kDynamic is not treated as such.
// expected-error @below{{expected strictly positive integer, got -9223372036854775808}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<?xf32, strides=[-9223372036854775808]>

// -----

// expected-error @below{{expected either 64-bit integer or question mark}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<?x32xf32, strides=[, 32]>

// -----

// expected-error @below{{expected 'strides'}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<2xf32>

// -----

// expected-error @below{{expected token after element type in 0-d tensor_view}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<f16,>
 
// -----

// expected-error @below{{dimensions must have strictly positive constant sizes but got [0]}}
%0 = "use_type"() : () -> !cuda_tile.tensor_view<0xf32, strides=[1]>

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_too_many_dyn_shapes(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{expected 0 dynamic shape operands, got 1}}
    "cuda_tile.make_tensor_view"(%base, %ci64) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<i64>) -> !cuda_tile.tensor_view<32xf32, strides=[1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_too_many_dyn_strides(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{expected 0 dynamic stride operands, got 1}}
    "cuda_tile.make_tensor_view"(%base, %ci64) <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (!cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<i64>) -> !cuda_tile.tensor_view<32xf32, strides=[1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_missing_dynamic_strides(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected 1 dynamic shape operands, got 0}}
    "cuda_tile.make_tensor_view"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!cuda_tile.tile<!cuda_tile.ptr<f32>>) -> !cuda_tile.tensor_view<?xf32, strides=[1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_missing_dynamic_strides(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected 1 dynamic stride operands, got 0}}
    "cuda_tile.make_tensor_view"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!cuda_tile.tile<!cuda_tile.ptr<f32>>) -> !cuda_tile.tensor_view<32xf32, strides=[?]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_wrong_type(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected pointer to 'f64' to build tensor_view of this type, got 'f32'}}
    "cuda_tile.make_tensor_view"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!cuda_tile.tile<!cuda_tile.ptr<f32>>) -> !cuda_tile.tensor_view<f64>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_amount(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected shape declaration to contain 2 elements due to tensor_view type, but 0 were provided}}
    cuda_tile.make_tensor_view %base, shape = [], strides = [32, 1] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_amount(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected stride declaration to contain 2 elements due to tensor_view type, but 0 were provided}}
    cuda_tile.make_tensor_view %base, shape = [32, 32], strides = [] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_value(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{input shape dimension 1 does not match tensor_view type (expected 32, got 64)}}
    cuda_tile.make_tensor_view %base, shape = [32, 64], strides = [32, 1] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_value(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{input stride dimension 0 does not match tensor_view type (expected 32, got 64)}}
    cuda_tile.make_tensor_view %base, shape = [32, 32], strides = [64, 1] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_kind(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input shape dimension 2 does not match tensor_view type (expected 32, got dynamic)}}
    cuda_tile.make_tensor_view %base, shape = [2, %ci64, %ci64], strides = [64, 32, 1] : tile<i64> -> tensor_view<2x?x32xf32, strides=[64, 32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_kind2(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input shape dimension 1 does not match tensor_view type (expected dynamic, got 32)}}
    cuda_tile.make_tensor_view %base, shape = [2, 32, 32], strides = [64, 32, 1] : tensor_view<2x?x32xf32, strides=[64, 32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_kind(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input stride dimension 1 does not match tensor_view type (expected 32, got dynamic)}}
    cuda_tile.make_tensor_view %base, shape = [2, %ci64, 32], strides = [64, %ci64, 1] : tile<i64> -> tensor_view<2x?x32xf32, strides=[64, 32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_kind2(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input stride dimension 1 does not match tensor_view type (expected dynamic, got 32)}}
    cuda_tile.make_tensor_view %base, shape = [2, %ci64, 32], strides = [64, 32, 1] : tile<i64> -> tensor_view<2x?x32xf32, strides=[64, ?, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_garbage_in(%base: !cuda_tile.tile<!cuda_tile.ptr<f64>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{expected either integer or SSA value}}
    cuda_tile.make_tensor_view %base, shape = [32, sdfsdffds], strides = [] : tensor_view<f32>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_wrong_type(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected pointer to 'f64' to build tensor_view of this type, got 'f32'}}
    "cuda_tile.make_tensor_view"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!cuda_tile.tile<!cuda_tile.ptr<f32>>) -> !cuda_tile.tensor_view<f64>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_amount(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected shape declaration to contain 2 elements due to tensor_view type, but 0 were provided}}
    cuda_tile.make_tensor_view %base, shape = [], strides = [32, 1] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_amount(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected stride declaration to contain 2 elements due to tensor_view type, but 0 were provided}}
    cuda_tile.make_tensor_view %base, shape = [32, 32], strides = [] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_value(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{input shape dimension 1 does not match tensor_view type (expected 32, got 64)}}
    cuda_tile.make_tensor_view %base, shape = [32, 64], strides = [32, 1] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_value(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{input stride dimension 0 does not match tensor_view type (expected 32, got 64)}}
    cuda_tile.make_tensor_view %base, shape = [32, 32], strides = [64, 1] : tensor_view<32x32xf32, strides=[32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_kind(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input shape dimension 2 does not match tensor_view type (expected 32, got dynamic)}}
    cuda_tile.make_tensor_view %base, shape = [2, %ci64, %ci64], strides = [64, 32, 1] : tile<i64> -> tensor_view<2x?x32xf32, strides=[64, 32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_shape_kind2(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input shape dimension 1 does not match tensor_view type (expected dynamic, got 32)}}
    cuda_tile.make_tensor_view %base, shape = [2, 32, 32], strides = [64, 32, 1] : tensor_view<2x?x32xf32, strides=[64, 32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_kind(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input stride dimension 1 does not match tensor_view type (expected 32, got dynamic)}}
    cuda_tile.make_tensor_view %base, shape = [2, %ci64, 32], strides = [64, %ci64, 1] : tile<i64> -> tensor_view<2x?x32xf32, strides=[64, 32, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_inconsistent_stride_kind2(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{input stride dimension 1 does not match tensor_view type (expected dynamic, got 32)}}
    cuda_tile.make_tensor_view %base, shape = [2, %ci64, 32], strides = [64, 32, 1] : tile<i64> -> tensor_view<2x?x32xf32, strides=[64, ?, 1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_garbage_in(%base: !cuda_tile.tile<!cuda_tile.ptr<f64>>, %ci64: !cuda_tile.tile<i64>) {
    // expected-error @below{{expected either integer or SSA value}}
    cuda_tile.make_tensor_view %base, shape = [32, sdfsdffds], strides = [] : tensor_view<f32>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_invalid_element_type(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error-re @below{{failed to verify 'elementType': f16 or bf16 or f32 or tf32 or f64 or f8E4M3FN or f8E5M2 {{(or f4E2M1FN )?}}or i1 or i8 or i16 or i32 or i64}}
    cuda_tile.make_tensor_view %arg0, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xptr<f32>, strides=[32,1]>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_tensor_view_wrong_index_type(%arg0: !cuda_tile.tile<ptr<f64>>) {
    // expected-error @below{{op operand #1 must be variadic of 0D tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<ptr<f64>>'}}
    %9 = make_tensor_view %arg0, shape = [%arg0, %arg0, %arg0, %arg0], strides = [%arg0, 1, %arg0, %arg0] : !cuda_tile.tile<ptr<f64>> -> !cuda_tile.tensor_view<?x?x?x?xf64, strides=[?,1,?,?]>
  }
}

// -----

// ****************** cuda_tile.make_partition_view ******************
// expected-error @below{{expected dim_map to map exactly all 2 dimensions of the tile, got 1 mappings}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[0]>

// -----

// expected-error @below{{target dimension is outside of tensor view dimensions, expected strictly less than 2, got 2}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[2, 1]>

// -----

// expected-error @below{{target dimension 0 mapped at least twice (for tile dimensions 0 and 1)}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[0, 0]>

// -----

// expected-error @below{{tile shape dimensions must have power of two length but got [5, 1024]}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(5x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

// expected-error @below{{tile dimension 0 exceeds i32 limitations (got 1099511627776, expected strictly positive and less than or equal to 2147483647)}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(1099511627776x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

// expected-error @below{{expected tensor_view rank and tile rank to match, got tensor_view of rank 3 and tiles of rank 2}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(1x1), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>

// -----

// expected-error @below{{0-dimension tile shape is not supported}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(), !cuda_tile.tensor_view<f32>>

// -----

// expected-error @below{{target dimension must not be negative, got -1}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[-1, 1]>

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_partition_view_wrong_tensor_view_elem(%tensor_view: !cuda_tile.tensor_view<4096x4096xf64, strides=[4096,1]>) {
    // expected-note @above{{prior use here}}
    // expected-error @below{{expects different type than prior uses}}
    cuda_tile.make_partition_view %tensor_view : !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_partition_view_wrong_tensor_view_shape(%tensor_view: !cuda_tile.tensor_view<4096x2048xf32, strides=[4096,1]>) {
    // expected-note @above{{prior use here}}
    // expected-error @below{{expects different type than prior uses}}
    cuda_tile.make_partition_view %tensor_view : !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>
  }
}

// -----

// ****************** cuda_tile.load_view_tko ******************
cuda_tile.module @module {
  cuda_tile.testing$func @tile_partition_wrong_load_type(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected tile type to be '!cuda_tile.tile<1024x1024xf32>' (based on view type), got '!cuda_tile.tile<8xf32>'}}
    load_view_tko weak %view[%c0, %c0] : partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, tile<i32> -> tile<8xf32>, token
  }
}

// -----

// This test uses generic format to test the verifier itself, as the parser already requires this property.
cuda_tile.module @module {
  cuda_tile.testing$func @tile_partition_wrong_load_rank(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 index operands (based on view type), got 1}}
    "cuda_tile.load_view_tko"(%view, %c0) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!cuda_tile.partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>) -> (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @load_view_tko_non_view_type(%tile: !cuda_tile.tile<32xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{operand #0 must be TileView instance, but got '!cuda_tile.tile<32xf32>'}}
    %x, %t = load_view_tko weak %tile[%c0] : !cuda_tile.tile<32xf32>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    cuda_tile.print "%f\n", %x : !cuda_tile.tile<8xf32>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @load_view_tko_index_type_mismatch(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c0_i64 = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
    // expected-error @below{{expected index type 1 to be the same as other index types ('!cuda_tile.tile<i32>'), got '!cuda_tile.tile<i64>'}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %c0_i32, %c0_i64) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 2, 0>}> : (!cuda_tile.partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i64>) -> (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @load_view_tko_invalid_memory_ordering(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expect one of: weak, relaxed, or acquire, but got: release}}
    %x, %t = load_view_tko release %view[%c0, %c0] : partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, tile<i32> -> tile<1024x1024xf32>, token
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.testing$func @load_missing_index(%memref_i8: !cuda_tile.tensor_view<1024xi8, strides=[1]>) {
    %view_i8 = make_partition_view %memref_i8 : partition_view<tile=(128), tensor_view<1024xi8, strides=[1]>>
    // expected-error @below{{expected 1 index operands (based on view type), got 0}}
    %tile_i8_l, %tok_i8 = load_view_tko weak %view_i8[] : partition_view<tile=(128), tensor_view<1024xi8, strides=[1]>>, tile<i32> -> tile<128xi8>, token
  }
}

// -----

// ****************** cuda_tile.store_view_tko ******************

// This test uses generic format to test the verifier itself, as the parser already requires this property.
cuda_tile.module @module {
  cuda_tile.testing$func @tile_partition_wrong_store_rank(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>, %tile: !cuda_tile.tile<1024x1024xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 index operands (based on view type), got 1}}
    "cuda_tile.store_view_tko"(%tile, %view, %c0) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>) -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @store_view_tko_non_view_type(%tile: !cuda_tile.tile<32xf32>, %non_view: !cuda_tile.tile<32xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{operand #1 must be TileView instance, but got '!cuda_tile.tile<32xf32>'}}
    %t = store_view_tko weak %tile, %non_view[%c0] : !cuda_tile.tile<32xf32>, !cuda_tile.tile<32xf32>, tile<i32> -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @store_view_tko_index_type_mismatch(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>, %tile: !cuda_tile.tile<1024x1024xf32>) {
    %c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c0_i64 = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
    // expected-error @below{{expected index type 1 to be the same as other index types ('!cuda_tile.tile<i32>'), got '!cuda_tile.tile<i64>'}}
    %t = "cuda_tile.store_view_tko"(%tile, %view, %c0_i32, %c0_i64) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 0>}> : (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i64>) -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @store_view_tko_invalid_memory_ordering_acquire(%view: !cuda_tile.partition_view<tile=(1024x1024), !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>, %tile: !cuda_tile.tile<1024x1024xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expect one of: weak, relaxed, or release, but got: acquire}}
    %t = store_view_tko acquire %tile, %view[%c0, %c0] : tile<1024x1024xf32>, partition_view<tile=(1024x1024), tensor_view<4096x4096xf32, strides=[4096,1]>>, tile<i32> -> token
  }
}
