// RUN: cuda-tile-opt %s -split-input-file | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt -split-input-file | FileCheck %s

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers StoreViewTkoOp::parse/print roundtrip
// Covers AtomicRedViewTkoOp::parse/print roundtrip
// Covers LoadViewTkoOp parse/print with various memory orderings
// Covers parseViewTkoCommon, parseIndexTypes, finalizeViewTkoParse helpers
// Covers printViewTkoCommon, printViewTkoOptionals, printIndexTypes helpers

// StoreViewTkoOp roundtrip with partition_view
cuda_tile.module @store_view_tko_roundtrip {
  // CHECK-LABEL: entry @test_store_view_tko
  entry @test_store_view_tko(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128], strides = [1] : !cuda_tile.tensor_view<128xf32, strides = [1]>
    %pv = cuda_tile.make_partition_view %tv : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>
    %val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<8xf32>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: store_view_tko weak
    %tok = cuda_tile.store_view_tko weak %val, %pv[%c0] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    return
  }
}

// -----

// AtomicRedViewTkoOp roundtrip
cuda_tile.module @atomic_red_view_tko_roundtrip {
  // CHECK-LABEL: entry @test_atomic_red_view_tko
  entry @test_atomic_red_view_tko(%p : !cuda_tile.tile<ptr<i32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [64, 64], strides = [64, 1] : !cuda_tile.tensor_view<64x64xi32, strides = [64, 1]>
    %pv = cuda_tile.make_partition_view %tv : partition_view<tile=(8x8), tensor_view<64x64xi32, strides=[64,1]>>
    %val = cuda_tile.constant <i32: 1> : !cuda_tile.tile<8x8xi32>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: atomic_red_view_tko relaxed device
    // CHECK-SAME: add
    %tok = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], add, %val : tile<8x8xi32>, partition_view<tile=(8x8), tensor_view<64x64xi32, strides=[64,1]>>, tile<i32> -> token
    return
  }
}

// -----

// LoadViewTkoOp roundtrip with different memory orderings
cuda_tile.module @load_view_tko_roundtrip {
  // CHECK-LABEL: entry @test_load_view_tko
  entry @test_load_view_tko(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128], strides = [1] : !cuda_tile.tensor_view<128xf32, strides = [1]>
    %pv = cuda_tile.make_partition_view %tv : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: load_view_tko weak
    %val, %tok = cuda_tile.load_view_tko weak %pv[%c0] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> tile<8xf32>, token
    return
  }
}

// -----

// StoreViewTkoOp with strided_view
cuda_tile.module @store_view_tko_strided {
  // CHECK-LABEL: entry @test_store_strided
  entry @test_store_strided(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128, 128], strides = [128, 1] : !cuda_tile.tensor_view<128x128xf32, strides = [128, 1]>
    %sv = cuda_tile.make_strided_view %tv : strided_view<tile=(8x8), traversal_strides=[8,8], tensor_view<128x128xf32, strides=[128,1]>>
    %val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<8x8xf32>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: store_view_tko weak
    %tok = cuda_tile.store_view_tko weak %val, %sv[%c0, %c0] : tile<8x8xf32>, strided_view<tile=(8x8), traversal_strides=[8,8], tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> token
    return
  }
}
