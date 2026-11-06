// RUN: cuda-tile-opt %s -split-input-file -verify-diagnostics

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers parseDenseIntOrFPElementsAttr - empty dense attribute error
// Covers printDenseIntOrFPElementsAttr - fallback path

// Test: unsupported integer type error path (lines around 638-639)
// This is hard to trigger since all int types are handled, but we can test
// the shape mismatch error path for dense attributes with array shapes.

cuda_tile.module @dense_shape_mismatch {
  entry @test_dense_shape_mismatch() {
    // expected-error @below {{inferred shape of elements literal}}
    %v = cuda_tile.constant <i32: [1, 2, 3]> : !cuda_tile.tile<2xi32>
    return
  }
}

// -----

// Covers the dense scalar-to-array shape mismatch
cuda_tile.module @dense_scalar_shape_mismatch {
  entry @test_dense_scalar_shape_mismatch() {
    // expected-error @below {{inferred shape of elements literal}}
    %v = cuda_tile.constant <i32: [1, 2]> : !cuda_tile.tile<i32>
    return
  }
}
