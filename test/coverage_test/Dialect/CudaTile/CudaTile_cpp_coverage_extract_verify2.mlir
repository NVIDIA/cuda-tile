// RUN: cuda-tile-opt %s -split-input-file -verify-diagnostics -allow-unregistered-dialect

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ExtractOp::verify() - source and result element type mismatch
// Covers ExtractOp::verify() - wrong number of indices

// ExtractOp: element type mismatch
cuda_tile.module @extract_elem_type_mismatch {
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<8xf32>
    %idx = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{source and result element type do not match}}
    %r = cuda_tile.extract %src[%idx] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<4xi32>
  }
}

// -----

// ExtractOp: wrong number of indices (2D source, only 1 index)
cuda_tile.module @extract_wrong_index_count {
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<8x8xf32>
    %idx = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 indices, but got 1}}
    %r = cuda_tile.extract %src[%idx] : !cuda_tile.tile<8x8xf32> -> !cuda_tile.tile<4x4xf32>
  }
}
