// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers GetIndexSpaceShapeOp::parse() - non-tile-view type error
// Covers GetTensorShapeOp::parse() - non-tensor-view type error

// GetIndexSpaceShapeOp: source is not a tile view
cuda_tile.module @get_idx_shape_not_view {
  entry @test(%v : !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expected tile view, got}}
    %0 = cuda_tile.get_index_space_shape %v : tile<8xf32> -> tile<i32>
    return
  }
}

// -----

// GetTensorShapeOp: source is not a tensor_view
cuda_tile.module @get_tensor_shape_not_view {
  entry @test(%v : !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expected tensor_view, got}}
    %0 = cuda_tile.get_tensor_shape %v : tile<8xf32> -> tile<i32>
    return
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ReturnOp::verify() - entry function returns mismatch path

cuda_tile.module @return_with_operand_in_entry {
  entry @test_entry_nonzero_return(%a : !cuda_tile.tile<i32>) {
    // expected-error @below{{has 1 operands, but enclosing function (@test_entry_nonzero_return) returns 0}}
    cuda_tile.return %a : tile<i32>
  }
}
