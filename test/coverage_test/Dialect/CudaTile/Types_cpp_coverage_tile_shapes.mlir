// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// Types.cpp: TileType::verify shape rules on concrete shapes. The symbolic-shape
// case is preview-gated and lives in preview_Types_cpp_coverage_tile_shapes.mlir.

// Non-power-of-2 dimension.
cuda_tile.module @test {
  cuda_tile.testing$func @decl_non_pow2_dim(
      // expected-error @below {{all dimensions must be powers of two, got 3}}
      %arg0: !cuda_tile.tile<3xf32>) {
    cuda_tile.return
  }
}

// -----

// Negative dimension.
cuda_tile.module @test {
  cuda_tile.testing$func @decl_neg_dim(
      // expected-error @below {{all dimensions must be positive constants, got -1}}
      %arg0: !cuda_tile.tile<-1xf32>) {
    cuda_tile.return
  }
}

// -----

// F4E2M1FN with odd static elements: single dim 1 is odd.
cuda_tile.module @test {
  cuda_tile.testing$func @decl_f4_odd(
      // expected-error @below {{F4E2M1FN tiles must have an even number of elements}}
      %arg0: !cuda_tile.tile<1xf4E2M1FN>) {
    cuda_tile.return
  }
}

// -----

// F4E2M1FN with even static elements (valid, no error).
cuda_tile.module @test {
  cuda_tile.testing$func @decl_f4_even(%arg0: !cuda_tile.tile<2xf4E2M1FN>) {
    cuda_tile.return
  }
}

// -----

// Second dimension non-power-of-2 (tests i>0 path in enumerate).
cuda_tile.module @test {
  cuda_tile.testing$func @decl_second_dim_bad(
      // expected-error @below {{all dimensions must be powers of two, got 128, 5}}
      %arg0: !cuda_tile.tile<128x5xf32>) {
    cuda_tile.return
  }
}
