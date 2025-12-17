// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

cuda_tile.module @kernels {
  testing$func @permute_different_rank(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{failed to verify that all of {source, result} have same rank}}
    %0 = permute %arg0 [0, 1] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x1x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_different_element_type(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{failed to verify that all of {source, result} have the same element type}}
    %0 = permute %arg0 [0, 1] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xf64>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_small_rank(%arg0: !cuda_tile.tile<2xf32>) {
    // expected-error @below{{expects at least rank 2, but got: 1}}
    %0 = permute %arg0 [0] : !cuda_tile.tile<2xf32> -> !cuda_tile.tile<2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_too_many_element_in_perm(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{expect permutation size (3) to equal the rank of the source (2)}}
    %0 = permute %arg0 [0, 1, 100] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_not_complete_perm(%arg0: !cuda_tile.tile<1x2x4xf32>) {
    // expected-error @below{{expect permutation size (2) to equal the rank of the source (3)}}
    %0 = permute %arg0 [0, 1] : !cuda_tile.tile<1x2x4xf32> -> !cuda_tile.tile<1x2x4xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_perm_is_oob(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{permutation element at index 1 (100) is out of bound [0, 2)}}
    %0 = permute %arg0 [0, 100] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_perm_is_oob(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{permutation element at index 0 (-1) is out of bound [0, 2)}}
    %0 = permute %arg0 [-1, 1] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_perm_is_not_unique(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{expect permutation elements to be unique}}
    %0 = permute %arg0 [0, 0] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xf32>
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @permute_output_shape_invalid(%arg0: !cuda_tile.tile<1x2xf32>) {
    // expected-error @below{{result shape invalid at index 0, expected: 2, but got: 1}}
    %0 = permute %arg0 [1, 0] : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x1xf32>
  }
}
