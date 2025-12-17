// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// ****************** cuda_tile.get_tensor_shape ******************

cuda_tile.module @test_dim_tensor_view_oob {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) {
    // expected-error @below{{operation defines 2 results but was provided 3 to bind}}
    %0:3 = cuda_tile.get_tensor_shape %tensor_view : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]> -> !cuda_tile.tile<i32>
  }
}

// -----

// This test uses generic format to test the verifier itself.
cuda_tile.module @test_dim_tensor_view_oob_generic {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) {
    // expected-error @below{{expected 2 results due to tensor rank, but got 3}}
    %0:3 = "cuda_tile.get_tensor_shape"(%tensor_view) : (!cuda_tile.tensor_view<64x64xf16, strides=[1,1]>) -> (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>)
  }
}

// -----

cuda_tile.module @test_dim_invalid_input_type {
  testing$func @kernel(%value : !cuda_tile.tile<8x8x!cuda_tile.ptr<i32>>) {
    // expected-error @below{{'cuda_tile.get_tensor_shape' expected tensor_view, got '!cuda_tile.tile<8x8xptr<i32>>'}}
    %0 = cuda_tile.get_tensor_shape %value : !cuda_tile.tile<8x8x!cuda_tile.ptr<i32>> -> !cuda_tile.tile<i32>
  }
}

// -----

cuda_tile.module @test_dim_invalid_output_type {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xi32, strides=[1,1]>) {
    // expected-error @below{{'cuda_tile.get_tensor_shape' op result #0 must be variadic of 0D tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<2xi32>'}}
    %0:2 = cuda_tile.get_tensor_shape %tensor_view : !cuda_tile.tensor_view<64x64xi32, strides=[1,1]> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @test_dim_invalid_result_element_type {
  testing$func @kernel(%tensor_view : !cuda_tile.tensor_view<64x64xi32, strides=[1,1]>) {
    // expected-error @below{{'cuda_tile.get_tensor_shape' op result #0 must be variadic of 0D tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<f32>'}}
    %0:2 = cuda_tile.get_tensor_shape %tensor_view : !cuda_tile.tensor_view<64x64xi32, strides=[1,1]> -> !cuda_tile.tile<f32>
  }
}

// -----

// ****************** cuda_tile.get_index_space_shape ******************

// Test that get_index_space_shape op fails when the index is out of bounds for the tile view.
cuda_tile.module @test_get_index_space_shape_oob {
  testing$func @kernel(%view: !cuda_tile.partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>>) {
    // expected-error @below{{operation defines 2 results but was provided 1 to bind}}
    %0 = get_index_space_shape %view : partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>> -> tile<i32>
  }
}

// -----

// Test that get_index_space_shape op fails when the index is out of bounds for the tile view.
// This test uses generic format to test the verifier itself.
cuda_tile.module @test_get_index_space_shape_oob {
  testing$func @kernel(%view: !cuda_tile.partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>>) {
    // expected-error @below{{'cuda_tile.get_index_space_shape' op expected 2 results due to view index space rank, but got 1}}
    "cuda_tile.get_index_space_shape"(%view) : (!cuda_tile.partition_view<tile=(4x4), tensor_view<?x?xf32, strides=[1,1]>>) -> (!cuda_tile.tile<i32>)
  }
}
