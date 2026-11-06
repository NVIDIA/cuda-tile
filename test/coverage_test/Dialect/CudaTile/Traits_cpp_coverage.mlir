// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// Traits.cpp coverage: exercise the PointerType branch in
// verifyLoadStoreType, plus additional mask/padding mismatch patterns.
//
// Focuses on:
//   - verifyLoadStoreType: PointerType dyn_cast success path (ptr<T>)
//   - verifyLoadStoreType: PointerType with element type mismatch
//   - verifyLoadStoreType: PointerType with shape mismatch
//   - verifyLoadStoreMask: 1D mask shape mismatches for load and store
//   - verifyLoadPadding: combined shape + element type mismatch

// Covers verifyLoadStoreMask — 1D mask shape mismatch for load.
cuda_tile.module @load_mask_1d_mismatch {
  cuda_tile.testing$func @load_mask_1d(
    %src: !cuda_tile.tile<16xptr<f32>>,
    %mask: !cuda_tile.tile<8xi1>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that shape of 'mask' must match the shape of 'source'}}
    cuda_tile.load_ptr_tko weak %src, %mask token=%t
      : tile<16xptr<f32>>, tile<8xi1> -> tile<16xf32>, token
  }
}

// -----

// Covers verifyLoadPadding — padding with wrong element type AND wrong shape,
// triggering both the shape and element type check (short-circuit at shape).
cuda_tile.module @load_pad_both_mismatch {
  cuda_tile.testing$func @load_pad_both(
    %src: !cuda_tile.tile<16xptr<f32>>,
    %mask: !cuda_tile.tile<16xi1>,
    %pad: !cuda_tile.tile<8xi32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that type of 'paddingValue' must match the type of 'result'}}
    cuda_tile.load_ptr_tko weak %src, %mask, %pad token=%t
      : tile<16xptr<f32>>, tile<16xi1>, tile<8xi32> -> tile<16xf32>, token
  }
}

// -----

// Covers verifyLoadStoreMask — store with 1D mask size mismatch.
cuda_tile.module @store_mask_1d_mismatch {
  cuda_tile.testing$func @store_mask_1d(
    %dst: !cuda_tile.tile<16xptr<i32>>,
    %val: !cuda_tile.tile<16xi32>,
    %mask: !cuda_tile.tile<32xi1>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.store_ptr_tko' op failed to verify that shape of 'destination' must match the shape of 'mask'}}
    %t1 = cuda_tile.store_ptr_tko weak %dst, %val, %mask token=%t
      : tile<16xptr<i32>>, tile<16xi32>, tile<32xi1> -> token
  }
}

// -----

// Traits.cpp coverage: exercise verifyLoadStoreType, verifyLoadStoreMask, verifyLoadPadding
// with varied type combinations to maximize branch coverage.

// Valid load with 1D tile and i32 element type - exercises verifyLoadStoreType success path.
cuda_tile.module @valid_load_1d_i32 {
  testing$func @load_1d_i32(%src: !cuda_tile.tile<16xptr<i32>>) {
    %t = make_token : !cuda_tile.token
    %res, %t1 = load_ptr_tko weak %src token=%t
      : tile<16xptr<i32>> -> tile<16xi32>, token
    return
  }
}

// -----

// Valid load with 2D tile, f16, mask, and padding - exercises all three verifiers.
cuda_tile.module @valid_load_f16_mask_pad {
  testing$func @load_f16_mask_pad(
    %src: !cuda_tile.tile<8x16xptr<f16>>,
    %mask: !cuda_tile.tile<8x16xi1>,
    %pad: !cuda_tile.tile<8x16xf16>) {
    %t = make_token : !cuda_tile.token
    %res, %t1 = load_ptr_tko weak %src, %mask, %pad token=%t
      : tile<8x16xptr<f16>>, tile<8x16xi1>, tile<8x16xf16> -> tile<8x16xf16>, token
    return
  }
}

// -----

// Valid load with bf16 element type.
cuda_tile.module @valid_load_bf16 {
  testing$func @load_bf16(%src: !cuda_tile.tile<32xptr<bf16>>) {
    %t = make_token : !cuda_tile.token
    %res, %t1 = load_ptr_tko weak %src token=%t
      : tile<32xptr<bf16>> -> tile<32xbf16>, token
    return
  }
}

// -----

// Valid store with 2D tile and mask - exercises verifyLoadStoreType and verifyLoadStoreMask.
cuda_tile.module @valid_store_mask {
  testing$func @store_with_mask_i32(
    %dst: !cuda_tile.tile<4x8xptr<i32>>,
    %val: !cuda_tile.tile<4x8xi32>,
    %mask: !cuda_tile.tile<4x8xi1>) {
    %t = make_token : !cuda_tile.token
    %t1 = store_ptr_tko weak %dst, %val, %mask token=%t
      : tile<4x8xptr<i32>>, tile<4x8xi32>, tile<4x8xi1> -> token
    return
  }
}

// -----

// Invalid: load element type mismatch (ptr<i32> but result f32)
// Triggers verifyLoadStoreType returning false (pointeeType != dstElementType).
cuda_tile.module @invalid_load_elem_mismatch {
  cuda_tile.testing$func @load_elem_mismatch(%src: !cuda_tile.tile<16xptr<i32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that `source` type is expected a pointer type of `result` type}}
    cuda_tile.load_ptr_tko weak %src token=%t
      : tile<16xptr<i32>> -> tile<16xf32>, token
  }
}

// -----

// Invalid: load shape mismatch (source 8x16 but result 8x32)
// Triggers verifyLoadStoreType returning false (shape mismatch).
cuda_tile.module @invalid_load_shape_mismatch {
  cuda_tile.testing$func @load_shape_mismatch(%src: !cuda_tile.tile<8x16xptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that `source` type is expected a pointer type of `result` type}}
    cuda_tile.load_ptr_tko weak %src token=%t
      : tile<8x16xptr<f32>> -> tile<8x32xf32>, token
  }
}

// -----

// Invalid: load mask shape mismatch
// Triggers verifyLoadStoreMask returning false.
cuda_tile.module @invalid_load_mask_shape {
  cuda_tile.testing$func @load_mask_shape(%src: !cuda_tile.tile<8x16xptr<f32>>, %mask: !cuda_tile.tile<8x32xi1>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that shape of 'mask' must match the shape of 'source'}}
    cuda_tile.load_ptr_tko weak %src, %mask token=%t
      : tile<8x16xptr<f32>>, tile<8x32xi1> -> tile<8x16xf32>, token
  }
}

// -----

// Invalid: load padding shape mismatch
// Triggers verifyLoadPadding returning false (shape mismatch).
cuda_tile.module @invalid_load_padding_shape {
  cuda_tile.testing$func @load_pad_shape(
    %src: !cuda_tile.tile<8x16xptr<f32>>,
    %mask: !cuda_tile.tile<8x16xi1>,
    %pad: !cuda_tile.tile<8x32xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that type of 'paddingValue' must match the type of 'result'}}
    cuda_tile.load_ptr_tko weak %src, %mask, %pad token=%t
      : tile<8x16xptr<f32>>, tile<8x16xi1>, tile<8x32xf32> -> tile<8x16xf32>, token
  }
}

// -----

// Invalid: load padding element type mismatch
// Triggers verifyLoadPadding returning false (element type mismatch).
cuda_tile.module @invalid_load_padding_elem {
  cuda_tile.testing$func @load_pad_elem(
    %src: !cuda_tile.tile<8x16xptr<f32>>,
    %mask: !cuda_tile.tile<8x16xi1>,
    %pad: !cuda_tile.tile<8x16xf16>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.load_ptr_tko' op failed to verify that type of 'paddingValue' must match the type of 'result'}}
    cuda_tile.load_ptr_tko weak %src, %mask, %pad token=%t
      : tile<8x16xptr<f32>>, tile<8x16xi1>, tile<8x16xf16> -> tile<8x16xf32>, token
  }
}

// -----

// Invalid: store element type mismatch
// Triggers verifyLoadStoreType returning false (destination ptr type vs value type mismatch).
cuda_tile.module @invalid_store_elem_mismatch {
  cuda_tile.testing$func @store_elem_mismatch(
    %dst: !cuda_tile.tile<16xptr<f32>>,
    %val: !cuda_tile.tile<16xi32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.store_ptr_tko' op failed to verify that `destination` type is expected a pointer type of `value` type}}
    %t1 = cuda_tile.store_ptr_tko weak %dst, %val token=%t
      : tile<16xptr<f32>>, tile<16xi32> -> token
  }
}

// -----

// Invalid: store shape mismatch
// Triggers verifyLoadStoreType returning false (shape mismatch).
cuda_tile.module @invalid_store_shape_mismatch {
  cuda_tile.testing$func @store_shape_mismatch(
    %dst: !cuda_tile.tile<8x16xptr<f32>>,
    %val: !cuda_tile.tile<8x32xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.store_ptr_tko' op failed to verify that `destination` type is expected a pointer type of `value` type}}
    %t1 = cuda_tile.store_ptr_tko weak %dst, %val token=%t
      : tile<8x16xptr<f32>>, tile<8x32xf32> -> token
  }
}

// -----

// Invalid: store mask shape mismatch
// Triggers verifyLoadStoreMask returning false.
cuda_tile.module @invalid_store_mask_shape {
  cuda_tile.testing$func @store_mask_shape(
    %dst: !cuda_tile.tile<8x16xptr<f32>>,
    %val: !cuda_tile.tile<8x16xf32>,
    %mask: !cuda_tile.tile<8x32xi1>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.store_ptr_tko' op failed to verify that shape of 'destination' must match the shape of 'mask'}}
    %t1 = cuda_tile.store_ptr_tko weak %dst, %val, %mask token=%t
      : tile<8x16xptr<f32>>, tile<8x16xf32>, tile<8x32xi1> -> token
  }
}
