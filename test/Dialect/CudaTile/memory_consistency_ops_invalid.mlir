// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

cuda_tile.module @invalid_new_token {
  testing$func @make_token_wrong_result_type() -> !cuda_tile.tile<i32> {
    // expected-error @+1 {{'cuda_tile.make_token' op result #0 must be cuda tile token type, but got '!cuda_tile.tile<i32>'}}
    %0 = make_token : tile<i32>
    return %0 : !cuda_tile.tile<i32>
  }
} // invalid_new_token

// -----

cuda_tile.module @invalid_join {
  testing$func @join_tokens_no_tokens() -> !cuda_tile.token {
    // expected-error @below{{expect two or more tokens}}
    %0 = join_tokens : token
    return %0 : !cuda_tile.token
  }
} // invalid_join

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @funcload(%arg0: !cuda_tile.tile<16x32xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{operand #0 must be tile of Pointer type values, but got '!cuda_tile.tile<16x32xf32>'}}
    load_ptr_tko weak %arg0 token=%t : tile<16x32xf32> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @load(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<i32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{`source` type is expected a pointer type of `result` type}}
    cuda_tile.load_ptr_tko weak %arg0 token=%t : tile<16x32xptr<i32>> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @load(%arg0: !cuda_tile.tile<16x64x!cuda_tile.ptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{`source` type is expected a pointer type of `result` type}}
    cuda_tile.load_ptr_tko weak %arg0 token=%t : tile<16x64xptr<f32>> -> tile<16x32xf32>, token
  }
}


// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @load_with_mask(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{operand #1 must be tile of i1 values, but got '!cuda_tile.tile<16x32xptr<f32>>'}}
    cuda_tile.load_ptr_tko weak %arg0, %arg1 token=%t 
      : tile<16x32xptr<f32>>, tile<16x32xptr<f32>> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @load_with_mask(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<16x64xi1>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{shape of 'mask' must match the shape of 'source'}}
    cuda_tile.load_ptr_tko weak %arg0, %arg1 token=%t 
      : tile<16x32xptr<f32>>, tile<16x64xi1> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @load_with_mask(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<16x32xi1>, %arg2: !cuda_tile.tile<16x64xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{type of 'paddingValue' must match the type of 'result'}}
    cuda_tile.load_ptr_tko weak %arg0, %arg1, %arg2 token=%t
      : tile<16x32xptr<f32>>, tile<16x32xi1>, tile<16x64xf32> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @load_with_mask(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<16x32xi1>, %arg2: !cuda_tile.tile<16x32xf16>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{type of 'paddingValue' must match the type of 'result'}}
    cuda_tile.load_ptr_tko weak %arg0, %arg1, %arg2 token=%t
      : tile<16x32xptr<f32>>, tile<16x32xi1>, tile<16x32xf16> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_store_ptr_tko {
  cuda_tile.testing$func @store(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1 : !cuda_tile.tile<16x64xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{op failed to verify that `destination` type is expected a pointer type of `value` type}}
    %t1 = store_ptr_tko weak %arg0, %arg1 token=%t : tile<16x32xptr<f32>>, tile<16x64xf32> -> token
  }
}

// -----

cuda_tile.module @invalid_store_ptr_tko {
  cuda_tile.testing$func @store(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1 : !cuda_tile.tile<16x32xf16>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{op failed to verify that `destination` type is expected a pointer type of `value` type}}
    %t1 = store_ptr_tko weak %arg0, %arg1 token=%t
      : tile<16x32xptr<f32>>, tile<16x32xf16> -> token
  }
}

// -----

cuda_tile.module @invalid_store_ptr_tko {
  cuda_tile.testing$func @store_with_mask(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<16x32xf32>, %arg2 : !cuda_tile.tile<16x64xi1>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{op failed to verify that shape of 'destination' must match the shape of 'mask'}}
    %t1 = store_ptr_tko weak %arg0, %arg1, %arg2 token=%t
      : tile<16x32xptr<f32>>, tile<16x32xf32>, tile<16x64xi1> -> token
  }
}

// -----

cuda_tile.module @invalid_store_ptr_tko {
  cuda_tile.testing$func @store_with_mask(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1: !cuda_tile.tile<16x32xf32>, %arg2 : !cuda_tile.tile<16x32xi8>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{'cuda_tile.store_ptr_tko' op operand #2 must be tile of i1 values}}
    %t1 = store_ptr_tko weak %arg0, %arg1, %arg2 token=%t
      : tile<16x32xptr<f32>>, tile<16x32xf32>, tile<16x32xi8> -> token
  }
}

// -----

cuda_tile.module @weak_token_ordered_load {
  testing$func @invalid_weak_load_with_scope(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below {{weak load must not have memory scope}}
    %0, %new_t = load_ptr_tko weak device %ptr token=%t
      : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
    return
  }
}

// -----

cuda_tile.module @token_ordered_load {
  testing$func @invalid_weak_load_with_scope(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error@below {{expect one of: weak, relaxed, or acquire, but got: release}}
    %0, %new_t = load_ptr_tko release device %ptr token=%t
      : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
    return
  }
}

// -----

cuda_tile.module @weak_token_ordered_store {
  testing$func @invalid_weak_store_with_scope(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %val: !cuda_tile.tile<16x32xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error@below {{weak store must not have memory scope}}
    %new_t = store_ptr_tko weak device %ptr, %val token=%t
      : tile<16x32xptr<f32>>, tile<16x32xf32> -> token
    return
  }
}

// -----

cuda_tile.module @invalid_store_ordering {
  testing$func @store_with_invalid_ordering(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %val: !cuda_tile.tile<16x32xf32>) {
    %t = make_token : !cuda_tile.token
    // expected-error@below {{expect one of: weak, relaxed, or release, but got: acquire}}
    %new_t = store_ptr_tko acquire device %ptr, %val token=%t
      : tile<16x32xptr<f32>>, tile<16x32xf32> -> token
    return
  }
}

// -----

cuda_tile.module @release_token_ordered_load {
  testing$func @invalid_weak_load_with_scope(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // expected-error@below {{weak load must not have memory scope}}
    %0, %new_t = load_ptr_tko weak device %ptr token=%t 
      : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
    return
  }
}

// -----

cuda_tile.module @release_token_ordered_load {
  testing$func @invalid_weak_load_with_scope(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    %t = make_token : !cuda_tile.token
    // The error here is not really great but that's the best we can do using assembly format.
    // expected-error@below {{expected SSA operand}}
    %0, %new_t = load_ptr_tko weak blah %ptr token=%t 
      : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
    return
  }
}

// -----

cuda_tile.module @tiled_view_load {
  // expected-note@below{{prior use here}}
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: i32) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{expects different type than prior uses: '!cuda_tile.token' vs 'i32'}}
    %tile_2, %tok_out = load_view_tko weak %arg0[%0, %0, %0] token = %arg1 : !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> !cuda_tile.tile<1024x1024x8xf32>, token
    return
  }
}

// -----

cuda_tile.module @tiled_view_load {
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: i32) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{op result #1 must be cuda tile token type, but got 'i32'}}
    %tile_2, %tok_out = load_view_tko weak %arg0[%0, %0, %0] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, i32
    return
  }
}

// -----

cuda_tile.module @tiled_view_load {
  // expected-note@below {{prior use here}}
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: i32) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{use of value '%arg1' expects different type than prior uses: '!cuda_tile.token' vs 'i32'}}
    %tile_1, %tok_out = load_view_tko weak %arg0[%0, %0, %0] token = %arg1 : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, i32
    return
  }
}

// -----

cuda_tile.module @tiled_view_store {
  testing$func @tiled_view_store(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: i32) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{result #0 must be cuda tile token type, but got 'i32'}}
    %1 = store_view_tko weak %arg0, %arg1[%0] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> i32
  }
}

// -----

cuda_tile.module @tiled_view_store {
  testing$func @tiled_view_store(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{invalid memory_ordering_semantics attribute specification. Got "invalid" but expect one of: weak, relaxed, acquire, release, acq_rel}}
    %1 = store_view_tko invalid %arg0, %arg1[%0] : !cuda_tile.tile<8xf32>, !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>> -> token
  }
}

// -----

cuda_tile.module @tiled_view_store {
  testing$func @tiled_view_store(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{expect one of: weak, relaxed, or release, but got: acquire}}
    %1 = store_view_tko acquire device %arg0, %arg1[%0] : !cuda_tile.tile<8xf32>, !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  }
}

// -----

cuda_tile.module @tiled_view_load {
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{invalid memory_ordering_semantics attribute specification. Got "invalid" but expect one of: weak, relaxed, acquire, release, acq_rel}}
    %tile_1, %tok_out = load_view_tko invalid %arg0[%0, %0, %0] token = %arg1 : !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>> -> !cuda_tile.tile<1024x1024x8xf32>, token
    return
  }
}

// -----

cuda_tile.module @tiled_view_load {
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{expect one of: weak, relaxed, or acquire, but got: release}}
    %tile_1, %tok_out = load_view_tko release device %arg0[%0, %0, %0] token = %arg1 : !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> !cuda_tile.tile<1024x1024x8xf32>, token
    return
  }
}

// -----

cuda_tile.module @tiled_view_load {
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{invalid memory_scope attribute specification. Got "invalid" but expect one of: tl_blk, device, sys}}
    %tile_1, %tok_out = load_view_tko relaxed invalid %arg0[%0, %0, %0] token = %arg1 : !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>> -> !cuda_tile.tile<1024x1024x8xf32>, token
    return
  }
}

// -----

cuda_tile.module @tiled_view_store {
  testing$func @tiled_view_store(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{invalid memory_scope attribute specification. Got "invalid" but expect one of: tl_blk, device, sys}}
    %1 = store_view_tko relaxed invalid %arg0, %arg1[%0] : !cuda_tile.tile<8xf32>, !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>> -> token
  }
}

// -----

cuda_tile.module @tiled_view_load {
  testing$func @tiled_view(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{operation specifies weak memory ordering semantics, but then provides "device" scope, expected no memory scope.}}
    %tile_1, %tok_out = load_view_tko weak device %arg0[%0, %0, %0] token = %arg1 : !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>> -> !cuda_tile.tile<1024x1024x8xf32>, token
    return
  }
}
// -----

cuda_tile.module @tiled_view_store {
  testing$func @tiled_view_store(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@below {{operation specifies weak memory ordering semantics, but then provides "tl_blk" scope, expected no memory scope.}}
    %1 = store_view_tko weak tl_blk %arg0, %arg1[%0] : !cuda_tile.tile<8xf32>, !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>> -> token
  }
}

// -----

cuda_tile.module @memory_model {
  testing$func @store_ptr_tko(%arg0: !cuda_tile.tile<16x32xptr<i8>>, %arg1: !cuda_tile.tile<16x32xi8>) {
    // expected-error@below {{memory scope is required for relaxed store}}
    %0 = store_ptr_tko relaxed %arg0, %arg1 : tile<16x32xptr<i8>>, tile<16x32xi8> -> token
  }
}

// -----

cuda_tile.module @memory_model {
  testing$func @store_ptr_tko(%arg0: !cuda_tile.tile<16x32xptr<i8>>, %arg1: !cuda_tile.tile<16x32xi8>) {
    // expected-error@below {{memory scope is required for release store}}
    %0 = store_ptr_tko release %arg0, %arg1 : tile<16x32xptr<i8>>, tile<16x32xi8> -> token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @funcload(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    // expected-error @below{{memory scope is required for acquire load}}
    %0, %t = load_ptr_tko acquire %arg0 : tile<16x32x!cuda_tile.ptr<f32>> -> tile<16x32xf32>, token
  }
}

// -----

cuda_tile.module @invalid_load_ptr_tko {
  cuda_tile.testing$func @funcload(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>) {
    // expected-error @below{{memory scope is required for relaxed load}}
    %0, %t = load_ptr_tko relaxed %arg0 : tile<16x32x!cuda_tile.ptr<f32>> -> tile<16x32xf32>, token
  }
}
