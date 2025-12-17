// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @kernels {

// CHECK-LABEL: @make_token_basic
testing$func @make_token_basic() -> !cuda_tile.token {
  // CHECK: %[[TOKEN:.*]] = make_token : token
  %0 = make_token : token
  // CHECK: return %[[TOKEN]] : token
  return %0 : token
}

// CHECK-LABEL: @join_tokens_two_tokens
testing$func @join_tokens_two_tokens() -> !cuda_tile.token {
  // CHECK: %[[TOKEN0:.*]] = make_token : token
  // CHECK: %[[TOKEN1:.*]] = make_token : token
  // CHECK: %[[RESULT:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]] : token
  %0 = make_token : token
  %1 = make_token : token
  %2 = join_tokens %0, %1 : token
  // CHECK: return %[[RESULT]] : token
  return %2 : token
}

// CHECK-LABEL: @join_tokens_three_tokens
testing$func @join_tokens_three_tokens() -> !cuda_tile.token {
  // CHECK: %[[TOKEN0:.*]] = make_token : token
  // CHECK: %[[TOKEN1:.*]] = make_token : token
  // CHECK: %[[TOKEN2:.*]] = make_token : token
  // CHECK: %[[RESULT:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]], %[[TOKEN2]] : token
  %0 = make_token : token
  %1 = make_token : token
  %2 = make_token : token
  %3 = join_tokens %0, %1, %2 : token
  // CHECK: return %[[RESULT]] : token
  return %3 : token
}

// CHECK-LABEL: load_ptr_tko
testing$func @load_ptr_tko(%arg0: !cuda_tile.tile<16x32xptr<f32>>) {
  // CHECK: %[[T:.+]] = make_token : token
  %t = make_token : token
  // CHECK: load_ptr_tko weak %{{.+}} token=%[[T]]
  // CHECK-SAME:  tile<16x32xptr<f32>> -> tile<16x32xf32>, token
  %0, %new_t = load_ptr_tko weak %arg0 token = %t
    : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
}

// CHECK-LABEL: load_ptr_tko_scoped
testing$func @load_ptr_tko_scoped(%arg0: !cuda_tile.tile<16x32xptr<f32>>) {
  // CHECK: %[[T:.+]] = make_token : token
  %t = make_token : token
  // CHECK: load_ptr_tko acquire device %{{.+}} token=%[[T]]
  // CHECK-SAME:  tile<16x32xptr<f32>> -> tile<16x32xf32>, token
  %0, %new_t = load_ptr_tko acquire device %arg0 token = %t
    : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
}

// CHECK-LABEL: load_ptr_tko_with_no_token_as_input
testing$func @load_ptr_tko_with_no_token_as_input(%arg0: !cuda_tile.tile<16x32xptr<f32>>) {
  // CHECK: load_ptr_tko weak %{{.+}} : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
  %0, %new_t = load_ptr_tko weak %arg0
    : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
}

// CHECK-LABEL: load_with_mask
testing$func @load_with_mask(%arg0: !cuda_tile.tile<16x32xptr<f32>>, %arg1: !cuda_tile.tile<16x32xi1>) {
  // CHECK: %[[T:.+]] = make_token : token
  %t = make_token : token
  // CHECK: %{{.+}}, %{{.+}} = load_ptr_tko weak %{{.+}}, %{{.+}} token=%[[T]]
  // CHECK-SAME: : tile<16x32xptr<f32>>, tile<16x32xi1> -> tile<16x32xf32>, token
  %0, %new_t = load_ptr_tko weak %arg0, %arg1 token = %t
    : tile<16x32xptr<f32>>, tile<16x32xi1> -> tile<16x32xf32>, token
}

// CHECK-LABEL: load_with_mask_and_padding
testing$func @load_with_mask_and_padding(%arg0: !cuda_tile.tile<16x32xptr<f32>>, %arg1: !cuda_tile.tile<16x32xi1>, %arg2: !cuda_tile.tile<16x32xf32>) {
  // CHECK: %[[T:.+]] = make_token : token
  %t = make_token : token
  // CHECK: %{{.+}}, %{{.+}} = load_ptr_tko weak %{{.+}}, %{{.+}}, %{{.+}} token=%[[T]]
  // CHECK-SAME: : tile<16x32xptr<f32>>, tile<16x32xi1>, tile<16x32xf32> -> tile<16x32xf32>, token
  %0, %new_t = load_ptr_tko weak %arg0, %arg1, %arg2 token = %t
    : tile<16x32xptr<f32>>, tile<16x32xi1>, tile<16x32xf32> -> tile<16x32xf32>, token
}

// CHECK-LABEL: store
testing$func @store(%arg0: !cuda_tile.tile<16x32xptr<f32>>, %arg1 : !cuda_tile.tile<16x32xf32>) {
  // CHECK: %[[T:.+]] = make_token : token
  %t = make_token : token
  // CHECK: store_ptr_tko weak %{{.+}}, %{{.+}} token=%[[T]]
  // CHECK-SAME:  : tile<16x32xptr<f32>>, tile<16x32xf32> -> token
  %t1 = store_ptr_tko weak %arg0, %arg1 token = %t
    : tile<16x32xptr<f32>>, tile<16x32xf32> -> token
}

// CHECK-LABEL: store_with_mask
testing$func @store_with_mask(%arg0: !cuda_tile.tile<16x32xptr<f32>>, %arg1: !cuda_tile.tile<16x32xi1>, %arg2 : !cuda_tile.tile<16x32xf32>) {
  // CHECK: %[[T:.+]] = make_token : token
  %t = make_token : token
  // CHECK: store_ptr_tko weak %{{.+}}, %{{.+}}, %{{.+}} token=%[[T]]
  // CHECK-SAME:  : tile<16x32xptr<f32>>, tile<16x32xf32>, tile<16x32xi1> -> token
  %t1 = store_ptr_tko weak %arg0, %arg2, %arg1 token = %t
    : tile<16x32xptr<f32>>, tile<16x32xf32>, tile<16x32xi1> -> token
}

// CHECK-LABEL: load_ptr_tko_optional_token
testing$func @load_ptr_tko_optional_token(%arg0: !cuda_tile.tile<16x32xptr<f32>>) {
  // CHECK: load_ptr_tko weak %{{.+}} : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
  %0, %t = load_ptr_tko weak %arg0  
    : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
}

// CHECK-LABEL: tiled_view_load
testing$func @tiled_view_load(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: !cuda_tile.token) {
  %0 = constant <i32: 0> : !cuda_tile.tile<i32>
  // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] token = %{{.+}} : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32>
  // CHECK-SAME:  -> tile<1024x1024x8xf32>, token
  %tile_2, %tok_out = load_view_tko weak %arg0[%0, %0, %0] token = %arg1 : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
  
  // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32>
  // CHECK-SAME:  -> tile<1024x1024x8xf32>, token
  %tile_3, %tok_out_1 = load_view_tko weak %arg0[%0, %0, %0] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
  
  // CHECK: %{{.+}}, %{{.+}} = load_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] token = %{{.+}}: partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32>
  // CHECK-SAME: -> tile<1024x1024x8xf32>, token
  %tile_4, %tok_out_2 = load_view_tko relaxed device %arg0[%0, %0, %0] token = %arg1 : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
  return
}

// CHECK-LABEL: tiled_view_store
testing$func @tiled_view_store(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: !cuda_tile.token) {
  %0 = constant <i32: 0> : !cuda_tile.tile<i32>
  // CHECK: %{{.+}} = store_view_tko weak %{{.+}}, %{{.+}}[%{{.+}}] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  %1 = store_view_tko weak %arg0, %arg1[%0] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  
  // CHECK-NEXT: %{{.+}} = store_view_tko weak %{{.+}}, %{{.+}}[%{{.+}}] token = %{{.+}} : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  %2 = store_view_tko weak %arg0, %arg1[%0] token = %token : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  
  // CHECK-NEXT: %{{.+}} = store_view_tko relaxed device %{{.+}}, %{{.+}}[%{{.+}}] token = %{{.+}} : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  %3 = store_view_tko relaxed device %arg0, %arg1[%0] token = %token : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  return
}

} // end memory_consistency_test
