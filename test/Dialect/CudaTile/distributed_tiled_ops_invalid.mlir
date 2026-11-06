// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// Test 1: multimem_load_reduce_view_tko with zero coordinates triggers verifier error
// (verifyViewIndexTileMatch: index count must match partition view rank).
cuda_tile.module @test {
  testing$func @test_zero_coords_load_reduce(
      %src: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>) {
    // expected-error @+1 {{expected 2 index operands (based on view type), got 0}}
    %vals, %tok0 = cuda_tile.multimem_load_reduce_view_tko weak %src[]
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128x64xf32>, !cuda_tile.token
  }
}

// -----

// Test 2: multimem_reduce_view_tko with zero coordinates triggers verifier error.
cuda_tile.module @test {
  testing$func @test_zero_coords_reduce(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>) {
    // expected-error @+1 {{expected 1 index operands (based on view type), got 0}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[]
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 3: multimem_store_view_tko with zero coordinates triggers verifier error.
cuda_tile.module @test {
  testing$func @test_zero_coords_store(
      %dst: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xf32>) {
    // expected-error @+1 {{expected 2 index operands (based on view type), got 0}}
    %tok = cuda_tile.multimem_store_view_tko weak %dst, %vals[]
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with signed-min on f32 is rejected (f32 not supported).
cuda_tile.module @test {
  testing$func @test_load_reduce_min_f32(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{min/max supports f16 and bf16}}
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf32>, !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with unsigned-max on f32 is rejected (umax requires integer).
cuda_tile.module @test {
  testing$func @test_load_reduce_umax_f32(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{and/or/xor/umin/umax reductions require integer}}
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>}
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf32>, !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with signed-max on f32 is rejected (f32 not supported).
cuda_tile.module @test {
  testing$func @test_reduce_max_f32(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with unsigned-min on f32 is rejected (umin requires integer).
cuda_tile.module @test {
  testing$func @test_reduce_umin_f32(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{and/or/xor/umin/umax reductions require integer}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>}
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with umin on f16 is rejected (umin requires integer).
cuda_tile.module @test {
  testing$func @test_load_reduce_umin_f16(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{and/or/xor/umin/umax reductions require integer}}
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>}
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf16>, !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with umax on bf16 is rejected (umax requires integer).
cuda_tile.module @test {
  testing$func @test_reduce_umax_bf16(
      %dst: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xbf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{and/or/xor/umin/umax reductions require integer}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xbf16>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with signed-min on f64 is rejected (f64 not supported).
cuda_tile.module @test {
  testing$func @test_load_reduce_min_f64(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{min/max supports f16 and bf16}}
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf64>, !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with umin on f64 is rejected (umin requires integer).
cuda_tile.module @test {
  testing$func @test_load_reduce_umin_f64(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{and/or/xor/umin/umax reductions require integer}}
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>}
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf64>, !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with signed-min on f64 is rejected (f64 not supported).
cuda_tile.module @test {
  testing$func @test_reduce_min_f64(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf64>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xf64>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with umin on f64 is rejected (umin requires integer).
cuda_tile.module @test {
  testing$func @test_reduce_umin_f64(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf64>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{and/or/xor/umin/umax reductions require integer}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>}
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xf64>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 4: multimem_load_reduce_view_tko with weak ordering must not have memory scope.
// Use a 1D partition view so one index matches view rank (memory-model check is the point).
cuda_tile.module @test {
  testing$func @test_weak_with_scope(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{weak load must not have memory scope}}
    %vals, %tok1 = cuda_tile.multimem_load_reduce_view_tko weak device %src[%c0]
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf32>, !cuda_tile.token
  }
}

// -----

// Test 5: multimem_load_reduce_view_tko with relaxed ordering requires memory scope.
cuda_tile.module @test {
  testing$func @test_relaxed_without_scope(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{memory scope is required for relaxed load}}
    %vals, %tok2 = cuda_tile.multimem_load_reduce_view_tko relaxed %src[%c0]
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf32>, !cuda_tile.token
  }
}

// -----

// Test 6: multimem_load_reduce_view_tko with acquire ordering requires memory scope.
cuda_tile.module @test {
  testing$func @test_acquire_without_scope(
      %src: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{memory scope is required for acquire load}}
    %vals, %tok3 = cuda_tile.multimem_load_reduce_view_tko acquire %src[%c0]
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<128xf32>, !cuda_tile.token
  }
}

// -----

// Test 7: multimem_reduce_view_tko with weak ordering is not supported.
cuda_tile.module @test {
  testing$func @test_reduce_weak(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{memory ordering semantics must be one of: relaxed, release}}
    %tok = cuda_tile.multimem_reduce_view_tko weak sys %dst, %vals[%c0]
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 8: multimem_reduce_view_tko with acquire ordering is not supported.
cuda_tile.module @test {
  testing$func @test_reduce_acquire(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{memory ordering semantics must be one of: relaxed, release}}
    %tok = cuda_tile.multimem_reduce_view_tko acquire device %dst, %vals[%c0]
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 9: multimem_store_view_tko with weak ordering must not have memory scope.
cuda_tile.module @test {
  testing$func @test_store_weak_with_scope(
      %dst: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{weak store must not have memory scope}}
    %tok = cuda_tile.multimem_store_view_tko weak device %dst, %vals[%c0]
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 10: multimem_store_view_tko with relaxed ordering requires memory scope.
cuda_tile.module @test {
  testing$func @test_store_relaxed_without_scope(
      %dst: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{memory scope is required for relaxed store}}
    %tok = cuda_tile.multimem_store_view_tko relaxed %dst, %vals[%c0]
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 11: multimem_store_view_tko with release ordering requires memory scope.
cuda_tile.module @test {
  testing$func @test_store_release_without_scope(
      %dst: !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{memory scope is required for release store}}
    %tok = cuda_tile.multimem_store_view_tko release %dst, %vals[%c0]
        : !cuda_tile.partition_view<tile=(128), !cuda_tile.tensor_view<128xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128xf32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Test 12: multimem_load_reduce_view_tko `inbounds` rank must equal the index rank.
cuda_tile.module @test {
  testing$func @test_load_reduce_inbounds_size_mismatch(
      %src: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @+1 {{inbounds size (3) must match the number of index dimensions (2)}}
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0, %c1] inbounds = [true, false, true]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<128x64xf32>, !cuda_tile.token
  }
}

// -----

// Test 13: multimem_reduce_view_tko `inbounds` rank must equal the index rank.
cuda_tile.module @test {
  testing$func @test_reduce_inbounds_size_mismatch(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{inbounds size (2) must match the number of index dimensions (1)}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0] token = %token inbounds = [true, true]
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }
}

// -----

// Test 14: multimem_store_view_tko `inbounds` rank must equal the index rank.
cuda_tile.module @test {
  testing$func @test_store_inbounds_size_mismatch(
      %dst: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @+1 {{inbounds size (1) must match the number of index dimensions (2)}}
    %tok = cuda_tile.multimem_store_view_tko weak %dst, %vals[%c0, %c1] inbounds = [true]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xf32>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with 1D tile=(2) and f16 passes the
// static verifier. The PTX v2 atom minimum-byte requirement is not statically
// checked; violations are runtime undefined behaviour.
cuda_tile.module @test {
  testing$func @test_load_reduce_1d_inner_extent1_f16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xf16>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [2], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %vals_out, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<2xf16>, !cuda_tile.token
  }
}

// -----

// Test: multimem_load_reduce_view_tko with 1D tile=(2) and bf16 passes the
// static verifier. The PTX v2 atom minimum-byte requirement is not statically
// checked; violations are runtime undefined behaviour.
cuda_tile.module @test {
  testing$func @test_load_reduce_1d_inner_extent1_bf16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xbf16>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [2], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %vals_out, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<2xbf16>, !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with 1D tile=(1) and f16 is rejected because
// the tile's total size (16 bits) is below the 32-bit PTX multimem transaction
// granularity required for float element types.
cuda_tile.module @test {
  testing$func @test_reduce_1d_inner_extent1_f16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf16>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [2], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{multimem op requires tile total size to be a multiple of 4 bytes for float element types; got 1 element(s) of 16 bits (total 16 bits)}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %pv, %vals[%c0] token = %token
        attributes {reduction = #cuda_tile.multimem_reduction_kind<add>}
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xf16>, !cuda_tile.tile<i32>, !cuda_tile.token
        -> !cuda_tile.token
  }
}

// -----

// Test: multimem_reduce_view_tko with 1D tile=(1) and bf16 is rejected because
// the tile's total size (16 bits) is below the 32-bit PTX multimem transaction
// granularity required for float element types.
cuda_tile.module @test {
  testing$func @test_reduce_1d_inner_extent1_bf16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xbf16>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [2], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{multimem op requires tile total size to be a multiple of 4 bytes for float element types; got 1 element(s) of 16 bits (total 16 bits)}}
    %tok = cuda_tile.multimem_reduce_view_tko release device %pv, %vals[%c0] token = %token
        attributes {reduction = #cuda_tile.multimem_reduction_kind<add>}
        : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xbf16>, !cuda_tile.tile<i32>, !cuda_tile.token
        -> !cuda_tile.token
  }
}

// -----


// Test 18: multimem_store_view_tko with unsupported element type i8 is rejected.
// i8 is not in the supported set (i32, i64, f16, bf16, f32, f64).
cuda_tile.module @test {
  testing$func @test_store_unsupported_i8(
      %dst: !cuda_tile.partition_view<tile=(4), !cuda_tile.tensor_view<4xi8, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<4xi8>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{multimem requires 32- or 64-bit integer/index element types}}
    %tok = cuda_tile.multimem_store_view_tko weak %dst, %vals[%c0]
      : !cuda_tile.partition_view<tile=(4), !cuda_tile.tensor_view<4xi8, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<4xi8>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }
}

// -----

// Test 19: multimem_store_view_tko with non-unit inner stride (stride=2) and f16 is rejected.
// PTX v2 atoms require the innermost stride to be 1 so the two packed f16
// elements are physically adjacent. strides=[128,2] has minimum stride=2,
// which violates that requirement. The view is built via make_tensor_view +
// make_partition_view so the verifier can walk the use-def chain.
cuda_tile.module @test {
  testing$func @test_store_nonunit_inner_stride_f16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<4x2xf16>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [4, 2], strides = [128, 2]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<4x2xf16, strides=[128,2], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(4x2), !cuda_tile.tensor_view<4x2xf16, strides=[128,2], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{f16/bf16 multimem views must have exactly one element stride statically known to equal 1; got 0}}
    %tok = cuda_tile.multimem_store_view_tko weak %pv, %vals[%c0, %c0]
      : !cuda_tile.partition_view<tile=(4x2), !cuda_tile.tensor_view<4x2xf16, strides=[128,2], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<4x2xf16>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }
}

// -----

// Test 20: multimem_reduce_view_tko with f16 MIN is rejected.
// reduce_view_tko only supports ADD for floating-point types; MIN/MAX on float
// is not lowerable to PTX multimem.red.
cuda_tile.module @test {
  testing$func @multimem_reduce_min_f16_invalid(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xf16>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [4], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<4xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %pv, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xf16>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }
}

// -----

// Test 21: multimem_reduce_view_tko with bf16 MAX is rejected.
// reduce_view_tko only supports ADD for floating-point types; MIN/MAX on float
// is not lowerable to PTX multimem.red.
cuda_tile.module @test {
  testing$func @multimem_reduce_max_bf16_invalid(
      %base: !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xbf16>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [4], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<4xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko release device %pv, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xbf16>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }
}
