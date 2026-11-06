// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @distributed_ops_module {

  // CHECK-LABEL: testing$func @ptr_to_multicast_ptr_basic
  testing$func @ptr_to_multicast_ptr_basic(
      %ptr: !cuda_tile.tile<ptr<i8>>) {
    // CHECK: ptr_to_ptr %{{.*}} : tile<ptr<i8>> -> tile<ptr<i8, <multicast>>>
    %mc = cuda_tile.ptr_to_ptr %ptr
      : !cuda_tile.tile<ptr<i8>> -> !cuda_tile.tile<!cuda_tile.ptr<i8, #cuda_tile.ptr_attr<multicast>>>
  }

  // CHECK-LABEL: testing$func @ptr_to_unicast_ptr_basic
  testing$func @ptr_to_unicast_ptr_basic(
      %ptr: !cuda_tile.tile<ptr<i8>>) {
    // CHECK: ptr_to_ptr %{{.*}} : tile<ptr<i8>> -> tile<ptr<i8, <unicast>>>
    %uc = cuda_tile.ptr_to_ptr %ptr
      : !cuda_tile.tile<ptr<i8>> -> !cuda_tile.tile<!cuda_tile.ptr<i8, #cuda_tile.ptr_attr<unicast>>>
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_basic
  testing$func @multimem_load_reduce_basic(
      %src: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} -> tile<128x64xf32>, token
    %vals, %tok_lr0 = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0, %c1]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<128x64xf32>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_with_token
  testing$func @multimem_load_reduce_with_token(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %token: !cuda_tile.token) {
    // CHECK: make_tensor_view %{{.*}}, shape = [128, 64], strides = [64, 1] : tile<ptr<f16, <multicast>>> -> tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko relaxed device {{.*}} token = {{.*}} -> tile<128x64xf16>, token
    %vals, %tok_out = cuda_tile.multimem_load_reduce_view_tko relaxed device %pv[%c0, %c1] token = %token
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.tile<128x64xf16>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_2d_coords
  testing$func @multimem_load_reduce_2d_coords(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}}[%{{.*}}, %{{.*}}] :
    // CHECK-SAME: partition_view<tile=(128x64), tensor_view<128x64xf16
    // CHECK-SAME: -> tile<128x64xf16>, token
    %vals, %tok_lr1 = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c1]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<128x64xf16>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_umin_i32
  testing$func @multimem_load_reduce_umin_i32(
      %src: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>} {{.*}} -> tile<1xi32>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<1xi32>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_umax_i32
  testing$func @multimem_load_reduce_umax_i32(
      %src: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>} {{.*}} -> tile<1xi32>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<1xi32>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_min_signed_i32
  testing$func @multimem_load_reduce_min_signed_i32(
      %src: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<min>} {{.*}} -> tile<1xi32>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<1xi32>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_max_signed_i32
  testing$func @multimem_load_reduce_max_signed_i32(
      %src: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<max>} {{.*}} -> tile<1xi32>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<1xi32>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_min_f16
  // tile=(2) satisfies the even packing tile extent requirement.
  testing$func @multimem_load_reduce_min_f16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [4], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<4xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<min>} {{.*}} -> tile<2xf16>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<2xf16>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_max_bf16
  // tile=(2) satisfies the even packing tile extent requirement.
  testing$func @multimem_load_reduce_max_bf16(
      %base: !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [4], strides = [1]
        : !cuda_tile.tile<!cuda_tile.ptr<bf16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<4xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<max>} {{.*}} -> tile<2xbf16>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<2xbf16>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_add_default
  testing$func @multimem_reduce_add_default(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}} token = {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0] token = %token
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_add_explicit
  testing$func @multimem_reduce_add_explicit(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed device {{.*}} token = {{.*}} -> token
    // Note: reduction=add is the default and is elided by the printer
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<add>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_umin
  testing$func @multimem_reduce_umin(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<umin>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_umax
  testing$func @multimem_reduce_umax(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko release sys {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko release sys %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_min_signed
  testing$func @multimem_reduce_min_signed(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<min>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_max_signed
  testing$func @multimem_reduce_max_signed(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko release device {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<max>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko release device %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_min_signed_i64
  testing$func @multimem_load_reduce_min_signed_i64(
      %src: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<min>} {{.*}} -> tile<1xi64>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<1xi64>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_max_signed_i64
  testing$func @multimem_load_reduce_max_signed_i64(
      %src: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<max>} {{.*}} -> tile<1xi64>, token
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<1xi64>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_min_signed_i64
  testing$func @multimem_reduce_min_signed_i64(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi64>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<min>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi64>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_max_signed_i64
  testing$func @multimem_reduce_max_signed_i64(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi64>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko release device {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<max>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko release device %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi64, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi64>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }


  // CHECK-LABEL: testing$func @multimem_reduce_and
  testing$func @multimem_reduce_and(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<and>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<and>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_or
  testing$func @multimem_reduce_or(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<or>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<or>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_xor
  testing$func @multimem_reduce_xor(
      %dst: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko release device {{.*}} attributes {reduction = #cuda_tile.multimem_reduction_kind<xor>} {{.*}} -> token
    %tok = cuda_tile.multimem_reduce_view_tko release device %dst, %vals[%c0] token = %token attributes {reduction = #cuda_tile.multimem_reduction_kind<xor>}
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_2d_coords
  testing$func @multimem_reduce_2d_coords(
      %dst: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xi32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}}[%{{.*}}, %{{.*}}] token = {{.*}} :
    // CHECK-SAME: partition_view<tile=(128x64), tensor_view<128x64xi32
    // CHECK-SAME: -> token
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0, %c1] token = %token
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xi32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_store_basic
  testing$func @multimem_store_basic(
      %dst: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_store_view_tko weak {{.*}} : partition_view<tile=(128x64), tensor_view<128x64xf32
    // CHECK-SAME: -> token
    %tok_st0 = cuda_tile.multimem_store_view_tko weak %dst, %vals[%c0, %c1]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xf32>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_store_with_token
  testing$func @multimem_store_with_token(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xf16>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_store_view_tko relaxed device {{.*}} token = {{.*}} -> token
    %tok = cuda_tile.multimem_store_view_tko relaxed device %pv, %vals[%c0, %c1] token = %token
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xf16>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  //===--------------------------------------------------------------------===//
  // atomic_rmw_tko with unicast !cuda_tile.ptr (ptr_attr)
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_add_sys
  testing$func @atomic_rmw_unicast_add_sys(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xi32>) {
    // CHECK: atomic_rmw_tko relaxed sys %{{.*}}, add, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko relaxed sys %ptr, add, %arg
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_xchg_sys
  testing$func @atomic_rmw_unicast_xchg_sys(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xi32>) {
    // CHECK: atomic_rmw_tko acq_rel sys %{{.*}}, xchg, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko acq_rel sys %ptr, xchg, %arg
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_fadd_sys
  testing$func @atomic_rmw_unicast_fadd_sys(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xf32>) {
    // CHECK: atomic_rmw_tko relaxed sys %{{.*}}, addf, %{{.*}} : tile<128xptr<f32, <unicast>>>, tile<128xf32> -> tile<128xf32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko relaxed sys %ptr, addf, %arg
        : tile<128xptr<f32, <unicast>>>,
          tile<128xf32>
        -> tile<128xf32>, token
  }

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_and_sys
  testing$func @atomic_rmw_unicast_and_sys(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xi32>) {
    // CHECK: atomic_rmw_tko relaxed sys %{{.*}}, and, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko relaxed sys %ptr, and, %arg
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_max_acquire_sys
  testing$func @atomic_rmw_unicast_max_acquire_sys(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xi32>) {
    // CHECK: atomic_rmw_tko acquire sys %{{.*}}, max, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko acquire sys %ptr, max, %arg
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_with_mask
  testing$func @atomic_rmw_unicast_with_mask(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xi32>,
      %mask: !cuda_tile.tile<128xi1>) {
    // CHECK: atomic_rmw_tko relaxed sys %{{.*}}, add, %{{.*}}, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32>, tile<128xi1> -> tile<128xi32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko relaxed sys %ptr, add, %arg, %mask
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>,
          tile<128xi1>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_rmw_unicast_with_token
  testing$func @atomic_rmw_unicast_with_token(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %arg: !cuda_tile.tile<128xi32>,
      %token: !cuda_tile.token) {
    // CHECK: atomic_rmw_tko relaxed sys %{{.*}}, add, %{{.*}} token=%{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %val, %tok = cuda_tile.atomic_rmw_tko relaxed sys %ptr, add, %arg token = %token
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  //===--------------------------------------------------------------------===//
  // atomic_cas_tko with unicast !cuda_tile.ptr (ptr_attr)
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: testing$func @atomic_cas_unicast_sys_relaxed
  testing$func @atomic_cas_unicast_sys_relaxed(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %cmp: !cuda_tile.tile<128xi32>,
      %val: !cuda_tile.tile<128xi32>) {
    // CHECK: atomic_cas_tko relaxed sys %{{.*}}, %{{.*}}, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %res, %tok = cuda_tile.atomic_cas_tko relaxed sys %ptr, %cmp, %val
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_cas_unicast_sys_acq_rel
  testing$func @atomic_cas_unicast_sys_acq_rel(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %cmp: !cuda_tile.tile<128xi32>,
      %val: !cuda_tile.tile<128xi32>) {
    // CHECK: atomic_cas_tko acq_rel sys %{{.*}}, %{{.*}}, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %res, %tok = cuda_tile.atomic_cas_tko acq_rel sys %ptr, %cmp, %val
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_cas_unicast_f32_sys
  testing$func @atomic_cas_unicast_f32_sys(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<unicast>>>,
      %cmp: !cuda_tile.tile<128xf32>,
      %val: !cuda_tile.tile<128xf32>) {
    // CHECK: atomic_cas_tko acq_rel sys %{{.*}}, %{{.*}}, %{{.*}} : tile<128xptr<f32, <unicast>>>, tile<128xf32> -> tile<128xf32>, token
    %res, %tok = cuda_tile.atomic_cas_tko acq_rel sys %ptr, %cmp, %val
        : tile<128xptr<f32, <unicast>>>,
          tile<128xf32>
        -> tile<128xf32>, token
  }

  // CHECK-LABEL: testing$func @atomic_cas_unicast_with_mask
  testing$func @atomic_cas_unicast_with_mask(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %cmp: !cuda_tile.tile<128xi32>,
      %val: !cuda_tile.tile<128xi32>,
      %mask: !cuda_tile.tile<128xi1>) {
    // CHECK: atomic_cas_tko acq_rel sys %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32>, tile<128xi1> -> tile<128xi32>, token
    %res, %tok = cuda_tile.atomic_cas_tko acq_rel sys %ptr, %cmp, %val, %mask
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>,
          tile<128xi1>
        -> tile<128xi32>, token
  }

  // CHECK-LABEL: testing$func @atomic_cas_unicast_with_token
  testing$func @atomic_cas_unicast_with_token(
      %ptr: !cuda_tile.tile<128x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %cmp: !cuda_tile.tile<128xi32>,
      %val: !cuda_tile.tile<128xi32>,
      %token: !cuda_tile.token) {
    // CHECK: atomic_cas_tko acq_rel sys %{{.*}}, %{{.*}}, %{{.*}} token=%{{.*}} : tile<128xptr<i32, <unicast>>>, tile<128xi32> -> tile<128xi32>, token
    %res, %tok = cuda_tile.atomic_cas_tko acq_rel sys %ptr, %cmp, %val token = %token
        : tile<128xptr<i32, <unicast>>>,
          tile<128xi32>
        -> tile<128xi32>, token
  }

  //===--------------------------------------------------------------------===//
  // atomic_red_view_tko end-to-end: unicast ptr → view → atomic reduction
  //===--------------------------------------------------------------------===//

  // End-to-end: unicast ptr → make_tensor_view → make_partition_view → atomic_red_view_tko
  // CHECK-LABEL: testing$func @atomic_red_view_unicast_e2e_add_i32
  testing$func @atomic_red_view_unicast_e2e_add_i32(
      %base: !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    // CHECK: make_tensor_view %{{.*}}, shape = [2, 2], strides = [2, 1] : tile<ptr<i32, <unicast>>> -> tensor_view<2x2xi32, strides=[2,1], #cuda_tile.ptr_attr<unicast>>
    %tv = cuda_tile.make_tensor_view %base, shape = [2, 2], strides = [2, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>
    // CHECK: make_partition_view
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], add, %{{.*}}
    %t = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], add, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
  }

  // End-to-end: unicast ptr → view → atomic_red_view_tko with addf (f32)
  // CHECK-LABEL: testing$func @atomic_red_view_unicast_e2e_addf_f32
  testing$func @atomic_red_view_unicast_e2e_addf_f32(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<unicast>>>,
      %value: !cuda_tile.tile<2x2xf32>) {
    // CHECK: make_tensor_view %{{.*}}, shape = [2, 2], strides = [2, 1]
    %tv = cuda_tile.make_tensor_view %base, shape = [2, 2], strides = [2, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<unicast>>>
        -> !cuda_tile.tensor_view<2x2xf32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>
    // CHECK: make_partition_view
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], addf, %{{.*}}
    %t = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], addf, %value
        : tile<2x2xf32>,
          partition_view<tile=(2x2), tensor_view<2x2xf32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
  }

  // End-to-end: ptr_to_ptr (regular → unicast) → view → atomic reduction (full chain)
  // CHECK-LABEL: testing$func @atomic_red_view_unicast_e2e_full_chain
  testing$func @atomic_red_view_unicast_e2e_full_chain(
      %raw_ptr: !cuda_tile.tile<ptr<i32>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    // CHECK: ptr_to_ptr %{{.*}} : tile<ptr<i32>> -> tile<ptr<i32, <unicast>>>
    %uc_ptr = cuda_tile.ptr_to_ptr %raw_ptr
        : !cuda_tile.tile<ptr<i32>>
        -> !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>
    // CHECK: make_tensor_view %{{.*}}, shape = [2, 2], strides = [2, 1]
    %tv = cuda_tile.make_tensor_view %uc_ptr, shape = [2, 2], strides = [2, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>
    // CHECK: make_partition_view
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], add, %{{.*}}
    %t = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], add, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
  }

  // End-to-end: unicast ptr → view → atomic_red_view_tko with token chaining + tl_blk scope
  // CHECK-LABEL: testing$func @atomic_red_view_unicast_e2e_token_tl_blk
  testing$func @atomic_red_view_unicast_e2e_token_tl_blk(
      %base: !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %value: !cuda_tile.tile<2x2xi32>,
      %in_token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [2, 2], strides = [2, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: atomic_red_view_tko relaxed tl_blk %{{.*}}[%{{.*}}, %{{.*}}], add, %{{.*}} token = %{{.*}}
    %t = cuda_tile.atomic_red_view_tko relaxed tl_blk %pv[%c0, %c0], add, %value token = %in_token
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
  }

  // End-to-end: unicast ptr → view → multiple atomic modes (bitwise + arithmetic)
  // CHECK-LABEL: testing$func @atomic_red_view_unicast_e2e_all_modes
  testing$func @atomic_red_view_unicast_e2e_all_modes(
      %base: !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [2, 2], strides = [2, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], and, %{{.*}}
    %t0 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], and, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], or, %{{.*}}
    %t1 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], or, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], xor, %{{.*}}
    %t2 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], xor, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], max, %{{.*}}
    %t3 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], max, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], min, %{{.*}}
    %t4 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], min, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], umax, %{{.*}}
    %t5 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], umax, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
    // CHECK: atomic_red_view_tko relaxed device %{{.*}}[%{{.*}}, %{{.*}}], umin, %{{.*}}
    %t6 = cuda_tile.atomic_red_view_tko relaxed device %pv[%c0, %c0], umin, %value
        : tile<2x2xi32>,
          partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2, 1], #cuda_tile.ptr_attr<unicast>>>,
          tile<i32> -> token
  }

  //===--------------------------------------------------------------------===//
  // inbounds round-trip for the multimem ops
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: testing$func @multimem_load_reduce_inbounds
  testing$func @multimem_load_reduce_inbounds(
      %src: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_load_reduce_view_tko weak {{.*}}[%{{.*}}, %{{.*}}] inbounds = [true, false]
    %vals0, %tok0 = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0, %c1] inbounds = [true, false]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<128x64xf32>, !cuda_tile.token
    // CHECK: multimem_load_reduce_view_tko weak {{.*}}[%{{.*}}, %{{.*}}] inbounds = [true, true]
    %vals1, %tok1 = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0, %c1] inbounds = [true, true]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<128x64xf32>, !cuda_tile.token
    // An explicit all-false annotation parses, but prints in canonical form.
    // CHECK: multimem_load_reduce_view_tko weak {{.*}}[%{{.*}}, %{{.*}}] :
    %vals2, %tok2 = cuda_tile.multimem_load_reduce_view_tko weak %src[%c0, %c1] inbounds = [false, false]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>
      -> !cuda_tile.tile<128x64xf32>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_load_reduce_inbounds_with_token
  testing$func @multimem_load_reduce_inbounds_with_token(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // The directive must appear between the optional `token = ...` clause and
    // the trailing attribute dict.
    // CHECK: multimem_load_reduce_view_tko relaxed device {{.*}} token = {{.*}} inbounds = [false, true]
    %vals, %tok = cuda_tile.multimem_load_reduce_view_tko relaxed device %pv[%c0, %c1] token = %token inbounds = [false, true]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.tile<128x64xf16>, !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_reduce_inbounds
  testing$func @multimem_reduce_inbounds(
      %dst: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xi32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_reduce_view_tko relaxed sys {{.*}}[%{{.*}}, %{{.*}}] token = {{.*}} inbounds = [true, false]
    %tok = cuda_tile.multimem_reduce_view_tko relaxed sys %dst, %vals[%c0, %c1] token = %token inbounds = [true, false]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xi32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_store_inbounds
  testing$func @multimem_store_inbounds(
      %dst: !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_store_view_tko weak {{.*}}[%{{.*}}, %{{.*}}] inbounds = [true, true]
    %tok = cuda_tile.multimem_store_view_tko weak %dst, %vals[%c0, %c1] inbounds = [true, true]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf32, strides=[?,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xf32>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }

  // CHECK-LABEL: testing$func @multimem_store_inbounds_with_token
  testing$func @multimem_store_inbounds_with_token(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<128x64xf16>,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
        : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: multimem_store_view_tko relaxed device {{.*}} token = {{.*}} inbounds = [true, false]
    %tok = cuda_tile.multimem_store_view_tko relaxed device %pv, %vals[%c0, %c1] token = %token inbounds = [true, false]
      : !cuda_tile.partition_view<tile=(128x64), !cuda_tile.tensor_view<128x64xf16, strides=[64,1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<128x64xf16>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }

} // end cuda_tile.module
