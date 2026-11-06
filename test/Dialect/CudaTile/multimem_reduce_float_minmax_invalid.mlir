// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

cuda_tile.module @reduce_f16_min {
  testing$func @test(
      %dst: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xf16>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @reduce_f16_max {
  testing$func @test(
      %dst: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xf16>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @reduce_bf16_min {
  testing$func @test(
      %dst: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xbf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xbf16>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @reduce_bf16_max {
  testing$func @test(
      %dst: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xbf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @+1 {{supports only add for floating-point element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %dst, %vals[%c0]
        attributes {reduction = #cuda_tile.multimem_reduction_kind<max>}
        : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xbf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>, !cuda_tile.tile<2xbf16>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}
