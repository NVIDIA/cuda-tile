// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.5 %s -o %t.mlir
// RUN: cuda-tile-opt --no-implicit-module %s -o %t.ref.mlir
// RUN: diff %t.mlir %t.ref.mlir

// Cover the complete wire shapes of the public unicast atomic ops: memory
// semantics, reductions, optional masks/tokens, and all results. The ops
// date to 13.1 for pointer atomics and 13.3 for view atomic reductions.
// `ptr_attr<unicast>` is a 13.5 enum case, so the module as a whole only
// serializes from 13.5 on.
cuda_tile.module @kernels {
  entry @ptr_atomics(
      %ptr: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>) {
    %ptr_1d = reshape %ptr
        : tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> tile<1xptr<i32, #cuda_tile.ptr_attr<unicast>>>
    %ptrs = broadcast %ptr_1d
        : tile<1xptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>
    %arg = constant <i32: 1> : tile<4xi32>
    %cmp = constant <i32: 0> : tile<4xi32>
    %mask = constant <i1: true> : tile<4xi1>
    %input_token = make_token : token

    %r0, %t0 = atomic_rmw_tko relaxed device %ptrs, and, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %r1, %t1 = atomic_rmw_tko acquire device %ptrs, or, %arg, %mask
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>, tile<4xi1>
        -> tile<4xi32>, token
    %r2, %t2 = atomic_rmw_tko release device %ptrs, xor, %arg token = %input_token
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %r3, %t3 = atomic_rmw_tko acq_rel sys %ptrs, add, %arg, %mask token = %t2
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>, tile<4xi1>
        -> tile<4xi32>, token
    %r4, %t4 = atomic_rmw_tko relaxed device %ptrs, max, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %r5, %t5 = atomic_rmw_tko relaxed device %ptrs, min, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %r6, %t6 = atomic_rmw_tko relaxed device %ptrs, umax, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %r7, %t7 = atomic_rmw_tko relaxed device %ptrs, umin, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %r8, %t8 = atomic_rmw_tko relaxed device %ptrs, xchg, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token

    %c0, %ct0 = atomic_cas_tko relaxed device %ptrs, %cmp, %arg
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %c1, %ct1 = atomic_cas_tko acquire device %ptrs, %cmp, %arg, %mask
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>, tile<4xi1>
        -> tile<4xi32>, token
    %c2, %ct2 = atomic_cas_tko release device %ptrs, %cmp, %arg token = %ct1
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>
        -> tile<4xi32>, token
    %c3, %ct3 = atomic_cas_tko acq_rel sys %ptrs, %cmp, %arg, %mask token = %ct2
        : tile<4xptr<i32, #cuda_tile.ptr_attr<unicast>>>, tile<4xi32>, tile<4xi1>
        -> tile<4xi32>, token

    // Keep value and token results live across subsequent instructions.
    %sum = addi %r0, %r1 : tile<4xi32>
    %sum2 = addi %sum, %c3 : tile<4xi32>
    %joined = join_tokens %t0, %t1, %t3, %t4, %t5, %t6, %t7, %t8,
        %ct0, %ct3 : token
    return
  }

  entry @view_atomics(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>) {
    %tv = make_tensor_view %ptr, shape = [8], strides = [1]
        : tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>
        -> tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>
    %view = make_partition_view %tv
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>
    %idx = constant <i32: 0> : tile<i32>
    %value = constant <i32: 1> : tile<4xi32>

    %t0 = atomic_red_view_tko relaxed tl_blk %view[%idx], and, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t1 = atomic_red_view_tko relaxed device %view[%idx], or, %value token = %t0
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t2 = atomic_red_view_tko relaxed device %view[%idx], xor, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t3 = atomic_red_view_tko relaxed device %view[%idx], add, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t4 = atomic_red_view_tko relaxed device %view[%idx], max, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t5 = atomic_red_view_tko relaxed device %view[%idx], min, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t6 = atomic_red_view_tko relaxed device %view[%idx], umax, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %t7 = atomic_red_view_tko relaxed device %view[%idx], umin, %value
        : tile<4xi32>, partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>>, tile<i32> -> token
    %joined = join_tokens %t1, %t2, %t3, %t4, %t5, %t6, %t7 : token
    return
  }

  entry @float_ptr_atomics(
      %ptr: tile<ptr<f32, #cuda_tile.ptr_attr<unicast>>>) {
    %ptr_1d = reshape %ptr
        : tile<ptr<f32, #cuda_tile.ptr_attr<unicast>>>
        -> tile<1xptr<f32, #cuda_tile.ptr_attr<unicast>>>
    %ptrs = broadcast %ptr_1d
        : tile<1xptr<f32, #cuda_tile.ptr_attr<unicast>>>
        -> tile<4xptr<f32, #cuda_tile.ptr_attr<unicast>>>
    %arg = constant <f32: 1.0> : tile<4xf32>
    %cmp = constant <f32: 0.0> : tile<4xf32>
    %mask = constant <i1: true> : tile<4xi1>
    %input_token = make_token : token

    %r, %rt = atomic_rmw_tko acq_rel device %ptrs, addf, %arg, %mask
        token = %input_token
        : tile<4xptr<f32, #cuda_tile.ptr_attr<unicast>>>, tile<4xf32>, tile<4xi1>
        -> tile<4xf32>, token
    %c, %ct = atomic_cas_tko relaxed device %ptrs, %cmp, %r, %mask token = %rt
        : tile<4xptr<f32, #cuda_tile.ptr_attr<unicast>>>, tile<4xf32>, tile<4xi1>
        -> tile<4xf32>, token
    %sum = addf %r, %c : tile<4xf32>
    %joined = join_tokens %rt, %ct : token
    return
  }
}
