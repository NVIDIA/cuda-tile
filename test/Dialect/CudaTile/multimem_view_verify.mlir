// RUN: cuda-tile-opt %s -split-input-file -verify-diagnostics
//
// Dialect verification for multimem view → make_tensor_view → ptr (ptr_attr)
// (multicast required). Architecture verification is tested separately.

// Traceable unicast base: multimem must see multicast

!ptr_uc_f32 = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<unicast>>>
!tv_22_f32 = !cuda_tile.tensor_view<2x2xf32, strides=[2,1], #cuda_tile.ptr_attr<unicast>>
!pv_22_f32 = !cuda_tile.partition_view<tile=(2x2), !tv_22_f32>

cuda_tile.module @multimem_unicast_base_rejected {
  testing$func @f(%ptr: !ptr_uc_f32) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [2, 2], strides = [2, 1]
      : !ptr_uc_f32 -> !tv_22_f32
    %pv = cuda_tile.make_partition_view %tv : !pv_22_f32
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error@+1 {{expected a multicast pointer, but got unicast}}
    %v, %tok_lr_uc = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c1]
      : !pv_22_f32, !cuda_tile.tile<i32> -> !cuda_tile.tile<2x2xf32>, !cuda_tile.token
  }
}

// -----

// Traceable multicast base: verification succeeds.

!ptr_mc_f32 = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
!tv_22_f32_ok = !cuda_tile.tensor_view<2x2xf32, strides=[2,1], #cuda_tile.ptr_attr<multicast>>
!pv_22_f32_ok = !cuda_tile.partition_view<tile=(2x2), !tv_22_f32_ok>

cuda_tile.module @multimem_multicast_base_ok {
  testing$func @g(%ptr: !ptr_mc_f32) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [2, 2], strides = [2, 1]
      : !ptr_mc_f32 -> !tv_22_f32_ok
    %pv = cuda_tile.make_partition_view %tv : !pv_22_f32_ok
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %v, %tok_lr_ok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c1]
      : !pv_22_f32_ok, !cuda_tile.tile<i32> -> !cuda_tile.tile<2x2xf32>, !cuda_tile.token
  }
}

// -----

// Reduced tile shape must match the view's tile type.

!ptr_mc_f32_mis = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
!tv_22_f32_mis = !cuda_tile.tensor_view<2x2xf32, strides=[2,1], #cuda_tile.ptr_attr<multicast>>
!pv_22_f32_mis = !cuda_tile.partition_view<tile=(2x2), !tv_22_f32_mis>

cuda_tile.module @multimem_tile_type_mismatch {
  testing$func @h(%ptr: !ptr_mc_f32_mis) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [2, 2], strides = [2, 1]
      : !ptr_mc_f32_mis -> !tv_22_f32_mis
    %pv = cuda_tile.make_partition_view %tv : !pv_22_f32_mis
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error@+1 {{expected tile type to be}}
    %v, %tok_lr_mis = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c1]
      : !pv_22_f32_mis, !cuda_tile.tile<i32> -> !cuda_tile.tile<2x1xf32>, !cuda_tile.token
  }
}

// -----

// Bitwise reduction on float is invalid for PTX multimem.

!ptr_mc_f32_xor = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
!tv_22_f32_xor = !cuda_tile.tensor_view<2x2xf32, strides=[2,1], #cuda_tile.ptr_attr<multicast>>
!pv_22_f32_xor = !cuda_tile.partition_view<tile=(2x2), !tv_22_f32_xor>

cuda_tile.module @multimem_xor_float_rejected {
  testing$func @f(%ptr: !ptr_mc_f32_xor) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [2, 2], strides = [2, 1]
      : !ptr_mc_f32_xor -> !tv_22_f32_xor
    %pv = cuda_tile.make_partition_view %tv : !pv_22_f32_xor
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error@+1 {{and/or/xor/umin/umax reductions require integer element types}}
    %v, %tok_lr_xor = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c1]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<xor>}
      : !pv_22_f32_xor, !cuda_tile.tile<i32> -> !cuda_tile.tile<2x2xf32>, !cuda_tile.token
  }
}

// -----

!ptr_mc_f32_r = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
!tv_1_f32 = !cuda_tile.tensor_view<1xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>
!pv_1_f32 = !cuda_tile.partition_view<tile=(1), !tv_1_f32>

cuda_tile.module @multimem_reduce_xor_float_rejected {
  testing$func @g(
      %ptr: !ptr_mc_f32_r,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [1], strides = [1]
      : !ptr_mc_f32_r -> !tv_1_f32
    %pv = cuda_tile.make_partition_view %tv : !pv_1_f32
    %vals = cuda_tile.constant <f32: [0.0]> : !cuda_tile.tile<1xf32>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{and/or/xor/umin/umax reductions require integer element types}}
    %t = cuda_tile.multimem_reduce_view_tko relaxed sys %pv, %vals[%c0] token = %token
      attributes {reduction = #cuda_tile.multimem_reduction_kind<xor>}
      : !pv_1_f32, !cuda_tile.tile<1xf32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }
}

// -----

// Min reduction on f32 is invalid (only f16/bf16 supported for min/max).

!ptr_mc_f32_min = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
!tv_1_f32_min = !cuda_tile.tensor_view<1xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>
!pv_1_f32_min = !cuda_tile.partition_view<tile=(1), !tv_1_f32_min>

cuda_tile.module @multimem_load_reduce_min_f32_rejected {
  testing$func @f(%ptr: !ptr_mc_f32_min) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [1], strides = [1]
      : !ptr_mc_f32_min -> !tv_1_f32_min
    %pv = cuda_tile.make_partition_view %tv : !pv_1_f32_min
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{min/max supports f16 and bf16}}
    %v, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
      attributes {reduction = #cuda_tile.multimem_reduction_kind<min>}
      : !pv_1_f32_min, !cuda_tile.tile<i32> -> !cuda_tile.tile<1xf32>, !cuda_tile.token
  }
}

// -----

// Unsigned-max reduction on f32 is invalid (umax requires integer element type).

!ptr_mc_f32_umax = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<multicast>>>
!tv_1_f32_umax = !cuda_tile.tensor_view<1xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>
!pv_1_f32_umax = !cuda_tile.partition_view<tile=(1), !tv_1_f32_umax>

cuda_tile.module @multimem_reduce_umax_f32_rejected {
  testing$func @g(
      %ptr: !ptr_mc_f32_umax,
      %token: !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %ptr, shape = [1], strides = [1]
      : !ptr_mc_f32_umax -> !tv_1_f32_umax
    %pv = cuda_tile.make_partition_view %tv : !pv_1_f32_umax
    %vals = cuda_tile.constant <f32: [0.0]> : !cuda_tile.tile<1xf32>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{and/or/xor/umin/umax reductions require integer}}
    %t = cuda_tile.multimem_reduce_view_tko relaxed sys %pv, %vals[%c0] token = %token
      attributes {reduction = #cuda_tile.multimem_reduction_kind<umax>}
      : !pv_1_f32_umax, !cuda_tile.tile<1xf32>, !cuda_tile.tile<i32>, !cuda_tile.token
      -> !cuda_tile.token
  }
}

// -----

// f16 + token + larger tile: traceable multicast base (positive).

!ptr_mcast_f16 = !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
!tv_f16 = !cuda_tile.tensor_view<128x64xf16, strides=[64, 1], #cuda_tile.ptr_attr<multicast>>
!pv_f16 = !cuda_tile.partition_view<tile=(128x64), !tv_f16>

cuda_tile.module @multimem_traceable_multicast_ok_f16_token {
  testing$func @f(
      %base: !ptr_mcast_f16,
      %token: !cuda_tile.token) -> (!cuda_tile.tile<128x64xf16>, !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !ptr_mcast_f16 -> !tv_f16
    %pv = cuda_tile.make_partition_view %tv : !pv_f16
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %a, %b = cuda_tile.multimem_load_reduce_view_tko relaxed device %pv[%c0, %c1] token = %token
      : !pv_f16, !cuda_tile.tile<i32>, !cuda_tile.token -> !cuda_tile.tile<128x64xf16>, !cuda_tile.token
    return %a, %b : !cuda_tile.tile<128x64xf16>, !cuda_tile.token
  }
}

// -----

// The odd packing extent and odd mapped tile extent violate the documented
// execution-time contract and make an executed access UB. The compiler does
// not prove these preconditions, so the CudaTile verifier does not reject it.

cuda_tile.module @multimem_odd_packing_extents_not_rejected {
  testing$func @f(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = cuda_tile.make_tensor_view %base, shape = [3, 8], strides = [1, 8]
      : !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<multicast>>>
        -> !cuda_tile.tensor_view<3x8xf16, strides=[1,8], #cuda_tile.ptr_attr<multicast>>
    %pv = cuda_tile.make_partition_view %tv
      : !cuda_tile.partition_view<tile=(2x1), !cuda_tile.tensor_view<3x8xf16, strides=[1,8], #cuda_tile.ptr_attr<multicast>>, dim_map=[1,0]>
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %v, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c0]
      : !cuda_tile.partition_view<tile=(2x1), !cuda_tile.tensor_view<3x8xf16, strides=[1,8], #cuda_tile.ptr_attr<multicast>>, dim_map=[1,0]>, !cuda_tile.tile<i32>
        -> !cuda_tile.tile<2x1xf16>, !cuda_tile.token
  }
}

!ptr_ucast_f16 = !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<unicast>>>
!tv_uc_f16 = !cuda_tile.tensor_view<128x64xf16, strides=[64, 1], #cuda_tile.ptr_attr<unicast>>
!pv_uc_f16 = !cuda_tile.partition_view<tile=(128x64), !tv_uc_f16>

cuda_tile.module @multimem_traceable_unicast_rejected_f16_token {
  testing$func @g(
      %base: !ptr_ucast_f16,
      %token: !cuda_tile.token) -> (!cuda_tile.tile<128x64xf16>, !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !ptr_ucast_f16 -> !tv_uc_f16
    %pv = cuda_tile.make_partition_view %tv : !pv_uc_f16
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{expected a multicast pointer, but got unicast}}
    %a, %b = cuda_tile.multimem_load_reduce_view_tko relaxed device %pv[%c0, %c1] token = %token
      : !pv_uc_f16, !cuda_tile.tile<i32>, !cuda_tile.token -> !cuda_tile.tile<128x64xf16>, !cuda_tile.token
    return %a, %b : !cuda_tile.tile<128x64xf16>, !cuda_tile.token
  }
}

// -----

!ptr_none_f16 = !cuda_tile.tile<!cuda_tile.ptr<f16, #cuda_tile.ptr_attr<none>>>
!tv_ne_f16 = !cuda_tile.tensor_view<128x64xf16, strides=[64, 1], #cuda_tile.ptr_attr<none>>
!pv_ne_f16 = !cuda_tile.partition_view<tile=(128x64), !tv_ne_f16>

cuda_tile.module @multimem_traceable_none_attr_rejected_f16_token {
  testing$func @h(
      %base: !ptr_none_f16,
      %token: !cuda_tile.token) -> (!cuda_tile.tile<128x64xf16>, !cuda_tile.token) {
    %tv = cuda_tile.make_tensor_view %base, shape = [128, 64], strides = [64, 1]
        : !ptr_none_f16 -> !tv_ne_f16
    %pv = cuda_tile.make_partition_view %tv : !pv_ne_f16
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{expected a multicast pointer, but got none}}
    %a, %b = cuda_tile.multimem_load_reduce_view_tko relaxed device %pv[%c0, %c1] token = %token
      : !pv_ne_f16, !cuda_tile.tile<i32>, !cuda_tile.token -> !cuda_tile.tile<128x64xf16>, !cuda_tile.token
    return %a, %b : !cuda_tile.tile<128x64xf16>, !cuda_tile.token
  }
}

// -----

!tv_no_attr = !cuda_tile.tensor_view<2x2xf32, strides=[2,1]>
!pv_no_attr = !cuda_tile.partition_view<tile=(2x2), !tv_no_attr>

cuda_tile.module @multimem_missing_view_attr_rejected {
  testing$func @i(%pv: !pv_no_attr) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error@+1 {{requires multicast ptr_attr, got none}}
    %v, %tok_lr_na = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0, %c1]
      : !pv_no_attr, !cuda_tile.tile<i32> -> !cuda_tile.tile<2x2xf32>, !cuda_tile.token
  }
}

// -----

// MultimemReduceViewTkoOp — unicast ptr_attr rejected.

!tv_uc_reduce = !cuda_tile.tensor_view<2xi32, strides=[1], #cuda_tile.ptr_attr<unicast>>
!pv_uc_reduce = !cuda_tile.partition_view<tile=(1), !tv_uc_reduce>

cuda_tile.module @multimem_reduce_unicast_rejected {
  testing$func @f(
      %pv: !pv_uc_reduce,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{requires multicast ptr_attr, got #cuda_tile.ptr_attr<unicast>}}
    %t = cuda_tile.multimem_reduce_view_tko relaxed device %pv, %vals[%c0] token = %token
      : !pv_uc_reduce, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
        -> !cuda_tile.token
  }
}

// -----

// MultimemReduceViewTkoOp — view without ptr_attr rejected.

!tv_none_reduce = !cuda_tile.tensor_view<2xi32, strides=[1]>
!pv_none_reduce = !cuda_tile.partition_view<tile=(1), !tv_none_reduce>

cuda_tile.module @multimem_reduce_none_rejected {
  testing$func @g(
      %pv: !pv_none_reduce,
      %vals: !cuda_tile.tile<1xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{requires multicast ptr_attr, got none}}
    %t = cuda_tile.multimem_reduce_view_tko relaxed device %pv, %vals[%c0] token = %token
      : !pv_none_reduce, !cuda_tile.tile<1xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
        -> !cuda_tile.token
  }
}

// -----

// MultimemStoreViewTkoOp — unicast ptr_attr rejected.

!tv_uc_store = !cuda_tile.tensor_view<128x64xf32, strides=[64,1], #cuda_tile.ptr_attr<unicast>>
!pv_uc_store = !cuda_tile.partition_view<tile=(128x64), !tv_uc_store>

cuda_tile.module @multimem_store_unicast_rejected {
  testing$func @f(
      %pv: !pv_uc_store,
      %vals: !cuda_tile.tile<128x64xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error@+1 {{requires multicast ptr_attr, got #cuda_tile.ptr_attr<unicast>}}
    %tok_st_uc = cuda_tile.multimem_store_view_tko weak %pv, %vals[%c0, %c1]
      : !pv_uc_store, !cuda_tile.tile<128x64xf32>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }
}

// -----

// MultimemStoreViewTkoOp — view without ptr_attr rejected.

!tv_none_store = !cuda_tile.tensor_view<128x64xf32, strides=[64,1]>
!pv_none_store = !cuda_tile.partition_view<tile=(128x64), !tv_none_store>

cuda_tile.module @multimem_store_none_rejected {
  testing$func @g(
      %pv: !pv_none_store,
      %vals: !cuda_tile.tile<128x64xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error@+1 {{requires multicast ptr_attr, got none}}
    %tok_st_na = cuda_tile.multimem_store_view_tko weak %pv, %vals[%c0, %c1]
      : !pv_none_store, !cuda_tile.tile<128x64xf32>, !cuda_tile.tile<i32>
      -> !cuda_tile.token
  }
}

// -----

// MultimemLoadReduceViewTkoOp — strided_view rejected.

!tv_mc_load_sv = !cuda_tile.tensor_view<16x16xi32, strides=[16,1], #cuda_tile.ptr_attr<multicast>>
!sv_mc_load = !cuda_tile.strided_view<tile=(16x16), traversal_strides=[16,1], !tv_mc_load_sv>

cuda_tile.module @multimem_load_reduce_strided_view_rejected {
  testing$func @f(%sv: !sv_mc_load) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem view operations support only partition_view; got}}
    %v, %tok = cuda_tile.multimem_load_reduce_view_tko weak %sv[%c0, %c0]
      : !sv_mc_load, !cuda_tile.tile<i32> -> !cuda_tile.tile<16x16xi32>, !cuda_tile.token
  }
}

// -----

// MultimemReduceViewTkoOp — strided_view rejected.

!tv_mc_reduce_sv = !cuda_tile.tensor_view<16x16xi32, strides=[16,1], #cuda_tile.ptr_attr<multicast>>
!sv_mc_reduce = !cuda_tile.strided_view<tile=(16x16), traversal_strides=[16,1], !tv_mc_reduce_sv>

cuda_tile.module @multimem_reduce_strided_view_rejected {
  testing$func @f(
      %sv: !sv_mc_reduce,
      %vals: !cuda_tile.tile<16x16xi32>,
      %token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem view operations support only partition_view; got}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %sv, %vals[%c0, %c0]
      token = %token
      : !sv_mc_reduce, !cuda_tile.tile<16x16xi32>, !cuda_tile.tile<i32>, !cuda_tile.token
        -> !cuda_tile.token
  }
}

// -----

// MultimemStoreViewTkoOp — strided_view rejected.

!tv_mc_store_sv = !cuda_tile.tensor_view<16x16xi32, strides=[16,1], #cuda_tile.ptr_attr<multicast>>
!sv_mc_store = !cuda_tile.strided_view<tile=(16x16), traversal_strides=[16,1], !tv_mc_store_sv>

cuda_tile.module @multimem_store_strided_view_rejected {
  testing$func @f(
      %sv: !sv_mc_store,
      %vals: !cuda_tile.tile<16x16xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem view operations support only partition_view; got}}
    %tok = cuda_tile.multimem_store_view_tko weak %sv, %vals[%c0, %c0]
      : !sv_mc_store, !cuda_tile.tile<16x16xi32>, !cuda_tile.tile<i32>
        -> !cuda_tile.token
  }
}

// -----

// Each multimem op rejects integer element widths other than 32 and 64 bits.

cuda_tile.module @multimem_load_i16_rejected {
  testing$func @f(
      %pv: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xi16, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem requires 32- or 64-bit integer/index element types; got 16 bits}}
    %v, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xi16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
        !cuda_tile.tile<i32> -> !cuda_tile.tile<2xi16>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @multimem_reduce_i16_rejected {
  testing$func @f(
      %pv: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xi16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xi16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem requires 32- or 64-bit integer/index element types; got 16 bits}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %pv, %vals[%c0]
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xi16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
        !cuda_tile.tile<2xi16>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @multimem_store_i16_rejected {
  testing$func @f(
      %pv: !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xi16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<2xi16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem requires 32- or 64-bit integer/index element types; got 16 bits}}
    %tok = cuda_tile.multimem_store_view_tko weak %pv, %vals[%c0]
      : !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<2xi16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
        !cuda_tile.tile<2xi16>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Each float multimem op rejects a tile whose total size is below the
// 4-byte (32-bit) PTX transaction granularity. A 1xf16 tile is 16 bits.

cuda_tile.module @multimem_load_reduce_f16_undersized_tile_rejected {
  testing$func @f(
      %pv: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<1xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem op requires tile total size to be a multiple of 4 bytes for float element types}}
    %v, %tok = cuda_tile.multimem_load_reduce_view_tko weak %pv[%c0]
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<1xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
        !cuda_tile.tile<i32> -> !cuda_tile.tile<1xf16>, !cuda_tile.token
  }
}

// -----

cuda_tile.module @multimem_reduce_f16_undersized_tile_rejected {
  testing$func @f(
      %pv: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<1xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem op requires tile total size to be a multiple of 4 bytes for float element types}}
    %tok = cuda_tile.multimem_reduce_view_tko relaxed device %pv, %vals[%c0]
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<1xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
        !cuda_tile.tile<1xf16>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @multimem_store_f16_undersized_tile_rejected {
  testing$func @f(
      %pv: !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<1xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
      %vals: !cuda_tile.tile<1xf16>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error@+1 {{multimem op requires tile total size to be a multiple of 4 bytes for float element types}}
    %tok = cuda_tile.multimem_store_view_tko weak %pv, %vals[%c0]
      : !cuda_tile.partition_view<tile=(1), !cuda_tile.tensor_view<1xf16, strides=[1], #cuda_tile.ptr_attr<multicast>>>,
        !cuda_tile.tile<1xf16>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}
