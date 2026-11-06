// RUN: cuda-tile-opt %s -allow-unregistered-dialect -split-input-file | FileCheck %s

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakeTensorViewOp::print paths that emit the base pointer type
// explicitly because the pointer has a ptr_attr.
// Without ptr_attr the printer uses the compact form; with ptr_attr it
// prints `base, indexType -> result` or `base -> result`.

// Covers MakeTensorViewOp::print with a ptr_attr and no dynamic shape/stride:
// takes the printExplicitBase branch but skips the dynamic type printing.
!ptr_none_f32 = !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<none>>>
!tv_none_f32 = !cuda_tile.tensor_view<4x4xf32, strides=[4, 1], #cuda_tile.ptr_attr<none>>

// CHECK-LABEL: @mtv_print_none_static
cuda_tile.module @kernels {
  testing$func @mtv_print_none_static(%p: !ptr_none_f32) {
    // CHECK: make_tensor_view %{{.+}}, shape = [4, 4], strides = [4, 1]
    // CHECK-SAME: tile<ptr<f32, <none>>>
    %tv = cuda_tile.make_tensor_view %p, shape = [4, 4], strides = [4, 1] : !ptr_none_f32 -> !tv_none_f32
  }
}

// -----

// Covers MakeTensorViewOp::print compact branch (no ptr_attr, dynamic
// strides path): printExplicitBase is false but dynamic types still get
// printed.
!ptr_f32 = !cuda_tile.tile<!cuda_tile.ptr<f32>>
!tv_dyn_stride = !cuda_tile.tensor_view<4x4xf32, strides=[?, 1]>

// CHECK-LABEL: @mtv_print_compact_dyn_stride
cuda_tile.module @kernels {
  testing$func @mtv_print_compact_dyn_stride(%p: !ptr_f32, %d: !cuda_tile.tile<i32>) {
    // CHECK: make_tensor_view %{{.+}}, shape = [4, 4], strides = [%{{.+}}, 1]
    // CHECK-SAME: tensor_view<4x4xf32
    %tv = cuda_tile.make_tensor_view %p, shape = [4, 4], strides = [%d, 1] : !cuda_tile.tile<i32> -> !tv_dyn_stride
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers verifyAggregateOp's bf16<->f16 identity tolerance path:
// when operand is bf16 and identity is f16 (or vice versa), the verifier
// does NOT report an element-type mismatch.

// Covers verifyAggregateOp tolerance: bf16 operand + f16 identity attribute.
// CHECK-LABEL: @reduce_bf16_with_f16_identity
cuda_tile.module @test {
  testing$func @reduce_bf16_with_f16_identity(%arg0: !cuda_tile.tile<8xbf16>) {
    // The identity is `f16: 0.0` while the operand element type is bf16.
    // This is permitted by verifyAggregateOp's bf16<->f16 tolerance.
    // CHECK: reduce %{{.+}} dim=0 identities=
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.0 : f16]
      : !cuda_tile.tile<8xbf16> -> !cuda_tile.tile<bf16>
      (%lhs : !cuda_tile.tile<bf16>, %rhs : !cuda_tile.tile<bf16>) {
        %r = cuda_tile.addf %lhs, %rhs rounding<nearest_even> : !cuda_tile.tile<bf16>
        cuda_tile.yield %r : !cuda_tile.tile<bf16>
      }
  }
}

// -----

// Covers verifyAggregateOp tolerance: f16 operand + bf16 identity attribute.
// CHECK-LABEL: @reduce_f16_with_bf16_identity
cuda_tile.module @test {
  testing$func @reduce_f16_with_bf16_identity(%arg0: !cuda_tile.tile<8xf16>) {
    // The identity is `bf16: 0.0` while the operand element type is f16.
    // CHECK: reduce %{{.+}} dim=0 identities=
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.0 : bf16]
      : !cuda_tile.tile<8xf16> -> !cuda_tile.tile<f16>
      (%lhs : !cuda_tile.tile<f16>, %rhs : !cuda_tile.tile<f16>) {
        %r = cuda_tile.addf %lhs, %rhs rounding<nearest_even> : !cuda_tile.tile<f16>
        cuda_tile.yield %r : !cuda_tile.tile<f16>
      }
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers parseViewTkoCommon and printViewTkoOptionals optional-token /
// optional-optimization_hints / attr-dict-with-keyword branches.

// Covers LoadViewTkoOp::parse and ::print with both 'token = %tok' and
// 'optimization_hints = <...>' optional fields, exercising
// parseViewTkoCommon's hasToken-true branch and the
// optimization_hints success branch.
// CHECK-LABEL: @load_view_tko_with_token_and_hints
cuda_tile.module @kernels {
  testing$func @load_view_tko_with_token_and_hints(
      %view: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>,
      %in_token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: load_view_tko weak %{{.+}}[%{{.+}}] token = %{{.+}} optimization_hints
    %t, %tok = cuda_tile.load_view_tko weak %view[%c0]
        token = %in_token
        optimization_hints = <sm_100 = {num_cta_in_cga = 2}>
        : !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>,
          !cuda_tile.tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
  }
}

// -----

// Covers StoreViewTkoOp::parse with 'token = %tok' and
// 'optimization_hints = <...>' optional fields.
// CHECK-LABEL: @store_view_tko_with_token_and_hints
cuda_tile.module @kernels {
  testing$func @store_view_tko_with_token_and_hints(
      %tile: !cuda_tile.tile<8xf32>,
      %view: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>,
      %in_token: !cuda_tile.token) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: store_view_tko weak %{{.+}}, %{{.+}}[%{{.+}}] token = %{{.+}} optimization_hints
    %tok = cuda_tile.store_view_tko weak %tile, %view[%c0]
        token = %in_token
        optimization_hints = <sm_100 = {num_cta_in_cga = 2}>
        : !cuda_tile.tile<8xf32>,
          !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>,
          !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

