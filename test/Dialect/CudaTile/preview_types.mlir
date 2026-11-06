// RUN: cuda-tile-opt --preview=simt -allow-unregistered-dialect -split-input-file %s \
// RUN:   | cuda-tile-opt --preview=simt -allow-unregistered-dialect -split-input-file \
// RUN:   | FileCheck %s
// RUN: cuda-tile-opt --preview=simt -allow-unregistered-dialect -split-input-file \
// RUN:   -mlir-print-op-generic %s \
// RUN:   | cuda-tile-opt --preview=simt -allow-unregistered-dialect -split-input-file \
// RUN:   | FileCheck %s
// RUN: not cuda-tile-opt -allow-unregistered-dialect -split-input-file %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-PREVIEW

// Parse/print round-trip for the public-preview types. The stable types live in
// types.mlir; only the SIMT-gated ones are here.

//===----------------------------------------------------------------------===//
// preview$sym_t and symbolic tile shapes
//===----------------------------------------------------------------------===//

cuda_tile.module @kernels {

// CHECK-LABEL: testing$func @test_symbolic_types
testing$func @test_symbolic_types(
    // CHECK-SAME: preview$sym_t<T>
    %arg0: !cuda_tile.preview$sym_t<T>,
    // A lower-case single letter is also valid.
    // CHECK-SAME: preview$sym_t<k>
    %arg1: !cuda_tile.preview$sym_t<k>,
    // CHECK-SAME: preview$sym_t<TILE_M>
    %arg2: !cuda_tile.preview$sym_t<TILE_M>,
    // CHECK-SAME: preview$sym_t<TILE_N>
    %arg3: !cuda_tile.preview$sym_t<TILE_N>,
    // CHECK-SAME: preview$sym_t<_>
    %arg4: !cuda_tile.preview$sym_t<_>,
    // CHECK-SAME: preview$sym_t<Explicit>
    %arg5: !cuda_tile.preview$sym_t<#cuda_tile.preview$expr_var<"Explicit">>,
    // CHECK-SAME: preview$sym_t<"T-1">
    %arg6: !cuda_tile.preview$sym_t<"T-1">) {
  return
}

// CHECK-LABEL: testing$func @test_symbolic_tile_types
testing$func @test_symbolic_tile_types(
    // CHECK-SAME: tile<[M]x[N]xf32>
    %arg0: !cuda_tile.tile<[M]x[N]xf32>,
    // CHECK-SAME: tile<[M+1]x8xi32>
    %arg1: !cuda_tile.tile<[M+1]x8xi32>,
    // CHECK-SAME: tile<8x[M]xptr<f32>>
    %arg2: !cuda_tile.tile<8x[M]xptr<f32>>,
    // CHECK-SAME: tile<[(M+1)*2]xbf16>
    %arg3: !cuda_tile.tile<[(M+1)*2]xbf16>) {
  return
}
}

// -----

//===----------------------------------------------------------------------===//
// preview$fragment
//===----------------------------------------------------------------------===//

// The `fragment` type describes one external-function invocation's view of a
// tile at the ABI boundary. It is exercised here on unregistered ops because it
// is not yet wired into any cuda_tile operation (it round-trips standalone).

cuda_tile.module @fragment_type_test {
  cuda_tile.testing$func @roundtrip() {
    // Scalar local fragment; alignment defaults to 1 and is elided (the type
    // ends right after fragmap, with no `alignment` suffix).
    // CHECK: preview$fragment<1xf32, local,{{.*}}fragmap = <{{.*}}>>
    %0 = "test.frag_scalar"() : () -> !cuda_tile.preview$fragment<1xf32, local, mem_layout = <(1):(1)>, fragmap = <(1024, 1, 1):(1, 0, 0)>>

    // 8x8 row-major shared-memory fragment.
    // CHECK: preview$fragment<8x8xf32, smem
    %1 = "test.frag_smem"() : () -> !cuda_tile.preview$fragment<8x8xf32, smem, mem_layout = <(8, 8):(8, 1)>, fragmap = <(1, 1, (8, 8)):(0, 0, (1, 8))>>

    // Local fragment with a different element type.
    // CHECK: preview$fragment<4x8xf16, local
    %2 = "test.frag_f16"() : () -> !cuda_tile.preview$fragment<4x8xf16, local, mem_layout = <(4, 8):(8, 1)>, fragmap = <(1, 1, 1):(1, 0, 0)>>

    // Non-default alignment must appear in the printed output.
    // CHECK: preview$fragment<2xf32, local,{{.*}}alignment = 4
    %3 = "test.frag_aligned"() : () -> !cuda_tile.preview$fragment<2xf32, local, mem_layout = <(2):(1)>, fragmap = <(512, 1, 2):(2, 0, 1)>, alignment = 4>

    // Symbolic shape dimension (written in brackets) with symbolic mem_layout
    // and fragmap. Symbolic layouts are now permitted.
    // CHECK: preview$fragment<[N]xf32, local
    %4 = "test.frag_symbolic_shape"() : () -> !cuda_tile.preview$fragment<[N]xf32, local, mem_layout = <(N):(1)>, fragmap = <(1024, 1, N):(N, 0, 1)>>

    // Mixed concrete/symbolic shape.
    // CHECK: preview$fragment<2x[F]xf32, local
    %5 = "test.frag_mixed_shape"() : () -> !cuda_tile.preview$fragment<2x[F]xf32, local, mem_layout = <(2, F):(F, 1)>, fragmap = <(1, 1, (2, F)):(0, 0, (F, 1))>>

    // Multiple symbolic dimensions.
    // CHECK: preview$fragment<[M]x[N]xf32, smem
    %6 = "test.frag_two_symbolic"() : () -> !cuda_tile.preview$fragment<[M]x[N]xf32, smem, mem_layout = <(M, N):(N, 1)>, fragmap = <(1, 1, (M, N)):(0, 0, (N, 1))>>

    // Multiple symbolic dimensions interleaved with a concrete dimension.
    // CHECK: preview$fragment<[M]x4x[N]xf32, local
    %7 = "test.frag_three_mixed"() : () -> !cuda_tile.preview$fragment<[M]x4x[N]xf32, local, mem_layout = <(M, 4, N):(N, 1, 1)>, fragmap = <(1, 1, (M, 4, N)):(0, 0, (N, 1, 1))>>

    // Symbolic alignment (variable).
    // CHECK: preview$fragment<4xf32, local,{{.*}}alignment = A
    %8 = "test.frag_sym_align"() : () -> !cuda_tile.preview$fragment<4xf32, local, mem_layout = <(4):(1)>, fragmap = <(256, 1, 4):(4, 0, 1)>, alignment = A>

    // Symbolic alignment expression.
    // CHECK: preview$fragment<8xf32, local,{{.*}}alignment = N*4
    %9 = "test.frag_sym_align_expr"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(8):(1)>, fragmap = <(128, 1, 8):(8, 0, 1)>, alignment = N * 4>

    // A mem_layout only has to be a permutation of the footprint, so any mode
    // order is fine. This one is column-major over the same 4x8 footprint.
    // CHECK: preview$fragment<4x8xf32, local, mem_layout = <(4, 8):(1, 4)>
    %10 = "test.frag_col_major"() : () -> !cuda_tile.preview$fragment<4x8xf32, local, mem_layout = <(4, 8):(1, 4)>, fragmap = <(1, 1, (4, 8)):(0, 0, (1, 4))>>

    // A hierarchical mem_layout is flattened before the permutation rules are
    // applied, so nesting round-trips unchanged.
    // CHECK: mem_layout = <(2, (2, 2)):(1, (2, 4))>
    %11 = "test.frag_nested_mem_layout"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(2, (2, 2)):(1, (2, 4))>, fragmap = <(128, 1, 8):(8, 0, 1)>>

    // A unit mode is never stepped, so its stride is unconstrained and a
    // stride-zero unit mode is not treated as a broadcast.
    // CHECK: mem_layout = <(1, 8):(0, 1)>
    %12 = "test.frag_unit_mode"() : () -> !cuda_tile.preview$fragment<8xf32, local, mem_layout = <(1, 8):(0, 1)>, fragmap = <(128, 1, 8):(8, 0, 1)>>

    // A rank-0 mem_layout has an empty product, which matches a one-element
    // fragment.
    // CHECK: mem_layout = <():()>
    %13 = "test.frag_rank0_mem_layout"() : () -> !cuda_tile.preview$fragment<1xf32, local, mem_layout = <():()>, fragmap = <(1024, 1, 1):(1, 0, 0)>>

    // A negation of a literal is a negative literal, whatever parentheses it
    // was written with, so `-(-4)` is the alignment 4 and prints as such. The
    // printer has no way to spell a negated literal other than `-4`, so the
    // parser has to fold it for the type to round-trip.
    // CHECK: preview$fragment<2xf32, local,{{.*}}alignment = 4
    %14 = "test.frag_neg_neg_align"() : () -> !cuda_tile.preview$fragment<2xf32, local, mem_layout = <(2):(1)>, fragmap = <(512, 1, 2):(2, 0, 1)>, alignment = -(-4)>
    return
  }
}

// NO-PREVIEW: uses preview feature 'simt', which is not enabled

// -----

//===----------------------------------------------------------------------===//
// index: a public-preview builtin type in cuda_tile IR
//===----------------------------------------------------------------------===//

// The builtin `index` type is public-preview surface. These check that the
// loop terminators (yield, break, continue) accept `index` operands, so that
// frontends can embed non-CudaTile values inside CudaTile control flow.
// The equivalent i32/f32 cases are stable and live in ops.mlir.

cuda_tile.module @kernels {

  // CHECK-LABEL: testing$func @yield_non_cuda_tile_type_in_if
  // CHECK-SAME: (%[[VAL:.*]]: index, %[[COND:.*]]: tile<i1>)
  testing$func @yield_non_cuda_tile_type_in_if(%val: index, %cond: !cuda_tile.tile<i1>) {
    // CHECK: if %[[COND]] -> (index)
    %r = if %cond -> (index) {
      // CHECK: yield %[[VAL]] : index
      yield %val : index
    } else {
      yield %val : index
    }
  }

  // CHECK-LABEL: testing$func @break_non_cuda_tile_type_in_loop
  // CHECK-SAME: (%[[VAL:.*]]: index, %[[COND:.*]]: tile<i1>)
  testing$func @break_non_cuda_tile_type_in_loop(%val: index, %cond: !cuda_tile.tile<i1>) {
    // CHECK: loop : index
    %r = loop : index {
      if %cond {
        // CHECK: break %[[VAL]] : index
        break %val : index
      }
      break %val : index
    }
  }

  // CHECK-LABEL: testing$func @continue_non_cuda_tile_type_in_loop_index
  // CHECK-SAME: (%[[VAL:.*]]: index)
  testing$func @continue_non_cuda_tile_type_in_loop_index(%val: index) {
    // CHECK: loop iter_values(%{{.*}} = %[[VAL]]) : index
    loop iter_values(%arg = %val) : index {
      // CHECK: continue %{{.*}} : index
      continue %arg : index
    }
  }

}
