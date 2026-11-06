// RUN: %round_trip_test %s %t --preview=simt
// RUN: cuda-tile-opt --preview=simt %s -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: cuda-tile-opt %s --preview=simt | FileCheck %s
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=NO-PREVIEW
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   --preview=simt -bytecode-version=13.4 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OLD-VERSION
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   --preview=simt %s -o %t.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-NO-PREVIEW

// CHECK: preview$extern_impl @foreign_sm90<[M: index]>
// CHECK-SAME: implements @foreign<[M]>
// CHECK-SAME: preview$fragment<1xf32, local
// CHECK-SAME: fragmap = <(1, M, 1):(0, 1, 0)>
// CHECK-SAME: cuda_tile.preview$access = #cuda_tile<preview$access<inout>>
// CHECK-SAME: synchronization = #cuda_tile<preview$synchronization<cta>>
// CHECK-SAME: arch = ["sm90"]
// CHECK-SAME: scratch {smem.size = #cuda_tile<preview$expr_mul<
// CHECK-SAME: smem.alignment = #cuda_tile<preview$expr_int<128>>}
// CHECK-SAME: frontendABI = {f32 = "i32"}
// CHECK: preview$func extern @foreign<[M: index]>
// CHECK-SAME: cuda_tile.preview$access = #cuda_tile<preview$access<inout>>
// CHECK: %[[TOKEN:.+]] = make_token : token
// CHECK: %{{.+}}:2 = preview$call @foreign<[128]>(%[[TOKEN]], %{{[^)]+}}) : (!cuda_tile.token, !cuda_tile.tile<128xf32>)
// CHECK-SAME: -> (!cuda_tile.token, !cuda_tile.tile<128xf32>)

// NO-PREVIEW: uses preview feature 'simt', which is not enabled
// READ-NO-PREVIEW: public-preview bytecode requires at least one enabled preview feature
// OLD-VERSION: only valid at the current bytecode version 13.5

// GENERIC: "cuda_tile.preview$extern_impl"
// GENERIC-SAME: synchronization = #cuda_tile<preview$synchronization<cta>>

// Symbol order is preserved across the bytecode round-trip.
// CHECK: entry @before_impl() {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: preview$extern_impl @impl implements @decl
// CHECK-NEXT: preview$func extern @decl(
// CHECK: entry @after_impl() {
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK: preview$func extern @callee(
// Verify preview calls round-trip with zero, one, and multiple token values.
// CHECK: %[[ZERO:.+]] = preview$call @callee(%[[ARG:.+]]) : (!cuda_tile.tile<f32>) -> !cuda_tile.tile<f32>
// CHECK: %[[TOKEN0:.+]] = make_token : token
// CHECK: %[[ONE:.+]]:2 = preview$call @callee(%[[TOKEN0]], %[[ZERO]]) : (!cuda_tile.token, !cuda_tile.tile<f32>) -> (!cuda_tile.token, !cuda_tile.tile<f32>)
// CHECK: %[[TOKEN2:.+]] = make_token : token
// CHECK: %{{.+}}:3 = preview$call @callee(%[[ONE]]#0, %[[TOKEN2]], %[[ONE]]#1) : (!cuda_tile.token, !cuda_tile.token, !cuda_tile.tile<f32>) -> (!cuda_tile.token, !cuda_tile.token, !cuda_tile.tile<f32>)

cuda_tile.module @kernels {

  preview$extern_impl @foreign_sm90<[M: index]>
    implements @foreign<[M]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, M, 1):(0, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
    synchronization = #cuda_tile.preview$synchronization<cta>
    arch = ["sm90"]
    scratch {
      smem.size = M * 4,
      smem.alignment = 128
    }
    {frontendABI = {f32 = "i32"}}

  preview$func extern @foreign<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>
      }) -> !cuda_tile.tile<[M]xf32>

  entry @caller() {
    %arg0 = constant <f32: 1.0> : tile<128xf32>
    %token = make_token : !cuda_tile.token
    %next_token, %result = preview$call @foreign<[128]>(%token, %arg0)
      : (!cuda_tile.token, !cuda_tile.tile<128xf32>)
        -> (!cuda_tile.token, !cuda_tile.tile<128xf32>)
    return
  }
  cuda_tile.entry @before_impl() {
    cuda_tile.return
  }
  cuda_tile.preview$extern_impl @impl implements @decl(
      %arg0: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 1, 0)>> {
          cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
    synchronization = #cuda_tile.preview$synchronization<warp>
  cuda_tile.preview$func extern @decl(
      %arg0: !cuda_tile.tile<32xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<32xf32>
  cuda_tile.entry @after_impl() {
    cuda_tile.return
  }
  preview$func extern @callee(
      %arg0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<f32>

  entry @call_caller(%arg0: !cuda_tile.tile<f32>) {
    %zero = preview$call @callee(%arg0)
        : (!cuda_tile.tile<f32>) -> !cuda_tile.tile<f32>
    %token0 = make_token : !cuda_tile.token
    %token1, %one = preview$call @callee(%token0, %zero)
        : (!cuda_tile.token, !cuda_tile.tile<f32>)
          -> (!cuda_tile.token, !cuda_tile.tile<f32>)
    %token2 = make_token : !cuda_tile.token
    %next0, %next1, %multiple = preview$call
        @callee(%token1, %token2, %one)
        : (!cuda_tile.token, !cuda_tile.token, !cuda_tile.tile<f32>)
          -> (!cuda_tile.token, !cuda_tile.token, !cuda_tile.tile<f32>)
    return
  }

  //===--------------------------------------------------------------------===//
  // FunctionTableWriter: bodiless extern func / extern_impl serialization and
  // the fragment type's dependent-attribute registration.
  //===--------------------------------------------------------------------===//

  // Bodiless implementations are emitted before extern declarations in the
  // bytecode function table.
  cuda_tile.preview$extern_impl @my_impl<[M: index, N: index]>
    implements @my_extern<[M, N]>
    (%arg0: !cuda_tile.preview$fragment<128xf32, local,
       mem_layout = #cuda_tile.preview$layout<(128):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 128):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
     %arg1: !cuda_tile.preview$fragment<64xf16, local,
       mem_layout = #cuda_tile.preview$layout<(64):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 64):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
    synchronization = #cuda_tile.preview$synchronization<none>
    arch = ["sm90"]

  cuda_tile.preview$func extern @my_extern<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %arg1: !cuda_tile.tile<64xf16> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<64xf16>

  // A second bodiless declaration independently exercises isExternFunc.
  cuda_tile.preview$func extern @my_decl<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %arg1: !cuda_tile.tile<64xf16> {
        cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    -> !cuda_tile.tile<64xf16>

  cuda_tile.entry @kernel_extern_funcs(
      %a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<nearest_even> : tile<f32>
    cuda_tile.return
  }

  //===--------------------------------------------------------------------===//
  // SymbolicExprVarAttr / SymbolicExprIntAttr serializers, reached through
  // extern_impl const_args and concrete tile shapes in the type table.
  //===--------------------------------------------------------------------===//

  // A separate bodiless declaration exercises the extern-function writer path.
  cuda_tile.preview$func extern @decl_var<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  // Extern preview$func that the native implementation implements; the
  // implements<[...]> const_args are matched against its const_params.
  cuda_tile.preview$func extern @extern_var<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> !cuda_tile.tile<128xf32>

  cuda_tile.preview$extern_impl @impl_var<[M: index, N: index]>
    implements @extern_var<[#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_var<"N">]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
       mem_layout = #cuda_tile.preview$layout<(1):(1)>,
       fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.entry @kernel_symbolic_exprs() {
    cuda_tile.return
  }
}
