// RUN: cuda-tile-opt --preview=simt -split-input-file %s \
// RUN:   | cuda-tile-opt --preview=simt -split-input-file | FileCheck %s
// RUN: cuda-tile-opt --preview=simt -split-input-file -mlir-print-op-generic %s \
// RUN:   | cuda-tile-opt --preview=simt -split-input-file | FileCheck %s
// RUN: not cuda-tile-opt -split-input-file %s 2>&1 | FileCheck %s --check-prefix=NO-PREVIEW

// Parse/print round-trip for the public-preview SIMT operations:
// cuda_tile.preview$func, cuda_tile.preview$extern_impl and cuda_tile.preview$call.
// Verifier diagnostics live in preview_invalid.mlir; the types and attributes
// these ops carry are covered by preview_types.mlir and preview_attrs.mlir.

//===----------------------------------------------------------------------===//
// The complete gated textual SIMT surface
//===----------------------------------------------------------------------===//

// Cover the complete gated textual SIMT surface.

cuda_tile.module @simt_preview_surface {
  // CHECK: preview$func extern @transform<[M: index, T: !cuda_tile<preview$Datatype>, Tag: !cuda_tile<preview$string>]>
  preview$func extern @transform<[
      M: index,
      T: !cuda_tile.preview$Datatype,
      Tag: !cuda_tile.preview$string]>(
      %value: !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>
      }) -> !cuda_tile.tile<[M]x!cuda_tile.preview$sym_t<T>>

  // CHECK: preview$extern_impl @transform_sm90<[?, ?, ?]>
  // CHECK-SAME: implements @transform<[4, f32, "nvrtc"]>
  // CHECK-SAME: synchronization = #cuda_tile<preview$synchronization<warp>>
  // CHECK-SAME: arch = ["sm90", "sm100"]
  // CHECK-SAME: scratch {smem.size = #cuda_tile<preview$expr_int<128>>, smem.alignment = #cuda_tile<preview$expr_int<16>>}
  // CHECK-SAME: frontendABI = {f32 = "i32"}
  preview$extern_impl @transform_sm90<[?, ?, ?]>
    implements @transform<[4, f32, "nvrtc"]>
    (%value: !cuda_tile.preview$fragment<4xf32, local,
       mem_layout = #cuda_tile.preview$layout<(4):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 32, 4):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
    synchronization = #cuda_tile.preview$synchronization<warp>
    arch = ["sm90", "sm100"]
    scratch {
      smem.size = 128,
      smem.alignment = 16
    }
    {frontendABI = {f32 = "i32"}}

  entry @caller() {
    %value = constant <f32: 1.0> : tile<4xf32>
    %token = make_token : !cuda_tile.token
    // CHECK: %[[RESULT:.+]]:2 = preview$call @transform<[4, f32, "nvrtc"]>
    // CHECK-SAME: (!cuda_tile.token, !cuda_tile.tile<4xf32>)
    // CHECK-SAME: (!cuda_tile.token, !cuda_tile.tile<4xf32>)
    %next, %result = preview$call
      @transform<[4, f32, "nvrtc"]>(%token, %value)
      : (!cuda_tile.token, !cuda_tile.tile<4xf32>)
        -> (!cuda_tile.token, !cuda_tile.tile<4xf32>)
    return
  }
}

// NO-PREVIEW: uses preview feature 'simt', which is not enabled

// -----

//===----------------------------------------------------------------------===//
// preview$func: printing and naming
//===----------------------------------------------------------------------===//

// Parse/print round-trip for the public-preview SIMT operations.

cuda_tile.module @kernels {

    // CHECK: direct_callsite_fn
    preview$func @direct_callsite_fn(%arg0: !cuda_tile.tile<2xi16>) {}

    // CHECK: direct_callsite_fn_multi_operands
    preview$func @direct_callsite_fn_multi_operands(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) {}

    // CHECK: direct_callsite_fn_with_results
    preview$func @direct_callsite_fn_with_results() -> !cuda_tile.tile<2xf32> {
      %0 = constant <f32: 2.0> : !cuda_tile.tile<2xf32>
      return %0 : !cuda_tile.tile<2xf32>
    }

    // CHECK: direct_callsite_fn_multi_operands_and_results
    preview$func @direct_callsite_fn_multi_operands_and_results(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) -> (!cuda_tile.tile<2xf32>, !cuda_tile.tile<2xi16>)  {
      return %arg1, %arg0 : !cuda_tile.tile<2xf32>, !cuda_tile.tile<2xi16>
    }

    // CHECK-LABEL: preview$func extern @extern_callsite_fn(
    // CHECK-SAME: %{{.*}}: tile<2xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>})
    // CHECK-SAME: -> tile<2xf32>
    preview$func extern @extern_callsite_fn(
        %arg0: !cuda_tile.tile<2xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
      -> !cuda_tile.tile<2xf32>

    // Result attributes of preview$func must survive repeated textual parsing.
    // CHECK: preview$func extern @produce
    // CHECK-SAME: -> (tile<32xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>})
    preview$func extern @produce(
        %arg0: tile<32xf32> {
          cuda_tile.preview$access = #cuda_tile.preview$access<in>})
      -> (tile<32xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<out>})

    // CHECK: preview$func
    cuda_tile.preview$func @preview$func(%arg0: !cuda_tile.tile<2x2xf32>) {}

    // CHECK: preview$func @foo() {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    "cuda_tile.preview$func"() ({
    ^bb0:
        return
    }) {const_params = [], function_type = () -> (), sym_name = "foo"} : () -> ()

    // CHECK: preview$func @foo2() {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    preview$func @foo2() {}

    // CHECK: preview$func @foo3(%{{.+}}: tile<2x2xf32>) {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    cuda_tile.preview$func @foo3(%arg0: !cuda_tile.tile<2x2xf32>) {}
    // CHECK-NEXT: preview$func @foo1(
    // CHECK-SAME: %{{.+}}: tile<4x2xf32>, %{{.+}}: tile<4x2xf32>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    cuda_tile.preview$func @foo1(%arg0: !cuda_tile.tile<4x2xf32>, %arg1: !cuda_tile.tile<4x2xf32>) {}


    // CHECK-LABEL: @preview$func_with_result
    // CHECK-SAME: %[[ARG0:.+]]: tile<2x2xf32>) -> tile<2x2xf32>
    preview$func @preview$func_with_result(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xf32> {
        // CHECK: return %[[ARG0]] : tile<2x2xf32>
        return %arg0 : tile<2x2xf32>
    }

    // CHECK: preview$func @preview$func_with_kernel_scope_device
    preview$func @preview$func_with_kernel_scope_device() {}

}

// -----

//===----------------------------------------------------------------------===//
// preview$call: compact constant-argument lists
//===----------------------------------------------------------------------===//

// CHECK: preview$call @f<[(1,), 1, 1+3, 1+3+4]>

cuda_tile.module @m {
  cuda_tile.preview$func extern @f<[
      Singleton: !cuda_tile.preview$shape,
      GroupedVar: index,
      GroupedAdd: index,
      GroupedConst: index]>(
      %x: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>
      }) -> !cuda_tile.tile<f32>

  cuda_tile.entry @k() {
    %x = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    %r = cuda_tile.preview$call @f<[(1,), (1), (1 + 3), (1 + 3 + 4)]>(%x)
        : (!cuda_tile.tile<f32>) -> !cuda_tile.tile<f32>
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// preview$extern_impl: scratch requests
//===----------------------------------------------------------------------===//

cuda_tile.module @extern_impl_scratch {
  cuda_tile.preview$func extern @scratch_decl(
      %arg0: !cuda_tile.tile<128xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> ()

  // CHECK: preview$extern_impl @scratch_impl
  // CHECK-SAME: scratch {smem.size = #cuda_tile<preview$expr_int<1024>>, smem.alignment = #cuda_tile<preview$expr_int<256>>}
  cuda_tile.preview$extern_impl @scratch_impl
    implements @scratch_decl
    (%arg0: !cuda_tile.preview$fragment<128xf32, local,
       mem_layout = #cuda_tile.preview$layout<(128):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, 128):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    scratch {
      smem.size = #cuda_tile.preview$expr_int<1024>,
      smem.alignment = #cuda_tile.preview$expr_int<256>
    }

  cuda_tile.preview$func extern @parametric_scratch_decl<[M: index]>(
      %arg0: !cuda_tile.tile<[M]xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>})
    -> ()

  // CHECK: preview$extern_impl @parametric_scratch_impl
  // CHECK-SAME: scratch {smem.size = #cuda_tile<preview$expr_mul<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_int<4>>>>, smem.alignment = #cuda_tile<preview$expr_int<128>>}
  cuda_tile.preview$extern_impl @parametric_scratch_impl<[M: index]>
    implements @parametric_scratch_decl<[M]>
    (%arg0: !cuda_tile.preview$fragment<[M]xf32, local,
       mem_layout = #cuda_tile.preview$layout<(M):(1)>,
       fragmap = #cuda_tile.preview$layout<(1, 1, M):(0, 0, 1)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
    scratch {
      smem.size = M * 4,
      smem.alignment = 128
    }
}

// -----

//===----------------------------------------------------------------------===//
// preview$extern_impl: fragment ABI
//===----------------------------------------------------------------------===//

// Parse-print roundtrip for fragment-typed extern_impl ops.
// Each extern_impl carries fragment-typed arguments rather than tile-typed
// ones; outputs are expressed as inout/out fragment args, not as results.

// Single in-access local fragment.

cuda_tile.preview$func extern @single_in_decl(
    %data: !cuda_tile.tile<8192xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @single_in
// CHECK-SAME:  implements @single_in_decl
// CHECK-SAME:  preview$fragment<64xf32, local
// CHECK-SAME:  cuda_tile.preview$access = #cuda_tile<preview$access<in>>
cuda_tile.preview$extern_impl @single_in
  implements @single_in_decl
  (%data: !cuda_tile.preview$fragment<64xf32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Single inout local fragment (output expressed as inout, no results).

cuda_tile.preview$func extern @single_inout_decl(
    %data: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
  -> !cuda_tile.tile<64xf32>

// CHECK-LABEL: preview$extern_impl @single_inout
// CHECK-SAME:  implements @single_inout_decl
// CHECK-SAME:  cuda_tile.preview$access = #cuda_tile<preview$access<inout>>
cuda_tile.preview$extern_impl @single_inout
  implements @single_inout_decl
  (%data: !cuda_tile.preview$fragment<64xf32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 32, 64):(0, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()

// -----

// Warp synchronization accepts any positive whole-warp participant count.

cuda_tile.preview$func extern @warp_p64_decl(
    %data: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @warp_p64
// CHECK-SAME: synchronization = #cuda_tile<preview$synchronization<warp>>
cuda_tile.preview$extern_impl @warp_p64
  implements @warp_p64_decl
  (%data: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 64, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  synchronization = #cuda_tile.preview$synchronization<warp>

// -----

// Warpgroup synchronization accepts whole multiples of 128 participants.

cuda_tile.preview$func extern @warpgroup_decl(
    %data: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @warpgroup_p256
// CHECK-SAME: synchronization = #cuda_tile<preview$synchronization<warpgroup>>
cuda_tile.preview$extern_impl @warpgroup_p256
  implements @warpgroup_decl
  (%data: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 256, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  synchronization = #cuda_tile.preview$synchronization<warpgroup>

// -----

// Two args: in then inout (contiguous out/inout run at the end).

cuda_tile.preview$func extern @in_then_inout_decl(
    %src: !cuda_tile.tile<8192xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %dst: !cuda_tile.tile<8192xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
  -> !cuda_tile.tile<8192xf32>

// CHECK-LABEL: preview$extern_impl @in_then_inout
// CHECK-SAME:  implements @in_then_inout_decl
cuda_tile.preview$extern_impl @in_then_inout
  implements @in_then_inout_decl
  (%src: !cuda_tile.preview$fragment<64xf32, local,
      mem_layout = #cuda_tile.preview$layout<(64):(1)>,
      fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %dst: !cuda_tile.preview$fragment<64xf32, local,
      mem_layout = #cuda_tile.preview$layout<(64):(1)>,
      fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
    {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()

// -----

// SMEM participation is measured in physical threads, just like LOCAL
// participation. All 128 threads receive the same shared-buffer pointer.

cuda_tile.preview$func extern @smem_decl(
    %data: !cuda_tile.tile<8x8xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @smem_in
// CHECK-SAME:  preview$fragment<8x8xf32, smem
cuda_tile.preview$extern_impl @smem_in
  implements @smem_decl
  (%data: !cuda_tile.preview$fragment<8x8xf32, smem,
     mem_layout = #cuda_tile.preview$layout<(8, 8):(8, 1)>,
     fragmap = #cuda_tile.preview$layout<(1, 128, (8, 8)):(0, 0, (1, 8))>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// One implementation may combine per-thread LOCAL fragments with an SMEM
// broadcast buffer when their invocation and physical-participant sizes agree.

cuda_tile.preview$func extern @mixed_memory_decl(
    %local: !cuda_tile.tile<32xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
    %shared: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @mixed_memory_impl
// CHECK-SAME: fragment<1xf32, local
// CHECK-SAME: fragment<1xf32, smem
cuda_tile.preview$extern_impl @mixed_memory_impl
  implements @mixed_memory_decl
  (%local: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 1, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
   %shared: !cuda_tile.preview$fragment<1xf32, smem,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// Symbolic fragment: shape and fragmap reference a declared const_param.
// (Only var-presence checks run; concrete fragmap checks are skipped.)

cuda_tile.preview$func extern @symbolic_decl(
    %data: !cuda_tile.tile<64xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @symbolic_fragment
// CHECK-SAME:  [N: index]
// CHECK-SAME:  preview$fragment<[N]xf32, local
cuda_tile.preview$extern_impl @symbolic_fragment<[N: index]>
  implements @symbolic_decl
  (%data: !cuda_tile.preview$fragment<[N]xf32, local,
     mem_layout = #cuda_tile.preview$layout<(N):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, N):(N, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

// A symbolic participant count is checked after parameter concretization.
cuda_tile.preview$func extern @symbolic_participant_decl(
    %data: !cuda_tile.tile<1xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK-LABEL: preview$extern_impl @symbolic_participant
cuda_tile.preview$extern_impl @symbolic_participant<[P: index]>
  implements @symbolic_participant_decl
  (%data: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, P, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
  synchronization = #cuda_tile.preview$synchronization<warp>

// -----

//===----------------------------------------------------------------------===//
// Library linking: unresolved declarations
//===----------------------------------------------------------------------===//

// An implementation-only SIMT catalog may leave its frontend declaration
// unresolved until the catalog is linked into the consumer module.
// CHECK: preview$extern_impl @library_impl
// CHECK-SAME: implements @library_provided_func
cuda_tile.preview$extern_impl @library_impl
  implements @library_provided_func
   (%arg0: !cuda_tile.preview$fragment<[64] x f32, local,
     mem_layout = #cuda_tile.preview$layout<(64):(1)>,
     fragmap = #cuda_tile.preview$layout<(128, 1, 64):(64, 0, 1)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()

// -----

// A standalone SIMT library body may likewise leave a direct callee unresolved
// until linking.
// CHECK: preview$func @library_caller
// CHECK: preview$call @library_provided_func
cuda_tile.preview$func @library_caller(
    %arg0: !cuda_tile.tile<64xf32>) -> !cuda_tile.tile<64xf32> {
  %0 = cuda_tile.preview$call @library_provided_func(%arg0)
    : (!cuda_tile.tile<64xf32>) -> !cuda_tile.tile<64xf32>
  cuda_tile.return %0 : !cuda_tile.tile<64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// preview$call inside reduce / scan regions
//===----------------------------------------------------------------------===//

// Verify the printed output can be parsed.

// A preview$call (SIMT interop) is permitted inside a reduce/scan region.

// CHECK-LABEL: @reduce_with_simt_call
cuda_tile.module @reduce_with_simt_call_mod {
  cuda_tile.preview$func extern @simt_combine(
      %arg0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>
      },
      %arg1: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<f32>

  testing$func @reduce_with_simt_call(%arg0: !cuda_tile.tile<8xf32>) {
    // CHECK: reduce
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+00 : f32]
        : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
        (%lhs: !cuda_tile.tile<f32>, %rhs: !cuda_tile.tile<f32>) {
      // CHECK: preview$call
      %r = cuda_tile.preview$call @simt_combine(%lhs, %rhs)
          : (!cuda_tile.tile<f32>, !cuda_tile.tile<f32>) -> !cuda_tile.tile<f32>
      cuda_tile.yield %r : !cuda_tile.tile<f32>
    }
  }
}

// -----

// CHECK-LABEL: @scan_with_simt_call
cuda_tile.module @scan_with_simt_call_mod {
  cuda_tile.preview$func extern @simt_prefix_sum(
      %arg0: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<inout>
      },
      %arg1: !cuda_tile.tile<f32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>
      }) -> !cuda_tile.tile<f32>

  testing$func @scan_with_simt_call(%arg0: !cuda_tile.tile<8xf32>) {
    // CHECK: scan
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+00 : f32]
        : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
        (%lhs: !cuda_tile.tile<f32>, %rhs: !cuda_tile.tile<f32>) {
      // CHECK: preview$call
      %r = cuda_tile.preview$call @simt_prefix_sum(%lhs, %rhs)
          : (!cuda_tile.tile<f32>, !cuda_tile.tile<f32>) -> !cuda_tile.tile<f32>
      cuda_tile.yield %r : !cuda_tile.tile<f32>
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// PreviewOps.cpp: native parse / print round-trip
//===----------------------------------------------------------------------===//

// PreviewOps.cpp: native fragment signature parsing and printing.
// This case also covers preservation of an arbitrary implementation argument
// attribute alongside the required access mode.
// CHECK-LABEL: cuda_tile.module @extra_attr_module
cuda_tile.module @extra_attr_module {
  cuda_tile.preview$func extern @extra_attr_decl(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>

  // CHECK: preview$extern_impl @extra_attr_impl implements @extra_attr_decl
  // CHECK-SAME: preview$fragment<1xf32, local
  // CHECK-SAME: cuda_tile.my_extra = 7 : i64
  cuda_tile.preview$extern_impl @extra_attr_impl
    implements @extra_attr_decl
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>,
       cuda_tile.my_extra = 7 : i64}) -> ()
}

// -----

// Test: extern function without any constParams or arch —
// exercises setPrivateVisibilityIfAbsent default-private branch and the
// "params == nullptr → skip" print path.
// CHECK-LABEL: cuda_tile.module @bare_decl_module
cuda_tile.module @bare_decl_module {
  // CHECK: preview$func extern @bare_decl
  // CHECK-SAME: (%arg0: tile<256xi32> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>})
  // CHECK-SAME: -> tile<256xi32>
  cuda_tile.preview$func extern @bare_decl(
      %arg0: !cuda_tile.tile<256xi32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<256xi32>
}

// -----

// Test: extern_impl without const_params and without const_args. The native
// fragment signature and required access mode still round-trip when both
// optional constant clauses are absent.
// CHECK-LABEL: cuda_tile.module @impl_no_optional_module
cuda_tile.module @impl_no_optional_module {
  cuda_tile.preview$func extern @decl_for_no_optional(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>

  // CHECK: preview$extern_impl @impl_no_optional implements @decl_for_no_optional
  // CHECK-SAME: preview$fragment<1xf32, local
  // CHECK-SAME: cuda_tile.preview$access = #cuda_tile<preview$access<inout>>
  cuda_tile.preview$extern_impl @impl_no_optional
    implements @decl_for_no_optional
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
}

// -----

// PreviewOps.cpp: FuncOp/ExternImplOp native parse and print roundtrip. These
// cases cover compact constant params/args, native fragment signatures,
// access modes, synchronization, architecture, and default visibility.
// Test: extern function with const_params and full roundtrip
cuda_tile.module @roundtrip_decl {
  cuda_tile.preview$func extern @my_decl<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>},
      %arg1: !cuda_tile.tile<64xf16> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> (!cuda_tile.tile<128xf32>, !cuda_tile.tile<64xf16>)
}

// CHECK-LABEL: cuda_tile.module @roundtrip_decl
// CHECK: preview$func extern @my_decl<[M: index, N: index]>
// CHECK-SAME: (%arg0: tile<128xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>}, %arg1: tile<64xf16> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>})
// CHECK-SAME: -> (tile<128xf32>, tile<64xf16>)

// -----

// Test: extern_impl with const_params, const_args, a native fragment signature,
// synchronization, and architecture roundtrip.
cuda_tile.module @roundtrip_impl {
  cuda_tile.preview$func extern @decl_for_impl<[M: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>

  cuda_tile.preview$extern_impl @my_impl<[M: index]>
    implements @decl_for_impl<[M]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(1, 128, 1):(0, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
    synchronization = #cuda_tile.preview$synchronization<cta> arch = ["sm90"]
}

// CHECK-LABEL: cuda_tile.module @roundtrip_impl
// CHECK: preview$func extern @decl_for_impl<[M: index]>
// CHECK: preview$extern_impl @my_impl<[M: index]> implements @decl_for_impl<[M]>
// CHECK-SAME: (%arg0: preview$fragment<1xf32, local
// CHECK-SAME: cuda_tile.preview$access = #cuda_tile<preview$access<inout>>
// CHECK-SAME: synchronization = #cuda_tile<preview$synchronization<cta>>
// CHECK-SAME: arch = ["sm90"]

// -----

// Test: extern_impl without constant parameters
cuda_tile.module @roundtrip_impl_bare {
  cuda_tile.preview$func extern @bare_decl(
      %arg0: !cuda_tile.tile<64xi32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<64xi32>

  cuda_tile.preview$extern_impl @bare_impl
    implements @bare_decl
    (%arg0: !cuda_tile.preview$fragment<1xi32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(2, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
}

// CHECK-LABEL: cuda_tile.module @roundtrip_impl_bare
// CHECK: preview$extern_impl @bare_impl implements @bare_decl
// CHECK-SAME: (%arg0: preview$fragment<1xi32, local
// CHECK-SAME: cuda_tile.preview$access = #cuda_tile<preview$access<inout>>

// -----

// Test: extern_impl with variable const_args (SymbolicExprVarAttr)
cuda_tile.module @roundtrip_impl_var_args {
  cuda_tile.preview$func extern @decl_var<[M: index, N: index]>(
      %arg0: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>

  cuda_tile.preview$extern_impl @impl_var<[M: index, N: index]>
    implements @decl_var<[#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_var<"N">]>
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
      mem_layout = #cuda_tile.preview$layout<(1):(1)>,
      fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
      {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
}

// CHECK-LABEL: cuda_tile.module @roundtrip_impl_var_args
// CHECK: preview$extern_impl @impl_var<[M: index, N: index]> implements @decl_var<[M, N]>

// const_args must be symbolic expression attributes
// (CudaTile_SymbolicExprAttrConstraint); the string form is rejected.
