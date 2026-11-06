// RUN: not cuda-tile-opt %s --preview=simt -split-input-file 2>&1 | FileCheck %s

// Diagnostics that are not attached to an operation location, so they cannot be
// matched with -verify-diagnostics. Op-attached verifier errors belong in
// preview_invalid.mlir.

//===----------------------------------------------------------------------===//
// preview$extern_impl: constant-argument diagnostics
//===----------------------------------------------------------------------===//

cuda_tile.preview$func extern @const_arg_datatype_diag_decl<[
    T: !cuda_tile.preview$Datatype]>(
    %arg0: !cuda_tile.tile<1xf32> {
      cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  -> ()

// CHECK: constant argument #0 for constant parameter 'T' must be a valid datatype expression: expected datatype expression, got #cuda_tile<preview$expr_int<128>>
cuda_tile.preview$extern_impl @const_arg_datatype_diag<[
    T: !cuda_tile.preview$Datatype]>
  implements @const_arg_datatype_diag_decl<[128]>
  (%arg0: !cuda_tile.preview$fragment<1xf32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

// -----

//===----------------------------------------------------------------------===//
// preview$extern_impl: string constant-argument diagnostics
//===----------------------------------------------------------------------===//

cuda_tile.preview$func extern @const_arg_escaped_diag_decl<[
    T: !cuda_tile.preview$Datatype]>(
        !cuda_tile.tile<!cuda_tile.preview$sym_t<T>> {
          cuda_tile.preview$access = #cuda_tile.preview$access<in>})

// CHECK: constant argument #0 for constant parameter 'T' must be a valid datatype expression: expected datatype expression, got "fo\22o"
cuda_tile.preview$extern_impl @const_arg_escaped_diag<[
    T: !cuda_tile.preview$Datatype]>
  implements @const_arg_escaped_diag_decl<["fo\"o"]>
  (!cuda_tile.preview$fragment<1xi32, local,
     mem_layout = #cuda_tile.preview$layout<(1):(1)>,
     fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>>
   {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
