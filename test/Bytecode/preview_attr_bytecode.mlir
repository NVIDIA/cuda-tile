// Tests public-preview attribute bytecode serialization and gating.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt,test_only %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module --preview=simt --preview=test_only %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s 2>&1 | FileCheck %s --check-prefix=WRITE-NOFLAG
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt %s 2>&1 | FileCheck %s --check-prefix=WRITE-PARTIAL
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 --preview=simt,test_only %s 2>&1 | FileCheck %s --check-prefix=WRITE-VERSION-CONFLICT
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --preview=simt,test_only %s -o %t.cutile
// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.cutile 2>&1 | FileCheck %s --check-prefix=READ-NOFLAG
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module --preview=simt,test_only %t.cutile | FileCheck %s --check-prefix=ROUNDTRIP

cuda_tile.module @preview_attribute_test {
  entry @f() {
    preview$bytecode_test_preview_attribute
      preview_attr = #cuda_tile.preview$bytecode_test_attr<7>
      nested_attrs = [
        #cuda_tile.preview$bytecode_test_attr<9>,
        #cuda_tile.preview$expr_int<42>,
        #cuda_tile.preview$expr_var<"M">,
        #cuda_tile.preview$expr_builtin<"context.num_threads", index>,
        #cuda_tile.preview$expr_const_str<"forward">,
        #cuda_tile.preview$concrete_type<f32>,
        #cuda_tile.preview$expr_mul<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>>,
        #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>>,
        #cuda_tile.preview$expr_add<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>>,
        #cuda_tile.preview$expr_neg<#cuda_tile.preview$expr_var<"M">>,
        #cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>>,
        #cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>>,
        #cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>>,
        #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<2>]>,
        #cuda_tile.preview$layout<(M, 2):(2, 1)>
      ]
    return
  }
}

// ROUNDTRIP: preview$bytecode_test_preview_attribute
// ROUNDTRIP-SAME: preview_attr = <7>
// ROUNDTRIP-SAME: nested_attrs = [#cuda_tile<preview$bytecode_test_attr<9>>,
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_int<42>>,
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_var<"M">>,
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_builtin<"context.num_threads", index>>,
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_const_str<"forward">>,
// ROUNDTRIP-SAME: #cuda_tile<preview$concrete_type<f32>>,
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_mul<
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_div<
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_add<
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_neg<
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_sub<
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_floor_div<
// ROUNDTRIP-SAME: #cuda_tile<preview$expr_ceil_div<
// ROUNDTRIP-SAME: #cuda_tile<preview$symbolic_tuple<
// ROUNDTRIP-SAME: #cuda_tile<preview$layout<(M, 2):(2, 1)>>]

// WRITE-NOFLAG: uses preview feature
// WRITE-PARTIAL: uses preview feature
// WRITE-VERSION-CONFLICT: only valid at the current bytecode version 13.5, but version 13.4 was explicitly requested
// READ-NOFLAG: public-preview bytecode requires at least one enabled preview feature
