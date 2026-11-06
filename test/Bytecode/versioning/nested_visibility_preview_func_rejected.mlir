// RUN: not cuda-tile-translate --preview=simt -mlir-to-cudatilebc -no-implicit-module %s 2>&1 | FileCheck %s

cuda_tile.module @kernels {
  "cuda_tile.preview$func"() <{
    arg_attrs = [{cuda_tile.preview$access = #cuda_tile.preview$access<in>}],
    const_params = [],
    function_type = (!cuda_tile.tile<128xf32>) -> (),
    is_extern,
    sym_name = "callee"
  }> ({}) {sym_visibility = "nested"} : () -> ()
}

// CHECK: error: symbol `callee` uses nested symbol visibility, which cannot be encoded in bytecode

