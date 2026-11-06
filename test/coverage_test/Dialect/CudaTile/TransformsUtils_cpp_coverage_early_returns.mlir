// RUN: cuda-tile-opt --test-debuginfo-update-symbol-name="symbol-to-rename=kernels::no_di_func new-name=_renamed_no_di update-uses=false" --mlir-print-debuginfo %s | FileCheck %s

// Test: entry function with loc(unknown) -- no DILocAttr on the symbol.
// (!diLoc is true -> early return) and.
// Also (symtab is null because update-uses=false).

cuda_tile.module @kernels {
  // CHECK: @_renamed_no_di()
  entry @no_di_func() {
    return loc(unknown)
  } loc(unknown)
} loc(unknown)
