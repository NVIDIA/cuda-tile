// RUN: cuda-tile-opt --test-debuginfo-update-symbol-name="symbol-to-rename=kernels::my_global new-name=renamed_global update-uses=false" --mlir-print-debuginfo %s | FileCheck %s

// Test: Rename a cuda_tile.global with loc(unknown), so it has no DILocAttr.
// (!diLoc is true -> early return).
// Also (symtab is null because update-uses=false).
//
// We use a global op because entry ops were already covered in other tests.
// This provides an additional code path exercising the no-DILocAttr early return.

cuda_tile.module @kernels {
  // CHECK: global @renamed_global
  global @my_global <i32: [42]> : !cuda_tile.tile<1xi32> loc(unknown)

  entry @dummy() {
    return loc(unknown)
  } loc(unknown)
} loc(unknown)
