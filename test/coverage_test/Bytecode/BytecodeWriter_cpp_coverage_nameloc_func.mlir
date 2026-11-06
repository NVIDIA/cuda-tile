// Test that NameLoc on a function location triggers registerDebugInfo
// NameLoc path through buildFunctionMap.
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s 2>&1 | FileCheck %s

// CHECK: failed to serialize debug info attribute
cuda_tile.module @kernels {
  entry @nameloc_func() {
    return
  } loc("my_named_function")
} loc(#loc)

#loc = loc(unknown)
