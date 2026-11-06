// Test that FusedLoc on a function location triggers the FusedLoc case in
// registerDebugInfo during debug info registration.
// The FusedLoc registration succeeds but serialization fails, exercising
// the serializeDebugInfo error path.
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s 2>&1 | FileCheck %s

// CHECK: failed to serialize debug info attribute
cuda_tile.module @kernels {
  entry @fused_func() {
    return
  } loc(fused["file.cu":10:5, "file.cu":20:3])
} loc(#loc)

#loc = loc(unknown)
