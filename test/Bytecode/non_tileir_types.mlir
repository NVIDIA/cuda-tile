// RUN: not cuda-tile-translate -mlir-to-cudatilebc %s -no-implicit-module 2>&1 | FileCheck %s

// CHECK: unsupported type in bytecode writer
cuda_tile.module @kernels {
  // Verify that we accept a non-tileir type in an entry arg, but the bytecode fails gracefully.
  cuda_tile.entry @nonTileIRTypeArg(%arg0 : tensor<2xi16>) {
    cuda_tile.return
  }
}
