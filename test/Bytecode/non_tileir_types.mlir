// RUN: not cuda-tile-translate -mlir-to-cudatilebc %s -no-implicit-module -bytecode-version=250.0 2>&1 | FileCheck %s

// CHECK: unsupported type in bytecode writer
cuda_tile.module @kernels {
  // Verify that the bytecode writer gracefully rejects a non-TileIR type.
  cuda_tile.testing$func @nonTileIRTypeArg(%arg0 : tensor<2xi16>) {
    cuda_tile.return
  }
}
