// Coverage for CommandLineOptions.cpp: valid version with explicit tag.
// Tests the path where consume_front(".") succeeds and consumeInteger parses a valid tag.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=13.1.0 %s -o %t && cuda-tile-translate -no-implicit-module -cudatilebc-to-mlir %t | FileCheck %s
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=13.4.0 %s -o %t && cuda-tile-translate -no-implicit-module -cudatilebc-to-mlir %t | FileCheck %s

// CHECK: cuda_tile.module @kernels
cuda_tile.module @test {
  cuda_tile.entry @dummy() {
    cuda_tile.return
  }
}
