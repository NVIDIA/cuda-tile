// Test that globals with non-public visibility or constant flag are rejected
// when targeting bytecode versions earlier than 13.3.

// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 -split-input-file %s 2>&1 | FileCheck %s

// Test: private global rejected in version 13.1
// CHECK: global `g_private` uses non-public symbol visibility
cuda_tile.module @test_private_global_old_version {
  cuda_tile.global private @g_private <i32: [10]> : !cuda_tile.tile<1xi32>
  cuda_tile.entry @dummy() {
    cuda_tile.return
  }
}

// -----

// Test: constant global rejected in version 13.1
// CHECK: constant global `g_const` cannot be encoded in bytecode version
cuda_tile.module @test_constant_global_old_version {
  cuda_tile.global constant @g_const <f32: [3.14]> : !cuda_tile.tile<1xf32>
  cuda_tile.entry @dummy() {
    cuda_tile.return
  }
}
