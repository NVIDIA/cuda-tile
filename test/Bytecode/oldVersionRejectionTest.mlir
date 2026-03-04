// Test for version rejection when targeting older bytecode versions with new features.
// This tests that when targeting 13.1 bytecode but using 13.2 features,
// appropriate errors are generated.

// RUN: not cuda-tile-translate -mlir-to-cudatilebc -bytecode-version=13.1 %s 2>&1 | FileCheck %s
// CHECK: attribute 'overflow' requires bytecode version 13.2+

cuda_tile.module @test_future_version_rejection {
  entry @test_13_2_feature_in_13_1() {
    %input = cuda_tile.constant <i32: [1, -2]> : !cuda_tile.tile<2xi32>
    %result = cuda_tile.negi %input overflow<no_signed_wrap> : !cuda_tile.tile<2xi32>
  }
}

