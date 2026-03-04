// Test forward compatibility: operations using base features work across bytecode versions.
// This validates that operations remain compatible when new features aren't used.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.0 %s | FileCheck %s --check-prefix=CHECK-250-0
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.1 %s | FileCheck %s --check-prefix=CHECK-250-1

cuda_tile.module @forward_compatibility_tests {
  // Test case 1: Base operands and results.
  entry @test_base_operation() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_out = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    // CHECK-250-0: %{{.*}} = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) -> token
    // CHECK-250-1: %{{.*}} = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) -> token
    cuda_tile.return
  }

  // Test case 2: Base attributes only.
  entry @test_base_attributes() {
    testing$bytecode_test_new_attribute
    // CHECK-250-0: bytecode_test_new_attribute{{$}}
    // CHECK-250-1: bytecode_test_new_attribute{{$}}
    return
  }

  // Test case 3: New attributes with default value.
  entry @test_new_attributes() {
    testing$bytecode_test_new_attribute new_param = 42
    // CHECK-250-0: bytecode_test_new_attribute{{$}}
    // CHECK-250-1: bytecode_test_new_attribute{{$}}
    return
  }
}
