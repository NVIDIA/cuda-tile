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

  // Test case 4: Backward compatibility for DefaultValuedParameter.
  // The newType parameter (added in 250.1) equals its default (i32).
  // This allows serialization to 250.0 which doesn't have these parameters.
  testing$func @test_explicit_default_equals(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32>) {
    // CHECK-250-0: @test_explicit_default_equals(%{{.*}}: testing$bytecode_test_evolved<f32, f32>)
    // CHECK-250-1: @test_explicit_default_equals(%{{.*}}: testing$bytecode_test_evolved<f32, f32>)
    return
  }

  // Test case 5: Type parameter without specifying optional params (all implicit defaults).
  testing$func @test_all_implicit_defaults(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32>) {
    // CHECK-250-0: @test_all_implicit_defaults(%{{.*}}: testing$bytecode_test_evolved<f32, f32>)
    // CHECK-250-1: @test_all_implicit_defaults(%{{.*}}: testing$bytecode_test_evolved<f32, f32>)
    return
  }

  // Test case 6: Enum attr with 250.0 value works when targeting 250.1.
  entry @test_enum_attr_250_0_value() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<low> -> !cuda_tile.token
    // CHECK-250-0: bytecode_test_evolution{{.*}}priority = <low>
    // CHECK-250-1: bytecode_test_evolution{{.*}}priority = <low>
    return
  }

  // Test case 7: Evolved attr with base value only works at both versions.
  entry @test_evolved_attr_base_only() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<42, 42> -> !cuda_tile.token
    // CHECK-250-0: bytecode_test_evolution{{.*}}test_attr = <42, 42>
    // CHECK-250-1: bytecode_test_evolution{{.*}}test_attr = <42, 42>
    return
  }

}
