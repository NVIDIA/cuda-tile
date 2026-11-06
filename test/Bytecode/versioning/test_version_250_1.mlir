// Test 250.1 features: operands, results, attributes, and dependent types.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.1 %s | FileCheck %s

cuda_tile.module @version_250_1_features {
  // Test case 1: Operand parsing - validates 250.1 optional operand are correctly parsed.
  entry @test_operand_parsing() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_in = make_token : !cuda_tile.token
    %token_out = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) 
      token = %token_in : !cuda_tile.token -> !cuda_tile.token
    // CHECK: %{{.*}} = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) token = %{{.*}} : token -> token
    return
  }

  // Test case 2: Result parsing - validates 250.1 results are correctly parsed and usable.
  entry @test_result_parsing() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token1 = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    // CHECK: %[[TOKEN1:.*]] = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) -> token
    %token2 = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    // CHECK: %[[TOKEN2:.*]] = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) -> token
    // Use parsed results to validate correct type preservation during deserialization
    %joined_tokens = join_tokens %token1, %token2 : !cuda_tile.token
    // CHECK: %{{.*}} = join_tokens %[[TOKEN1]], %[[TOKEN2]] : token
    return
  }

  // Test case 3: Attribute parsing - validates 250.1 non-default attributes are correctly parsed.
  entry @test_attribute_parsing() {
    testing$bytecode_test_new_attribute new_flag new_param = 123
    // CHECK: bytecode_test_new_attribute new_flag new_param = 123
    return
  }

  // Test case 4: Type parameter with DefaultValuedParameter from 250.1.
  testing$func @test_type_explicit_default(
      %arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, f16>) {
    // CHECK: testing$bytecode_test_evolved<f32, f32, f16>
    return
  }

  // Test case 5: OptionalParameter<Attr> with value.
  testing$func @test_optional_attr(%arg0: !cuda_tile.testing$bytecode_test_evolved<f64, f64, i32, zero>) {
    // CHECK: testing$bytecode_test_evolved<f64, f64, zero>
    return
  }

  // Test case 6: OptionalParameter<Type> with value.
  testing$func @test_optional_type_present(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32, zero, f16>) {
    // CHECK: testing$bytecode_test_evolved<f32, f32, zero, f16>
    return
  }

  // Test case 7: OptionalParameter<Type> NULL.
  testing$func @test_optional_type_null(%arg0: !cuda_tile.testing$bytecode_test_evolved<f64, f64, i32>) {
    // CHECK: testing$bytecode_test_evolved<f64, f64>
    return
  }

  // Test case 8: All defaults/nulls.
  testing$func @test_all_defaults(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32>) {
    // CHECK: testing$bytecode_test_evolved<f32, f32>
    return
  }

  // Test case 9: Evolution op with enum attr (priority).
  entry @test_evolution_with_priority() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<high> -> !cuda_tile.token
    // CHECK: bytecode_test_evolution{{.*}}priority = <high>
    return
  }

  // Test case 10: Evolved attr with base value only (no optional).
  entry @test_evolved_attr_base_only() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<123, 123> -> !cuda_tile.token
    // CHECK: bytecode_test_evolution{{.*}}test_attr = <123, 123>
    return
  }

  // Test case 11: Evolved attr with optional value (250.1 feature).
  entry @test_evolved_attr_with_optional() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<123, 123, 456> -> !cuda_tile.token
    // CHECK: bytecode_test_evolution{{.*}}test_attr = <123, 123, 456>
    return
  }

  // Test case 12: Evolved attr with enum and optional value.
  entry @test_evolved_attr_with_enum_and_optional() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<low> test_attr = #cuda_tile.bytecode_test_evolved_attr<789, 789, 1000> -> !cuda_tile.token
    // CHECK: bytecode_test_evolution{{.*}}priority = <low> test_attr = <789, 789, 1000>
    return
  }

  // Test case 13: Dependent type differs from fallback baseType.
  testing$func @test_dependent_type_diff(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, i64>) {
    // CHECK: testing$bytecode_test_evolved<f32, i64>
    return
  }
}
