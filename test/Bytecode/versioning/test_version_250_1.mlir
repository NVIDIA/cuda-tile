// Test 250.1 features: operands, results, and attributes.

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
}
