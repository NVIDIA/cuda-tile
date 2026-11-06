// This validates that proper errors are generated when version requirements aren't met.

// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=250.0 %s -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-250-0
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-13-1


// Test case 1: Attribute version error
cuda_tile.module @attribute_version_error_test {
  entry @test_attribute_error() {
    testing$bytecode_test_new_attribute new_param = 123
    return
  }
}

// CHECK-250-0: attribute 'new_param' requires bytecode version 250.1+, but targeting 250.0

// -----

// Test case 2: Optional attribute version error
cuda_tile.module @optional_attribute_version_error_test {
  entry @test_optional_attr_error() {
    testing$bytecode_test_new_attribute new_flag
    return
  }
}

// CHECK-250-0: optional attribute 'new_flag' is provided but requires bytecode version 250.1, targeting 250.0

// -----

// Test case 3: Operand version error
cuda_tile.module @operand_version_error_test {
  entry @test_operand_error() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_in = make_token : !cuda_tile.token
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) token = %token_in : !cuda_tile.token -> !cuda_tile.token
    return
  }
}

// CHECK-250-0: optional operand 'optional_token' is provided but requires bytecode version 250.1, targeting 250.0

// -----

// Test case 4: Result version error
cuda_tile.module @result_version_error_test {
  entry @test_result_error() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    %joined = join_tokens %token, %token : !cuda_tile.token
    return
  }
}

// CHECK-250-0: result 'result_token' requires bytecode version 250.1 but is being used and targeting 250.0

// -----

// Test case 5: Op version error
cuda_tile.module @op_version_error_test {
  entry @test_op_error() {
    testing$bytecode_test_new_attribute new_param = 123
    return
  }
}

// CHECK-13-1: operation 'cuda_tile.testing$bytecode_test_new_attribute' is not available in bytecode version 13.1

// -----

// Test case 6: Type parameter version error - DefaultValuedParameter.
cuda_tile.module @type_param_explicit_default_error {
  testing$func @test_explicit_default_error(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, f16>) {
    return
  }
}

// CHECK-250-0: type parameter 'newType' requires bytecode version 250.1+
// CHECK-250-0-SAME: but targeting 250.0

// -----

// Test case 7: Optional parameter version error.
cuda_tile.module @optional_attr_error {
  testing$func @test_optional_attr_error(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32, zero>) {
    return
  }
}

// CHECK-250-0: type parameter 'optionalAttr' requires bytecode version 250.1+
// CHECK-250-0-SAME: but targeting 250.0

// -----

// Test case 8: Enum value version error - 'high' case requires 250.1.
// The enum type is available in 250.0, but the 'high' value was added in 250.1.
cuda_tile.module @enum_value_version_error {
  entry @test_enum_value_version_error() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<high> -> !cuda_tile.token
    return
  }
}

// CHECK-250-0: enum value for attribute 'priority' is not available in bytecode version 250.0

// -----

// Test case 9: Evolved attr parameter version error.
// BytecodeTestEvolvedAttr is available at 250.0, but its optionalValue
// parameter requires 250.1. Providing a value at 250.0 must be an error.
cuda_tile.module @evolved_attr_param_version_error {
  entry @test_evolved_attr_param_error() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<42, 42, 100> -> !cuda_tile.token
    return
  }
}

// CHECK-250-0: attribute parameter 'optionalValue' requires bytecode version 250.1+
// CHECK-250-0-SAME: but targeting 250.0

// -----

// Test case 10: Dependent type version error (bytecodeFallbackValue).
// dependentType (added in 250.1) is f16, which != baseType (f32).
// Writing to 250.0 must fail because the fallback value would be f32.
cuda_tile.module @dependent_type_version_error {
  testing$func @test_dependent_type_error(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f16>) {
    return
  }
}

// CHECK-250-0: type parameter 'dependentType' requires bytecode version 250.1+
// CHECK-250-0-SAME: but targeting 250.0

// -----

// Test case 11: Dependent attr parameter version error (bytecodeFallbackValue).
// dependentValue (added in 250.1) is 999, which != baseValue (42).
// Writing to 250.0 must fail because the fallback value would be 42.
cuda_tile.module @dependent_attr_param_version_error {
  entry @test_dependent_attr_param_error() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<42, 999> -> !cuda_tile.token
    return
  }
}

// CHECK-250-0: attribute parameter 'dependentValue' requires bytecode version 250.1+
// CHECK-250-0-SAME: but targeting 250.0
