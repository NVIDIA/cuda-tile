// This validates that proper errors are generated when version requirements aren't met.

// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=250.0 %s -split-input-file 2>&1 | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-OPTIONAL-ATTR,CHECK-OPERAND,CHECK-RESULT
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-OP-NOT-AVAILABLE


// Test case 1: Attribute version error
cuda_tile.module @attribute_version_error_test {
  entry @test_attribute_error() {
    testing$bytecode_test_new_attribute new_param = 123
    return
  }
}

// CHECK-ATTR: attribute 'new_param' requires bytecode version 250.1+, but targeting 250.0

// -----

// Test case 2: Optional attribute version error
cuda_tile.module @optional_attribute_version_error_test {
  entry @test_optional_attr_error() {
    testing$bytecode_test_new_attribute new_flag
    return
  }
}

// CHECK-OPTIONAL-ATTR: optional attribute 'new_flag' is provided but requires bytecode version 250.1, targeting 250.0

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

// CHECK-OPERAND: optional operand 'optional_token' is provided but requires bytecode version 250.1, targeting 250.0

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

// CHECK-RESULT: result 'result_token' requires bytecode version 250.1 but is being used and targeting 250.0

// -----

// Test case 5: Op version error
cuda_tile.module @op_version_error_test {
  entry @test_op_error() {
    testing$bytecode_test_new_attribute new_param = 123
    return
  }
}

// CHECK-OP-NOT-AVAILABLE: operation 'cuda_tile.testing$bytecode_test_new_attribute' is not available in bytecode version 13.1
