// Round-trips an attribute that carries a self type.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.0 -mlir-print-op-generic %s | FileCheck %s

cuda_tile.module @self_type_attr {
  // Self type is a builtin type.
  entry @builtin_self_type() {
    testing$bytecode_test_new_attribute self_type = #cuda_tile.bytecode_test_self_type<5> : i32
    // CHECK: self_type_attr = #cuda_tile.bytecode_test_self_type<5> : i32
    return
  }

  // Self type is a CudaTile dialect type. Exercises the type table and
  // dependent-type registration for a raw Type attribute parameter.
  entry @dialect_self_type() {
    testing$bytecode_test_new_attribute self_type = #cuda_tile.bytecode_test_self_type<7> : !cuda_tile.tile<4xf32>
    // CHECK: self_type_attr = #cuda_tile.bytecode_test_self_type<7> : !cuda_tile.tile<4xf32>
    return
  }
}
