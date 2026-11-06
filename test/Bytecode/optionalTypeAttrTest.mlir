// Round-trips an attribute with optional raw Type parameters.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.0 -mlir-print-op-generic %s | FileCheck %s

cuda_tile.module @optional_type_attr {
  // Both optional types present.
  entry @both_present() {
    testing$bytecode_test_new_attribute opt_type = #cuda_tile.bytecode_test_optional_type<5 elem = f16 int_type = i32>
    // CHECK: opt_type_attr = #cuda_tile.bytecode_test_optional_type<5 elem = f16 int_type = i32>
    return
  }

  // Both optional types absent (bit-clear path; nothing written/read).
  entry @both_absent() {
    testing$bytecode_test_new_attribute opt_type = #cuda_tile.bytecode_test_optional_type<5>
    // CHECK: opt_type_attr = #cuda_tile.bytecode_test_optional_type<5>
    return
  }

  // Only the bare Type present (mixed -> exercises flag bit indexing).
  entry @elem_only() {
    testing$bytecode_test_new_attribute opt_type = #cuda_tile.bytecode_test_optional_type<5 elem = f32>
    // CHECK: opt_type_attr = #cuda_tile.bytecode_test_optional_type<5 elem = f32>
    return
  }

  // Only the concrete IntegerType present (exercises the dyn_cast read path).
  entry @int_type_only() {
    testing$bytecode_test_new_attribute opt_type = #cuda_tile.bytecode_test_optional_type<5 int_type = i8>
    // CHECK: opt_type_attr = #cuda_tile.bytecode_test_optional_type<5 int_type = i8>
    return
  }
}
