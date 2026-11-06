// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -split-input-file -verify-diagnostics -bytecode-version=250.0 %s | FileCheck %s --check-prefix=CHECK-250-0
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -split-input-file -bytecode-version=250.1 %s | FileCheck %s --check-prefix=CHECK-250-1

// Tests round-trip of a type with an evolved OptionalParameter (`middle`, added
// in 250.1) sandwiched between two non-evolved OptionalParameters (`before` and
// `after`, both 250.0). This mirrors test_sandwiched_optional_attr.mlir for
// type parameter encoding.

// Case 1: before + after present, middle absent — works at both versions.
// CHECK-250-0-LABEL: testing$func @k_before_after
// CHECK-250-0: testing$bytecode_test_sandwiched_optional<before = f32, after = f64>
// CHECK-250-1-LABEL: testing$func @k_before_after
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<before = f32, after = f64>
cuda_tile.module @case_before_after {
  testing$func @k_before_after(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<before = f32, after = f64>) {
    return
  }
}

// -----

// Case 2: only the leading non-evolved optional present.
// CHECK-250-0-LABEL: testing$func @k_only_before
// CHECK-250-0: testing$bytecode_test_sandwiched_optional<before = f16>
// CHECK-250-1-LABEL: testing$func @k_only_before
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<before = f16>
cuda_tile.module @case_only_before {
  testing$func @k_only_before(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<before = f16>) {
    return
  }
}

// -----

// Case 3: only the trailing non-evolved optional present — exercises the bit
// position that comes after the evolved slot.
// CHECK-250-0-LABEL: testing$func @k_only_after
// CHECK-250-0: testing$bytecode_test_sandwiched_optional<after = i32>
// CHECK-250-1-LABEL: testing$func @k_only_after
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<after = i32>
cuda_tile.module @case_only_after {
  testing$func @k_only_after(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<after = i32>) {
    return
  }
}

// -----

// Case 4: all three optionals absent.
// CHECK-250-0-LABEL: testing$func @k_all_absent
// CHECK-250-0: testing$bytecode_test_sandwiched_optional<>
// CHECK-250-1-LABEL: testing$func @k_all_absent
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<>
cuda_tile.module @case_all_absent {
  testing$func @k_all_absent(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<>) {
    return
  }
}

// -----

// Case 5: only the evolved middle present — 250.1-only.
// CHECK-250-1-LABEL: testing$func @k_only_middle
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<middle = bf16>
// expected-error@unknown {{type parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_only_middle {
  testing$func @k_only_middle(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<middle = bf16>) {
    return
  }
}

// -----

// Case 6: all three present — 250.1-only.
// CHECK-250-1-LABEL: testing$func @k_all_present
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<before = f32, middle = bf16, after = i64>
// expected-error@unknown {{type parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_all_present {
  testing$func @k_all_present(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<before = f32, middle = bf16, after = i64>) {
    return
  }
}

// -----

// Case 7: leading non-evolved + evolved middle — 250.1-only.
// CHECK-250-1-LABEL: testing$func @k_before_middle
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<before = f16, middle = f32>
// expected-error@unknown {{type parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_before_middle {
  testing$func @k_before_middle(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<before = f16, middle = f32>) {
    return
  }
}

// -----

// Case 8: evolved middle + trailing non-evolved — 250.1-only.
// CHECK-250-1-LABEL: testing$func @k_middle_after
// CHECK-250-1: testing$bytecode_test_sandwiched_optional<middle = f64, after = i32>
// expected-error@unknown {{type parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_middle_after {
  testing$func @k_middle_after(%arg0: !cuda_tile.testing$bytecode_test_sandwiched_optional<middle = f64, after = i32>) {
    return
  }
}
