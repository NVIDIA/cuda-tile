// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -split-input-file -verify-diagnostics -bytecode-version=250.0 %s | FileCheck %s --check-prefix=CHECK-250-0
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -split-input-file -bytecode-version=250.1 %s | FileCheck %s --check-prefix=CHECK-250-1

// Tests round-trip of an attribute with an evolved OptionalParameter
// (`middle`, added in 250.1) sandwiched between two non-evolved
// OptionalParameters (`before` and `after`, both 250.0). Validates that the
// bitfield encoding for OptionalParameter stays consistent when interleaving
// evolved and non-evolved optionals across bytecode versions.
//
// At 250.0, modules where `middle` is set must produce a writer error
// (the parameter requires 250.1 to differ from its null fallback). The 250.0
// RUN uses `-verify-diagnostics` so those errors are matched via
// `// expected-error@unknown` annotations rather than stderr scraping. The
// 250.1 RUN does not pass `-verify-diagnostics`, so the annotations are inert
// for it; every case there must round-trip cleanly.

// Case 1: before + after present, middle absent — works at both versions.
// CHECK-250-0-LABEL: @k_before_after
// CHECK-250-0: sandwich = <before = 1, after = 3>
// CHECK-250-1-LABEL: @k_before_after
// CHECK-250-1: sandwich = <before = 1, after = 3>
cuda_tile.module @case_before_after {
  entry @k_before_after() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<before = 1, after = 3> -> !cuda_tile.token
    return
  }
}

// -----

// Case 2: only the leading non-evolved optional present.
// CHECK-250-0-LABEL: @k_only_before
// CHECK-250-0: sandwich = <before = 7>
// CHECK-250-1-LABEL: @k_only_before
// CHECK-250-1: sandwich = <before = 7>
cuda_tile.module @case_only_before {
  entry @k_only_before() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<before = 7> -> !cuda_tile.token
    return
  }
}

// -----

// Case 3: only the trailing non-evolved optional present — exercises the bit
// position that comes after the evolved slot.
// CHECK-250-0-LABEL: @k_only_after
// CHECK-250-0: sandwich = <after = 9>
// CHECK-250-1-LABEL: @k_only_after
// CHECK-250-1: sandwich = <after = 9>
cuda_tile.module @case_only_after {
  entry @k_only_after() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<after = 9> -> !cuda_tile.token
    return
  }
}

// -----

// Case 4: all three optionals absent.
// CHECK-250-0-LABEL: @k_all_absent
// CHECK-250-0: sandwich = <>
// CHECK-250-1-LABEL: @k_all_absent
// CHECK-250-1: sandwich = <>
cuda_tile.module @case_all_absent {
  entry @k_all_absent() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<> -> !cuda_tile.token
    return
  }
}

// -----

// Case 5: only the evolved middle present — 250.1-only.
// CHECK-250-1-LABEL: @k_only_middle
// CHECK-250-1: sandwich = <middle = 5>
// expected-error@unknown {{attribute parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_only_middle {
  entry @k_only_middle() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    // expected-error@+1 {{unsupported attribute type encountered during serialization}}
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<middle = 5> -> !cuda_tile.token
    return
  }
}

// -----

// Case 6: all three present — 250.1-only.
// CHECK-250-1-LABEL: @k_all_present
// CHECK-250-1: sandwich = <before = 10, middle = 20, after = 30>
// expected-error@unknown {{attribute parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_all_present {
  entry @k_all_present() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    // expected-error@+1 {{unsupported attribute type encountered during serialization}}
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<before = 10, middle = 20, after = 30> -> !cuda_tile.token
    return
  }
}

// -----

// Case 7: leading non-evolved + evolved middle — 250.1-only.
// CHECK-250-1-LABEL: @k_before_middle
// CHECK-250-1: sandwich = <before = 11, middle = 22>
// expected-error@unknown {{attribute parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_before_middle {
  entry @k_before_middle() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    // expected-error@+1 {{unsupported attribute type encountered during serialization}}
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<before = 11, middle = 22> -> !cuda_tile.token
    return
  }
}

// -----

// Case 8: evolved middle + trailing non-evolved — 250.1-only.
// CHECK-250-1-LABEL: @k_middle_after
// CHECK-250-1: sandwich = <middle = 33, after = 44>
// expected-error@unknown {{attribute parameter 'middle' requires bytecode version 250.1+}}
cuda_tile.module @case_middle_after {
  entry @k_middle_after() {
    %i = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    // expected-error@+1 {{unsupported attribute type encountered during serialization}}
    %t = testing$bytecode_test_evolution (%i : !cuda_tile.tile<1xf32>) sandwich = #cuda_tile.bytecode_test_sandwiched_optional_attr<middle = 33, after = 44> -> !cuda_tile.token
    return
  }
}
