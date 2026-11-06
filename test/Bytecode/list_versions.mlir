// Test that --list-versions outputs the supported bytecode versions.
// RUN: cuda-tile-translate --list-versions | FileCheck %s

// Production versions.
// CHECK:      13.1
// CHECK-NEXT: 13.2
// CHECK-NEXT: 13.3
// CHECK-NEXT: 13.4
// CHECK-NEXT: 13.5
// Testing versions (only present when TILE_IR_INCLUDE_TESTS is enabled).
// CHECK-NEXT: 250.0
// CHECK-NEXT: 250.1
