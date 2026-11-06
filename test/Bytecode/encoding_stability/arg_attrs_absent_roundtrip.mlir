// Control for the `arg_attrs` version gate (kMinArgAttrsVersion, 13.5): a
// function with no argument attributes must stay writable below the gate, since
// the parser normalizes the all-empty attribute away and nothing is lost.
// Counterpart to arg_attrs_roundtrip.mlir, which pins the refusal.

// RUN: %round_trip_test %s %t

// Below the gate the module still serializes, and reads back unchanged.
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=13.4 %s | FileCheck %s

// Nothing here forces 13.5.
// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module %s | FileCheck %s --check-prefix=MINVER
// MINVER: 13.1

cuda_tile.module @kernels {
  cuda_tile.entry @entry_no_arg_attrs(
      %a: !cuda_tile.tile<f32>,
      %b: !cuda_tile.tile<i32>) {
    cuda_tile.return
  }
}

// CHECK: entry @entry_no_arg_attrs(%arg0: tile<f32>, %arg1: tile<i32>)
