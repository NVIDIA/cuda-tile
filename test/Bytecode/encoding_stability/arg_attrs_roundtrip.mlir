// Encoding-stability / version-gate guard for function `arg_attrs`, which is
// only serialized from 13.5 on (kMinArgAttrsVersion). The absence case is
// covered by arg_attrs_absent_roundtrip.mlir.

// Default round-trip (writes at the current version, which is >= 13.5).
// RUN: %round_trip_test %s %t

// Pin the introduction version so the gate boundary is exercised regardless of
// what the default bytecode version happens to be.
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=13.5 %s -o %t.13_5.mlir
// RUN: cuda-tile-opt --no-implicit-module %s -o %t.ref.mlir
// RUN: diff %t.13_5.mlir %t.ref.mlir

// Below the gate the attributes cannot be represented, so the writer refuses
// instead of dropping them silently.
// RUN: not cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=13.4 %s 2>&1 | FileCheck %s --check-prefix=REFUSE

// The reported minimum version must name the same gate the writer enforces.
// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module %s | FileCheck %s --check-prefix=MINVER
// MINVER: 13.5

cuda_tile.module @kernels {
  // Every argument carries a (distinct kind of) arg attr.
  cuda_tile.entry @entry_all_arg_attrs(
      %a: !cuda_tile.tile<f32> {cuda_tile.alignment = 32 : i64},
      %b: !cuda_tile.tile<i32> {cuda_tile.alignment = 16 : i64}) {
    cuda_tile.return
  }

  // Control: no arg attrs at all -- the empty case must still round-trip.
  cuda_tile.entry @entry_no_arg_attrs(
      %a: !cuda_tile.tile<f32>,
      %b: !cuda_tile.tile<i32>) {
    cuda_tile.return
  }

  // Mixed: some args have attrs and some do not, exercising per-argument
  // (positional) preservation.
  cuda_tile.entry @entry_mixed_arg_attrs(
      %x: !cuda_tile.tile<f32> {cuda_tile.alignment = 32 : i64},
      %y: !cuda_tile.tile<f32>,
      %z: !cuda_tile.tile<f32> {cuda_tile.alignment = 8 : i64}) {
    cuda_tile.return
  }
}

// REFUSE: error: argument attributes require bytecode version 13.5, targeting 13.4
