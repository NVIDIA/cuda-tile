// Covers BytecodeWriter writeGlobalSection() at the 13.3 boundary —
// the first version where private-visibility and `constant` flags are
// encoded. Existing tests pin the legacy-reject path (13.1/13.2) and
// the default (13.4) path; this test exercises the extended-fields
// branch (`version >= kMinGlobalExtendedFields`) using each combination
// of {public, private} x {non-constant, constant} explicitly at v13.3.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   -bytecode-version=13.3 %s -o %t.mlir
// RUN: FileCheck %s < %t.mlir

cuda_tile.module @kernels {
  // Public + non-constant — exercises the writeGlobalSection() flow
  // serialising defaults for visibility & constant flag.
  cuda_tile.global @pub_data <i32: [1, 2, 3, 4]> : !cuda_tile.tile<4xi32>

  // Private + non-constant — exercises private visibility encoding
  // (bit set in the visibility varint, not the constant varint).
  cuda_tile.global private @priv_data <i32: [10, 20]> : !cuda_tile.tile<2xi32>

  // Public + constant — exercises constant encoding without changing
  // visibility.
  cuda_tile.global constant @const_data <f32: [3.140000e+00]> : !cuda_tile.tile<1xf32>

  // Private + constant — both extended fields take their non-default
  // values, exercising both varints in the >=13.3 branch.
  cuda_tile.global private constant @priv_const <i32: 99> : !cuda_tile.tile<1xi32>

  // Custom alignment — exercises writeGlobalSection's alignment varint
  // independently of visibility/constant. Default alignment paths are
  // already exercised by the four globals above.
  cuda_tile.global @aligned_data alignment = 128 <i32: 42> : !cuda_tile.tile<1xi32>

  cuda_tile.entry @use_globals() {
    cuda_tile.return
  }
}

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK-DAG:     global  @pub_data <i32: [1, 2, 3, 4]>
// CHECK-DAG:     global private  @priv_data <i32: [10, 20]>
// CHECK-DAG:     global  constant @const_data <f32: 3.140000e+00>
// CHECK-DAG:     global private  constant @priv_const <i32: 99>
// CHECK-DAG:     global  @aligned_data alignment = 128 <i32: 42>
// CHECK:         entry @use_globals
