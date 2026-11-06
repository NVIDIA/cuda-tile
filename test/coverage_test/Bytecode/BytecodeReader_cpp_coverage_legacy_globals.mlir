// Roundtrip test for legacy bytecode versions (13.1 and 13.2) to exercise the
// version-specific global section parsing paths in BytecodeReader.
// Version 13.1/13.2 have 4-varint global entries; 13.3 has 6-varint entries.

// Test with version 13.1 (legacy 4-varint format, no visibility/constant)
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -o %t.v131.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.v131.bc | FileCheck %s --check-prefix=LEGACY

// Test with version 13.2 (legacy 4-varint format, no visibility/constant)
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.2 %s -o %t.v132.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.v132.bc | FileCheck %s --check-prefix=LEGACY

// Test with current version (13.3, has visibility and constant fields)
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s -o %t.v133.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.v133.bc | FileCheck %s --check-prefix=CURRENT

cuda_tile.module @kernels {
  cuda_tile.global @public_val <f32: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf32>
  cuda_tile.global @aligned_val alignment = 64 <i32: [42]> : !cuda_tile.tile<1xi32>

  cuda_tile.entry @use_globals() {
    %0 = cuda_tile.get_global @public_val : tile<ptr<f32>>
    %1 = cuda_tile.get_global @aligned_val : tile<ptr<i32>>
    cuda_tile.return
  }
}

// LEGACY: cuda_tile.module @kernels
// LEGACY: global @public_val
// LEGACY: global @aligned_val
// LEGACY: entry @use_globals

// CURRENT: cuda_tile.module @kernels
// CURRENT: global @public_val
// CURRENT: global @aligned_val
// CURRENT: entry @use_globals
