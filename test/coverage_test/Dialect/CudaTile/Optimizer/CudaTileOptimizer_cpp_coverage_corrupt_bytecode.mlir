// Coverage for CudaTileOptimizer.cpp: corrupt bytecode triggers readBytecode failure
// Exercises parseTileIRBytecode — readBytecode returns null path

// Create a file with valid TileIR bytecode magic header but corrupt payload
// Magic: 0x7F 'T' 'i' 'l' 'e' 'I' 'R' 0x00, followed by garbage
// RUN: printf '\x7fTileIR\x00CORRUPT_PAYLOAD_DATA' > %t.corrupt.bc
// RUN: not cuda-tile-optimize %t.corrupt.bc --opt-level 0 2>&1 | FileCheck %s --check-prefix=CHECK-CORRUPT

// Also test bytecode round-trip with verbose to cover verbose + bytecode path
// RUN: cuda-tile-optimize %s --opt-level 0 --emit-bytecode -o %t.valid.bc 2>&1
// RUN: cuda-tile-optimize %t.valid.bc --opt-level 1 --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-BC-VERBOSE

cuda_tile.module @test_corrupt_bc {
  entry @test_entry() {
    return
  }
}

// CHECK-CORRUPT: Failed to parse input
// CHECK-BC-VERBOSE: Pipeline:
// CHECK-BC-VERBOSE: cuda_tile.module
