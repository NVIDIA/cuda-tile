// Test output mode paths in CudaTileOptimizer.cpp
// Covers: emitOutputs (MlirFile, BytecodeFile, MlirStdout, verbose+file),
//         writeBytecodeToBuffer, loadInputModule (File path, MLIR text),
//         validateConfig (valid config)

// --- MLIR file output ---
// RUN: cuda-tile-optimize %s --opt-level 0 -o %t.mlir 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-MLIR-FILE < %t.mlir

// --- Bytecode file output, then read back ---
// RUN: cuda-tile-optimize %s --opt-level 0 --emit-bytecode -o %t.bc 2>&1
// RUN: cuda-tile-optimize %t.bc --opt-level 0 | FileCheck %s --check-prefix=CHECK-BC-ROUNDTRIP

// --- Verbose + file output: MlirStdout added alongside file ---
// RUN: cuda-tile-optimize %s --opt-level 0 -o %t.verbose.mlir --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-VERBOSE-FILE
// RUN: FileCheck %s --check-prefix=CHECK-VERBOSE-WRITTEN < %t.verbose.mlir

cuda_tile.module @output_test {
  entry @output_entry() {
    return
  }
}

// CHECK-MLIR-FILE: cuda_tile.module @output_test
// CHECK-MLIR-FILE:   entry @output_entry()
// CHECK-MLIR-FILE:     return

// CHECK-BC-ROUNDTRIP: cuda_tile.module
// CHECK-BC-ROUNDTRIP:   entry @output_entry()
// CHECK-BC-ROUNDTRIP:     return

// CHECK-VERBOSE-FILE: Pipeline:
// CHECK-VERBOSE-FILE: cuda_tile.module @output_test

// CHECK-VERBOSE-WRITTEN: cuda_tile.module @output_test
// CHECK-VERBOSE-WRITTEN:   entry @output_entry()
