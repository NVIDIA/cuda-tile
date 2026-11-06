// Coverage for CudaTileOptimizer.cpp: loadInputModule error paths
// Exercises file-not-found, parse failure,
// and verbose + file-not-found for error severity in diag handler

// --- File not found (non-verbose) covers loadInputModule file error ---
// RUN: rm -f %t.nonexistent.mlir
// RUN: not cuda-tile-optimize %t.nonexistent.mlir --opt-level 0 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FILE

// --- File not found in verbose mode covers error severity in diag handler ---
// RUN: not cuda-tile-optimize %t.nonexistent.mlir --opt-level 0 --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FILE-VERBOSE

// --- Bad MLIR parse covers !outMod path ---
// RUN: echo "this is not valid MLIR" > %t.bad.mlir
// RUN: not cuda-tile-optimize %t.bad.mlir --opt-level 0 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-PARSE

// --- Bad MLIR parse in verbose mode ---
// RUN: not cuda-tile-optimize %t.bad.mlir --opt-level 0 --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-PARSE-VERBOSE

// This file itself is valid MLIR (used only by other RUN lines that don't reference %s)
cuda_tile.module @dummy_for_file {
  entry @dummy_entry() {
    return
  }
}

// CHECK-NO-FILE: Failed to read file
// CHECK-NO-FILE-VERBOSE: Failed to read file
// CHECK-BAD-PARSE: Failed to parse input
// CHECK-BAD-PARSE-VERBOSE: Failed to parse input
