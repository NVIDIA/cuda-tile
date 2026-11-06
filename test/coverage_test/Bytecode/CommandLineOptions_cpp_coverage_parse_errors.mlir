// Test coverage for CommandLineOptions.cpp: BytecodeVersionParser::parse error paths.
// Each RUN line triggers a different invalid version format to cover uncovered branches.

// --- consumeInteger fails for verMajor (non-numeric input) ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=abc %s 2>&1 | FileCheck --check-prefix=CHECK-BAD-MAJOR %s
// CHECK-BAD-MAJOR: Invalid argument 'abc'

// --- consumeInteger fails for verMajor (leading dot) ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=.1 %s 2>&1 | FileCheck --check-prefix=CHECK-DOT-FIRST %s
// CHECK-DOT-FIRST: Invalid argument '.1'

// --- consume_front(".") fails (no dot after major) ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=13 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-DOT %s
// CHECK-NO-DOT: Invalid argument '13'

// --- consumeInteger fails for verMinor (non-numeric minor) ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=13.abc %s 2>&1 | FileCheck --check-prefix=CHECK-BAD-MINOR %s
// CHECK-BAD-MINOR: Invalid argument '13.abc'

// --- tag parse error (dot present but non-numeric tag) ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=13.1.abc %s 2>&1 | FileCheck --check-prefix=CHECK-BAD-TAG %s
// CHECK-BAD-TAG: Invalid argument '13.1.abc'

// --- trailing characters after valid version ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=13.1xyz %s 2>&1 | FileCheck --check-prefix=CHECK-TRAILING %s
// CHECK-TRAILING: Invalid argument '13.1xyz'

// --- out-of-range version (valid parse, fromVersion returns nullopt) ---
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module --bytecode-version=99.0 %s 2>&1 | FileCheck --check-prefix=CHECK-RANGE %s
// CHECK-RANGE: Invalid argument '99.0': the supported versions are

// Minimal valid MLIR input (required by cuda-tile-translate but not the focus of this test).
cuda_tile.module @test {
  cuda_tile.entry @dummy() {
    cuda_tile.return
  }
}
