// Coverage for CudaTileOptimizer.cpp: verbose diagnostic handler severity branches
// (Warning/Error severity in verbose handler)

// --- Verbose + invalid pass triggers error diagnostic through the handler ---
// RUN: not cuda-tile-optimize %s --opt-level 0 --before "nonexistent-pass-xyz" --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-VERBOSE-ERR

// --- Verbose + multithread combined ---
// RUN: cuda-tile-optimize %s --opt-level 1 --verbose --enable-multithread 2>&1 | FileCheck %s --check-prefix=CHECK-VERBOSE-MT

cuda_tile.module @test_verbose_diag {
  entry @test_entry() {
    return
  }
}

// CHECK-VERBOSE-ERR: does not refer to a registered pass or pass pipeline
// CHECK-VERBOSE-MT: Pipeline:
// CHECK-VERBOSE-MT: cuda_tile.module @test_verbose_diag
