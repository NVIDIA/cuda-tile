// Test invalid pipeline text error path in CudaTileOptimizer.cpp
// Covers: parseTextInto (failed parsePassPipeline), emitConfigError

// --- invalid pre-pipeline text triggers parseTextInto failure ---
// RUN: not cuda-tile-optimize %s --opt-level 0 --before "invalid-nonexistent-pass" 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-PRE

// --- invalid post-pipeline text triggers parseTextInto failure ---
// RUN: not cuda-tile-optimize %s --opt-level 0 --after "another-bad-pass" 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-POST

cuda_tile.module @test_module {
  entry @test_entry() {
    return
  }
}

// CHECK-BAD-PRE: does not refer to a registered pass or pass pipeline
// CHECK-BAD-POST: does not refer to a registered pass or pass pipeline
