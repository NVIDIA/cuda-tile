// Test pre/post pipeline text and quiet mode in CudaTileOptimizer.cpp
// Covers: parseTextInto (valid text), buildOptimizationPipelineWithOverrides
//         (pre/post text paths), optimizeTileIR quiet mode (None output)

// --- pre-pipeline pass ---
// RUN: cuda-tile-optimize %s --opt-level 0 --before "canonicalize" --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-PRE

// --- post-pipeline pass ---
// RUN: cuda-tile-optimize %s --opt-level 0 --after "cse" --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-POST

// --- both pre and post ---
// RUN: cuda-tile-optimize %s --opt-level 0 --before "canonicalize" --after "cse" --verbose 2>&1 | FileCheck %s --check-prefix=CHECK-BOTH

// --- quiet mode: output.mode == None, early return ---
// RUN: cuda-tile-optimize %s --opt-level 0 --quiet 2>&1 | FileCheck %s --check-prefix=CHECK-QUIET --allow-empty

cuda_tile.module @test_prepost {
  entry @test_entry() {
    return
  }
}

// CHECK-PRE: Pipeline:
// CHECK-PRE: canonicalize
// CHECK-PRE: cuda_tile.module @test_prepost

// CHECK-POST: Pipeline:
// CHECK-POST: cse
// CHECK-POST: cuda_tile.module @test_prepost

// CHECK-BOTH: Pipeline:
// CHECK-BOTH: canonicalize
// CHECK-BOTH: cse
// CHECK-BOTH: cuda_tile.module @test_prepost

// CHECK-QUIET-NOT: cuda_tile.module
