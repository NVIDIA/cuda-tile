// Coverage for CudaTileOptimizer.cpp: enableMultithread flag
// Exercises Threading::ENABLED path in optimizeTileIR

// RUN: cuda-tile-optimize %s --opt-level 0 --enable-multithread | FileCheck %s

cuda_tile.module @test_multithread {
  entry @test_entry() {
    return
  }
}

// CHECK: cuda_tile.module @test_multithread
// CHECK:   entry @test_entry()
// CHECK:     return
