// Coverage for CudaTileOptimizer.cpp: output file error paths
// Exercises openOutputFile failure for MLIR file and bytecode file

// --- MLIR file output to non-existent directory ---
// RUN: not cuda-tile-optimize %s --opt-level 0 -o %t.dir_does_not_exist/subdir/out.mlir 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-MLIR-OUT

// --- Bytecode file output to non-existent directory ---
// RUN: not cuda-tile-optimize %s --opt-level 0 --emit-bytecode -o %t.dir_does_not_exist/subdir/out.bc 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-BC-OUT

cuda_tile.module @test_output_err {
  entry @test_entry() {
    return
  }
}

// CHECK-BAD-MLIR-OUT: cannot open output file
// CHECK-BAD-BC-OUT: cannot open output file
