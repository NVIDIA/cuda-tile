// Test error paths in CudaTileOptimizer.cpp
// Covers: parseTextInto (invalid text), emitConfigError,
//         extractCudaTileModuleOp failure

// --- no cuda_tile.module in input ---
// RUN: not cuda-tile-optimize %s --opt-level 0 2>&1 | FileCheck %s --check-prefix=CHECK-NO-MODULE

module {
}

// CHECK-NO-MODULE: cuda_tile.ModuleOp not found in input
