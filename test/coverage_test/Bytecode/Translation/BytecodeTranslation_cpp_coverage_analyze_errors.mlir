// Test error paths in getCudaTileModuleOp when called from analyze-min-version.
// Covers BytecodeTranslation.cpp (failed(moduleOp) in min-version
// analysis) via both empty-module and non-module-op error triggers.
//
// RUN: not cuda-tile-translate -analyze-min-version -split-input-file %s 2>&1 | FileCheck %s

// An empty builtin module has no cuda_tile.module child.
// CHECK: expected a single CUDA Tile IR module in the MLIR module
module {
}

// -----

// A builtin module with multiple cuda_tile.module children is also invalid.
// CHECK: expected a single CUDA Tile IR module in the MLIR module
module {
  cuda_tile.module @mod1 {
  }
  cuda_tile.module @mod2 {
  }
}
