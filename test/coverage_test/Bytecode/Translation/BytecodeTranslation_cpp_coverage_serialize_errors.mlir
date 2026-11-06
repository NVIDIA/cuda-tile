// Test error paths in getCudaTileModuleOp when called from mlir-to-cudatilebc.
// Covers BytecodeTranslation.cpp (mlir::ModuleOp without valid
// cuda_tile::ModuleOp child) and (failed(moduleOp) in serialization).
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -split-input-file %s 2>&1 | FileCheck %s

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
