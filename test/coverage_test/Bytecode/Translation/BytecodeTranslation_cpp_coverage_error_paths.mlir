// Test getCudaTileModuleOp error when mlir::ModuleOp contains a single element
// that is NOT a cuda_tile::ModuleOp. This specifically targets
// in BytecodeTranslation.cpp (isa<cuda_tile::ModuleOp> returns false while
// hasSingleElement is true). Also covers the analyze-min-version path.
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -allow-unregistered-dialect -split-input-file %s 2>&1 | FileCheck --check-prefix=CHECK-SERIALIZE %s
// RUN: not cuda-tile-translate -analyze-min-version -no-implicit-module -allow-unregistered-dialect -split-input-file %s 2>&1 | FileCheck --check-prefix=CHECK-ANALYZE %s

// mlir::ModuleOp with a single non-cuda_tile child op.
// CHECK-SERIALIZE: expected a single CUDA Tile IR module in the MLIR module
// CHECK-ANALYZE: expected a single CUDA Tile IR module in the MLIR module
module {
  "test.non_cuda_tile_op"() : () -> ()
}
