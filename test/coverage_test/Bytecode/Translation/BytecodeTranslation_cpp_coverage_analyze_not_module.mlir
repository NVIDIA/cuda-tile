// Test analyze-min-version with a top-level op that is neither cuda_tile::ModuleOp
// nor mlir::ModuleOp. Covers the getCudaTileModuleOp error path
// exercised through the analyze-min-version translation.
//
// RUN: not cuda-tile-translate -analyze-min-version -no-implicit-module -allow-unregistered-dialect %s 2>&1 | FileCheck %s

// CHECK: expected a CUDA Tile IR module, but got a test.unknown_op
"test.unknown_op"() : () -> ()
