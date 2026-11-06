// Test getCudaTileModuleOp error when top-level op is neither cuda_tile::ModuleOp
// nor mlir::ModuleOp. Covers BytecodeTranslation.cpp ( false
// branch) and (serialization failure path).
//
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -allow-unregistered-dialect %s 2>&1 | FileCheck %s

// CHECK: expected a CUDA Tile IR module, but got a test.unknown_op
"test.unknown_op"() : () -> ()
