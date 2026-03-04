// This file ensures that a checked-in 13.1 bytecode fixture can be parsed
// and yields the expected IR.

// COM: bytecode contains
// COM: cuda_tile.module @test {
// COM:   entry @basic() {
// COM:     %input = cuda_tile.constant <i32: [1, 2]> : !cuda_tile.tile<2xi32>
// COM:     %result = cuda_tile.negi %input : !cuda_tile.tile<2xi32>
// COM:     %result2 = cuda_tile.negi %input overflow <none> : !cuda_tile.tile<2xi32>
// COM:   }
// COM: }

// RUN: cuda-tile-translate -cudatilebc-to-mlir %S/Inputs/13.1/negi-op-13.1.tileirbc | FileCheck %s

// CHECK: entry @basic() {
// CHECK: %{{.*}} = constant <i32: [1, 2]> : tile<2xi32>
// CHECK: %{{.*}} = negi %{{.*}} : tile<2xi32>
// CHECK: %{{.*}} = negi %{{.*}} : tile<2xi32>
// CHECK: }
