// Coverage for CommandLineOptions.cpp: --list-versions callback false path.
// Passing -list-versions=0 invokes the callback with val=false, covering the else branch.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -list-versions=0 %s -o %t && cuda-tile-translate -no-implicit-module -cudatilebc-to-mlir %t | FileCheck %s

// CHECK: cuda_tile.module @kernels
cuda_tile.module @test {
  cuda_tile.entry @dummy() {
    cuda_tile.return
  }
}
