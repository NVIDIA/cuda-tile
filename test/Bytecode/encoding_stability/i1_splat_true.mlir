// Encoding-stability guard for splat-true `tile<i1>` constants.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/i1_splat_true.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @i1_splat_true() {
    %0 = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    cuda_tile.return
  }
}
