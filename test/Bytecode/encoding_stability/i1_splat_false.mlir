// Encoding-stability guard for splat-false `tile<i1>` constants. Pairs with
// i1_splat_true.mlir to pin both halves of the i1 raw-buffer contract.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/i1_splat_false.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @i1_splat_false() {
    %0 = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
    cuda_tile.return
  }
}
