// Encoding-stability guard for `tile<f8E4M3FN>` splat.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/f8E4M3FN_splat.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @f8E4M3FN_splat() {
    %0 = cuda_tile.constant <f8E4M3FN: 2.5> : !cuda_tile.tile<f8E4M3FN>
    cuda_tile.return
  }
}
