// Encoding-stability guard for `tile<f16>` (IEEE binary16) splat.
// Uses an asymmetric value so any single-bit shift in the raw layout flips
// it to a clearly-wrong number.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/f16_splat.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @f16_splat() {
    %0 = cuda_tile.constant <f16: -1.5> : !cuda_tile.tile<f16>
    cuda_tile.return
  }
}
