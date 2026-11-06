// Encoding-stability guard for `tile<f64>` (IEEE binary64) splat with the
// sign bit set. Asymmetric value flips it to obviously-wrong on any single
// bit shift in the encoding.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/f64_negative.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @f64_negative() {
    %0 = cuda_tile.constant <f64: -12.3456> : !cuda_tile.tile<f64>
    cuda_tile.return
  }
}
