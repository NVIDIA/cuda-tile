// Encoding-stability guard for signed `tile<i8>` constants with the high
// bit set. Catches regressions in sign-extension / signedness handling on
// the writer side.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/i8_signed_negative.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @i8_signed_negative() {
    %0 = cuda_tile.constant <i8: -42> : !cuda_tile.tile<i8>
    cuda_tile.return
  }
}
