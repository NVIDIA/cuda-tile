// Encoding-stability guard for `tile<f8E5M2>` splat.
//
// f8E5M2's bit layout is MLIR-defined. Negative value picked so the sign
// bit is set and the mantissa is non-zero, making single-bit shifts in
// the encoding obvious.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/f8E5M2_splat.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @f8E5M2_splat() {
    %0 = cuda_tile.constant <f8E5M2: -1.5> : !cuda_tile.tile<f8E5M2>
    cuda_tile.return
  }
}
