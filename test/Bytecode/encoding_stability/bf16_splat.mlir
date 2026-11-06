// Encoding-stability guard for `tile<bf16>` splat.
//
// bf16's bit layout is MLIR's responsibility (1 sign + 8 exp + 7 mantissa).
// If MLIR ever changes how APFloat::bitcastToAPInt yields the bytes for
// bf16, this fixture will diverge and the test fires.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/bf16_splat.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @bf16_splat() {
    %0 = cuda_tile.constant <bf16: 5.5> : !cuda_tile.tile<bf16>
    cuda_tile.return
  }
}
