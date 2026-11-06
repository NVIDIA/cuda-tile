// Encoding-stability guard for `tile<tf32>` splat.
//
// tf32 is a 19-bit value padded into 32 bits; the padding-bit convention
// is MLIR-defined. If MLIR changes how the padding bits are placed, this
// fixture diverges.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/tf32_splat.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @tf32_splat() {
    %0 = cuda_tile.constant <tf32: 3.14> : !cuda_tile.tile<tf32>
    cuda_tile.return
  }
}
