// Encoding-stability guard for multi-element `tile<Nxi32>` constants.
// Catches both endianness shifts in the raw buffer and any change in how
// non-splat dense ints are laid out.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/i32_array.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @i32_array() {
    %0 = cuda_tile.constant <i32: [1, 2, 3, 4]> : !cuda_tile.tile<4xi32>
    cuda_tile.return
  }
}
