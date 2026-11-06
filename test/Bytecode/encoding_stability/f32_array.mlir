// Encoding-stability guard for multi-element `tile<Nxf32>` constants.
// Catches endianness shifts and non-splat dense-float layout changes.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/f32_array.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @f32_array() {
    %0 = cuda_tile.constant <f32: [5.0, 6.0, 7.0, 8.0]> : !cuda_tile.tile<4xf32>
    cuda_tile.return
  }
}
