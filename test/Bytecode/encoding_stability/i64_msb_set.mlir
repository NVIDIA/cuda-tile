// Encoding-stability guard for `tile<i64>` constants with the high bit set.
// Catches regressions in 64-bit value encoding (the path that uses
// IntegerAttr's getZExtValue for signless integers).

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/i64_msb_set.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @i64_msb_set() {
    %0 = cuda_tile.constant <i64: -1> : !cuda_tile.tile<i64>
    cuda_tile.return
  }
}
