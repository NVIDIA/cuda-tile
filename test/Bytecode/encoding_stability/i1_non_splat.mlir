// Encoding-stability guard for non-splat `tile<NxI1>` constants. Picks
// patterns that catch both bit-ordering and byte-ordering regressions
// in the bit-packed `i1` writer:
//
//   * `@i1_alt_4`: `[T,F,T,F]` -- bit-reversed differs (0101 vs 1010).
//   * `@i1_ff_tt`: `[F,F,T,T]` -- visually obvious asymmetry; bit-
//     reversed (`[T,T,F,F]`) is a different byte (0x0c vs 0x03).
//   * `@i1_endpoints_16`: `[T, F*14, T]` -- spans two bytes so the
//     *byte* order is also exercised; bit 0 must land in byte 0 and
//     bit 15 in byte 1, not the other way around.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.4 %s -o %t.bc
// RUN: cmp %t.bc %S/goldens/i1_non_splat.golden.tileirbc

cuda_tile.module @kernels {
  cuda_tile.entry @i1_alt_4() {
    %0 = cuda_tile.constant <i1: [true, false, true, false]> : !cuda_tile.tile<4xi1>
    cuda_tile.return
  }
  cuda_tile.entry @i1_ff_tt() {
    %0 = cuda_tile.constant <i1: [false, false, true, true]> : !cuda_tile.tile<4xi1>
    cuda_tile.return
  }
  cuda_tile.entry @i1_endpoints_16() {
    %0 = cuda_tile.constant <i1: [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]> : !cuda_tile.tile<16xi1>
    cuda_tile.return
  }
}
