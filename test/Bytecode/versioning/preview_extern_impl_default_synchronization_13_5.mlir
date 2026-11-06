// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module \
// RUN:   --preview=simt %s \
// RUN:   | FileCheck %s --check-prefix=MIN
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   --preview=simt -bytecode-version=13.5 %s \
// RUN:   | FileCheck %s --check-prefix=ROUNDTRIP

// MIN: 13.5
// ROUNDTRIP: preview$extern_impl @legacy_impl
// ROUNDTRIP-NOT: synchronization =

cuda_tile.module @kernels {
  preview$func extern @legacy(
      %arg0: !cuda_tile.tile<1xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<1xf32>

  preview$extern_impl @legacy_impl implements @legacy
      (%arg0: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(1, 1, 1):(0, 0, 0)>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
}
