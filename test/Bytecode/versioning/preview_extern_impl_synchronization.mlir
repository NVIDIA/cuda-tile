// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module \
// RUN:   --preview=simt %s \
// RUN:   | FileCheck %s --check-prefix=MIN
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-PREVIEW
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   --preview=simt -bytecode-version=13.5 %s \
// RUN:   | FileCheck %s

// MIN: 13.5
// NO-PREVIEW: uses preview feature 'simt', which is not enabled
// CHECK: preview$func extern @native(
// CHECK: preview$extern_impl @native_impl implements @native

cuda_tile.module @kernels {
  preview$func extern @native(
      %input: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  preview$extern_impl @native_impl implements @native
      (%input: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
      synchronization = #cuda_tile.preview$synchronization<warp>
}
