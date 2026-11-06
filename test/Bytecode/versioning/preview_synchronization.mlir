// RUN: cuda-tile-translate -analyze-min-version -no-implicit-module \
// RUN:   --preview=simt %s \
// RUN:   | FileCheck %s --check-prefix=MIN
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-PREVIEW
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s \
// RUN:   --preview=simt -bytecode-version=13.4 \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OLD-VERSION
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module \
// RUN:   --preview=simt -bytecode-version=13.5 %s \
// RUN:   | FileCheck %s --check-prefix=ROUNDTRIP

// MIN: 13.5
// NO-PREVIEW: uses preview feature 'simt', which is not enabled
// OLD-VERSION: public-preview content, which is only valid at the current bytecode version 13.5
// ROUNDTRIP: preview$extern_impl @synchronization_version
// ROUNDTRIP-NOT: optimization_hints
// ROUNDTRIP-SAME: synchronization = #cuda_tile<preview$synchronization<warp>>

cuda_tile.module @kernels {
  cuda_tile.preview$func extern @target(
      %input: !cuda_tile.tile<1xf32> {
        cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()

  cuda_tile.preview$extern_impl @synchronization_version implements @target(
      %input: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(1, 32, 1):(0, 0, 0)>> {
          cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> ()
      synchronization = #cuda_tile.preview$synchronization<warp>
}
