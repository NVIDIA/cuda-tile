// Backward-compat regression: legacy v13.3-era binaries wrote i1 splat
// `DenseElementsAttr` payloads as a single byte regardless of the tensor
// shape, with `0xff` for true and `0x00` for false.

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-splat-legacy-13.3.tileirbc | FileCheck %s

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK:   entry @i1_splat
// CHECK:     constant <i1: true> : tile<16xi1>
// CHECK:     constant <i1: false> : tile<16xi1>
