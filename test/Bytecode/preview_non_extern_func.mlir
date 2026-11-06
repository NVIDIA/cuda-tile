// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   --preview=simt %s 2>&1 | FileCheck %s

// CHECK: 'cuda_tile.preview$func' op must be extern when serialized to bytecode

cuda_tile.module @kernels {
  preview$func @identity(%arg0: !cuda_tile.tile<f32>)
      -> !cuda_tile.tile<f32> {
    return %arg0 : !cuda_tile.tile<f32>
  }
}
