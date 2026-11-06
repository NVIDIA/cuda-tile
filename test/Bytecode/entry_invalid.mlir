// RUN: not cuda-tile-translate -mlir-to-cudatilebc %s -no-implicit-module 2>&1 | FileCheck %s

cuda_tile.module @kernels {
  // CHECK: unsupported type in bytecode for entry argument #0: '!cuda_tile.tensor_view<?x?xf32, strides=[?,?]>'; expected a rank-0 !cuda_tile.tile type
  cuda_tile.entry @kernel(
      %arg0: !cuda_tile.tensor_view<?x?xf32, strides=[?,?]>) {
    cuda_tile.return
  }
}
