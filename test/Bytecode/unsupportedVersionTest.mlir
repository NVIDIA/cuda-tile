// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=12.0 %s 2>&1 | FileCheck %s
// CHECK: Invalid argument '12.0': the supported versions are [13.1, 13.1]

cuda_tile.module @kernels {
  cuda_tile.entry @unsupported_version_func(%arg0: !cuda_tile.tile<2xi32>) -> !cuda_tile.tile<i32> {
    %0 = cuda_tile.constant <i32 : 5> : !cuda_tile.tile<i32>
    cuda_tile.return %0 : !cuda_tile.tile<i32>
  }
}
