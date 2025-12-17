// RUN: %round_trip_test %s %t

// Check that we correctly round-trip when forcing the version to 13.1
// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=13.1 %s -o %t.mlir
// RUN: cuda-tile-opt --no-implicit-module %s -o %t.ref.mlir
// RUN: diff %t.mlir %t.ref.mlir

cuda_tile.module @kernels {
  cuda_tile.entry @simple_function(%a: !cuda_tile.tile<i32>) {
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %result = cuda_tile.addi %a, %c1 : !cuda_tile.tile<i32>
    cuda_tile.return
  }
}
