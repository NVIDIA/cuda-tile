// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 -verify-diagnostics -split-input-file %s

// expected-error@unknown {{type 'F8E8M0FNU' requires bytecode version 13.2+, targeting 13.1}}
cuda_tile.module @f8e8m0fnu_version_test {
  entry @test_f8e8m0fnu_version(%ptr: tile<f8E8M0FNU>) {
    cuda_tile.return
  }
}
