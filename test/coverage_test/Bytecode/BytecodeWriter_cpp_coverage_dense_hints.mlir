// Test bytecode writer with DenseIntOrFPElementsAttr in optimization hints
// to exercise DenseElementsAttr self-contained path.
// Serialize-only test: roundtrip may differ due to dense repr in hints,
// but the writer path is exercised.
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module %s -o %t.tilebc

cuda_tile.module @kernels {
  cuda_tile.entry @dense_in_hints(%a: !cuda_tile.tile<f32>) optimization_hints=<sm_100={weights=dense<[1, 2, 3, 4]> : !cuda_tile.tile<4xi32>, biases=dense<[0.1, 0.2]> : !cuda_tile.tile<2xf32>}> {
    cuda_tile.return
  }

  // Entry with i64 dense elements in hints
  cuda_tile.entry @dense_i64_hints(%a: !cuda_tile.tile<i32>) optimization_hints=<sm_100={offsets=dense<[100, 200, 300, 400]> : !cuda_tile.tile<4xi64>}> {
    cuda_tile.return
  }
}
