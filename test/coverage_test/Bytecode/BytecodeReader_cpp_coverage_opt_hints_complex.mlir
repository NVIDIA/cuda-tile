// Roundtrip test for complex optimization hints and producer string to exercise
// more self-contained attribute parsing paths in BytecodeReader:
// parseSelfContainedOpAttribute (Integer, String, Dictionary branches),
// parseFunctionTableSection (optimization hints flag), and parseProducerSection.
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels attributes {producer = "test-coverage-producer v1.0 --debug"} {
  // Entry with multiple optimization hint keys to exercise nested dictionary parsing
  cuda_tile.entry @kernel_with_hints(%arg0: !cuda_tile.tile<ptr<f32>>,
                                      %arg1: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=2, occupancy=128}> {
    %0 = cuda_tile.constant <f32: 1.0> : tile<f32>
    cuda_tile.return
  }

  // Entry without hints to exercise the "no hints" path
  cuda_tile.entry @kernel_no_hints(%arg0: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.addi %arg0, %arg0 : tile<i32>
    cuda_tile.return
  }

  // Entry with single hint key
  cuda_tile.entry @kernel_single_hint(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={occupancy=64}> {
    cuda_tile.return
  }
}
