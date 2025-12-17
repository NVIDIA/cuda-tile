// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // Check EntryInfo with three SMs with different params
  // CHECK:      entry @test_optimization_hints(%arg0: tile<ptr<f32>>)
  // CHECK-SAME: optimization_hints=<sm_100 = {num_cta_in_cga = 2}, sm_120 = {num_cta_in_cga = 2, occupancy = 2}> {
  entry @test_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 2}, sm_120 = {num_cta_in_cga = 2, occupancy = 2}> {
    return
  }
  // Check processing of empty EntryInfo
  // CHECK: entry @empty_optimization_hints(%arg0: tile<ptr<f32>>) {
  entry @empty_optimization_hints(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<> {
    return
  }
}
