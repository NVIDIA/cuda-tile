// Test that SSA value indexing is correct when versioned results are not serialized.
// print_tko's result_token requires 13.2 - when targeting 13.1, it's not serialized.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -o %t.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.bc | FileCheck %s

// CHECK: cuda_tile.module @kernels
cuda_tile.module @kernels {
  global @mutex <i32: 1> : tile<1xi32>

  entry @test_print_then_more_values() {
    %cst = constant <i32: 1> : tile<i32>
    %ptr = get_global @mutex : tile<ptr<i32>>
    // CHECK: print_tko "%d"
    %print_token = print_tko "%d", %cst : tile<i32> -> token
    // CHECK: atomic_rmw_tko acq_rel device
    %result, %token = atomic_rmw_tko acq_rel device %ptr, xchg, %cst : tile<ptr<i32>>, tile<i32> -> tile<i32>, token
    // More values after print_tko
    %cst2 = constant <i32: 2> : tile<i32>
    // CHECK: print_tko "%d"
    %print_token2 = print_tko "%d", %cst2 : tile<i32> -> token
    return
  }
}
