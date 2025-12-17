// RUN: %round_trip_test %s %t

cuda_tile.module @kernels{
  // Test function with no parameters
  cuda_tile.entry @no_parameters() {
    %0 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    cuda_tile.return
  }

  // Test function with many parameters
  cuda_tile.entry @many_parameters(
    %p0: !cuda_tile.tile<i32>, %p1: !cuda_tile.tile<i32>, %p2: !cuda_tile.tile<i32>, 
    %p3: !cuda_tile.tile<i32>, %p4: !cuda_tile.tile<i32>, %p5: !cuda_tile.tile<i32>, 
    %p6: !cuda_tile.tile<i32>, %p7: !cuda_tile.tile<i32>, %p8: !cuda_tile.tile<i32>, 
    %p9: !cuda_tile.tile<i32>
  ) {
    %0 = cuda_tile.addi %p0, %p1 : !cuda_tile.tile<i32>
    %1 = cuda_tile.addi %0, %p2 : !cuda_tile.tile<i32>
    %2 = cuda_tile.addi %1, %p3 : !cuda_tile.tile<i32>
    %3 = cuda_tile.addi %2, %p4 : !cuda_tile.tile<i32>
    %4 = cuda_tile.addi %3, %p5 : !cuda_tile.tile<i32>
    %5 = cuda_tile.addi %4, %p6 : !cuda_tile.tile<i32>
    %6 = cuda_tile.addi %5, %p7 : !cuda_tile.tile<i32>
    %7 = cuda_tile.addi %6, %p8 : !cuda_tile.tile<i32>
    %8 = cuda_tile.addi %7, %p9 : !cuda_tile.tile<i32>
    cuda_tile.return
  }

  // Test function with many intermediate values
  cuda_tile.entry @multiple_returns(%p0: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %1 = cuda_tile.addi %p0, %0 : !cuda_tile.tile<i32>
    %2 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %3 = cuda_tile.addi %p0, %2 : !cuda_tile.tile<i32>
    %4 = cuda_tile.addi %1, %3 : !cuda_tile.tile<i32>
    cuda_tile.return
  }

  // Test with long function name (string table handling)
  cuda_tile.entry @long_function_name_that_tests_string_table_with_longer_than_usual_identifiers() {
    %0 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    cuda_tile.return
  }
}
