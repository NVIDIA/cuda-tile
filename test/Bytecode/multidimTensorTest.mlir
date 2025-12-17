// RUN: %round_trip_test %s %t

// Test bytecode serialization/deserialization of multi-element constants

cuda_tile.module @kernels {
  cuda_tile.entry @array_constants() {
    %0 = cuda_tile.constant <i32: [1, 2, 3, 4]> : !cuda_tile.tile<4xi32>
    %1 = cuda_tile.constant <f32: [5.0, 6.0, 7.0, 8.0]> : !cuda_tile.tile<4xf32>
    %2 = cuda_tile.constant <i1: [true, false, true, false]> : !cuda_tile.tile<4xi1>
    %3 = cuda_tile.constant <i16: [10, 20, 30, 40]> : !cuda_tile.tile<4xi16>
    %4 = cuda_tile.constant <f64: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf64>
    %5 = cuda_tile.constant <i32: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : !cuda_tile.tile<2x2x2xi32>
    %6 = cuda_tile.constant <i8: [9, 10, 11, 12]> : !cuda_tile.tile<4xi8>
    %7 = cuda_tile.constant <i64: [100, 200, 300, 400]> : !cuda_tile.tile<4xi64>
    %8 = cuda_tile.constant <f16: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf16>
    %9 = cuda_tile.constant <bf16: [5.0, 6.0, 7.0, 8.0]> : !cuda_tile.tile<4xbf16>
    %10 = cuda_tile.constant <tf32: [9.0, 10.0, 11.0, 12.0]> : !cuda_tile.tile<4xtf32>
    %11 = cuda_tile.constant <f8E4M3FN: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf8E4M3FN>
    %12 = cuda_tile.constant <f8E5M2: [5.0, 6.0, 7.0, 8.0]> : !cuda_tile.tile<4xf8E5M2>
    cuda_tile.return
  }

}
