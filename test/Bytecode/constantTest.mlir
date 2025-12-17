// RUN: %round_trip_test %s %t

// Test bytecode serialization/deserialization of different constants

cuda_tile.module @kernels {
  cuda_tile.entry @constants() {
    %0 = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
    %1 = cuda_tile.constant <i1: 0> : !cuda_tile.tile<i1>
    %2 = cuda_tile.constant <i8: 42> : !cuda_tile.tile<i8>
    %3 = cuda_tile.constant <i8: -42> : !cuda_tile.tile<i8>
    %4 = cuda_tile.constant <i16: 1000> : !cuda_tile.tile<i16>
    %5 = cuda_tile.constant <i16: -1000> : !cuda_tile.tile<i16>
    %6 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %7 = cuda_tile.constant <i64: 1> : !cuda_tile.tile<i64>
    %8 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    %9 = cuda_tile.constant <i32: -1> : !cuda_tile.tile<i32>
    %10 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    %11 = cuda_tile.constant <i32: 2147483647> : !cuda_tile.tile<i32>  // INT32_MAX
    %12 = cuda_tile.constant <i32: -2147483647> : !cuda_tile.tile<i32> // INT32_MIN+1
    %13 = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
    %14 = cuda_tile.constant <i64: -1> : !cuda_tile.tile<i64>
    %15 = cuda_tile.constant <f64: 12.3456> : !cuda_tile.tile<f64>
    %16 = cuda_tile.constant <f64: -12.3456> : !cuda_tile.tile<f64>
    %17 = cuda_tile.constant <bf16: 5.5> : !cuda_tile.tile<bf16>
    %18 = cuda_tile.constant <f8E4M3FN: 2.5> : !cuda_tile.tile<f8E4M3FN>
    %19 = cuda_tile.constant <f8E5M2: -1.0> : !cuda_tile.tile<f8E5M2>
    %20 = cuda_tile.constant <tf32: 3.14> : !cuda_tile.tile<tf32>
    cuda_tile.return
  }
}
