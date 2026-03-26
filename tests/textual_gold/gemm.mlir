cuda_tile.module @basic_program {
  entry @main(%0: tile<f32>, %1: tile<f32>) {
    // GEMM: C(M,N) = A(M,K) * B(K,N)
    %2 = constant <f32: 0.0> : tile<512xf32>
    %3 = constant <f32: 0.0> : tile<512xf32>
    %4 = constant <f32: 0.0> : tile<512xf32>
    %5 = constant <f32: 0.000000e+00> : tile<f32>
    %6 = constant <i32: 4> : tile<i32>
    %8 = itof %6 : tile<i32> -> tile<f32>
    %7 = divf %5, %8 rounding<nearest_even> : tile<f32>
    %9 = constant <i32: 4> : tile<i32>
    %11 = itof %9 : tile<i32> -> tile<f32>
    %10 = remf %5, %11 : tile<f32>
    %12 = constant <f32: 0.000000e+00> : tile<f32>
    %13 = constant <i32: 0> : tile<i32>
    %14 = constant <i32: 15> : tile<i32>
    %15 = constant <i32: 1> : tile<i32>
    %16 = addi %14, %15 : tile<i32>
    for %17 in (%13 to %16, step %15) : tile<i32> {
      continue
    }
    return
    return
  }
}
