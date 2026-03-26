cuda_tile.module @basic_program {
  entry @main(%0: tile<f32>, %1: tile<f32>) {
    // Vector Add: C = A + B
    %2 = constant <f32: 0.0> : tile<128xf32>
    %3 = constant <f32: 0.0> : tile<128xf32>
    %4 = constant <f32: 0.0> : tile<128xf32>
    %5 = constant <f32: 0.000000e+00> : tile<f32>
    %6 = extract %2[%5] : tile<f32>
    %7 = extract %3[%5] : tile<f32>
    %8 = addf %6, %7 rounding<nearest_even> : tile<f32>
    // store %8 into C[%5]
    return
    return
  }
}
