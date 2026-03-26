cuda_tile.module @basic_program {
  entry @main() {
    // Hello World in BASIC
    print "Hello, World!\n"
    %0 = constant <f32: 4.200000e+01> : tile<f32>
    %1 = constant <f32: 2.000000e+00> : tile<f32>
    %2 = mulf %0, %1 rounding<nearest_even> : tile<f32>
    print "X = %f\n", %0 : tile<f32>
    print "Y = %f\n", %2 : tile<f32>
    %3 = constant <i32: 80> : tile<i32>
    %5 = itof %3 : tile<i32> -> tile<f32>
    %4 = cmpf greater_than ordered %2, %5 : tile<f32>
    if %4 {
      print "Y is large\n"
    } else {
      print "Y is small\n"
    }
    %6 = constant <i32: 1> : tile<i32>
    %7 = constant <i32: 5> : tile<i32>
    %8 = constant <i32: 1> : tile<i32>
    %9 = addi %7, %8 : tile<i32>
    for %10 in (%6 to %9, step %8) : tile<i32> {
      print "I = %d\n", %10 : tile<i32>
      continue
    }
    return
    return
  }
}
