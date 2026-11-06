// Roundtrip test for arithmetic and logic operations to exercise BytecodeReader
// generated operation readers (parseOperation dispatch).
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // Test integer arithmetic operations
  cuda_tile.entry @int_arith_ops(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.addi %a, %b : tile<i32>
    %1 = cuda_tile.subi %a, %b : tile<i32>
    %2 = cuda_tile.muli %a, %b : tile<i32>
    %3 = cuda_tile.divi %a, %b signed : tile<i32>
    %4 = cuda_tile.divi %a, %b unsigned : tile<i32>
    %5 = cuda_tile.remi %a, %b signed : tile<i32>
    %6 = cuda_tile.remi %a, %b unsigned : tile<i32>
    %7 = cuda_tile.negi %a : tile<i32>
    %8 = cuda_tile.absi %a : tile<i32>
    %9 = cuda_tile.maxi %a, %b signed : tile<i32>
    %10 = cuda_tile.maxi %a, %b unsigned : tile<i32>
    %11 = cuda_tile.mini %a, %b signed : tile<i32>
    %12 = cuda_tile.mini %a, %b unsigned : tile<i32>
    cuda_tile.return
  }

  // Test floating point arithmetic operations
  cuda_tile.entry @float_arith_ops(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.addf %a, %b rounding<nearest_even> : tile<f32>
    %1 = cuda_tile.subf %a, %b rounding<nearest_even> : tile<f32>
    %2 = cuda_tile.mulf %a, %b rounding<nearest_even> : tile<f32>
    %3 = cuda_tile.divf %a, %b rounding<approx> : tile<f32>
    %4 = cuda_tile.negf %a : tile<f32>
    %5 = cuda_tile.absf %a : tile<f32>
    %6 = cuda_tile.maxf %a, %b : tile<f32>
    %7 = cuda_tile.minf %a, %b : tile<f32>
    %8 = cuda_tile.fma %a, %b, %a rounding<nearest_even> : tile<f32>
    %9 = cuda_tile.remf %a, %b : tile<f32>
    cuda_tile.return
  }

  // Test bitwise logic operations
  cuda_tile.entry @bitwise_ops(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %0 = cuda_tile.andi %a, %b : tile<i32>
    %1 = cuda_tile.ori %a, %b : tile<i32>
    %2 = cuda_tile.xori %a, %b : tile<i32>
    %3 = cuda_tile.shli %a, %b : tile<i32>
    %4 = cuda_tile.shri %a, %b signed : tile<i32>
    %5 = cuda_tile.shri %a, %b unsigned : tile<i32>
    cuda_tile.return
  }

  // Test comparison operations
  cuda_tile.entry @comparison_ops(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>,
                                   %c: !cuda_tile.tile<f32>, %d: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.cmpi equal %a, %b, signed : tile<i32> -> tile<i1>
    %1 = cuda_tile.cmpi less_than %a, %b, signed : tile<i32> -> tile<i1>
    %2 = cuda_tile.cmpi greater_than %a, %b, unsigned : tile<i32> -> tile<i1>
    %3 = cuda_tile.cmpf equal ordered %c, %d : tile<f32> -> tile<i1>
    %4 = cuda_tile.cmpf less_than ordered %c, %d : tile<f32> -> tile<i1>
    %5 = cuda_tile.select %0, %a, %b : tile<i1>, tile<i32>
    cuda_tile.return
  }

  // Test type conversion operations
  cuda_tile.entry @type_conversion_ops(%i32: !cuda_tile.tile<i32>, %f32: !cuda_tile.tile<f32>,
                                        %i16: !cuda_tile.tile<i16>) {
    %0 = cuda_tile.itof %i32 signed : tile<i32> -> tile<f32>
    %1 = cuda_tile.ftoi %f32 signed : tile<f32> -> tile<i32>
    %2 = cuda_tile.exti %i16 signed : tile<i16> -> tile<i32>
    %3 = cuda_tile.trunci %i32 : tile<i32> -> tile<i16>
    %4 = cuda_tile.ftof %f32 : tile<f32> -> tile<f16>
    cuda_tile.return
  }

  // Test math operations
  cuda_tile.entry @math_ops(%a: !cuda_tile.tile<f32>) {
    %0 = cuda_tile.sqrt %a : tile<f32>
    %1 = cuda_tile.rsqrt %a : tile<f32>
    %2 = cuda_tile.exp %a : tile<f32>
    %3 = cuda_tile.exp2 %a : tile<f32>
    %4 = cuda_tile.log %a : tile<f32>
    %5 = cuda_tile.log2 %a : tile<f32>
    %6 = cuda_tile.sin %a : tile<f32>
    %7 = cuda_tile.cos %a : tile<f32>
    %8 = cuda_tile.floor %a : tile<f32>
    %9 = cuda_tile.ceil %a : tile<f32>
    %10 = cuda_tile.tanh %a : tile<f32>
    cuda_tile.return
  }
}
