// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s
// RUN: %round_trip_test %s %t

//===----------------------------------------------------------------------===//
// Integer Arithmetic Operations
//===----------------------------------------------------------------------===//

cuda_tile.module @kernels {
  entry @addi() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: addi %[[c1_i1]], %[[c1_i1]] : tile<i1>
      %add_i1 = cuda_tile.addi %c1_i1, %c1_i1 : tile<i1>

      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      // CHECK: addi %[[c42_i8]], %[[c42_i8]] : tile<i8>
      %add_i8 = cuda_tile.addi %c42_i8, %c42_i8 : tile<i8>

      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      // CHECK: addi %[[c42_i16]], %[[c42_i16]] : tile<i16>
      %add_i16 = cuda_tile.addi %c42_i16, %c42_i16 : tile<i16>

      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      // CHECK: addi %[[c42_i32]], %[[c42_i32]] : tile<i32>
      %add_i32 = cuda_tile.addi %c42_i32, %c42_i32 : tile<i32>

      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
      // CHECK: addi %[[c42_i64]], %[[c42_i64]] : tile<i64>
      %add_i64 = cuda_tile.addi %c42_i64, %c42_i64 : tile<i64>
  }

  entry @cmpi() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      // CHECK: cmpi less_than %[[c1_i1]], %[[c1_i1]], signed : tile<i1>
      // CHECK: cmpi less_than %[[c1_i1]], %[[c1_i1]], signed : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      %cmpi_i1_asm = cmpi less_than %c1_i1, %c1_i1, signed : tile<i1> -> tile<i1> 
      %cmpi_i1_generic = "cuda_tile.cmpi"(%c1_i1, %c1_i1) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i1>, !cuda_tile.tile<i1>) -> !cuda_tile.tile<i1>

      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      // CHECK: cmpi less_than %[[c42_i8]], %[[c42_i8]], signed : tile<i8>
      // CHECK: cmpi less_than %[[c42_i8]], %[[c42_i8]], signed : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      %cmpi_i8_asm = cmpi less_than %c42_i8, %c42_i8, signed : tile<i8> -> tile<i1> 
      %cmpi_i8_generic = "cuda_tile.cmpi"(%c42_i8, %c42_i8) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i8>, !cuda_tile.tile<i8>) -> !cuda_tile.tile<i1>

      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      // CHECK: cmpi less_than %[[c42_i16]], %[[c42_i16]], signed : tile<i16>
      // CHECK: cmpi less_than %[[c42_i16]], %[[c42_i16]], signed : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      %cmpi_i16_asm = cmpi less_than %c42_i16, %c42_i16, signed : tile<i16> -> tile<i1> 
      %cmpi_i16_generic = "cuda_tile.cmpi"(%c42_i16, %c42_i16) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i16>, !cuda_tile.tile<i16>) -> !cuda_tile.tile<i1>

      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      // CHECK: cmpi less_than %[[c42_i32]], %[[c42_i32]], signed : tile<i32>
      // CHECK: cmpi less_than %[[c42_i32]], %[[c42_i32]], signed : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      %cmpi_i32_asm = cmpi less_than %c42_i32, %c42_i32, signed : tile<i32> -> tile<i1> 
      %cmpi_i32_generic = "cuda_tile.cmpi"(%c42_i32, %c42_i32) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>) -> !cuda_tile.tile<i1>

      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      // CHECK: cmpi less_than %[[c42_i64]], %[[c42_i64]], signed : tile<i64>
      // CHECK: cmpi less_than %[[c42_i64]], %[[c42_i64]], signed : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
      %cmpi_i64_asm = cmpi less_than %c42_i64, %c42_i64, signed : tile<i64> -> tile<i1> 
      %cmpi_i64_generic = "cuda_tile.cmpi"(%c42_i64, %c42_i64) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i64>, !cuda_tile.tile<i64>) -> !cuda_tile.tile<i1>

      // CHECK: %[[v0_i32:.*]] = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
      // CHECK: cmpi less_than %[[v0_i32]], %[[v0_i32]], signed : tile<4xi32>
      // CHECK: cmpi less_than %[[v0_i32]], %[[v0_i32]], signed : tile<4xi32>
      %v0_i32 = constant <i32: [1, 2, 3, 4]> : !cuda_tile.tile<4xi32>
      %cmpi_vector_asm = cmpi less_than %v0_i32, %v0_i32, signed : tile<4xi32> -> tile<4xi1> 
      %cmpi_vector_generic = "cuda_tile.cmpi"(%v0_i32, %v0_i32) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<4xi32>, !cuda_tile.tile<4xi32>) -> !cuda_tile.tile<4xi1>

      // CHECK: %[[t0_i64:.*]] = constant <i64: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi64>
      // CHECK: cmpi equal %[[t0_i64]], %[[t0_i64]], signed : tile<2x2xi64>
      // CHECK: cmpi equal %[[t0_i64]], %[[t0_i64]], signed : tile<2x2xi64>
      %t0_i64 = constant <i64: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi64>
      %cmpi_tensor_asm = cmpi equal %t0_i64, %t0_i64, signed : tile<2x2xi64> -> tile<2x2xi1> 
      %cmpi_tensor_generic = "cuda_tile.cmpi"(%t0_i64, %t0_i64) {comparison_predicate = #cuda_tile.comparison_predicate<equal>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<2x2xi64>, !cuda_tile.tile<2x2xi64>) -> !cuda_tile.tile<2x2xi1>

  }

  entry @divi() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: divi %[[c1_i1]], %[[c1_i1]] signed : tile<i1>
      %divi_i1_signed = cuda_tile.divi %c1_i1, %c1_i1 signed : tile<i1>
      // CHECK: divi %[[c1_i1]], %[[c1_i1]] unsigned : tile<i1>
      %divi_i1_unsigned = cuda_tile.divi %c1_i1, %c1_i1 unsigned : tile<i1>

      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      // CHECK: divi %[[c42_i8]], %[[c42_i8]] signed : tile<i8>
      %divi_i8_signed = cuda_tile.divi %c42_i8, %c42_i8 signed : tile<i8>
      // CHECK: divi %[[c42_i8]], %[[c42_i8]] unsigned : tile<i8>
      %divi_i8_unsigned = cuda_tile.divi %c42_i8, %c42_i8 unsigned : tile<i8>

      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      // CHECK: divi %[[c42_i16]], %[[c42_i16]] signed : tile<i16>
      %divi_i16_signed = cuda_tile.divi %c42_i16, %c42_i16 signed : tile<i16>
      // CHECK: divi %[[c42_i16]], %[[c42_i16]] unsigned : tile<i16>
      %divi_i16_unsigned = cuda_tile.divi %c42_i16, %c42_i16 unsigned : tile<i16>

      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      // CHECK: divi %[[c42_i32]], %[[c42_i32]] signed : tile<i32>
      %divi_i32_signed = cuda_tile.divi %c42_i32, %c42_i32 signed : tile<i32>
      // CHECK: divi %[[c42_i32]], %[[c42_i32]] unsigned : tile<i32>
      %divi_i32_unsigned = cuda_tile.divi %c42_i32, %c42_i32 unsigned : tile<i32>

      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
      // CHECK: divi %[[c42_i64]], %[[c42_i64]] signed : tile<i64>
      %divi_i64_signed = cuda_tile.divi %c42_i64, %c42_i64 signed : tile<i64>
      // CHECK: divi %[[c42_i64]], %[[c42_i64]] unsigned : tile<i64>
      %divi_i64_unsigned = cuda_tile.divi %c42_i64, %c42_i64 unsigned : tile<i64>

      // CHECK: %[[t0_i32:.*]] = constant <i32: {{\[\[}}1, 2], [4, 5]]> : tile<2x2xi32>
      %t0_i32 = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>
      // CHECK: divi %[[t0_i32]], %[[t0_i32]] signed : tile<2x2xi32>
      %divi_tensor_signed = cuda_tile.divi %t0_i32, %t0_i32 signed : tile<2x2xi32>
      // CHECK: divi %[[t0_i32]], %[[t0_i32]] unsigned : tile<2x2xi32>
      %divi_tensor_unsigned = cuda_tile.divi %t0_i32, %t0_i32 unsigned : tile<2x2xi32>
  }

entry @floordivi() {
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: divi %[[c1_i1]], %[[c1_i1]] signed rounding<negative_inf> : tile<i1>
    %floordivi_i1 = divi %c1_i1, %c1_i1 signed rounding<negative_inf> : tile<i1>

    // CHECK: %[[s8:.*]] = constant <i8: 42> : tile<i8>
    // CHECK: divi %[[s8]], %[[s8]] signed rounding<negative_inf> : tile<i8>
    %s8 = constant <i8: 42> : !cuda_tile.tile<i8>
    %floordivi_scalar_i8 = divi %s8, %s8 signed rounding<negative_inf> : tile<i8>

    // CHECK: %[[s16:.*]] = constant <i16: 42> : tile<i16>
    // CHECK: divi %[[s16]], %[[s16]] signed rounding<negative_inf> : tile<i16>
    %s16 = constant <i16: 42> : !cuda_tile.tile<i16>
    %floordivi_scalar_i16 = divi %s16, %s16 signed rounding<negative_inf> : tile<i16>

    // CHECK: %[[s32:.*]] = constant <i32: 42> : tile<i32>
    // CHECK: divi %[[s32]], %[[s32]] signed rounding<negative_inf> : tile<i32>
    %s32 = constant <i32: 42> : !cuda_tile.tile<i32>
    %floordivi_scalar_i32 = divi %s32, %s32 signed rounding<negative_inf> : tile<i32>

    // CHECK: %[[s64:.*]] = constant <i64: 42> : tile<i64>
    // CHECK: divi %[[s64]], %[[s64]] signed rounding<negative_inf> : tile<i64>
    %s64 = constant <i64: 42> : !cuda_tile.tile<i64>
    %floordivi_scalar_i64 = divi %s64, %s64 signed rounding<negative_inf> : tile<i64>

    // CHECK: %[[v0:.*]] = constant <i32: {{\[.*\]}}> : tile<4xi32>
    // CHECK: divi %[[v0]], %[[v0]] signed rounding<negative_inf> : tile<4xi32>
    %v0 = constant <i32: [1, 2, 3, 4]> : !cuda_tile.tile<4xi32>
    %floordivi_vector = divi %v0, %v0 signed rounding<negative_inf> : tile<4xi32>

    // CHECK: %[[t0:.*]] = constant <i64: {{\[.*\]}}> : tile<2x2xi64>
    // CHECK: divi %[[t0]], %[[t0]] signed rounding<negative_inf> : tile<2x2xi64>
    %t0 = constant <i64: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi64>
    %floordivi_tensor = divi %t0, %t0 signed rounding<negative_inf> : tile<2x2xi64>
}

  entry @maxi() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: maxi %[[c1_i1]], %[[c1_i1]] signed : tile<i1>
      %maxi_i1_signed = cuda_tile.maxi %c1_i1, %c1_i1 signed : tile<i1>
      // CHECK: maxi %[[c1_i1]], %[[c1_i1]] unsigned : tile<i1>
      %maxi_i1_unsigned = cuda_tile.maxi %c1_i1, %c1_i1 unsigned : tile<i1>

      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      // CHECK: maxi %[[c42_i8]], %[[c42_i8]] signed : tile<i8>
      %maxi_i8_signed = cuda_tile.maxi %c42_i8, %c42_i8 signed : tile<i8>
      // CHECK: maxi %[[c42_i8]], %[[c42_i8]] unsigned : tile<i8>
      %maxi_i8_unsigned = cuda_tile.maxi %c42_i8, %c42_i8 unsigned : tile<i8>

      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      // CHECK: maxi %[[c42_i16]], %[[c42_i16]] signed : tile<i16>
      %maxi_i16_signed = cuda_tile.maxi %c42_i16, %c42_i16 signed : tile<i16>
      // CHECK: maxi %[[c42_i16]], %[[c42_i16]] unsigned : tile<i16>
      %maxi_i16_unsigned = cuda_tile.maxi %c42_i16, %c42_i16 unsigned : tile<i16>

      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      // CHECK: maxi %[[c42_i32]], %[[c42_i32]] signed : tile<i32>
      %maxi_i32_signed = cuda_tile.maxi %c42_i32, %c42_i32 signed : tile<i32>
      // CHECK: maxi %[[c42_i32]], %[[c42_i32]] unsigned : tile<i32>
      %maxi_i32_unsigned = cuda_tile.maxi %c42_i32, %c42_i32 unsigned : tile<i32>

      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
      // CHECK: maxi %[[c42_i64]], %[[c42_i64]] signed : tile<i64>
      %maxi_i64_signed = cuda_tile.maxi %c42_i64, %c42_i64 signed : tile<i64>
      // CHECK: maxi %[[c42_i64]], %[[c42_i64]] unsigned : tile<i64>
      %maxi_i64_unsigned = cuda_tile.maxi %c42_i64, %c42_i64 unsigned : tile<i64>

      // CHECK: %[[c_itensor:.*]] = constant <i32: {{\[\[}}1, 2], [4, 5]]> : tile<2x2xi32>
      %c_itensor = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>
      // CHECK: maxi %[[c_itensor]], %[[c_itensor]] signed : tile<2x2xi32>
      %maxi_tensor_signed = cuda_tile.maxi %c_itensor, %c_itensor signed : tile<2x2xi32>
      // CHECK: maxi %[[c_itensor]], %[[c_itensor]] unsigned : tile<2x2xi32>
      %maxi_tensor_unsigned = cuda_tile.maxi %c_itensor, %c_itensor unsigned : tile<2x2xi32>
  }

  entry @mini() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: mini %[[c1_i1]], %[[c1_i1]] signed : tile<i1>
      %mini_i1_signed = cuda_tile.mini %c1_i1, %c1_i1 signed : tile<i1>
      // CHECK: mini %[[c1_i1]], %[[c1_i1]] unsigned : tile<i1>
      %mini_i1_unsigned = cuda_tile.mini %c1_i1, %c1_i1 unsigned : tile<i1>

      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      // CHECK: mini %[[c42_i8]], %[[c42_i8]] signed : tile<i8>
      %mini_i8_signed = cuda_tile.mini %c42_i8, %c42_i8 signed : tile<i8>
      // CHECK: mini %[[c42_i8]], %[[c42_i8]] unsigned : tile<i8>
      %mini_i8_unsigned = cuda_tile.mini %c42_i8, %c42_i8 unsigned : tile<i8>

      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      // CHECK: mini %[[c42_i16]], %[[c42_i16]] signed : tile<i16>
      %mini_i16_signed = cuda_tile.mini %c42_i16, %c42_i16 signed : tile<i16>
      // CHECK: mini %[[c42_i16]], %[[c42_i16]] unsigned : tile<i16>
      %mini_i16_unsigned = cuda_tile.mini %c42_i16, %c42_i16 unsigned : tile<i16>

      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      // CHECK: mini %[[c42_i32]], %[[c42_i32]] signed : tile<i32>
      %mini_i32_signed = cuda_tile.mini %c42_i32, %c42_i32 signed : tile<i32>
      // CHECK: mini %[[c42_i32]], %[[c42_i32]] unsigned : tile<i32>
      %mini_i32_unsigned = cuda_tile.mini %c42_i32, %c42_i32 unsigned : tile<i32>

      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
      // CHECK: mini %[[c42_i64]], %[[c42_i64]] signed : tile<i64>
      %mini_i64_signed = cuda_tile.mini %c42_i64, %c42_i64 signed : tile<i64>
      // CHECK: mini %[[c42_i64]], %[[c42_i64]] unsigned : tile<i64>
      %mini_i64_unsigned = cuda_tile.mini %c42_i64, %c42_i64 unsigned : tile<i64>

      // CHECK: %[[c_itensor:.*]] = constant <i32: {{\[\[}}1, 2], [4, 5]]> : tile<2x2xi32>
      %c_itensor = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>
      // CHECK: mini %[[c_itensor]], %[[c_itensor]] signed : tile<2x2xi32>
      %mini_tensor_signed = cuda_tile.mini %c_itensor, %c_itensor signed : tile<2x2xi32>
      // CHECK: mini %[[c_itensor]], %[[c_itensor]] unsigned : tile<2x2xi32>
      %mini_tensor_unsigned = cuda_tile.mini %c_itensor, %c_itensor unsigned : tile<2x2xi32>
  }

  entry @muli() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: muli %[[c1_i1]], %[[c1_i1]] : tile<i1>
      %mul_i1 = cuda_tile.muli %c1_i1, %c1_i1 : tile<i1>

      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      // CHECK: muli %[[c42_i8]], %[[c42_i8]] : tile<i8>
      %mul_i8 = cuda_tile.muli %c42_i8, %c42_i8 : tile<i8>

      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      // CHECK: muli %[[c42_i16]], %[[c42_i16]] : tile<i16>
      %mul_i16 = cuda_tile.muli %c42_i16, %c42_i16 : tile<i16>

      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      // CHECK: muli %[[c42_i32]], %[[c42_i32]] : tile<i32>
      %mul_i32 = cuda_tile.muli %c42_i32, %c42_i32 : tile<i32>

      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
      // CHECK: muli %[[c42_i64]], %[[c42_i64]] : tile<i64>
      %mul_i64 = cuda_tile.muli %c42_i64, %c42_i64 : tile<i64>

      // CHECK: %[[c_itensor:.*]] = constant <i32: {{\[\[}}1, 2], [4, 5]]> : tile<2x2xi32>
      %c_itensor = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>
      // CHECK: muli %[[c_itensor]], %[[c_itensor]] : tile<2x2xi32>
      %mul_tensor = cuda_tile.muli %c_itensor, %c_itensor : tile<2x2xi32>
  }

  entry @mulhii() {
      // CHECK: %[[c4_i8:.*]] = constant <i8: 4> : tile<i8>
      %c4_i8 = constant <i8: 4> : !cuda_tile.tile<i8>
      // CHECK: %[[c4_i16:.*]] = constant <i16: 4> : tile<i16>
      %c4_i16 = constant <i16: 4> : !cuda_tile.tile<i16>
      // CHECK: %[[c4_i32:.*]] = constant <i32: 4> : tile<i32>
      %c4_i32 = constant <i32: 4> : !cuda_tile.tile<i32>
      // CHECK: %[[c4_i64:.*]] = constant <i64: 4> : tile<i64>
      %c4_i64 = constant <i64: 4> : !cuda_tile.tile<i64>

      // CHECK: %[[c_i8tensor:.*]] = constant <i8: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi8>
      %c_i8tensor = constant <i8: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi8>
      // CHECK: %[[c_i16tensor:.*]] = constant <i16: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi16>
      %c_i16tensor = constant <i16: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi16>
      // CHECK: %[[c_i32tensor:.*]] = constant <i32: {{\[\[}}1, 2], [4, 5]]> : tile<2x2xi32>
      %c_i32tensor = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>
      // CHECK: %[[c_i64tensor:.*]] = constant <i64: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi64>
      %c_i64tensor = constant <i64: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi64>

      // CHECK: mulhii %[[c4_i8]], %[[c4_i8]] : tile<i8>
      %mulhii_scalar_i8 = cuda_tile.mulhii %c4_i8, %c4_i8 : !cuda_tile.tile<i8>
      // CHECK: mulhii %[[c4_i16]], %[[c4_i16]] : tile<i16>
      %mulhii_scalar_i16 = cuda_tile.mulhii %c4_i16, %c4_i16 : !cuda_tile.tile<i16>
      // CHECK: mulhii %[[c4_i32]], %[[c4_i32]] : tile<i32>
      %mulhii_scalar_i32 = cuda_tile.mulhii %c4_i32, %c4_i32 : !cuda_tile.tile<i32>
      // CHECK: mulhii %[[c4_i64]], %[[c4_i64]] : tile<i64>
      %mulhii_scalar_i64 = cuda_tile.mulhii %c4_i64, %c4_i64 : !cuda_tile.tile<i64>

      // CHECK: mulhii %[[c_i8tensor]], %[[c_i8tensor]] : tile<2x2xi8>
      %mulhii_tensor_i8 = cuda_tile.mulhii %c_i8tensor, %c_i8tensor : !cuda_tile.tile<2x2xi8>
      // CHECK: mulhii %[[c_i16tensor]], %[[c_i16tensor]] : tile<2x2xi16>
      %mulhii_tensor_i16 = cuda_tile.mulhii %c_i16tensor, %c_i16tensor : !cuda_tile.tile<2x2xi16>
      // CHECK: mulhii %[[c_i32tensor]], %[[c_i32tensor]] : tile<2x2xi32>
      %mulhii_tensor_i32 = cuda_tile.mulhii %c_i32tensor, %c_i32tensor : tile<2x2xi32>
      // CHECK: mulhii %[[c_i64tensor]], %[[c_i64tensor]] : tile<2x2xi64>
      %mulhii_tensor_i64 = cuda_tile.mulhii %c_i64tensor, %c_i64tensor : tile<2x2xi64>
  }

  entry @subi() {
      // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
      %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
      %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
      // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
      %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
      // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
      %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
      // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
      %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

      // CHECK: %[[c_i1tensor:.*]] = constant <i1: {{\[\[}}true, false], [true, true]]> : tile<2x2xi1>
      %c_i1tensor = constant <i1: [[true, false], [true, true]]> : !cuda_tile.tile<2x2xi1>
      // CHECK: %[[c_i8tensor:.*]] = constant <i8: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi8>
      %c_i8tensor = constant <i8: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi8>
      // CHECK: %[[c_i16tensor:.*]] = constant <i16: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi16>
      %c_i16tensor = constant <i16: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi16>
      // CHECK: %[[c_i32tensor:.*]] = constant <i32: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi32>
      %c_i32tensor = constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>
      // CHECK: %[[c_i64tensor:.*]] = constant <i64: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi64>
      %c_i64tensor = constant <i64: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi64>

      // CHECK: subi %[[c1_i1]], %[[c1_i1]] : tile<i1>
      %sub_scalar_i1 = cuda_tile.subi %c1_i1, %c1_i1 : tile<i1>
      // CHECK: subi %[[c42_i8]], %[[c42_i8]] : tile<i8>
      %sub_scalar_i8 = cuda_tile.subi %c42_i8, %c42_i8 : tile<i8>
      // CHECK: subi %[[c42_i16]], %[[c42_i16]] : tile<i16>
      %sub_scalar_i16 = cuda_tile.subi %c42_i16, %c42_i16 : tile<i16>
      // CHECK: subi %[[c42_i32]], %[[c42_i32]] : tile<i32>
      %sub_scalar_i32 = cuda_tile.subi %c42_i32, %c42_i32 : tile<i32>
      // CHECK: subi %[[c42_i64]], %[[c42_i64]] : tile<i64>
      %sub_scalar_i64 = cuda_tile.subi %c42_i64, %c42_i64 : tile<i64>

      // CHECK: subi %[[c_i1tensor]], %[[c_i1tensor]] : tile<2x2xi1>
      %sub_tensor_i1 = cuda_tile.subi %c_i1tensor, %c_i1tensor : tile<2x2xi1>
      // CHECK: subi %[[c_i8tensor]], %[[c_i8tensor]] : tile<2x2xi8>
      %sub_tensor_i8 = cuda_tile.subi %c_i8tensor, %c_i8tensor : tile<2x2xi8>
      // CHECK: subi %[[c_i16tensor]], %[[c_i16tensor]] : tile<2x2xi16>
      %sub_tensor_i16 = cuda_tile.subi %c_i16tensor, %c_i16tensor : tile<2x2xi16>
      // CHECK: subi %[[c_i32tensor]], %[[c_i32tensor]] : tile<2x2xi32>
      %sub_tensor_i32 = cuda_tile.subi %c_i32tensor, %c_i32tensor : tile<2x2xi32>
      // CHECK: subi %[[c_i64tensor]], %[[c_i64tensor]] : tile<2x2xi64>
      %sub_tensor_i64 = cuda_tile.subi %c_i64tensor, %c_i64tensor : tile<2x2xi64>
  }

  entry @andi() {
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
    %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
    %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
    %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
    %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // CHECK: andi %[[c1_i1]], %[[c1_i1]] : tile<i1>
    %res_i1 = andi %c1_i1, %c1_i1 : tile<i1>
    // CHECK: andi %[[c42_i8]], %[[c42_i8]] : tile<i8>
    %res_i8 = andi %c42_i8, %c42_i8 : tile<i8>
    // CHECK: andi %[[c42_i16]], %[[c42_i16]] : tile<i16>
    %res_i16 = andi %c42_i16, %c42_i16 : tile<i16>
    // CHECK: andi %[[c42_i32]], %[[c42_i32]] : tile<i32>
    %res_i32 = andi %c42_i32, %c42_i32 : tile<i32>
    // CHECK: andi %[[c42_i64]], %[[c42_i64]] : tile<i64>
    %res_i64 = andi %c42_i64, %c42_i64 : tile<i64>
  }

  entry @ori() {
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
    %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
    %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
    %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
    %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // CHECK: ori %[[c1_i1]], %[[c1_i1]] : tile<i1>
    %res_i1 = ori %c1_i1, %c1_i1 : tile<i1>
    // CHECK: ori %[[c42_i8]], %[[c42_i8]] : tile<i8>
    %res_i8 = ori %c42_i8, %c42_i8 : tile<i8>
    // CHECK: ori %[[c42_i16]], %[[c42_i16]] : tile<i16>
    %res_i16 = ori %c42_i16, %c42_i16 : tile<i16>
    // CHECK: ori %[[c42_i32]], %[[c42_i32]] : tile<i32>
    %res_i32 = ori %c42_i32, %c42_i32 : tile<i32>
    // CHECK: ori %[[c42_i64]], %[[c42_i64]] : tile<i64>
    %res_i64 = ori %c42_i64, %c42_i64 : tile<i64>
  }

  entry @shli() {
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
    %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
    %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
    %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
    %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // CHECK: shli %[[c1_i1]], %[[c1_i1]] : tile<i1>
    %res_i1 = shli %c1_i1, %c1_i1 : tile<i1>
    // CHECK: shli %[[c42_i8]], %[[c42_i8]] : tile<i8>
    %res_i8 = shli %c42_i8, %c42_i8 : tile<i8>
    // CHECK: shli %[[c42_i16]], %[[c42_i16]] : tile<i16>
    %res_i16 = shli %c42_i16, %c42_i16 : tile<i16>
    // CHECK: shli %[[c42_i32]], %[[c42_i32]] : tile<i32>
    %res_i32 = shli %c42_i32, %c42_i32 : tile<i32>
    // CHECK: shli %[[c42_i64]], %[[c42_i64]] : tile<i64>
    %res_i64 = shli %c42_i64, %c42_i64 : tile<i64>
  }

  entry @shri_signed() {
    // CHECK-LABEL: entry @shri_signed
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
    %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
    %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
    %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
    %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // CHECK: shri %[[c1_i1]], %[[c1_i1]] signed : tile<i1>
    %res_i1 = shri %c1_i1, %c1_i1 signed : tile<i1>
    // CHECK: shri %[[c42_i8]], %[[c42_i8]] signed : tile<i8>
    %res_i8 = shri %c42_i8, %c42_i8 signed : tile<i8>
    // CHECK: shri %[[c42_i16]], %[[c42_i16]] signed : tile<i16>
    %res_i16 = shri %c42_i16, %c42_i16 signed : tile<i16>
    // CHECK: shri %[[c42_i32]], %[[c42_i32]] signed : tile<i32>
    %res_i32 = shri %c42_i32, %c42_i32 signed : tile<i32>
    // CHECK: shri %[[c42_i64]], %[[c42_i64]] signed : tile<i64>
    %res_i64 = shri %c42_i64, %c42_i64 signed : tile<i64>
  }

  entry @shri_unsigned() {
    // CHECK-LABEL: entry @shri_unsigned
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
    %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
    %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
    %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
    %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // CHECK: shri %[[c1_i1]], %[[c1_i1]] unsigned : tile<i1>
    %res_i1 = shri %c1_i1, %c1_i1 unsigned : tile<i1>
    // CHECK: shri %[[c42_i8]], %[[c42_i8]] unsigned : tile<i8>
    %res_i8 = shri %c42_i8, %c42_i8 unsigned : tile<i8>
    // CHECK: shri %[[c42_i16]], %[[c42_i16]] unsigned : tile<i16>
    %res_i16 = shri %c42_i16, %c42_i16 unsigned : tile<i16>
    // CHECK: shri %[[c42_i32]], %[[c42_i32]] unsigned : tile<i32>
    %res_i32 = shri %c42_i32, %c42_i32 unsigned : tile<i32>
    // CHECK: shri %[[c42_i64]], %[[c42_i64]] unsigned : tile<i64>
    %res_i64 = shri %c42_i64, %c42_i64 unsigned : tile<i64>
  }

  entry @xori() {
    // CHECK-LABEL: entry @xori
    // CHECK: %[[c1_i1:.*]] = constant <i1: true> : tile<i1>
    %c1_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c42_i8:.*]] = constant <i8: 42> : tile<i8>
    %c42_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c42_i16:.*]] = constant <i16: 42> : tile<i16>
    %c42_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c42_i32:.*]] = constant <i32: 42> : tile<i32>
    %c42_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c42_i64:.*]] = constant <i64: 42> : tile<i64>
    %c42_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // CHECK: xori %[[c1_i1]], %[[c1_i1]] : tile<i1>
    %res_i1 = xori %c1_i1, %c1_i1 : tile<i1>
    // CHECK: xori %[[c42_i8]], %[[c42_i8]] : tile<i8>
    %res_i8 = xori %c42_i8, %c42_i8 : tile<i8>
    // CHECK: xori %[[c42_i16]], %[[c42_i16]] : tile<i16>
    %res_i16 = xori %c42_i16, %c42_i16 : tile<i16>
    // CHECK: xori %[[c42_i32]], %[[c42_i32]] : tile<i32>
    %res_i32 = xori %c42_i32, %c42_i32 : tile<i32>
    // CHECK: xori %[[c42_i64]], %[[c42_i64]] : tile<i64>
    %res_i64 = xori %c42_i64, %c42_i64 : tile<i64>
  }

  entry @xori_tensor() {
    // CHECK-LABEL: entry @xori_tensor
    // CHECK: %[[c_itensor:.*]] = constant <i32: {{\[}}[1, 2], [4, 5]]> : tile<2x2xi32>
    %c_itensor = constant <i32: [[1, 2], [4, 5]]> : !cuda_tile.tile<2x2xi32>

    // CHECK: xori %[[c_itensor]], %[[c_itensor]] : tile<2x2xi32>
    %res_itensor = xori %c_itensor, %c_itensor : tile<2x2xi32>
  }

//===----------------------------------------------------------------------===//
// Floating Point Arithmetic Operations
//===----------------------------------------------------------------------===//

  entry @addf() {
    // CHECK-LABEL: entry @addf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: addf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %add_f16 = cuda_tile.addf %c42_f16, %c42_f16 rounding<nearest_even> : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: addf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %add_bf16 = cuda_tile.addf %c42_bf16, %c42_bf16 rounding<nearest_even> : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: addf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %add_f32 = cuda_tile.addf %c42_f32, %c42_f32 rounding<nearest_even> : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: addf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %add_f64 = cuda_tile.addf %c42_f64, %c42_f64 rounding<nearest_even> : tile<f64>
  }
  
  entry @addf_tensor() {
    // CHECK-LABEL: entry @addf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: addf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = cuda_tile.addf %c_f16tensor, %c_f16tensor rounding<nearest_even> : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: addf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = cuda_tile.addf %c_bf16tensor, %c_bf16tensor rounding<nearest_even> : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: addf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = cuda_tile.addf %c_f32tensor, %c_f32tensor rounding<nearest_even> : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: addf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = cuda_tile.addf %c_f64tensor, %c_f64tensor rounding<nearest_even> : tile<2x2xf64>
  }

  entry @absf() {
    // CHECK-LABEL: entry @absf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: absf %[[c42_f16]] : tile<f16>
    %abs_f16 = absf %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: absf %[[c42_bf16]] : tile<bf16>
    %abs_bf16 = absf %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: absf %[[c42_f32]] : tile<f32>
    %abs_f32 = absf %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: absf %[[c42_f64]] : tile<f64>
    %abs_f64 = absf %c42_f64 : tile<f64>
  }

  entry @absf_tensor() {
    // CHECK-LABEL: entry @absf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: absf %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = absf %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: absf %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = absf %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: absf %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = absf %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: absf %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = absf %c_f64tensor : tile<2x2xf64>
  }

  entry @cos() {
    // CHECK-LABEL: entry @cos
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: cos %[[c42_f16]] : tile<f16>
    %cos_f16 = cos %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: cos %[[c42_bf16]] : tile<bf16>
    %cos_bf16 = cos %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: cos %[[c42_f32]] : tile<f32>
    %cos_f32 = cos %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: cos %[[c42_f64]] : tile<f64>
    %cos_f64 = cos %c42_f64 : tile<f64>
  }

  entry @cos_tensor() {
    // CHECK-LABEL: entry @cos_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: cos %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = cos %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: cos %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = cos %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: cos %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = cos %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: cos %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = cos %c_f64tensor : tile<2x2xf64>
  }

  entry @cosh() {
    // CHECK-LABEL: entry @cosh
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: cosh %[[c42_f16]] : tile<f16>
    %cosh_f16 = cosh %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: cosh %[[c42_bf16]] : tile<bf16>
    %cosh_bf16 = cosh %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: cosh %[[c42_f32]] : tile<f32>
    %cosh_f32 = cosh %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: cosh %[[c42_f64]] : tile<f64>
    %cosh_f64 = cosh %c42_f64 : tile<f64>
  }

  entry @cosh_tensor() {
    // CHECK-LABEL: entry @cosh_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: cosh %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = cosh %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: cosh %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = cosh %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: cosh %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = cosh %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: cosh %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = cosh %c_f64tensor : tile<2x2xf64>
  }

  entry @ceil() {
    // CHECK-LABEL: entry @ceil
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: ceil %[[c42_f16]] : tile<f16>
    %ceil_f16 = ceil %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: ceil %[[c42_bf16]] : tile<bf16>
    %ceil_bf16 = ceil %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: ceil %[[c42_f32]] : tile<f32>
    %ceil_f32 = ceil %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: ceil %[[c42_f64]] : tile<f64>
    %ceil_f64 = ceil %c42_f64 : tile<f64>
  }

  entry @ceil_tensor() {
    // CHECK-LABEL: entry @ceil_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: ceil %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = ceil %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: ceil %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = ceil %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: ceil %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = ceil %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: ceil %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = ceil %c_f64tensor : tile<2x2xf64>
  }

  entry @cmpf() {
    // CHECK-LABEL: entry @cmpf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: cmpf less_than ordered %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %cmp_f16 = cmpf less_than ordered %c42_f16, %c42_f16 : tile<f16> -> tile<i1> 

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: cmpf less_than ordered %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %cmp_bf16 = cmpf less_than ordered %c42_bf16, %c42_bf16 : tile<bf16> -> tile<i1> 

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: cmpf less_than ordered %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %cmp_f32 = cmpf less_than ordered %c42_f32, %c42_f32 : tile<f32> -> tile<i1> 

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: cmpf less_than ordered %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %cmp_f64 = cmpf less_than ordered %c42_f64, %c42_f64 : tile<f64> -> tile<i1> 
  }

  entry @cmpf_tensor() {
    // CHECK-LABEL: entry @cmpf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: cmpf less_than ordered %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = cmpf less_than ordered %c_f16tensor, %c_f16tensor : tile<2x2xf16> -> tile<2x2xi1>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: cmpf less_than ordered %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = cmpf less_than ordered %c_bf16tensor, %c_bf16tensor : tile<2x2xbf16> -> tile<2x2xi1>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: cmpf less_than ordered %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = cmpf less_than ordered %c_f32tensor, %c_f32tensor : tile<2x2xf32> -> tile<2x2xi1>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: cmpf less_than ordered %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = cmpf less_than ordered %c_f64tensor, %c_f64tensor : tile<2x2xf64> -> tile<2x2xi1>
  }

  entry @divf() {
    // CHECK-LABEL: entry @divf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: divf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %div_f16 = divf %c42_f16, %c42_f16 rounding<nearest_even> : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: divf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %div_bf16 = divf %c42_bf16, %c42_bf16 rounding<nearest_even> : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: divf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %div_f32 = divf %c42_f32, %c42_f32 rounding<nearest_even> : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: divf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %div_f64 = divf %c42_f64, %c42_f64 rounding<nearest_even> : tile<f64>
  }

  entry @divf_tensor() {
    // CHECK-LABEL: entry @divf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: divf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = divf %c_f16tensor, %c_f16tensor rounding<nearest_even> : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: divf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = divf %c_bf16tensor, %c_bf16tensor rounding<nearest_even> : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: divf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = divf %c_f32tensor, %c_f32tensor rounding<nearest_even> : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: divf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = divf %c_f64tensor, %c_f64tensor rounding<nearest_even> : tile<2x2xf64>
  }

  entry @exp2() {
    // CHECK-LABEL: entry @exp2
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: exp2 %[[c42_f16]] : tile<f16>
    %exp2_f16 = exp2 %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: exp2 %[[c42_bf16]] : tile<bf16>
    %exp2_bf16 = exp2 %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: exp2 %[[c42_f32]] : tile<f32>
    %exp2_f32 = exp2 %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: exp2 %[[c42_f64]] : tile<f64>
    %exp2_f64 = exp2 %c42_f64 : tile<f64>
  }

  entry @exp2_tensor() {
    // CHECK-LABEL: entry @exp2_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: exp2 %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = exp2 %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: exp2 %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = exp2 %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: exp2 %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = exp2 %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: exp2 %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = exp2 %c_f64tensor : tile<2x2xf64>
  }

  entry @floor() {
    // CHECK-LABEL: entry @floor
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: floor %[[c42_f16]] : tile<f16>
    %floor_f16 = floor %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: floor %[[c42_bf16]] : tile<bf16>
    %floor_bf16 = floor %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: floor %[[c42_f32]] : tile<f32>
    %floor_f32 = floor %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: floor %[[c42_f64]] : tile<f64>
    %floor_f64 = floor %c42_f64 : tile<f64>
  }

  entry @floor_tensor() {
    // CHECK-LABEL: entry @floor_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: floor %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = floor %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: floor %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = floor %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: floor %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = floor %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: floor %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = floor %c_f64tensor : tile<2x2xf64>
  }

  entry @log() {
    // CHECK-LABEL: entry @log
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: log %[[c42_f16]] : tile<f16>
    %log_f16 = log %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: log %[[c42_bf16]] : tile<bf16>
    %log_bf16 = log %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: log %[[c42_f32]] : tile<f32>
    %log_f32 = log %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: log %[[c42_f64]] : tile<f64>
    %log_f64 = log %c42_f64 : tile<f64>
  }

  entry @log_tensor() {
    // CHECK-LABEL: entry @log_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: log %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = log %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: log %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = log %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: log %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = log %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: log %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = log %c_f64tensor : tile<2x2xf64>
  }

  entry @log2() {
    // CHECK-LABEL: entry @log2
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: log2 %[[c42_f16]] : tile<f16>
    %log2_f16 = log2 %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: log2 %[[c42_bf16]] : tile<bf16>
    %log2_bf16 = log2 %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: log2 %[[c42_f32]] : tile<f32>
    %log2_f32 = log2 %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: log2 %[[c42_f64]] : tile<f64>
    %log2_f64 = log2 %c42_f64 : tile<f64>
  }

  entry @log2_tensor() {
    // CHECK-LABEL: entry @log2_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: log2 %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = log2 %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: log2 %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = log2 %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: log2 %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = log2 %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: log2 %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = log2 %c_f64tensor : tile<2x2xf64>
  }

  entry @maxf() {
    // CHECK-LABEL: entry @maxf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: maxf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %max_f16 = maxf %c42_f16, %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: maxf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %max_bf16 = maxf %c42_bf16, %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: maxf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %max_f32 = maxf %c42_f32, %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: maxf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %max_f64 = maxf %c42_f64, %c42_f64 : tile<f64>
  }

  entry @maxf_tensor() {
    // CHECK-LABEL: entry @maxf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: maxf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = maxf %c_f16tensor, %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: maxf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = maxf %c_bf16tensor, %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: maxf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = maxf %c_f32tensor, %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: maxf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = maxf %c_f64tensor, %c_f64tensor : tile<2x2xf64>
  }

  entry @minf() {
    // CHECK-LABEL: entry @minf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: minf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %min_f16 = minf %c42_f16, %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: minf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %min_bf16 = minf %c42_bf16, %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: minf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %min_f32 = minf %c42_f32, %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: minf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %min_f64 = minf %c42_f64, %c42_f64 : tile<f64>
  }

  entry @minf_tensor() {
    // CHECK-LABEL: entry @minf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: minf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = minf %c_f16tensor, %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: minf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = minf %c_bf16tensor, %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: minf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = minf %c_f32tensor, %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: minf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = minf %c_f64tensor, %c_f64tensor : tile<2x2xf64>
  }

  entry @mulf() {
    // CHECK-LABEL: entry @mulf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: mulf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %mul_f16 = mulf %c42_f16, %c42_f16 rounding<nearest_even> : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: mulf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %mul_bf16 = mulf %c42_bf16, %c42_bf16 rounding<nearest_even> : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: mulf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %mul_f32 = mulf %c42_f32, %c42_f32 rounding<nearest_even> : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: mulf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %mul_f64 = mulf %c42_f64, %c42_f64 rounding<nearest_even> : tile<f64>
  }

  entry @mulf_tensor() {
    // CHECK-LABEL: entry @mulf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: mulf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = mulf %c_f16tensor, %c_f16tensor rounding<nearest_even> : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: mulf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = mulf %c_bf16tensor, %c_bf16tensor rounding<nearest_even> : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: mulf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = mulf %c_f32tensor, %c_f32tensor rounding<nearest_even> : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: mulf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = mulf %c_f64tensor, %c_f64tensor rounding<nearest_even> : tile<2x2xf64>
  }

  entry @negf() {
    // CHECK-LABEL: entry @negf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: negf %[[c42_f16]] : tile<f16>
    %neg_f16 = negf %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: negf %[[c42_bf16]] : tile<bf16>
    %neg_bf16 = negf %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: negf %[[c42_f32]] : tile<f32>
    %neg_f32 = negf %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: negf %[[c42_f64]] : tile<f64>
    %neg_f64 = negf %c42_f64 : tile<f64>
  }

  entry @negf_tensor() {
    // CHECK-LABEL: entry @negf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: negf %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = negf %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: negf %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = negf %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: negf %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = negf %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: negf %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = negf %c_f64tensor : tile<2x2xf64>
  }

  entry @powf() {
    // CHECK-LABEL: entry @powf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: pow %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %pow_f16 = pow %c42_f16, %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: pow %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %pow_bf16 = pow %c42_bf16, %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: pow %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %pow_f32 = pow %c42_f32, %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: pow %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %pow_f64 = pow %c42_f64, %c42_f64 : tile<f64>
  }

  entry @powf_tensor() {
    // CHECK-LABEL: entry @powf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: pow %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = pow %c_f16tensor, %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: pow %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = pow %c_bf16tensor, %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: pow %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = pow %c_f32tensor, %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: pow %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = pow %c_f64tensor, %c_f64tensor : tile<2x2xf64>
  }

  entry @rsqrtf() {
    // CHECK-LABEL: entry @rsqrtf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: rsqrt %[[c42_f16]] : tile<f16>
    %rsqrt_f16 = rsqrt %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: rsqrt %[[c42_bf16]] : tile<bf16>
    %rsqrt_bf16 = rsqrt %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: rsqrt %[[c42_f32]] : tile<f32>
    %rsqrt_f32 = rsqrt %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: rsqrt %[[c42_f64]] : tile<f64>
    %rsqrt_f64 = rsqrt %c42_f64 : tile<f64>
  }

  entry @rsqrtf_tensor() {
    // CHECK-LABEL: entry @rsqrtf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: rsqrt %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = rsqrt %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: rsqrt %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = rsqrt %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: rsqrt %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = rsqrt %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: rsqrt %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = rsqrt %c_f64tensor : tile<2x2xf64>
  }

  entry @remf() {
    // CHECK-LABEL: entry @remf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: remf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %rem_f16 = remf %c42_f16, %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: remf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %rem_bf16 = remf %c42_bf16, %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: remf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %rem_f32 = remf %c42_f32, %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: remf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %rem_f64 = remf %c42_f64, %c42_f64 : tile<f64>
  }

  entry @remf_tensor() {
    // CHECK-LABEL: entry @remf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: remf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = remf %c_f16tensor, %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: remf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = remf %c_bf16tensor, %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: remf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = remf %c_f32tensor, %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: remf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = remf %c_f64tensor, %c_f64tensor : tile<2x2xf64>
  }

  entry @sin() {
    // CHECK-LABEL: entry @sin
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: sin %[[c42_f16]] : tile<f16>
    %sin_f16 = sin %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: sin %[[c42_bf16]] : tile<bf16>
    %sin_bf16 = sin %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: sin %[[c42_f32]] : tile<f32>
    %sin_f32 = sin %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: sin %[[c42_f64]] : tile<f64>
    %sin_f64 = sin %c42_f64 : tile<f64>
  }

  entry @sin_tensor() {
    // CHECK-LABEL: entry @sin_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: sin %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = sin %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: sin %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = sin %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: sin %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = sin %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: sin %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = sin %c_f64tensor : tile<2x2xf64>
  }

  entry @sinh() {
    // CHECK-LABEL: entry @sinh
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: sinh %[[c42_f16]] : tile<f16>
    %sinh_f16 = sinh %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: sinh %[[c42_bf16]] : tile<bf16>
    %sinh_bf16 = sinh %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: sinh %[[c42_f32]] : tile<f32>
    %sinh_f32 = sinh %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: sinh %[[c42_f64]] : tile<f64>
    %sinh_f64 = sinh %c42_f64 : tile<f64>
  }

  entry @sinh_tensor() {
    // CHECK-LABEL: entry @sinh_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: sinh %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = sinh %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: sinh %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = sinh %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: sinh %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = sinh %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: sinh %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = sinh %c_f64tensor : tile<2x2xf64>
  }

  entry @sqrt() {
    // CHECK-LABEL: entry @sqrt
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: sqrt %[[c42_f16]] : tile<f16>
    %sqrt_f16 = sqrt %c42_f16 rounding<nearest_even> : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: sqrt %[[c42_bf16]] : tile<bf16>
    %sqrt_bf16 = sqrt %c42_bf16 rounding<nearest_even> : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: sqrt %[[c42_f32]] : tile<f32>
    %sqrt_f32 = sqrt %c42_f32 rounding<nearest_even> : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: sqrt %[[c42_f64]] : tile<f64>
    %sqrt_f64 = sqrt %c42_f64 rounding<nearest_even> : tile<f64>
  }

  entry @sqrt_tensor() {
    // CHECK-LABEL: entry @sqrt_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: sqrt %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = sqrt %c_f16tensor rounding<nearest_even> : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: sqrt %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = sqrt %c_bf16tensor rounding<nearest_even> : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: sqrt %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = sqrt %c_f32tensor rounding<nearest_even> : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: sqrt %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = sqrt %c_f64tensor rounding<nearest_even> : tile<2x2xf64>
  }

  entry @subf() {
    // CHECK-LABEL: entry @subf
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: subf %[[c42_f16]], %[[c42_f16]] : tile<f16>
    %sub_f16 = subf %c42_f16, %c42_f16 rounding<nearest_even> : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: subf %[[c42_bf16]], %[[c42_bf16]] : tile<bf16>
    %sub_bf16 = subf %c42_bf16, %c42_bf16 rounding<nearest_even> : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: subf %[[c42_f32]], %[[c42_f32]] : tile<f32>
    %sub_f32 = subf %c42_f32, %c42_f32 rounding<nearest_even> : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: subf %[[c42_f64]], %[[c42_f64]] : tile<f64>
    %sub_f64 = subf %c42_f64, %c42_f64 rounding<nearest_even> : tile<f64>
  }

  entry @subf_tensor() {
    // CHECK-LABEL: entry @subf_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: subf %[[c_f16tensor]], %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = subf %c_f16tensor, %c_f16tensor rounding<nearest_even> : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: subf %[[c_bf16tensor]], %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = subf %c_bf16tensor, %c_bf16tensor rounding<nearest_even> : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: subf %[[c_f32tensor]], %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = subf %c_f32tensor, %c_f32tensor rounding<nearest_even> : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: subf %[[c_f64tensor]], %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = subf %c_f64tensor, %c_f64tensor rounding<nearest_even> : tile<2x2xf64>
  }

  entry @tan() {
    // CHECK-LABEL: entry @tan
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: tan %[[c42_f16]] : tile<f16>
    %tan_f16 = tan %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: tan %[[c42_bf16]] : tile<bf16>
    %tan_bf16 = tan %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: tan %[[c42_f32]] : tile<f32>
    %tan_f32 = tan %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: tan %[[c42_f64]] : tile<f64>
    %tan_f64 = tan %c42_f64 : tile<f64>
  }

  entry @tan_tensor() {
    // CHECK-LABEL: entry @tan_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: tan %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = tan %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: tan %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = tan %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: tan %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = tan %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: tan %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = tan %c_f64tensor : tile<2x2xf64>
  }

  entry @tanh() {
    // CHECK-LABEL: entry @tanh
    // CHECK: %[[c42_f16:.*]] = constant <f16: 4.200000e+01> : tile<f16>
    %c42_f16 = constant <f16: 42.000000e+00> : !cuda_tile.tile<f16>
    // CHECK: tanh %[[c42_f16]] : tile<f16>
    %tanh_f16 = tanh %c42_f16 : tile<f16>

    // CHECK: %[[c42_bf16:.*]] = constant <bf16: 4.200000e+01> : tile<bf16>
    %c42_bf16 = constant <bf16: 42.000000e+00> : !cuda_tile.tile<bf16>
    // CHECK: tanh %[[c42_bf16]] : tile<bf16>
    %tanh_bf16 = tanh %c42_bf16 : tile<bf16>

    // CHECK: %[[c42_f32:.*]] = constant <f32: 4.200000e+01> : tile<f32>
    %c42_f32 = constant <f32: 42.000000e+00> : !cuda_tile.tile<f32>
    // CHECK: tanh %[[c42_f32]] : tile<f32>
    %tanh_f32 = tanh %c42_f32 : tile<f32>

    // CHECK: %[[c42_f64:.*]] = constant <f64: 4.200000e+01> : tile<f64>
    %c42_f64 = constant <f64: 42.000000e+00> : !cuda_tile.tile<f64>
    // CHECK: tanh %[[c42_f64]] : tile<f64>
    %tanh_f64 = tanh %c42_f64 : tile<f64>
  }

  entry @tanh_tensor() {
    // CHECK-LABEL: entry @tanh_tensor
    // CHECK: %[[c_f16tensor:.*]] = constant <f16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf16>
    %c_f16tensor = constant <f16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: tanh %[[c_f16tensor]] : tile<2x2xf16>
    %res_f16tensor = tanh %c_f16tensor : tile<2x2xf16>

    // CHECK: %[[c_bf16tensor:.*]] = constant <bf16: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xbf16>
    %c_bf16tensor = constant <bf16: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: tanh %[[c_bf16tensor]] : tile<2x2xbf16>
    %res_bf16tensor = tanh %c_bf16tensor : tile<2x2xbf16>

    // CHECK: %[[c_f32tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
    %c_f32tensor = constant <f32: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: tanh %[[c_f32tensor]] : tile<2x2xf32>
    %res_f32tensor = tanh %c_f32tensor : tile<2x2xf32>

    // CHECK: %[[c_f64tensor:.*]] = constant <f64: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf64>
    %c_f64tensor = constant <f64: [[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : !cuda_tile.tile<2x2xf64>
    // CHECK: tanh %[[c_f64tensor]] : tile<2x2xf64>
    %res_f64tensor = tanh %c_f64tensor : tile<2x2xf64>
  }
}
