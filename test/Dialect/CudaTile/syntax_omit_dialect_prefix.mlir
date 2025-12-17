// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @constant {
  entry @constant() {
    // === Basic Integer Types ===
    // CHECK: %{{.*}} = constant <i8: 127> : tile<i8>
    %i8_scalar = constant <i8: 127> : tile<i8>
    // CHECK: %{{.*}} = constant <i8: -128> : tile<i8>
    %i8_negative = constant <i8: -128> : tile<i8>
    // CHECK: %{{.*}} = constant <i16: 32767> : tile<i16>
    %i16_scalar = constant <i16: 32767> : tile<i16>
    // CHECK: %{{.*}} = constant <i16: -32768> : tile<i16>
    %i16_negative = constant <i16: -32768> : tile<i16>
    // CHECK: %{{.*}} = constant <i32: 1> : tile<i32>
    %i32_positive_one = constant <i32: 1> : tile<i32>
    // CHECK: %{{.*}} = constant <i32: -1> : tile<i32>
    %i32_negative_one = constant <i32: -1> : tile<i32>
    // CHECK: %{{.*}} = constant <i64: 9223372036854775807> : tile<i64>
    %i64_scalar = constant <i64: 9223372036854775807> : tile<i64>
    // CHECK: %{{.*}} = constant <i64: -9223372036854775808> : tile<i64>
    %i64_negative = constant <i64: -9223372036854775808> : tile<i64>

    // === Float Types ===
    // CHECK: %{{.*}} = constant <f16: 1.500000e+00> : tile<f16>
    %f16_scalar = constant <f16: 1.5> : tile<f16>
    // CHECK: %{{.*}} = constant <f16: -3.140630e+00> : tile<f16>
    %f16_negative = constant <f16: -3.14159> : tile<f16>
    // CHECK: %{{.*}} = constant <f32: 1.000000e+00> : tile<f32>
    %f32_positive_one = constant <f32: 1.0> : tile<f32>
    // CHECK: %{{.*}} = constant <f32: -1.000000e+00> : tile<f32>
    %f32_negative_one = constant <f32: -1.0> : tile<f32>
    // CHECK: %{{.*}} = constant <f64: 2.7182818284590451> : tile<f64>
    %f64_scalar = constant <f64: 2.718281828459045> : tile<f64>
    // CHECK: %{{.*}} = constant <f64: -1.4142135623730951> : tile<f64>
    %f64_negative = constant <f64: -1.4142135623730951> : tile<f64>

    // === Hex Literals ===
    // CHECK: %{{.*}} = constant <i32: 2147483647> : tile<i32>
    %i32_hex = constant <i32: 0x7FFFFFFF> : tile<i32>
    // CHECK: %{{.*}} = constant <i32: -2147483648> : tile<i32>
    %i32_hex_negative = constant <i32: 0x80000000> : tile<i32>
    // CHECK: %{{.*}} = constant <i64: 9223372036854775807> : tile<i64>
    %i64_hex = constant <i64: 0x7FFFFFFFFFFFFFFF> : tile<i64>
    // CHECK: %{{.*}} = constant <f32: 0x7F800000> : tile<f32>
    %f32_positive_inf = constant <f32: 0x7F800000> : tile<f32>
    // CHECK: %{{.*}} = constant <f32: 0xFF800000> : tile<f32>
    %f32_negative_inf = constant <f32: 0xFF800000> : tile<f32>
    // CHECK: %{{.*}} = constant <f32: 0x7FC00000> : tile<f32>
    %f32_nan = constant <f32: 0x7FC00000> : tile<f32>
    // CHECK: %{{.*}} = constant <f64: 0x7FF0000000000000> : tile<f64>
    %f64_positive_inf = constant <f64: 0x7FF0000000000000> : tile<f64>

    // === Zero Values ===
    // CHECK: %{{.*}} = constant <i32: 0> : tile<i32>
    %i32_zero = constant <i32: 0> : tile<i32>
    // CHECK: %{{.*}} = constant <f32: 0.000000e+00> : tile<f32>
    %f32_zero = constant <f32: 0.0> : tile<f32>
    // CHECK: %{{.*}} = constant <f32: -0.000000e+00> : tile<f32>
    %f32_negative_zero = constant <f32: -0.0> : tile<f32>

    // === 1D Arrays ===
    // CHECK: %{{.*}} = constant <i8: {{\[}}1, 2, 3, 4{{\]}}> : tile<4xi8>
    %i8_array = constant <i8: [1, 2, 3, 4]> : tile<4xi8>
    // CHECK: %{{.*}} = constant <i16: {{\[}}100, 200, 300, 400{{\]}}> : tile<4xi16>
    %i16_array = constant <i16: [100, 200, 300, 400]> : tile<4xi16>
    // CHECK: %{{.*}} = constant <i16: {{\[}}1, 2{{\]}}> : tile<2xi16>
    %i32_array_brackets = constant <i16: [1, 2]> : tile<2xi16>
    // CHECK: %{{.*}} = constant <i32: {{\[}}0, -1, 42, 127, 10, 1000, -500, 255{{\]}}> : tile<8xi32>
    %i32_array_mixed = constant <i32: [0, -1, 42, 0x7F, 0xA, 1000, -500, 255]> : tile<8xi32>
    // CHECK: %{{.*}} = constant <i64: {{\[}}1000000000000, -1000000000000{{\]}}> : tile<2xi64>
    %i64_array = constant <i64: [1000000000000, -1000000000000]> : tile<2xi64>
    
    // CHECK: %{{.*}} = constant <f16: {{\[}}1.000000e+00, 2.500000e+00, -3.140630e+00, 0.000000e+00{{\]}}> : tile<4xf16>
    %f16_array = constant <f16: [1.0, 2.5, -3.14159, 0.0]> : tile<4xf16>
    // CHECK: %{{.*}} = constant <f32: {{\[}}1.000000e+00, 2.000000e+00{{\]}}> : tile<2xf32>
    %f32_array_brackets = constant <f32: [1.0, 2.0]> : tile<2xf32>
    // CHECK: %{{.*}} = constant <f32: 1.000000e+00> : tile<2xf32>
    %f321_array_brackets = constant <f32: [1.0, 1.0]> : tile<2xf32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}1.000000e+00, 2.000000e+00{{\]}}> : tile<2xf32>
    %f32_array_no_brackets = constant <f32: [1.0, 2.0]> : tile<2xf32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}0.000000e+00, -0.000000e+00, 1.000000e+00, -1.000000e+00{{\]}}> : tile<4xf32>
    %f32_array_special = constant <f32: [0.0, -0.0, 1.0, -1.0]> : tile<4xf32>
    // CHECK: %{{.*}} = constant <f64: {{\[}}2.7182818284590451, 3.1415926535897931{{\]}}> : tile<2xf64>
    %f64_array = constant <f64: [2.718281828459045, 3.141592653589793]> : tile<2xf64>
    
    // CHECK: %{{.*}} = constant <f32: {{\[}}0x7F800000, 0xFF800000{{\]}}> : tile<2xf32>
    %hex_array_brackets = constant <f32: [0x7F800000, 0xFF800000]> : tile<2xf32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}0.000000e+00, 0x7FC00000, 0x7F800000, 1.000000e+00{{\]}}> : tile<4xf32>
    %hex_array_mixed = constant <f32: [0x00000000, 0x7FC00000, 0x7F800000, 0x3F800000]> : tile<4xf32>

    // === 2D Arrays ===
    // CHECK: %{{.*}} = constant <i32: {{\[}}{{\[}}1, 2{{\]}}, {{\[}}3, 4{{\]}}{{\]}}> : tile<2x2xi32>
    %i32_2d = constant <i32: [[1, 2], [3, 4]]> : tile<2x2xi32>
    // CHECK: %{{.*}} = constant <i32: {{\[}}{{\[}}1, 2, 3, 4{{\]}}, {{\[}}5, 6, 7, 8{{\]}}{{\]}}> : tile<2x4xi32>
    %i32_2d_rect = constant <i32: [[1, 2, 3, 4], [5, 6, 7, 8]]> : tile<2x4xi32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}{{\[}}1.000000e+00, 2.000000e+00{{\]}}, {{\[}}3.000000e+00, 4.000000e+00{{\]}}{{\]}}> : tile<2x2xf32>
    %f32_2d = constant <f32: [[1.0, 2.0], [3.0, 4.0]]> : tile<2x2xf32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}{{\[}}0.000000e+00, 1.000000e+00, -1.000000e+00, 2.000000e+00{{\]}}, {{\[}}0x7F800000, 0xFF800000, 0x7FC00000, 1.000000e+00{{\]}}{{\]}}> : tile<2x4xf32>
    %f32_2d_mixed = constant <f32: [[0.0, 1.0, -1.0, 2.0], [0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000]]> : tile<2x4xf32>

    // === 3D Arrays ===
    // CHECK: %{{.*}} = constant <i32: {{\[}}{{\[}}{{\[}}1, 2{{\]}}, {{\[}}3, 4{{\]}}{{\]}}, {{\[}}{{\[}}5, 6{{\]}}, {{\[}}7, 8{{\]}}{{\]}}{{\]}}> : tile<2x2x2xi32>
    %i32_3d = constant <i32: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tile<2x2x2xi32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}{{\[}}{{\[}}1.000000e+00, 2.000000e+00{{\]}}, {{\[}}3.000000e+00, 4.000000e+00{{\]}}{{\]}}, {{\[}}{{\[}}5.000000e+00, 6.000000e+00{{\]}}, {{\[}}7.000000e+00, 8.000000e+00{{\]}}{{\]}}{{\]}}> : tile<2x2x2xf32>
    %f32_3d = constant <f32: [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tile<2x2x2xf32>

    // === Edge Cases ===
    // CHECK: %{{.*}} = constant <i32: 42> : tile<1xi32>
    %single_element_array = constant <i32: [42]> : tile<1xi32>
    // CHECK: %{{.*}} = constant <i32: {{\[}}1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16{{\]}}> : tile<16xi32>
    %large_array = constant <i32: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tile<16xi32>
    
    // === Mixed Number Formats in Arrays ===
    // CHECK: %{{.*}} = constant <i32: {{\[}}10, 10, 12, 12{{\]}}> : tile<4xi32>
    %mixed_format_array = constant <i32: [10, 0xA, 12, 0xC]> : tile<4xi32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00{{\]}}> : tile<4xf32>
    %mixed_float_array = constant <f32: [1.0, 0x3F800000, 2.0, 0x40000000]> : tile<4xf32>

    // === Long Form and Mixed Form Type Syntax ===
    // CHECK: %{{.*}} = constant <i32: 42> : tile<i32>
    %long_form_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %{{.*}} = constant <f32: 3.141590e+00> : tile<f32>
    %long_form_f32 = constant <f32: 3.14159> : !cuda_tile.tile<f32>
    // CHECK: %{{.*}} = constant <i16: {{\[}}32, 64{{\]}}> : tile<2xi16>
    %long_form_array = constant <i16: [32, 64]> : !cuda_tile.tile<2xi16>
    // CHECK: %{{.*}} = constant <i32: {{\[}}{{\[}}1, 2{{\]}}, {{\[}}3, 4{{\]}}{{\]}}> : tile<2x2xi32>
    %long_form_2d = constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>
    // CHECK: %{{.*}} = constant <i32: 2147483647> : tile<i32>
    %long_form_hex = constant <i32: 0x7FFFFFFF> : !cuda_tile.tile<i32>
    // CHECK: %{{.*}} = constant <f32: 0x7F800000> : tile<f32>
    %long_form_float_inf = constant <f32: 0x7F800000> : !cuda_tile.tile<f32>
    
    // Mixed short and long form in same test
    // CHECK: %{{.*}} = constant <i32: 100> : tile<i32>
    %mixed_short = constant <i32: 100> : tile<i32>
    // CHECK: %{{.*}} = constant <i32: 200> : tile<i32>
    %mixed_long = constant <i32: 200> : !cuda_tile.tile<i32>
    // CHECK: %{{.*}} = constant <i32: {{\[}}1, 2, 3, 4{{\]}}> : tile<4xi32>
    %mixed_short_array = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    // CHECK: %{{.*}} = constant <i32: {{\[}}5, 6, 7, 8{{\]}}> : tile<4xi32>
    %mixed_long_array = constant <i32: [5, 6, 7, 8]> : !cuda_tile.tile<4xi32>
  }
}

cuda_tile.module @global {
  // === 1D Arrays ===
  // CHECK: global @i8_array <i8: {{\[}}1, 2, 3, 4{{\]}}> : tile<4xi8>
  global @i8_array <i8 : [1, 2, 3, 4]> : tile<4xi8>
  // CHECK: global @i16_array <i16: {{\[}}100, 200, 300, 400{{\]}}> : tile<4xi16>
  global @i16_array <i16 : [100, 200, 300, 400]> : tile<4xi16>
  // CHECK: global @i32_array <i32: {{\[}}1, 2{{\]}}> : tile<2xi32>
  global @i32_array <i32 : [1, 2]> : tile<2xi32>
  // CHECK: global @i32_array_mixed <i32: {{\[}}0, -1, 42, 127, 10, 1000, -500, 255{{\]}}> : tile<8xi32>
  global @i32_array_mixed <i32 : [0, -1, 42, 0x7F, 0xA, 1000, -500, 255]> : tile<8xi32>
  // CHECK: global @i64_array <i64: {{\[}}1000000000000, -1000000000000{{\]}}> : tile<2xi64>
  global @i64_array <i64: [1000000000000, -1000000000000]> : tile<2xi64>
  
  // CHECK: global @f16_array <f16: {{\[}}1.000000e+00, 2.500000e+00, -3.140630e+00, 0.000000e+00{{\]}}> : tile<4xf16>
  global @f16_array <f16: [1.0, 2.5, -3.14159, 0.0]> : tile<4xf16>
  // CHECK: global @f32_array <f32: {{\[}}1.000000e+00, 2.000000e+00{{\]}}> : tile<2xf32>
  global @f32_array <f32: [1.0, 2.0]> : tile<2xf32>
  // CHECK: global @f32_array_special <f32: {{\[}}0.000000e+00, -0.000000e+00, 1.000000e+00, -1.000000e+00{{\]}}> : tile<4xf32>
  global @f32_array_special <f32: [0.0, -0.0, 1.0, -1.0]> : tile<4xf32>
  // CHECK: global @f64_array <f64: {{\[}}2.7182818284590451, 3.1415926535897931{{\]}}> : tile<2xf64>
  global @f64_array <f64: [2.718281828459045, 3.141592653589793]> : tile<2xf64>
  
  // CHECK: global @hex_array <f32: {{\[}}0x7F800000, 0xFF800000{{\]}}> : tile<2xf32>
  global @hex_array <f32: [0x7F800000, 0xFF800000]> : tile<2xf32>
  // CHECK: global @hex_array_mixed <f32: {{\[}}0.000000e+00, 0x7FC00000, 0x7F800000, 1.000000e+00{{\]}}> : tile<4xf32>
  global @hex_array_mixed <f32: [0x00000000, 0x7FC00000, 0x7F800000, 0x3F800000]> : tile<4xf32>
  // CHECK: global @val <f32: {{\[}}1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01{{\]}}> : tile<4xf32>
  global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

  // === Edge Cases ===
  // CHECK: global @single_element <i32: 42> : tile<1xi32>
  global @single_element <i32: [42]> : tile<1xi32>
  // CHECK: global @large_array <i32: {{\[}}1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16{{\]}}> : tile<16xi32>
  global @large_array <i32: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tile<16xi32>
  
  // === Mixed Number Formats in Arrays ===
  // CHECK: global @mixed_format_array <i32: {{\[}}10, 10, 12, 12{{\]}}> : tile<4xi32>
  global @mixed_format_array <i32: [10, 0xA, 12, 0xC]> : tile<4xi32>
  // CHECK: global @mixed_float_array <f32: {{\[}}1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00{{\]}}> : tile<4xf32>
  global @mixed_float_array <f32: [1.0, 0x3F800000, 2.0, 0x40000000]> : tile<4xf32>

  // === Long Form and Mixed Form Type Syntax ===
  // CHECK: global @long_form_array <i16: {{\[}}32, 64{{\]}}> : tile<2xi16>
  global @long_form_array <i16: [32, 64]> : !cuda_tile.tile<2xi16>
  // CHECK: global @long_form_hex_array <i32: {{\[}}2147483647, -2147483648{{\]}}> : tile<2xi32>
  global @long_form_hex_array <i32: [0x7FFFFFFF, 0x80000000]> : !cuda_tile.tile<2xi32>
  // CHECK: global @long_form_float_array <f32: {{\[}}0x7F800000, 0xFF800000{{\]}}> : tile<2xf32>
  global @long_form_float_array <f32: [0x7F800000, 0xFF800000]> : !cuda_tile.tile<2xf32>
  
  // Mixed short and long form in same test
  // CHECK: global @mixed_short_array <i32: {{\[}}1, 2, 3, 4{{\]}}> : tile<4xi32>
  global @mixed_short_array <i32: [1, 2, 3, 4]> : tile<4xi32>
  // CHECK: global @mixed_long_array <i32: {{\[}}5, 6, 7, 8{{\]}}> : tile<4xi32>
  global @mixed_long_array <i32: [5, 6, 7, 8]> : !cuda_tile.tile<4xi32>
}

cuda_tile.module @assume {
  // CHECK: entry @assume_predicate(%{{.*}}: tile<ptr<f32>>) {
  entry @assume_predicate(%ptr: tile<ptr<f32>>) {
    // === Basic Test Values ===
    // CHECK: %{{.*}} = constant <i32: {{\[}}64, 128, 256, 512{{\]}}> : tile<4xi32>
    %i32_tile = constant <i32: [64, 128, 256, 512]> : tile<4xi32>
    // CHECK: %{{.*}} = constant <i64: {{\[}}1024, 2048{{\]}}> : tile<2xi64>
    %i64_tile = constant <i64: [1024, 2048]> : tile<2xi64>
    
    // CHECK: %{{.*}} = reshape %{{.*}} : tile<ptr<f32>> -> tile<1xptr<f32>>
    %ptr_1d = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
    // CHECK: %{{.*}} = broadcast %{{.*}} : tile<1xptr<f32>> -> tile<16xptr<f32>>
    %ptr_flat = broadcast %ptr_1d : tile<1xptr<f32>> -> tile<16xptr<f32>>
    // CHECK: %{{.*}} = reshape %{{.*}} : tile<16xptr<f32>> -> tile<4x4xptr<f32>>
    %ptr_2d = reshape %ptr_flat : tile<16xptr<f32>> -> tile<4x4xptr<f32>>
    
    // === Short Form Syntax Tests ===
    
    // DivBy predicate - short form
    // CHECK: %{{.*}} = assume div_by<32>, %{{.*}} : tile<4xi32>
    %short_div_basic = assume div_by<32>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<8, every 2 along 0>, %{{.*}} : tile<4xi32>
    %short_div_pattern = assume div_by<8, every 2 along 0>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<16, every 4 along 0>, %{{.*}} : tile<4xi32>
    %short_div_unsigned = assume div_by<16, every 4 along 0>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<4>, %{{.*}} : tile<4x4xptr<f32>>
    %short_div_ptr = assume div_by<4>, %ptr_2d : tile<4x4xptr<f32>>
    
    // SameElements predicate - short form
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2{{\]}}>, %{{.*}} : tile<4xi32>
    %short_same_1d = assume same_elements<[2]>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2, 2{{\]}}>, %{{.*}} : tile<4x4xptr<f32>>
    %short_same_2d = assume same_elements<[2, 2]>, %ptr_2d : tile<4x4xptr<f32>>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}1, 4{{\]}}>, %{{.*}} : tile<4x4xptr<f32>>
    %short_same_mixed = assume same_elements<[1, 4]>, %ptr_2d : tile<4x4xptr<f32>>
    
    // Bounded predicate - short form
    // CHECK: %{{.*}} = assume bounded<0, 2>, %{{.*}} : tile<4xi32>
    %short_non_neg = assume bounded<0, 2>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume bounded<-2, 16>, %{{.*}} : tile<2xi64>
    %short_non_neg_i64 = assume bounded<-2, 16>, %i64_tile : tile<2xi64>
    
    // === Long Form Syntax Tests ===
    
    // DivBy predicate - long form
    // CHECK: %{{.*}} = assume div_by<32>, %{{.*}} : tile<4xi32>
    %long_div_basic = assume #cuda_tile.div_by<32>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<8, every 2 along 0>, %{{.*}} : tile<4xi32>
    %long_div_pattern = assume #cuda_tile.div_by<8, every 2 along 0>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<16, every 4 along 0>, %{{.*}} : tile<4xi32>
    %long_div_unsigned = assume #cuda_tile.div_by<16, every 4 along 0>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<4>, %{{.*}} : tile<4x4xptr<f32>>
    %long_div_ptr = assume #cuda_tile.div_by<4>, %ptr_2d : tile<4x4xptr<f32>>
    
    // SameElements predicate - long form
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2{{\]}}>, %{{.*}} : tile<4xi32>
    %long_same_1d = assume #cuda_tile.same_elements<[2]>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2, 2{{\]}}>, %{{.*}} : tile<4x4xptr<f32>>
    %long_same_2d = assume #cuda_tile.same_elements<[2, 2]>, %ptr_2d : tile<4x4xptr<f32>>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}1, 4{{\]}}>, %{{.*}} : tile<4x4xptr<f32>>
    %long_same_mixed = assume #cuda_tile.same_elements<[1, 4]>, %ptr_2d : tile<4x4xptr<f32>>
    
    // Bounded predicate - long form
    // CHECK: %{{.*}} = assume bounded<0, ?>, %{{.*}} : tile<4xi32>
    %long_non_neg = assume #cuda_tile.bounded<0, ?>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume bounded<?, ?>, %{{.*}} : tile<2xi64>
    %long_non_neg_i64 = assume #cuda_tile.bounded<?, ?>, %i64_tile : tile<2xi64>
    
    // === Mixed Form Usage Tests ===
    
    // Same predicate, different syntax
    // CHECK: %{{.*}} = assume div_by<64>, %{{.*}} : tile<4xi32>
    %mixed_div_short = assume div_by<64>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<64>, %{{.*}} : tile<4xi32>
    %mixed_div_long = assume #cuda_tile.div_by<64>, %i32_tile : tile<4xi32>
    
    // CHECK: %{{.*}} = assume same_elements<{{\[}}4{{\]}}>, %{{.*}} : tile<4xi32>
    %mixed_same_short = assume same_elements<[4]>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}4{{\]}}>, %{{.*}} : tile<4xi32>
    %mixed_same_long = assume #cuda_tile.same_elements<[4]>, %i32_tile : tile<4xi32>
    
    // CHECK: %{{.*}} = assume bounded<0, ?>, %{{.*}} : tile<2xi64>
    %mixed_neg_short = assume bounded<0, ?>, %i64_tile : tile<2xi64>
    // CHECK: %{{.*}} = assume bounded<0, ?>, %{{.*}} : tile<2xi64>
    %mixed_neg_long = assume #cuda_tile.bounded<0, ?>, %i64_tile : tile<2xi64>
    
    // === Extended Bounded Tests ===
    
    // Bounded with different integer types
    // CHECK: %{{.*}} = constant <i16: {{\[}}1, 2, 3, 4{{\]}}> : tile<4xi16>
    %non_neg_small = constant <i16: [1, 2, 3, 4]> : tile<4xi16>
    // CHECK: %{{.*}} = constant <i64: {{\[}}100, 200, 300, 400{{\]}}> : tile<4xi64>
    %non_neg_large = constant <i64: [100, 200, 300, 400]> : tile<4xi64>
    
    // CHECK: %{{.*}} = assume bounded<?, 4>, %{{.*}} : tile<4xi16>
    %short_non_neg_i16 = assume bounded<?, 4>, %non_neg_small : tile<4xi16>
    // CHECK: %{{.*}} = assume bounded<?, 4>, %{{.*}} : tile<4xi16>
    %long_non_neg_i16 = assume #cuda_tile.bounded<?, 4>, %non_neg_small : tile<4xi16>
    
    // CHECK: %{{.*}} = assume bounded<-16, 4>, %{{.*}} : tile<4xi64>
    %short_non_neg_i64_large = assume bounded<-16, 4>, %non_neg_large : tile<4xi64>
    // CHECK: %{{.*}} = assume bounded<-16, 4>, %{{.*}} : tile<4xi64>
    %long_non_neg_i64_large = assume #cuda_tile.bounded<-16, 4>, %non_neg_large : tile<4xi64>
    
    // Bounded in chains with other predicates
    // CHECK: %{{.*}} = assume bounded<-16, 4>, %{{.*}} : tile<4xi32>
    %chain_non_neg_1 = assume bounded<-16, 4>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<8>, %{{.*}} : tile<4xi32>
    %chain_non_neg_2 = assume div_by<8>, %chain_non_neg_1 : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2{{\]}}>, %{{.*}} : tile<4xi32>
    %chain_non_neg_3 = assume same_elements<[2]>, %chain_non_neg_2 : tile<4xi32>
    
    // Mixed syntax chains with bounded
    // CHECK: %{{.*}} = assume bounded<-16, 4>, %{{.*}} : tile<4xi32>
    %mixed_chain_non_neg_1 = assume #cuda_tile.bounded<-16, 4>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume div_by<4>, %{{.*}} : tile<4xi32>
    %mixed_chain_non_neg_2 = assume div_by<4>, %mixed_chain_non_neg_1 : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}1{{\]}}>, %{{.*}} : tile<4xi32>
    %mixed_chain_non_neg_3 = assume #cuda_tile.same_elements<[1]>, %mixed_chain_non_neg_2 : tile<4xi32>
    
    // === Chained Assumptions with Mixed Syntax ===
    
    // Chain short → long → short
    // CHECK: %{{.*}} = assume div_by<8>, %{{.*}} : tile<4xi32>
    %chain_short_1 = assume div_by<8>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume bounded<-16, 4>, %{{.*}} : tile<4xi32>
    %chain_long_1 = assume #cuda_tile.bounded<-16, 4>, %chain_short_1 : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2{{\]}}>, %{{.*}} : tile<4xi32>
    %chain_short_2 = assume same_elements<[2]>, %chain_long_1 : tile<4xi32>
    
    // Chain long → short → long
    // CHECK: %{{.*}} = assume div_by<16>, %{{.*}} : tile<4xi32>
    %chain_long_2 = assume #cuda_tile.div_by<16>, %i32_tile : tile<4xi32>
    // CHECK: %{{.*}} = assume bounded<-16, 4>, %{{.*}} : tile<4xi32>
    %chain_short_3 = assume bounded<-16, 4>, %chain_long_2 : tile<4xi32>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}1{{\]}}>, %{{.*}} : tile<4xi32>
    %chain_long_3 = assume #cuda_tile.same_elements<[1]>, %chain_short_3 : tile<4xi32>
    
    // === Complex Patterns with Both Syntaxes ===
    
    // Multi-dimensional patterns
    // CHECK: %{{.*}} = assume div_by<4, every 2 along 0>, %{{.*}} : tile<4x4xptr<f32>>
    %short_3d_pattern = assume div_by<4, every 2 along 0>, %ptr_2d : tile<4x4xptr<f32>>
    // CHECK: %{{.*}} = assume div_by<4, every 2 along 1>, %{{.*}} : tile<4x4xptr<f32>>
    %long_3d_pattern = assume #cuda_tile.div_by<4, every 2 along 1>, %ptr_2d : tile<4x4xptr<f32>>
    
    // Complex same elements
    // CHECK: %{{.*}} = assume same_elements<{{\[}}2, 4{{\]}}>, %{{.*}} : tile<4x4xptr<f32>>
    %short_complex_same = assume same_elements<[2, 4]>, %ptr_2d : tile<4x4xptr<f32>>
    // CHECK: %{{.*}} = assume same_elements<{{\[}}4, 1{{\]}}>, %{{.*}} : tile<4x4xptr<f32>>
    %long_complex_same = assume #cuda_tile.same_elements<[4, 1]>, %ptr_2d : tile<4x4xptr<f32>>
    
    return
  }
}

cuda_tile.module @function_signature {
  
  // === Basic Type Forms ===
  
  // Short form only
  // CHECK: entry @short_form_only(%{{.*}}: tile<i32>, %{{.*}}: tile<f32>) {
  entry @short_form_only(%arg0: tile<i32>, %arg1: tile<f32>) {
    return
  }
  
  // Long form only  
  // CHECK: entry @long_form_only(%{{.*}}: tile<i32>, %{{.*}}: tile<f32>) {
  entry @long_form_only(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<f32>) {
    return
  }
  
  // === Mixed Forms in Same Signature ===
  
  // CHECK: testing$func @mixed_args(%{{.*}}: tile<i32>, %{{.*}}: tile<f32>) -> tile<i32> {
  testing$func @mixed_args(%short: tile<i32>, %long: !cuda_tile.tile<f32>) -> tile<i32> {
    return %short : tile<i32>
  }
  
  // CHECK: testing$func @mixed_return_short(%{{.*}}: tile<i32>) -> tile<i32> {
  testing$func @mixed_return_short(%arg0: !cuda_tile.tile<i32>) -> tile<i32> {
    return %arg0 : tile<i32>
  }
  
  // CHECK: testing$func @mixed_return_long(%{{.*}}: tile<i32>) -> tile<i32> {
  testing$func @mixed_return_long(%arg0: tile<i32>) -> !cuda_tile.tile<i32> {
    return %arg0 : tile<i32>
  }
  
  // === Different Data Types ===
  
  // Integer types
  // CHECK: testing$func @integer_types_short(%{{.*}}: tile<i8>, %{{.*}}: tile<i16>, %{{.*}}: tile<i32>, %{{.*}}: tile<i64>) {
  testing$func @integer_types_short(%i8: tile<i8>, %i16: tile<i16>, %i32: tile<i32>, %i64: tile<i64>) {
    return
  }
  
  // CHECK: testing$func @integer_types_long(%{{.*}}: tile<i8>, %{{.*}}: tile<i16>, %{{.*}}: tile<i32>, %{{.*}}: tile<i64>) {
  testing$func @integer_types_long(%i8: !cuda_tile.tile<i8>, %i16: !cuda_tile.tile<i16>, 
                          %i32: !cuda_tile.tile<i32>, %i64: !cuda_tile.tile<i64>) {
    return
  }
  
  // Float types
  // CHECK: testing$func @float_types_short(%{{.*}}: tile<f16>, %{{.*}}: tile<f32>, %{{.*}}: tile<f64>) {
  testing$func @float_types_short(%f16: tile<f16>, %f32: tile<f32>, %f64: tile<f64>) {
    return
  }
  
  // CHECK: testing$func @float_types_long(%{{.*}}: tile<f16>, %{{.*}}: tile<f32>, %{{.*}}: tile<f64>) {
  testing$func @float_types_long(%f16: !cuda_tile.tile<f16>, %f32: !cuda_tile.tile<f32>, 
                        %f64: !cuda_tile.tile<f64>) {
    return
  }
  
  // Pointer types
  // CHECK: testing$func @pointer_types_short(%{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<i32>>) {
  testing$func @pointer_types_short(%ptr_f32: tile<ptr<f32>>, %ptr_i32: tile<ptr<i32>>) {
    return
  }
  
  // CHECK: testing$func @pointer_types_long(%{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<i32>>) {
  testing$func @pointer_types_long(%ptr_f32: !cuda_tile.tile<ptr<f32>>, %ptr_i32: !cuda_tile.tile<ptr<i32>>) {
    return
  }
  
  // === Dimensional Variations ===
  
  // 1D arrays
  // CHECK: testing$func @array_1d_short(%{{.*}}: tile<2xi32>, %{{.*}}: tile<4xf32>, %{{.*}}: tile<8xi64>) {
  testing$func @array_1d_short(%a1: tile<2xi32>, %a2: tile<4xf32>, %a3: tile<8xi64>) {
    return
  }
  
  // CHECK: testing$func @array_1d_long(%{{.*}}: tile<2xi32>, %{{.*}}: tile<4xf32>, %{{.*}}: tile<8xi64>) {
  testing$func @array_1d_long(%a1: !cuda_tile.tile<2xi32>, %a2: !cuda_tile.tile<4xf32>, 
                     %a3: !cuda_tile.tile<8xi64>) {
    return
  }
  
  // 2D arrays
  // CHECK: testing$func @array_2d_short(%{{.*}}: tile<2x2xi32>, %{{.*}}: tile<4x4xf32>, %{{.*}}: tile<2x8xf64>) {
  testing$func @array_2d_short(%m1: tile<2x2xi32>, %m2: tile<4x4xf32>, %m3: tile<2x8xf64>) {
    return
  }
  
  // CHECK: testing$func @array_2d_long(%{{.*}}: tile<2x2xi32>, %{{.*}}: tile<4x4xf32>, %{{.*}}: tile<2x8xf64>) {
  testing$func @array_2d_long(%m1: !cuda_tile.tile<2x2xi32>, %m2: !cuda_tile.tile<4x4xf32>, 
                     %m3: !cuda_tile.tile<2x8xf64>) {
    return
  }
  
  // 3D arrays
  // CHECK: testing$func @array_3d_short(%{{.*}}: tile<2x2x2xi32>, %{{.*}}: tile<1x4x8xf32>) {
  testing$func @array_3d_short(%t1: tile<2x2x2xi32>, %t2: tile<1x4x8xf32>) {
    return
  }
  
  // CHECK: testing$func @array_3d_long(%{{.*}}: tile<2x2x2xi32>, %{{.*}}: tile<1x4x8xf32>) {
  testing$func @array_3d_long(%t1: !cuda_tile.tile<2x2x2xi32>, %t2: !cuda_tile.tile<1x4x8xf32>) {
    return
  }
  
  // === Mixed Dimensional Types ===
  
  // CHECK: testing$func @mixed_dimensions(%{{.*}}: tile<i32>, %{{.*}}: tile<4xi32>, %{{.*}}: tile<2x2xi32>, %{{.*}}: tile<2x2x2xi32>) {
  testing$func @mixed_dimensions(%scalar: tile<i32>, %vec: tile<4xi32>, 
                        %matrix: tile<2x2xi32>, %tensor: tile<2x2x2xi32>) {
    return
  }
  
  // CHECK: testing$func @mixed_dimensions_long(%{{.*}}: tile<i32>, %{{.*}}: tile<4xi32>, %{{.*}}: tile<2x2xi32>, %{{.*}}: tile<2x2x2xi32>) {
  testing$func @mixed_dimensions_long(%scalar: !cuda_tile.tile<i32>, %vec: !cuda_tile.tile<4xi32>, 
                             %matrix: !cuda_tile.tile<2x2xi32>, %tensor: !cuda_tile.tile<2x2x2xi32>) {
    return
  }
  
  // === Complex Return Types ===
  
  // Multiple returns - short form
  // CHECK: testing$func @multi_return_short() -> (tile<i32>, tile<f32>, tile<2xi64>) {
  testing$func @multi_return_short() -> (tile<i32>, tile<f32>, tile<2xi64>) {
    // CHECK: %{{.*}} = constant <i32: 42> : tile<i32>
    %i = constant <i32: 42> : tile<i32>
    // CHECK: %{{.*}} = constant <f32: 3.140000e+00> : tile<f32>
    %f = constant <f32: 3.14> : tile<f32>
    // CHECK: %{{.*}} = constant <i64: [1, 2]> : tile<2xi64>
    %v = constant <i64: [1, 2]> : tile<2xi64>
    return %i, %f, %v : tile<i32>, tile<f32>, tile<2xi64>
  }
  
  // Multiple returns - long form
  // CHECK: testing$func @multi_return_long() -> (tile<i32>, tile<f32>, tile<2xi64>) {
  testing$func @multi_return_long() -> (!cuda_tile.tile<i32>, !cuda_tile.tile<f32>, !cuda_tile.tile<2xi64>) {
    // CHECK: %{{.*}} = constant <i32: 42> : tile<i32>
    %i = constant <i32: 42> : tile<i32>
    // CHECK: %{{.*}} = constant <f32: 3.140000e+00> : tile<f32>
    %f = constant <f32: 3.14> : tile<f32>
    // CHECK: %{{.*}} = constant <i64: [1, 2]> : tile<2xi64>
    %v = constant <i64: [1, 2]> : tile<2xi64>
    return %i, %f, %v : tile<i32>, tile<f32>, tile<2xi64>
  }
  
  // Multiple returns - mixed form
  // CHECK: testing$func @multi_return_mixed() -> (tile<i32>, tile<f32>, tile<2xi64>) {
  testing$func @multi_return_mixed() -> (tile<i32>, !cuda_tile.tile<f32>, tile<2xi64>) {
    // CHECK: %{{.*}} = constant <i32: 42> : tile<i32>
    %i = constant <i32: 42> : tile<i32>
    // CHECK: %{{.*}} = constant <f32: 3.140000e+00> : tile<f32>
    %f = constant <f32: 3.14> : tile<f32>
    // CHECK: %{{.*}} = constant <i64: [1, 2]> : tile<2xi64>
    %v = constant <i64: [1, 2]> : tile<2xi64>
    return %i, %f, %v : tile<i32>, tile<f32>, tile<2xi64>
  }
  
  // === Edge Cases ===
  
  // No arguments
  // CHECK: testing$func @no_args_short() -> tile<i32> {
  testing$func @no_args_short() -> tile<i32> {
    // CHECK: %{{.*}} = constant <i32: 0> : tile<i32>
    %result = constant <i32: 0> : tile<i32>
    return %result : tile<i32>
  }
  
  // CHECK: testing$func @no_args_long() -> tile<i32> {
  testing$func @no_args_long() -> !cuda_tile.tile<i32> {
    // CHECK: %{{.*}} = constant <i32: 0> : tile<i32>
    %result = constant <i32: 0> : tile<i32>
    return %result : tile<i32>
  }
  
  // Single argument
  // CHECK: testing$func @single_arg_short(%{{.*}}: tile<i32>) -> tile<i32> {
  testing$func @single_arg_short(%arg: tile<i32>) -> tile<i32> {
    return %arg : tile<i32>
  }
  
  // CHECK: testing$func @single_arg_long(%{{.*}}: tile<i32>) -> tile<i32> {
  testing$func @single_arg_long(%arg: !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    return %arg : tile<i32>
  }
  
  // Many arguments
  // CHECK: testing$func @many_args(%{{.*}}: tile<i32>, %{{.*}}: tile<i32>, %{{.*}}: tile<f32>, %{{.*}}: tile<f32>, %{{.*}}: tile<2xi32>, %{{.*}}: tile<2xi32>, %{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<f32>>) {
  testing$func @many_args(%a0: tile<i32>, %a1: !cuda_tile.tile<i32>, %a2: tile<f32>, %a3: !cuda_tile.tile<f32>,
                 %a4: tile<2xi32>, %a5: !cuda_tile.tile<2xi32>, %a6: tile<ptr<f32>>, %a7: !cuda_tile.tile<ptr<f32>>) {
    return
  }
  
  // === Entry Points with Both Forms ===
  
  // Basic entry forms
  // CHECK: entry @entry_short_args(%{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<i32>) {
  entry @entry_short_args(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
    return
  }
  
  // CHECK: entry @entry_long_args(%{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<i32>) {
  entry @entry_long_args(%arg0: !cuda_tile.tile<ptr<f32>>, %arg1: !cuda_tile.tile<i32>) {
    return
  }
  
  // CHECK: entry @entry_mixed_args(%{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<i32>) {
  entry @entry_mixed_args(%short: tile<ptr<f32>>, %long: !cuda_tile.tile<i32>) {
    return
  }
  
  // === Comprehensive Entry Testing ===
  // NOTE: Entry operations only support scalar types (rank 0 tiles)
  
  // Entry with different scalar data types - short form
  // CHECK: entry @entry_types_short(%{{.*}}: tile<i8>, %{{.*}}: tile<i16>, %{{.*}}: tile<i32>, %{{.*}}: tile<i64>, %{{.*}}: tile<f16>, %{{.*}}: tile<f32>, %{{.*}}: tile<f64>) {
  entry @entry_types_short(%i8: tile<i8>, %i16: tile<i16>, %i32: tile<i32>, %i64: tile<i64>,
                          %f16: tile<f16>, %f32: tile<f32>, %f64: tile<f64>) {
    return
  }
  
  // Entry with different scalar data types - long form  
  // CHECK: entry @entry_types_long(%{{.*}}: tile<i8>, %{{.*}}: tile<i16>, %{{.*}}: tile<i32>, %{{.*}}: tile<i64>, %{{.*}}: tile<f16>, %{{.*}}: tile<f32>, %{{.*}}: tile<f64>) {
  entry @entry_types_long(%i8: !cuda_tile.tile<i8>, %i16: !cuda_tile.tile<i16>, 
                         %i32: !cuda_tile.tile<i32>, %i64: !cuda_tile.tile<i64>,
                         %f16: !cuda_tile.tile<f16>, %f32: !cuda_tile.tile<f32>, 
                         %f64: !cuda_tile.tile<f64>) {
    return
  }
  
  // Entry with mixed scalar data types
  // CHECK: entry @entry_types_mixed(%{{.*}}: tile<i8>, %{{.*}}: tile<i16>, %{{.*}}: tile<i32>, %{{.*}}: tile<i64>, %{{.*}}: tile<f16>, %{{.*}}: tile<f32>, %{{.*}}: tile<f64>) {
  entry @entry_types_mixed(%i8: tile<i8>, %i16: !cuda_tile.tile<i16>, 
                          %i32: tile<i32>, %i64: !cuda_tile.tile<i64>,
                          %f16: tile<f16>, %f32: !cuda_tile.tile<f32>, 
                          %f64: tile<f64>) {
    return
  }
  
  // Entry with pointer types - short form
  // CHECK: entry @entry_ptrs_short(%{{.*}}: tile<ptr<i32>>, %{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<f64>>, %{{.*}}: tile<ptr<f16>>) {
  entry @entry_ptrs_short(%ptr_i32: tile<ptr<i32>>, %ptr_f32: tile<ptr<f32>>, 
                         %ptr_f64: tile<ptr<f64>>, %ptr_f16: tile<ptr<f16>>) {
    return
  }
  
  // Entry with pointer types - long form
  // CHECK: entry @entry_ptrs_long(%{{.*}}: tile<ptr<i32>>, %{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<f64>>, %{{.*}}: tile<ptr<f16>>) {
  entry @entry_ptrs_long(%ptr_i32: !cuda_tile.tile<ptr<i32>>, %ptr_f32: !cuda_tile.tile<ptr<f32>>, 
                        %ptr_f64: !cuda_tile.tile<ptr<f64>>, %ptr_f16: !cuda_tile.tile<ptr<f16>>) {
    return
  }
  
  // Entry with pointer types - mixed
  // CHECK: entry @entry_ptrs_mixed(%{{.*}}: tile<ptr<i32>>, %{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<f64>>, %{{.*}}: tile<ptr<f16>>) {
  entry @entry_ptrs_mixed(%ptr_i32: tile<ptr<i32>>, %ptr_f32: !cuda_tile.tile<ptr<f32>>, 
                         %ptr_f64: tile<ptr<f64>>, %ptr_f16: !cuda_tile.tile<ptr<f16>>) {
    return
  }
  
  // Entry with no arguments - short form
  // CHECK: entry @entry_no_args_short() {
  entry @entry_no_args_short() {
    return
  }
  
  // Entry with no arguments - long form (no args to show form)
  // CHECK: entry @entry_no_args_long() {
  entry @entry_no_args_long() {
    return
  }
  
  // Entry with single argument - short form
  // CHECK: entry @entry_single_short(%{{.*}}: tile<ptr<f32>>) {
  entry @entry_single_short(%arg: tile<ptr<f32>>) {
    return
  }
  
  // Entry with single argument - long form
  // CHECK: entry @entry_single_long(%{{.*}}: tile<ptr<f32>>) {
  entry @entry_single_long(%arg: !cuda_tile.tile<ptr<f32>>) {
    return
  }
  
  // Entry with many scalar arguments - mixed forms
  // CHECK: entry @entry_many_mixed(%{{.*}}: tile<i32>, %{{.*}}: tile<i32>, %{{.*}}: tile<f32>, %{{.*}}: tile<f32>, %{{.*}}: tile<i64>, %{{.*}}: tile<i64>, %{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<f32>>, %{{.*}}: tile<ptr<i32>>, %{{.*}}: tile<ptr<i32>>) {
  entry @entry_many_mixed(%a0: tile<i32>, %a1: !cuda_tile.tile<i32>, 
                         %a2: tile<f32>, %a3: !cuda_tile.tile<f32>,
                         %a4: tile<i64>, %a5: !cuda_tile.tile<i64>, 
                         %a6: tile<ptr<f32>>, %a7: !cuda_tile.tile<ptr<f32>>,
                         %a8: tile<ptr<i32>>, %a9: !cuda_tile.tile<ptr<i32>>) {
    return
  }
}
