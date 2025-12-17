// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  cuda_tile.entry @bitcast() {
    // **** 8-bit ****
    // i8 -> i8
    // CHECK: %[[const_i8:.*]] = constant <i8: [1, 2, 3, 4]> : tile<4xi8>
    %c_i8 = constant <i8: [1, 2, 3, 4]> : !cuda_tile.tile<4xi8>
    // CHECK: %[[bc_i8_i8:.*]] = bitcast %[[const_i8]] : tile<4xi8> -> tile<4xi8>
    %bc_i8_i8 = bitcast %c_i8 : tile<4xi8> -> tile<4xi8>

    // **** 16-bit ****
    // i16 -> i16
    // CHECK: %[[const_i16:.*]] = constant <i16: [1, 2, 3, 4]> : tile<4xi16>
    %c_i16 = constant <i16: [1, 2, 3, 4]> : !cuda_tile.tile<4xi16>
    // CHECK: %[[bc_i16_i16:.*]] = bitcast %[[const_i16]] : tile<4xi16> -> tile<4xi16>
    %bc_i16_i16 = bitcast %c_i16 : tile<4xi16> -> tile<4xi16>

    // i16 -> f16
    // CHECK: %[[bc_i16_f16:.*]] = bitcast %[[const_i16]] : tile<4xi16> -> tile<4xf16>
    %bc_i16_f16 = bitcast %c_i16 : tile<4xi16> -> tile<4xf16>

    // i16 -> bf16
    // CHECK: %[[bc_i16_bf16:.*]] = bitcast %[[const_i16]] : tile<4xi16> -> tile<4xbf16>
    %bc_i16_bf16 = bitcast %c_i16 : tile<4xi16> -> tile<4xbf16>

    // f16 -> f16
    // CHECK: %[[const_f16:.*]] = constant <f16: [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tile<4xf16>
    %c_f16 = constant <f16: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf16>
    // CHECK: %[[bc_f16_f16:.*]] = bitcast %[[const_f16]] : tile<4xf16> -> tile<4xf16>
    %bc_f16_f16 = bitcast %c_f16 : tile<4xf16> -> tile<4xf16>

    // f16 -> i16
    // CHECK: %[[bc_f16_i16:.*]] = bitcast %[[const_f16]] : tile<4xf16> -> tile<4xi16>
    %bc_f16_i16 = bitcast %c_f16 : tile<4xf16> -> tile<4xi16>

    // f16 -> bf16
    // CHECK: %[[bc_f16_bf16:.*]] = bitcast %[[const_f16]] : tile<4xf16> -> tile<4xbf16>
    %bc_f16_bf16 = bitcast %c_f16 : tile<4xf16> -> tile<4xbf16>

    // bf16 -> bf16
    // CHECK: %[[const_bf16:.*]] = constant <bf16: [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tile<4xbf16>
    %c_bf16 = constant <bf16: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xbf16>
    // CHECK: %[[bc_bf16_bf16:.*]] = bitcast %[[const_bf16]] : tile<4xbf16> -> tile<4xbf16>
    %bc_bf16_bf16 = bitcast %c_bf16 : tile<4xbf16> -> tile<4xbf16>

    // bf16 -> i16
    // CHECK: %[[bc_bf16_i16:.*]] = bitcast %[[const_bf16]] : tile<4xbf16> -> tile<4xi16>
    %bc_bf16_i16 = bitcast %c_bf16 : tile<4xbf16> -> tile<4xi16>

    // bf16 -> f16
    // CHECK: %[[bc_bf16_f16:.*]] = bitcast %[[const_bf16]] : tile<4xbf16> -> tile<4xf16>
    %bc_bf16_f16 = bitcast %c_bf16 : tile<4xbf16> -> tile<4xf16>

    // **** 32-bit ****
    // i32 -> i32
    // CHECK: %[[const_i32:.*]] = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    %c_i32 = constant <i32: [1, 2, 3, 4]> : !cuda_tile.tile<4xi32>
    // CHECK: %[[bc_i32_i32:.*]] = bitcast %[[const_i32]] : tile<4xi32> -> tile<4xi32>
    %bc_i32_i32 = bitcast %c_i32 : tile<4xi32> -> tile<4xi32>

    // i32 -> f32
    // CHECK: %[[bc_i32_f32:.*]] = bitcast %[[const_i32]] : tile<4xi32> -> tile<4xf32>
    %bc_i32_f32 = bitcast %c_i32 : tile<4xi32> -> tile<4xf32>

    // f32 -> f32
    // CHECK: %[[const_f32:.*]] = constant <f32: [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tile<4xf32>
    %c_f32 = constant <f32: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf32>
    // CHECK: %[[bc_f32_f32:.*]] = bitcast %[[const_f32]] : tile<4xf32> -> tile<4xf32>
    %bc_f32_f32 = bitcast %c_f32 : tile<4xf32> -> tile<4xf32>

    // f32 -> i32
    // CHECK: %[[bc_f32_i32:.*]] = bitcast %[[const_f32]] : tile<4xf32> -> tile<4xi32>
    %bc_f32_i32 = bitcast %c_f32 : tile<4xf32> -> tile<4xi32>

    // **** 64-bit ****
    // i64 -> i64
    // CHECK: %[[const_i64:.*]] = constant <i64: [1, 2, 3, 4]> : tile<4xi64>
    %c_i64 = constant <i64: [1, 2, 3, 4]> : !cuda_tile.tile<4xi64>
    // CHECK: %[[bc_i64_i64:.*]] = bitcast %[[const_i64]] : tile<4xi64> -> tile<4xi64>
    %bc_i64_i64 = bitcast %c_i64 : tile<4xi64> -> tile<4xi64>

    // i64 -> f64
    // CHECK: %[[bc_i64_f64:.*]] = bitcast %[[const_i64]] : tile<4xi64> -> tile<4xf64>
    %bc_i64_f64 = bitcast %c_i64 : tile<4xi64> -> tile<4xf64>

    // f64 -> f64
    // CHECK: %[[const_f64:.*]] = constant <f64: [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tile<4xf64>
    %c_f64 = constant <f64: [1.0, 2.0, 3.0, 4.0]> : !cuda_tile.tile<4xf64>
    // CHECK: %[[bc_f64_f64:.*]] = bitcast %[[const_f64]] : tile<4xf64> -> tile<4xf64>
    %bc_f64_f64 = bitcast %c_f64 : tile<4xf64> -> tile<4xf64>

    // f64 -> i64
    // CHECK: %[[bc_f64_i64:.*]] = bitcast %[[const_f64]] : tile<4xf64> -> tile<4xi64>
    %bc_f64_i64 = bitcast %c_f64 : tile<4xf64> -> tile<4xi64>

    // int64 to pointer back to int64
    // CHECK: %[[c2_i64:.*]] = constant <i64: 1> : tile<i64>
    %c2_i64 = constant <i64: 1> : !cuda_tile.tile<i64>
    // CHECK: %[[c3_ptr:.*]] = int_to_ptr %[[c2_i64]] : tile<i64> -> tile<ptr<i8>>
    %c3_ptr = int_to_ptr %c2_i64 : tile<i64> -> tile<ptr<i8>>
    // CHECK: %[[c4_i64:.*]] = ptr_to_int %[[c3_ptr]] : tile<ptr<i8>> -> tile<i64>
    %c4_i64 = ptr_to_int %c3_ptr : tile<ptr<i8>> -> tile<i64>

    // elementwise int64 to pointer
    // CHECK: %[[c5_i64:.*]] = constant <i64: [1, 2, 3, 4]> : tile<4xi64>
    %c5_i64 = constant <i64: [1, 2, 3, 4]> : !cuda_tile.tile<4xi64>
    // CHECK: %[[c6_ptr:.*]] = int_to_ptr %[[c5_i64]] : tile<4xi64> -> tile<4xptr<i8>>
    %c6_ptr = int_to_ptr %c5_i64 : tile<4xi64> -> tile<4xptr<i8>>

    // pointer to pointer
    // CHECK: %[[c7_ptr:.*]] = ptr_to_ptr %[[c6_ptr]] : tile<4xptr<i8>> -> tile<4xptr<f64>>
    %c7_ptr = ptr_to_ptr %c6_ptr : tile<4xptr<i8>> -> tile<4xptr<f64>>
  }

  cuda_tile.entry @ftof() {
    // Constants
    // CHECK: %[[c5_f16:.*]] = constant <f16: 5.000000e+00> : tile<f16>
    %c5_f16 = constant <f16: 5.0> : !cuda_tile.tile<f16>
    // CHECK: %[[c5_bf16:.*]] = constant <bf16: 5.000000e+00> : tile<bf16>
    %c5_bf16 = constant <bf16: 5.0> : !cuda_tile.tile<bf16>
    // CHECK: %[[c5_f32:.*]] = constant <f32: 5.000000e+00> : tile<f32>
    %c5_f32 = constant <f32: 5.0> : !cuda_tile.tile<f32>
    // CHECK: %[[c5_f64:.*]] = constant <f64: 5.000000e+00> : tile<f64>
    %c5_f64 = constant <f64: 5.0> : !cuda_tile.tile<f64>

    // CHECK: %[[c_tensor_f16:.*]] = constant <f16: {{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tile<2x2xf16>
    %c_tensor_f16 = constant <f16: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf16>
    // CHECK: %[[c_tensor_bf16:.*]] = constant <bf16: {{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tile<2x2xbf16>
    %c_tensor_bf16 = constant <bf16: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xbf16>
    // CHECK: %[[c_tensor_f32:.*]] = constant <f32: {{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tile<2x2xf32>
    %c_tensor_f32 = constant <f32: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: %[[c_tensor_f64:.*]] = constant <f64: {{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tile<2x2xf64>
    %c_tensor_f64 = constant <f64: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf64>

    // **** f16 input ****
    // CHECK: ftof %[[c5_f16]] : tile<f16> -> tile<bf16>
    %ftof_f16_bf16_s = ftof %c5_f16 : tile<f16> -> tile<bf16>
    // CHECK: ftof %[[c5_f16]] : tile<f16> -> tile<f32>
    %ftof_f16_f32_s = ftof %c5_f16 : tile<f16> -> tile<f32>
    // CHECK: ftof %[[c5_f16]] : tile<f16> -> tile<f64>
    %ftof_f16_f64_s = ftof %c5_f16 : tile<f16> -> tile<f64>
    // CHECK: ftof %[[c_tensor_f16]] : tile<2x2xf16> -> tile<2x2xf32>
    %ftof_f16_f32_t = ftof %c_tensor_f16 : tile<2x2xf16> -> tile<2x2xf32>

    // **** bf16 input ****
    // CHECK: ftof %[[c5_bf16]] : tile<bf16> -> tile<f16>
    %ftof_bf16_f16_s = ftof %c5_bf16 : tile<bf16> -> tile<f16>
    // CHECK: ftof %[[c5_bf16]] : tile<bf16> -> tile<f32>
    %ftof_bf16_f32_s = ftof %c5_bf16 : tile<bf16> -> tile<f32>
    // CHECK: ftof %[[c5_bf16]] : tile<bf16> -> tile<f64>
    %ftof_bf16_f64_s = ftof %c5_bf16 : tile<bf16> -> tile<f64>
    // CHECK: ftof %[[c_tensor_bf16]] : tile<2x2xbf16> -> tile<2x2xf32>
    %ftof_bf16_f32_t = ftof %c_tensor_bf16 : tile<2x2xbf16> -> tile<2x2xf32>

    // **** f32 input ****
    // CHECK: ftof %[[c5_f32]] : tile<f32> -> tile<f16>
    %ftof_f32_f16_s = ftof %c5_f32 : tile<f32> -> tile<f16>
    // CHECK: ftof %[[c5_f32]] : tile<f32> -> tile<bf16>
    %ftof_f32_bf16_s = ftof %c5_f32 : tile<f32> -> tile<bf16>
    // CHECK: ftof %[[c5_f32]] : tile<f32> -> tile<f64>
    %ftof_f32_f64_s = ftof %c5_f32 : tile<f32> -> tile<f64>
    // CHECK: ftof %[[c_tensor_f32]] : tile<2x2xf32> -> tile<2x2xf16>
    %ftof_f32_f16_t = ftof %c_tensor_f32 : tile<2x2xf32> -> tile<2x2xf16>
    // CHECK: ftof %[[c_tensor_f32]] : tile<2x2xf32> -> tile<2x2xbf16>
    %ftof_f32_bf16_t = ftof %c_tensor_f32 : tile<2x2xf32> -> tile<2x2xbf16>
    // CHECK: ftof %[[c_tensor_f32]] : tile<2x2xf32> -> tile<2x2xf64>
    %ftof_f32_f64_t = ftof %c_tensor_f32 : tile<2x2xf32> -> tile<2x2xf64>

    // **** f64 input ****
    // CHECK: ftof %[[c5_f64]] : tile<f64> -> tile<f16>
    %ftof_f64_f16_s = ftof %c5_f64 : tile<f64> -> tile<f16>
    // CHECK: ftof %[[c5_f64]] : tile<f64> -> tile<bf16>
    %ftof_f64_bf16_s = ftof %c5_f64 : tile<f64> -> tile<bf16>
    // CHECK: ftof %[[c5_f64]] : tile<f64> -> tile<f32>
    %ftof_f64_f32_s = ftof %c5_f64 : tile<f64> -> tile<f32>
    // CHECK: ftof %[[c_tensor_f64]] : tile<2x2xf64> -> tile<2x2xf32>
    %ftof_f64_f32_t = ftof %c_tensor_f64 : tile<2x2xf64> -> tile<2x2xf32>
  }

  cuda_tile.entry @ftoi() {
    // Constants
    // CHECK: %[[c5_f16:.*]] = constant <f16: 5.000000e+00> : tile<f16>
    %c5_f16 = constant <f16: 5.0> : !cuda_tile.tile<f16>
    // CHECK: %[[c5_bf16:.*]] = constant <bf16: 5.000000e+00> : tile<bf16>
    %c5_bf16 = constant <bf16: 5.0> : !cuda_tile.tile<bf16>
    // CHECK: %[[c_tensor_f32:.*]] = constant <f32: {{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tile<2x2xf32>
    %c_tensor_f32 = constant <f32: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf32>
    // CHECK: %[[c5_f64:.*]] = constant <f64: 5.000000e+00> : tile<f64>
    %c5_f64 = constant <f64: 5.0> : !cuda_tile.tile<f64>

    // **** f16 input ****
    // CHECK: ftoi %[[c5_f16]] signed : tile<f16> -> tile<i1>
    %ftoi_f16_i1_s = ftoi %c5_f16 signed : tile<f16> -> tile<i1>
    // CHECK: ftoi %[[c5_f16]] unsigned : tile<f16> -> tile<i1>
    %ftoi_f16_i1_u = ftoi %c5_f16 unsigned : tile<f16> -> tile<i1>
    // CHECK: ftoi %[[c5_f16]] signed : tile<f16> -> tile<i8>
    %ftoi_f16_i8_s = ftoi %c5_f16 signed : tile<f16> -> tile<i8>
    // CHECK: ftoi %[[c5_f16]] unsigned : tile<f16> -> tile<i8>
    %ftoi_f16_i8_u = ftoi %c5_f16 unsigned : tile<f16> -> tile<i8>
    // CHECK: ftoi %[[c5_f16]] signed : tile<f16> -> tile<i16>
    %ftoi_f16_i16_s = ftoi %c5_f16 signed : tile<f16> -> tile<i16>
    // CHECK: ftoi %[[c5_f16]] unsigned : tile<f16> -> tile<i16>
    %ftoi_f16_i16_u = ftoi %c5_f16 unsigned : tile<f16> -> tile<i16>
    // CHECK: ftoi %[[c5_f16]] signed : tile<f16> -> tile<i32>
    %ftoi_f16_i32_s = ftoi %c5_f16 signed : tile<f16> -> tile<i32>
    // CHECK: ftoi %[[c5_f16]] unsigned : tile<f16> -> tile<i32>
    %ftoi_f16_i32_u = ftoi %c5_f16 unsigned : tile<f16> -> tile<i32>
    // CHECK: ftoi %[[c5_f16]] signed : tile<f16> -> tile<i64>
    %ftoi_f16_i64_s = ftoi %c5_f16 signed : tile<f16> -> tile<i64>
    // CHECK: ftoi %[[c5_f16]] unsigned : tile<f16> -> tile<i64>
    %ftoi_f16_i64_u = ftoi %c5_f16 unsigned : tile<f16> -> tile<i64>

    // **** bf16 input ****
    // CHECK: ftoi %[[c5_bf16]] signed : tile<bf16> -> tile<i1>
    %ftoi_bf16_i1_s = ftoi %c5_bf16 signed : tile<bf16> -> tile<i1>
    // CHECK: ftoi %[[c5_bf16]] unsigned : tile<bf16> -> tile<i1>
    %ftoi_bf16_i1_u = ftoi %c5_bf16 unsigned : tile<bf16> -> tile<i1>
    // CHECK: ftoi %[[c5_bf16]] signed : tile<bf16> -> tile<i8>
    %ftoi_bf16_i8_s = ftoi %c5_bf16 signed : tile<bf16> -> tile<i8>
    // CHECK: ftoi %[[c5_bf16]] unsigned : tile<bf16> -> tile<i8>
    %ftoi_bf16_i8_u = ftoi %c5_bf16 unsigned : tile<bf16> -> tile<i8>
    // CHECK: ftoi %[[c5_bf16]] signed : tile<bf16> -> tile<i16>
    %ftoi_bf16_i16_s = ftoi %c5_bf16 signed : tile<bf16> -> tile<i16>
    // CHECK: ftoi %[[c5_bf16]] unsigned : tile<bf16> -> tile<i16>
    %ftoi_bf16_i16_u = ftoi %c5_bf16 unsigned : tile<bf16> -> tile<i16>
    // CHECK: ftoi %[[c5_bf16]] signed : tile<bf16> -> tile<i32>
    %ftoi_bf16_i32_s = ftoi %c5_bf16 signed : tile<bf16> -> tile<i32>
    // CHECK: ftoi %[[c5_bf16]] unsigned : tile<bf16> -> tile<i32>
    %ftoi_bf16_i32_u = ftoi %c5_bf16 unsigned : tile<bf16> -> tile<i32>
    // CHECK: ftoi %[[c5_bf16]] signed : tile<bf16> -> tile<i64>
    %ftoi_bf16_i64_s = ftoi %c5_bf16 signed : tile<bf16> -> tile<i64>
    // CHECK: ftoi %[[c5_bf16]] unsigned : tile<bf16> -> tile<i64>
    %ftoi_bf16_i64_u = ftoi %c5_bf16 unsigned : tile<bf16> -> tile<i64>

    // **** f32 input ****
    // CHECK: ftoi %[[c_tensor_f32]] signed : tile<2x2xf32> -> tile<2x2xi1>
    %ftoi_f32_i1_s = ftoi %c_tensor_f32 signed : tile<2x2xf32> -> tile<2x2xi1>
    // CHECK: ftoi %[[c_tensor_f32]] unsigned : tile<2x2xf32> -> tile<2x2xi1>
    %ftoi_f32_i1_u = ftoi %c_tensor_f32 unsigned : tile<2x2xf32> -> tile<2x2xi1>
    // CHECK: ftoi %[[c_tensor_f32]] signed : tile<2x2xf32> -> tile<2x2xi8>
    %ftoi_f32_i8_s = ftoi %c_tensor_f32 signed : tile<2x2xf32> -> tile<2x2xi8>
    // CHECK: ftoi %[[c_tensor_f32]] unsigned : tile<2x2xf32> -> tile<2x2xi8>
    %ftoi_f32_i8_u = ftoi %c_tensor_f32 unsigned : tile<2x2xf32> -> tile<2x2xi8>
    // CHECK: ftoi %[[c_tensor_f32]] signed : tile<2x2xf32> -> tile<2x2xi16>
    %ftoi_f32_i16_s = ftoi %c_tensor_f32 signed : tile<2x2xf32> -> tile<2x2xi16>
    // CHECK: ftoi %[[c_tensor_f32]] unsigned : tile<2x2xf32> -> tile<2x2xi16>
    %ftoi_f32_i16_u = ftoi %c_tensor_f32 unsigned : tile<2x2xf32> -> tile<2x2xi16>
    // CHECK: ftoi %[[c_tensor_f32]] signed : tile<2x2xf32> -> tile<2x2xi32>
    %ftoi_f32_i32_s = ftoi %c_tensor_f32 signed : tile<2x2xf32> -> tile<2x2xi32>
    // CHECK: ftoi %[[c_tensor_f32]] unsigned : tile<2x2xf32> -> tile<2x2xi32>
    %ftoi_f32_i32_u = ftoi %c_tensor_f32 unsigned : tile<2x2xf32> -> tile<2x2xi32>
    // CHECK: ftoi %[[c_tensor_f32]] signed : tile<2x2xf32> -> tile<2x2xi64>
    %ftoi_f32_i64_s = ftoi %c_tensor_f32 signed : tile<2x2xf32> -> tile<2x2xi64>
    // CHECK: ftoi %[[c_tensor_f32]] unsigned : tile<2x2xf32> -> tile<2x2xi64>
    %ftoi_f32_i64_u = ftoi %c_tensor_f32 unsigned : tile<2x2xf32> -> tile<2x2xi64>
    // CHECK: ftoi %[[c_tensor_f32]] unsigned : tile<2x2xf32> -> tile<2x2xi64>
    %ftoi_f32_i64_u_explicit_rnd = ftoi %c_tensor_f32 unsigned rounding<nearest_int_to_zero> : tile<2x2xf32> -> tile<2x2xi64>

    // **** f64 input ****
    // CHECK: ftoi %[[c5_f64]] signed : tile<f64> -> tile<i1>
    %ftoi_f64_i1_s = ftoi %c5_f64 signed : tile<f64> -> tile<i1>
    // CHECK: ftoi %[[c5_f64]] unsigned : tile<f64> -> tile<i1>
    %ftoi_f64_i1_u = ftoi %c5_f64 unsigned : tile<f64> -> tile<i1>
    // CHECK: ftoi %[[c5_f64]] signed : tile<f64> -> tile<i8>
    %ftoi_f64_i8_s = ftoi %c5_f64 signed : tile<f64> -> tile<i8>
    // CHECK: ftoi %[[c5_f64]] unsigned : tile<f64> -> tile<i8>
    %ftoi_f64_i8_u = ftoi %c5_f64 unsigned : tile<f64> -> tile<i8>
    // CHECK: ftoi %[[c5_f64]] signed : tile<f64> -> tile<i16>
    %ftoi_f64_i16_s = ftoi %c5_f64 signed : tile<f64> -> tile<i16>
    // CHECK: ftoi %[[c5_f64]] unsigned : tile<f64> -> tile<i16>
    %ftoi_f64_i16_u = ftoi %c5_f64 unsigned : tile<f64> -> tile<i16>
    // CHECK: ftoi %[[c5_f64]] signed : tile<f64> -> tile<i32>
    %ftoi_f64_i32_s = ftoi %c5_f64 signed : tile<f64> -> tile<i32>
    // CHECK: ftoi %[[c5_f64]] unsigned : tile<f64> -> tile<i32>
    %ftoi_f64_i32_u = ftoi %c5_f64 unsigned : tile<f64> -> tile<i32>
    // CHECK: ftoi %[[c5_f64]] signed : tile<f64> -> tile<i64>
    %ftoi_f64_i64_s = ftoi %c5_f64 signed : tile<f64> -> tile<i64>
    // CHECK: ftoi %[[c5_f64]] unsigned : tile<f64> -> tile<i64>
    %ftoi_f64_i64_u = ftoi %c5_f64 unsigned : tile<f64> -> tile<i64>
  }

  cuda_tile.entry @itof() {
    // Constants
    // CHECK: %[[c_i1:.*]] = constant <i1: true> : tile<i1>
    %c_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[c_i8:.*]] = constant <i8: 42> : tile<i8>
    %c_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[c_i16:.*]] = constant <i16: 42> : tile<i16>
    %c_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[c_i32:.*]] = constant <i32: 42> : tile<i32>
    %c_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[c_i64:.*]] = constant <i64: 42> : tile<i64>
    %c_i64 = constant <i64: 42> : !cuda_tile.tile<i64>

    // **** i1 input ****
    // CHECK: itof %[[c_i1]] signed : tile<i1> -> tile<f16>
    %itof_i1_f16_s = itof %c_i1 signed : tile<i1> -> tile<f16>
    // CHECK: itof %[[c_i1]] unsigned : tile<i1> -> tile<f16>
    %itof_i1_f16_u = itof %c_i1 unsigned : tile<i1> -> tile<f16>
    // CHECK: itof %[[c_i1]] signed : tile<i1> -> tile<bf16>
    %itof_i1_bf16_s = itof %c_i1 signed : tile<i1> -> tile<bf16>
    // CHECK: itof %[[c_i1]] unsigned : tile<i1> -> tile<bf16>
    %itof_i1_bf16_u = itof %c_i1 unsigned : tile<i1> -> tile<bf16>
    // CHECK: itof %[[c_i1]] signed : tile<i1> -> tile<f32>
    %itof_i1_f32_s = itof %c_i1 signed : tile<i1> -> tile<f32>
    // CHECK: itof %[[c_i1]] unsigned : tile<i1> -> tile<f32>
    %itof_i1_f32_u = itof %c_i1 unsigned : tile<i1> -> tile<f32>
    // CHECK: itof %[[c_i1]] signed : tile<i1> -> tile<f64>
    %itof_i1_f64_s = itof %c_i1 signed : tile<i1> -> tile<f64>
    // CHECK: itof %[[c_i1]] unsigned : tile<i1> -> tile<f64>
    %itof_i1_f64_u = itof %c_i1 unsigned : tile<i1> -> tile<f64>

    // **** i8 input ****
    // CHECK: itof %[[c_i8]] signed : tile<i8> -> tile<f16>
    %itof_i8_f16_s = itof %c_i8 signed : tile<i8> -> tile<f16>
    // CHECK: itof %[[c_i8]] unsigned : tile<i8> -> tile<f16>
    %itof_i8_f16_u = itof %c_i8 unsigned : tile<i8> -> tile<f16>
    // CHECK: itof %[[c_i8]] signed : tile<i8> -> tile<bf16>
    %itof_i8_bf16_s = itof %c_i8 signed : tile<i8> -> tile<bf16>
    // CHECK: itof %[[c_i8]] unsigned : tile<i8> -> tile<bf16>
    %itof_i8_bf16_u = itof %c_i8 unsigned : tile<i8> -> tile<bf16>
    // CHECK: itof %[[c_i8]] signed : tile<i8> -> tile<f32>
    %itof_i8_f32_s = itof %c_i8 signed : tile<i8> -> tile<f32>
    // CHECK: itof %[[c_i8]] unsigned : tile<i8> -> tile<f32>
    %itof_i8_f32_u = itof %c_i8 unsigned : tile<i8> -> tile<f32>
    // CHECK: itof %[[c_i8]] signed : tile<i8> -> tile<f64>
    %itof_i8_f64_s = itof %c_i8 signed : tile<i8> -> tile<f64>
    // CHECK: itof %[[c_i8]] unsigned : tile<i8> -> tile<f64>
    %itof_i8_f64_u = itof %c_i8 unsigned : tile<i8> -> tile<f64>

    // **** i16 input ****
    // CHECK: itof %[[c_i16]] signed : tile<i16> -> tile<f16>
    %itof_i16_f16_s = itof %c_i16 signed : tile<i16> -> tile<f16>
    // CHECK: itof %[[c_i16]] unsigned : tile<i16> -> tile<f16>
    %itof_i16_f16_u = itof %c_i16 unsigned : tile<i16> -> tile<f16>
    // CHECK: itof %[[c_i16]] signed : tile<i16> -> tile<bf16>
    %itof_i16_bf16_s = itof %c_i16 signed : tile<i16> -> tile<bf16>
    // CHECK: itof %[[c_i16]] unsigned : tile<i16> -> tile<bf16>
    %itof_i16_bf16_u = itof %c_i16 unsigned : tile<i16> -> tile<bf16>
    // CHECK: itof %[[c_i16]] signed : tile<i16> -> tile<f32>
    %itof_i16_f32_s = itof %c_i16 signed : tile<i16> -> tile<f32>
    // CHECK: itof %[[c_i16]] unsigned : tile<i16> -> tile<f32>
    %itof_i16_f32_u = itof %c_i16 unsigned : tile<i16> -> tile<f32>
    // CHECK: itof %[[c_i16]] signed : tile<i16> -> tile<f64>
    %itof_i16_f64_s = itof %c_i16 signed : tile<i16> -> tile<f64>
    // CHECK: itof %[[c_i16]] unsigned : tile<i16> -> tile<f64>
    %itof_i16_f64_u = itof %c_i16 unsigned : tile<i16> -> tile<f64>

    // **** i32 input ****
    // CHECK: itof %[[c_i32]] signed : tile<i32> -> tile<f16>
    %itof_i32_f16_s = itof %c_i32 signed : tile<i32> -> tile<f16>
    // CHECK: itof %[[c_i32]] unsigned : tile<i32> -> tile<f16>
    %itof_i32_f16_u = itof %c_i32 unsigned : tile<i32> -> tile<f16>
    // CHECK: itof %[[c_i32]] signed : tile<i32> -> tile<bf16>
    %itof_i32_bf16_s = itof %c_i32 signed : tile<i32> -> tile<bf16>
    // CHECK: itof %[[c_i32]] unsigned : tile<i32> -> tile<bf16>
    %itof_i32_bf16_u = itof %c_i32 unsigned : tile<i32> -> tile<bf16>
    // CHECK: itof %[[c_i32]] signed : tile<i32> -> tile<f32>
    %itof_i32_f32_s = itof %c_i32 signed : tile<i32> -> tile<f32>
    // CHECK: itof %[[c_i32]] unsigned : tile<i32> -> tile<f32>
    %itof_i32_f32_u = itof %c_i32 unsigned : tile<i32> -> tile<f32>
    // CHECK: itof %[[c_i32]] signed : tile<i32> -> tile<f64>
    %itof_i32_f64_s = itof %c_i32 signed : tile<i32> -> tile<f64>
    // CHECK: itof %[[c_i32]] unsigned : tile<i32> -> tile<f64>
    %itof_i32_f64_u = itof %c_i32 unsigned : tile<i32> -> tile<f64>

    // **** i64 input ****
    // CHECK: itof %[[c_i64]] signed : tile<i64> -> tile<f16>
    %itof_i64_f16_s = itof %c_i64 signed : tile<i64> -> tile<f16>
    // CHECK: itof %[[c_i64]] unsigned : tile<i64> -> tile<f16>
    %itof_i64_f16_u = itof %c_i64 unsigned : tile<i64> -> tile<f16>
    // CHECK: itof %[[c_i64]] signed : tile<i64> -> tile<bf16>
    %itof_i64_bf16_s = itof %c_i64 signed : tile<i64> -> tile<bf16>
    // CHECK: itof %[[c_i64]] unsigned : tile<i64> -> tile<bf16>
    %itof_i64_bf16_u = itof %c_i64 unsigned : tile<i64> -> tile<bf16>
    // CHECK: itof %[[c_i64]] signed : tile<i64> -> tile<f32>
    %itof_i64_f32_s = itof %c_i64 signed : tile<i64> -> tile<f32>
    // CHECK: itof %[[c_i64]] unsigned : tile<i64> -> tile<f32>
    %itof_i64_f32_u = itof %c_i64 unsigned : tile<i64> -> tile<f32>
    // CHECK: itof %[[c_i64]] signed : tile<i64> -> tile<f64>
    %itof_i64_f64_s = itof %c_i64 signed : tile<i64> -> tile<f64>
    // CHECK: itof %[[c_i64]] unsigned : tile<i64> -> tile<f64>
    %itof_i64_f64_u = itof %c_i64 unsigned : tile<i64> -> tile<f64>
  }

  cuda_tile.entry @itof_tensor() {
    // Constants
    // CHECK: %[[c_tensor_i1:.*]] = constant <i1: {{\[\[}}true, false], [true, true]]> : tile<2x2xi1>
    %c_tensor_i1 = constant <i1: [[true, false], [true, true]]> : !cuda_tile.tile<2x2xi1>
    // CHECK: %[[c_tensor_i8:.*]] = constant <i8: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi8>
    %c_tensor_i8 = constant <i8: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi8>
    // CHECK: %[[c_tensor_i16:.*]] = constant <i16: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi16>
    %c_tensor_i16 = constant <i16: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi16>
    // CHECK: %[[c_tensor_i32:.*]] = constant <i32: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi32>
    %c_tensor_i32 = constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>
    // CHECK: %[[c_tensor_i64:.*]] = constant <i64: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi64>
    %c_tensor_i64 = constant <i64: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi64>

    // **** i1 input ****
    // ** Tensor **
    // CHECK: itof %[[c_tensor_i1]] signed : tile<2x2xi1> -> tile<2x2xf16>
    %itof_tensor_i1_f16_s = itof %c_tensor_i1 signed : tile<2x2xi1> -> tile<2x2xf16>
    // CHECK: itof %[[c_tensor_i1]] unsigned : tile<2x2xi1> -> tile<2x2xf16>
    %itof_tensor_i1_f16_u = itof %c_tensor_i1 unsigned : tile<2x2xi1> -> tile<2x2xf16>
    // CHECK: itof %[[c_tensor_i1]] signed : tile<2x2xi1> -> tile<2x2xbf16>
    %itof_tensor_i1_bf16_s = itof %c_tensor_i1 signed : tile<2x2xi1> -> tile<2x2xbf16>
    // CHECK: itof %[[c_tensor_i1]] unsigned : tile<2x2xi1> -> tile<2x2xbf16>
    %itof_tensor_i1_bf16_u = itof %c_tensor_i1 unsigned : tile<2x2xi1> -> tile<2x2xbf16>
    // CHECK: itof %[[c_tensor_i1]] signed : tile<2x2xi1> -> tile<2x2xf32>
    %itof_tensor_i1_f32_s = itof %c_tensor_i1 signed : tile<2x2xi1> -> tile<2x2xf32>
    // CHECK: itof %[[c_tensor_i1]] unsigned : tile<2x2xi1> -> tile<2x2xf32>
    %itof_tensor_i1_f32_u = itof %c_tensor_i1 unsigned : tile<2x2xi1> -> tile<2x2xf32>
    // CHECK: itof %[[c_tensor_i1]] signed : tile<2x2xi1> -> tile<2x2xf64>
    %itof_tensor_i1_f64_s = itof %c_tensor_i1 signed : tile<2x2xi1> -> tile<2x2xf64>
    // CHECK: itof %[[c_tensor_i1]] unsigned : tile<2x2xi1> -> tile<2x2xf64>
    %itof_tensor_i1_f64_u = itof %c_tensor_i1 unsigned : tile<2x2xi1> -> tile<2x2xf64>

    // **** i8 input ****
    // ** Tensor **
    // CHECK: itof %[[c_tensor_i8]] signed : tile<2x2xi8> -> tile<2x2xf16>
    %itof_tensor_i8_f16_s = itof %c_tensor_i8 signed : tile<2x2xi8> -> tile<2x2xf16>
    // CHECK: itof %[[c_tensor_i8]] unsigned : tile<2x2xi8> -> tile<2x2xf16>
    %itof_tensor_i8_f16_u = itof %c_tensor_i8 unsigned : tile<2x2xi8> -> tile<2x2xf16>

    // **** i16 input ****
    // ** Tensor **
    // CHECK: itof %[[c_tensor_i16]] signed : tile<2x2xi16> -> tile<2x2xbf16>
    %itof_tensor_i16_bf16_s = itof %c_tensor_i16 signed : tile<2x2xi16> -> tile<2x2xbf16>
    // CHECK: itof %[[c_tensor_i16]] unsigned : tile<2x2xi16> -> tile<2x2xbf16>
    %itof_tensor_i16_bf16_u = itof %c_tensor_i16 unsigned : tile<2x2xi16> -> tile<2x2xbf16>

    // **** i32 input ****
    // ** Tensor **
    // CHECK: itof %[[c_tensor_i32]] signed : tile<2x2xi32> -> tile<2x2xf32>
    %itof_tensor_i32_f32_s = itof %c_tensor_i32 signed : tile<2x2xi32> -> tile<2x2xf32>
    // CHECK: itof %[[c_tensor_i32]] unsigned : tile<2x2xi32> -> tile<2x2xf32>
    %itof_tensor_i32_f32_u = itof %c_tensor_i32 unsigned : tile<2x2xi32> -> tile<2x2xf32>
    // CHECK: itof %[[c_tensor_i32]] signed : tile<2x2xi32> -> tile<2x2xf64>
    %itof_tensor_i32_f64_s = itof %c_tensor_i32 signed : tile<2x2xi32> -> tile<2x2xf64>
    // CHECK: itof %[[c_tensor_i32]] unsigned : tile<2x2xi32> -> tile<2x2xf64>
    %itof_tensor_i32_f64_u = itof %c_tensor_i32 unsigned : tile<2x2xi32> -> tile<2x2xf64>

    // **** i64 input ****
    // ** Tensor **
    // CHECK: itof %[[c_tensor_i64]] signed : tile<2x2xi64> -> tile<2x2xf64>
    %itof_tensor_i64_f64_s = itof %c_tensor_i64 signed : tile<2x2xi64> -> tile<2x2xf64>
    // CHECK: itof %[[c_tensor_i64]] unsigned : tile<2x2xi64> -> tile<2x2xf64>
    %itof_tensor_i64_f64_u = itof %c_tensor_i64 unsigned : tile<2x2xi64> -> tile<2x2xf64>
  }

  cuda_tile.entry @trunci_scalar() {
    // Constants
    // CHECK: %[[C_I64:.*]] = constant <i64: 42> : tile<i64>
    %c_i64 = constant <i64: 42> : !cuda_tile.tile<i64>
    // CHECK: %[[C_I32:.*]] = constant <i32: 42> : tile<i32>
    %c_i32 = constant <i32: 42> : !cuda_tile.tile<i32>
    // CHECK: %[[C_I16:.*]] = constant <i16: 42> : tile<i16>
    %c_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[C_I8:.*]] = constant <i8: 42> : tile<i8>
    %c_i8 = constant <i8: 42> : !cuda_tile.tile<i8>

    // Truncations
    // CHECK: trunci %[[C_I64]] : tile<i64> -> tile<i32>
    %trunci_i64_i32 = trunci %c_i64 : tile<i64> -> tile<i32>
    // CHECK: trunci %[[C_I64]] : tile<i64> -> tile<i16>
    %trunci_i64_i16 = trunci %c_i64 : tile<i64> -> tile<i16>
    // CHECK: trunci %[[C_I64]] : tile<i64> -> tile<i8>
    %trunci_i64_i8 = trunci %c_i64 : tile<i64> -> tile<i8>
    // CHECK: trunci %[[C_I64]] : tile<i64> -> tile<i1>
    %trunci_i64_i1 = trunci %c_i64 : tile<i64> -> tile<i1>

    // CHECK: trunci %[[C_I32]] : tile<i32> -> tile<i16>
    %trunci_i32_i16 = trunci %c_i32 : tile<i32> -> tile<i16>
    // CHECK: trunci %[[C_I32]] : tile<i32> -> tile<i8>
    %trunci_i32_i8 = trunci %c_i32 : tile<i32> -> tile<i8>
    // CHECK: trunci %[[C_I32]] : tile<i32> -> tile<i1>
    %trunci_i32_i1 = trunci %c_i32 : tile<i32> -> tile<i1>

    // CHECK: trunci %[[C_I16]] : tile<i16> -> tile<i8>
    %trunci_i16_i8 = trunci %c_i16 : tile<i16> -> tile<i8>
    // CHECK: trunci %[[C_I16]] : tile<i16> -> tile<i1>
    %trunci_i16_i1 = trunci %c_i16 : tile<i16> -> tile<i1>

    // CHECK: trunci %[[C_I8]] : tile<i8> -> tile<i1>
    %trunci_i8_i1 = trunci %c_i8 : tile<i8> -> tile<i1>
  }

  cuda_tile.entry @trunci_tensor() {
    // CHECK: %[[c_itensor_i64:.*]] = constant <i64: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi64>
    %c_itensor_i64 = constant <i64: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi64>
    // CHECK: %[[c_itensor_i32:.*]] = constant <i32: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi32>
    %c_itensor_i32 = constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>
    // CHECK: %[[c_itensor_i16:.*]] = constant <i16: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi16>
    %c_itensor_i16 = constant <i16: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi16>
    // CHECK: %[[c_itensor_i8:.*]] = constant <i8: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi8>
    %c_itensor_i8 = constant <i8: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi8>

    // CHECK: trunci %[[c_itensor_i64]] : tile<2x2xi64> -> tile<2x2xi32>
    %trunci_i64_i32 = trunci %c_itensor_i64 : tile<2x2xi64> -> tile<2x2xi32>
    // CHECK: trunci %[[c_itensor_i32]] : tile<2x2xi32> -> tile<2x2xi16>
    %trunci_i32_i16 = trunci %c_itensor_i32 : tile<2x2xi32> -> tile<2x2xi16>
    // CHECK: trunci %[[c_itensor_i16]] : tile<2x2xi16> -> tile<2x2xi8>
    %trunci_i16_i8 = trunci %c_itensor_i16 : tile<2x2xi16> -> tile<2x2xi8>
    // CHECK: trunci %[[c_itensor_i8]] : tile<2x2xi8> -> tile<2x2xi1>
    %trunci_i8_i1 = trunci %c_itensor_i8 : tile<2x2xi8> -> tile<2x2xi1>
  }

  cuda_tile.entry @exti_signed() {
    // Constants
    // CHECK: %[[C_I1:.*]] = constant <i1: true> : tile<i1>
    %c_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[C_I8:.*]] = constant <i8: 42> : tile<i8>
    %c_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[C_I16:.*]] = constant <i16: 42> : tile<i16>
    %c_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[C_I32:.*]] = constant <i32: 42> : tile<i32>
    %c_i32 = constant <i32: 42> : !cuda_tile.tile<i32>

    // Signed Extensions
    // CHECK: exti %[[C_I1]] signed : tile<i1> -> tile<i8>
    %exti_i1_i8_s = exti %c_i1 signed : tile<i1> -> tile<i8>
    // CHECK: exti %[[C_I1]] signed : tile<i1> -> tile<i16>
    %exti_i1_i16_s = exti %c_i1 signed : tile<i1> -> tile<i16>
    // CHECK: exti %[[C_I1]] signed : tile<i1> -> tile<i32>
    %exti_i1_i32_s = exti %c_i1 signed : tile<i1> -> tile<i32>
    // CHECK: exti %[[C_I1]] signed : tile<i1> -> tile<i64>
    %exti_i1_i64_s = exti %c_i1 signed : tile<i1> -> tile<i64>

    // CHECK: exti %[[C_I8]] signed : tile<i8> -> tile<i16>
    %exti_i8_i16_s = exti %c_i8 signed : tile<i8> -> tile<i16>
    // CHECK: exti %[[C_I8]] signed : tile<i8> -> tile<i32>
    %exti_i8_i32_s = exti %c_i8 signed : tile<i8> -> tile<i32>
    // CHECK: exti %[[C_I8]] signed : tile<i8> -> tile<i64>
    %exti_i8_i64_s = exti %c_i8 signed : tile<i8> -> tile<i64>

    // CHECK: exti %[[C_I16]] signed : tile<i16> -> tile<i32>
    %exti_i16_i32_s = exti %c_i16 signed : tile<i16> -> tile<i32>
    // CHECK: exti %[[C_I16]] signed : tile<i16> -> tile<i64>
    %exti_i16_i64_s = exti %c_i16 signed : tile<i16> -> tile<i64>

    // CHECK: exti %[[C_I32]] signed : tile<i32> -> tile<i64>
    %exti_i32_i64_s = exti %c_i32 signed : tile<i32> -> tile<i64>
  }

  cuda_tile.entry @exti_unsigned() {
    // Constants
    // CHECK: %[[C_I1:.*]] = constant <i1: true> : tile<i1>
    %c_i1 = constant <i1: true> : !cuda_tile.tile<i1>
    // CHECK: %[[C_I8:.*]] = constant <i8: 42> : tile<i8>
    %c_i8 = constant <i8: 42> : !cuda_tile.tile<i8>
    // CHECK: %[[C_I16:.*]] = constant <i16: 42> : tile<i16>
    %c_i16 = constant <i16: 42> : !cuda_tile.tile<i16>
    // CHECK: %[[C_I32:.*]] = constant <i32: 42> : tile<i32>
    %c_i32 = constant <i32: 42> : !cuda_tile.tile<i32>

    // Unsigned Extensions
    // CHECK: exti %[[C_I1]] unsigned : tile<i1> -> tile<i8>
    %exti_i1_i8_u = exti %c_i1 unsigned : tile<i1> -> tile<i8>
    // CHECK: exti %[[C_I1]] unsigned : tile<i1> -> tile<i16>
    %exti_i1_i16_u = exti %c_i1 unsigned : tile<i1> -> tile<i16>
    // CHECK: exti %[[C_I1]] unsigned : tile<i1> -> tile<i32>
    %exti_i1_i32_u = exti %c_i1 unsigned : tile<i1> -> tile<i32>
    // CHECK: exti %[[C_I1]] unsigned : tile<i1> -> tile<i64>
    %exti_i1_i64_u = exti %c_i1 unsigned : tile<i1> -> tile<i64>

    // CHECK: exti %[[C_I8]] unsigned : tile<i8> -> tile<i16>
    %exti_i8_i16_u = exti %c_i8 unsigned : tile<i8> -> tile<i16>
    // CHECK: exti %[[C_I8]] unsigned : tile<i8> -> tile<i32>
    %exti_i8_i32_u = exti %c_i8 unsigned : tile<i8> -> tile<i32>
    // CHECK: exti %[[C_I8]] unsigned : tile<i8> -> tile<i64>
    %exti_i8_i64_u = exti %c_i8 unsigned : tile<i8> -> tile<i64>

    // CHECK: exti %[[C_I16]] unsigned : tile<i16> -> tile<i32>
    %exti_i16_i32_u = exti %c_i16 unsigned : tile<i16> -> tile<i32>
    // CHECK: exti %[[C_I16]] unsigned : tile<i16> -> tile<i64>
    %exti_i16_i64_u = exti %c_i16 unsigned : tile<i16> -> tile<i64>

    // CHECK: exti %[[C_I32]] unsigned : tile<i32> -> tile<i64>
    %exti_i32_i64_u = exti %c_i32 unsigned : tile<i32> -> tile<i64>
  }

  cuda_tile.entry @exti_tensor_signed() {
    // CHECK: %[[c_itensor_i1:.*]] = constant <i1: {{\[\[}}true, false], [true, true]]> : tile<2x2xi1>
    %c_itensor_i1 = constant <i1: [[true, false], [true, true]]> : !cuda_tile.tile<2x2xi1>
    // CHECK: %[[c_itensor_i8:.*]] = constant <i8: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi8>
    %c_itensor_i8 = constant <i8: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi8>
    // CHECK: %[[c_itensor_i16:.*]] = constant <i16: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi16>
    %c_itensor_i16 = constant <i16: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi16>
    // CHECK: %[[c_itensor_i32:.*]] = constant <i32: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi32>
    %c_itensor_i32 = constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>

    // CHECK: exti %[[c_itensor_i1]] signed : tile<2x2xi1> -> tile<2x2xi8>
    %exti_i1_i8 = exti %c_itensor_i1 signed : tile<2x2xi1> -> tile<2x2xi8>
    // CHECK: exti %[[c_itensor_i8]] signed : tile<2x2xi8> -> tile<2x2xi16>
    %exti_i8_i16 = exti %c_itensor_i8 signed : tile<2x2xi8> -> tile<2x2xi16>
    // CHECK: exti %[[c_itensor_i16]] signed : tile<2x2xi16> -> tile<2x2xi32>
    %exti_i16_i32 = exti %c_itensor_i16 signed : tile<2x2xi16> -> tile<2x2xi32>
    // CHECK: exti %[[c_itensor_i32]] signed : tile<2x2xi32> -> tile<2x2xi64>
    %exti_i32_i64 = exti %c_itensor_i32 signed : tile<2x2xi32> -> tile<2x2xi64>
  }

  cuda_tile.entry @exti_tensor_unsigned() {
    // CHECK: %[[c_itensor_i1:.*]] = constant <i1: {{\[\[}}true, false], [true, true]]> : tile<2x2xi1>
    %c_itensor_i1 = constant <i1: [[true, false], [true, true]]> : !cuda_tile.tile<2x2xi1>
    // CHECK: %[[c_itensor_i8:.*]] = constant <i8: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi8>
    %c_itensor_i8 = constant <i8: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi8>
    // CHECK: %[[c_itensor_i16:.*]] = constant <i16: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi16>
    %c_itensor_i16 = constant <i16: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi16>
    // CHECK: %[[c_itensor_i32:.*]] = constant <i32: {{\[\[}}1, 2], [3, 4]]> : tile<2x2xi32>
    %c_itensor_i32 = constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>

    // CHECK: exti %[[c_itensor_i1]] unsigned : tile<2x2xi1> -> tile<2x2xi8>
    %exti_i1_i8_u = exti %c_itensor_i1 unsigned : tile<2x2xi1> -> tile<2x2xi8>
    // CHECK: exti %[[c_itensor_i8]] unsigned : tile<2x2xi8> -> tile<2x2xi16>
    %exti_i8_i16_u = exti %c_itensor_i8 unsigned : tile<2x2xi8> -> tile<2x2xi16>
    // CHECK: exti %[[c_itensor_i16]] unsigned : tile<2x2xi16> -> tile<2x2xi32>
    %exti_i16_i32_u = exti %c_itensor_i16 unsigned : tile<2x2xi16> -> tile<2x2xi32>
    // CHECK: exti %[[c_itensor_i32]] unsigned : tile<2x2xi32> -> tile<2x2xi64>
    %exti_i32_i64_u = exti %c_itensor_i32 unsigned : tile<2x2xi32> -> tile<2x2xi64>
  }

  cuda_tile.entry @iota_scalar() {
    // Generate sequences of different lengths
    // CHECK: %[[iota_4:.*]] = iota : tile<4xi32>
    %iota_4 = iota : !cuda_tile.tile<4xi32>
    // CHECK: %[[iota_8:.*]] = iota : tile<8xi32>
    %iota_8 = iota : !cuda_tile.tile<8xi32>
    // CHECK: %[[iota_16:.*]] = iota : tile<16xi32>
    %iota_16 = iota : !cuda_tile.tile<16xi32>
    // CHECK: %[[iota_32:.*]] = iota : tile<32xi32>
    %iota_32 = iota : !cuda_tile.tile<32xi32>
    // CHECK: %[[iota_64:.*]] = iota : tile<64xi32>
    %iota_64 = iota : !cuda_tile.tile<64xi32>

    // Generate sequences with different integer types
    // CHECK: %[[iota_i8:.*]] = iota : tile<4xi8>
    %iota_i8 = iota : !cuda_tile.tile<4xi8>
    // CHECK: %[[iota_i16:.*]] = iota : tile<4xi16>
    %iota_i16 = iota : !cuda_tile.tile<4xi16>
    // CHECK: %[[iota_i32:.*]] = iota : tile<4xi32>
    %iota_i32 = iota : !cuda_tile.tile<4xi32>
    // CHECK: %[[iota_i64:.*]] = iota : tile<4xi64>
    %iota_i64 = iota : !cuda_tile.tile<4xi64>
  }
}
