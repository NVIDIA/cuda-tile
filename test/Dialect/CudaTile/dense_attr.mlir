// RUN: cuda-tile-opt %s -split-input-file | FileCheck %s

// Test basic valid constants: hex strings, scalar splats, and arrays

cuda_tile.module @kernels {
  entry @kernel() {
    // Valid hex strings
    // CHECK: %{{.*}} = constant <i16: -1> : tile<i16>
    %1 = constant <i16: 0xFFFF> : tile<i16>
    // CHECK: %{{.*}} = constant <i32: 305419896> : tile<i32>
    %2 = constant <i32: 0x12345678> : tile<i32>
    // CHECK: %{{.*}} = constant <i16: 4267> : tile<i16>
    %3 = constant <i16: 0x10AB> : tile<i16>
    
    // Valid scalar splats
    // CHECK: %{{.*}} = constant <i32: 42> : tile<4x4xi32>
    %4 = constant <i32: 42> : tile<4x4xi32>
    // CHECK: %{{.*}} = constant <f32: 1.500000e+00> : tile<2x4x4xf32>
    %5 = constant <f32: 1.5> : tile<2x4x4xf32>
    // CHECK: %{{.*}} = constant <i1: true> : tile<8xi1>
    %6 = constant <i1: true> : tile<8xi1>
    
    // Valid arrays with matching shapes
    // CHECK: %{{.*}} = constant <i32: {{\[}}{{\[}}1, 2{{\]}}, {{\[}}3, 4{{\]}}{{\]}}> : tile<2x2xi32>
    %7 = constant <i32: [[1, 2], [3, 4]]> : tile<2x2xi32>
    // CHECK: %{{.*}} = constant <f32: {{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00{{\]}}> : tile<4xf32>
    %8 = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %{{.*}} = constant <i1: {{\[}}{{\[}}{{\[}}true, false{{\]}}{{\]}}, {{\[}}{{\[}}false, true{{\]}}{{\]}}{{\]}}> : tile<2x1x2xi1>
    %9 = constant <i1: [[[true, false]], [[false, true]]]> : tile<2x1x2xi1>
    return
  }
}

// -----
// Test integer bitwidth matching (with and without quotes)

cuda_tile.module @kernels {
  entry @kernel() {
    // i8 tests
    // CHECK: %{{.*}} = constant <i8: -1> : tile<i8>
    %1 = constant <i8: 0xFF> : tile<i8>
    
    // i16 tests
    // CHECK: %{{.*}} = constant <i16: 4660> : tile<i16>
    %3 = constant <i16: 0x1234> : tile<i16>
    
    // i32 tests
    // CHECK: %{{.*}} = constant <i32: 305419896> : tile<i32>
    %5 = constant <i32: 0x12345678> : tile<i32>
    
    // i64 tests
    // CHECK: %{{.*}} = constant <i64: 1311768467463790320> : tile<i64>
    %7 = constant <i64: 0x123456789ABCDEF0> : tile<i64>
    // CHECK: %{{.*}} = constant <i64: 9223372036854775807> : tile<i64>
    %8 = constant <i64: 9223372036854775807> : tile<i64>
    // CHECK: %{{.*}} = constant <i64: -9223372036854775808> : tile<i64>
    %9 = constant <i64: -9223372036854775808> : tile<i64>

    return
  }
}

// -----
// Test float bitwidth matching (with and without quotes)

cuda_tile.module @kernels {
  entry @kernel() {
    // f16 tests
    // CHECK: %{{.*}} = constant <f16: 1.000000e+00> : tile<f16>
    %1 = constant <f16: 0x3C00> : tile<f16>  // 1.0 in f16
    
    // f32 tests
    // CHECK: %{{.*}} = constant <f32: 1.000000e+00> : tile<f32>
    %3 = constant <f32: 0x3F800000> : tile<f32>  // 1.0 in f32
  
    // f64 tests
    // CHECK: %{{.*}} = constant <f64: 1.000000e+00> : tile<f64>
    %5 = constant <f64: 0x3FF0000000000000> : tile<f64>  // 1.0 in f64
    return
  }
}

// -----
// Test mixed valid hex constants with correct bitwidths

cuda_tile.module @kernels {
  entry @kernel() {
    // CHECK: %{{.*}} = constant <i16: -12817> : tile<i16>
    %1 = constant <i16: 0xCDEF> : tile<i16>
    // CHECK: %{{.*}} = constant <i32: -2023406815> : tile<i32>
    %2 = constant <i32: 0x87654321> : tile<i32>
    // CHECK: %{{.*}} = constant <f16: 2.000000e+00> : tile<f16>
    %4 = constant <f16: 0x4000> : tile<f16>  // 2.0 in f16
    // CHECK: %{{.*}} = constant <f32: 2.000000e+00> : tile<f32>
    %5 = constant <f32: 0x40000000> : tile<f32>  // 2.0 in f32
    // CHECK: %{{.*}} = constant <f64: 2.000000e+00> : tile<f64>
    %6 = constant <f64: 0x4000000000000000> : tile<f64>  // 2.0 in f64
    return
  }
}

// -----
// Test floating point overflow conditions

cuda_tile.module @kernels {
  entry @kernel() {
    // f16 overflow tests
    // CHECK: %{{.*}} = constant <f16: 0x7C00> : tile<f16>
    %0 = constant <f16: 70000.0> : tile<f16>
    // CHECK: %{{.*}} = constant <f16: 0xFC00> : tile<f16>
    %1 = constant <f16: -70000.0> : tile<f16>
    
    // f32 overflow tests
    // CHECK: %{{.*}} = constant <f32: 0x7F800000> : tile<f32>
    %2 = constant <f32: 10000000000000000000000000000000000000000.0> : tile<f32>
    // CHECK: %{{.*}} = constant <f32: 0xFF800000> : tile<f32>
    %3 = constant <f32: -10000000000000000000000000000000000000000.0> : tile<f32>
    
    // f64 overflow test
    // CHECK: %{{.*}} = constant <f64: 0x7FF0000000000000> : tile<f64>
    %4 = constant <f64: 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0> : tile<f64>
    return
  }
}
