// RUN: cuda-tile-opt %s -split-input-file -verify-diagnostics

// -----
// Test shape mismatch error for 2D array

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{inferred shape of elements literal ([2, 2]) does not match type ([4, 2])}}
    %0 = constant <i1: [[true, true], [true, true]]> : !cuda_tile.tile<4x2xi1>
    return
  }
}

// -----
// Test shape mismatch error for 4D array

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{inferred shape of elements literal ([1, 2, 2, 4]) does not match type ([2, 2, 2, 4])}}
    %0 = constant <i32: [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]]> : !cuda_tile.tile<2x2x2x4xi32>
    return
  }
}

// -----
// Test shape mismatch error for 1D array with too many elements

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@below {{unexpected decimal integer literal for a floating point value}}
    // expected-note@below {{add a trailing dot to make the literal a float}}
    %0 = constant <f32: [0.0, 2.0, -1.0, 0.99, 1.0, 0.01, -0.01, -1.0, 0.0, -0.01, 0.01, 5.0, 5.5, 0.001, 1.111, 0.0, 7.0, 8.0, 9.0, 2147483647, -2147483647, 9223372036854775807, -9223372036854775807, 34028234, -34028234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : !cuda_tile.tile<32xf32>
    return
  }
}

// -----
// Test shape mismatch error for 1D array with too many elements

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{inferred shape of elements literal ([36]) does not match type ([32])}}
    %0 = constant <f32: [0.0, 2.0, -1.0, 0.99, 1.0, 0.01, -0.01, -1.0, 0.0, -0.01, 0.01, 5.0, 5.5, 0.001, 1.111, 0.0, 7.0, 8.0, 9.0, 2147483647.0, -2147483647.0, 9223372036854775807.0, -9223372036854775807.0, 34028234.0, -34028234.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : !cuda_tile.tile<32xf32>
    return
  }
}

// -----
// Test inconsistent element ranks in 2D array

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{tensor literal is invalid; ranks are not consistent between elements}}
    %0 = constant <i1: [[true, true], [true]]> : !cuda_tile.tile<2x2xi1>
    return
  }
}

// -----
// Test inconsistent element ranks in 3D array

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{tensor literal is invalid; ranks are not consistent between elements}}
    %0 = constant <i1: [[[true, true], [true]]]> : !cuda_tile.tile<1x2x2xi1>
    return
  }
}

// -----
// Test inconsistent nested array shapes

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{tensor literal is invalid; ranks are not consistent between elements}}
    %0 = constant <i32: [[[1, 2]], [[3, 4], [5, 6]]]> : !cuda_tile.tile<2x2x2xi32>
    return
  }
}

// -----
// Test shape mismatch with 1D array - too few elements

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{inferred shape of elements literal ([3]) does not match type ([8])}}
    %0 = constant <i32: [1, 2, 3]> : !cuda_tile.tile<8xi32>
    return
  }
}

// -----
// Test shape mismatch with 3D array - wrong middle dimension

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{inferred shape of elements literal ([2, 3, 2]) does not match type ([2, 2, 2])}}
    %0 = constant <i32: [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : !cuda_tile.tile<2x2x2xi32>
    return
  }
}

// -----

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{expected integer value}}
    %0 = constant <i16: ABC> : !cuda_tile.tile<i16>
    return
  }
}

// -----
// Test inconsistent inner array lengths with floating point

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error @+1 {{tensor literal is invalid; ranks are not consistent between elements}}
    %0 = constant <f32: [[1.0, 2.0, 3.0], [4.0, 5.0]]> : !cuda_tile.tile<2x3xf32>
    return
  }
}

// -----
// Test hex string size mismatch - hex too large for i8

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{integer constant out of range for type}}
    %0 = constant <i8: 0x10AB> : !cuda_tile.tile<i8>
    return
  }
}

// -----
// Test integer out of bounds for i8 (positive overflow)

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{integer constant out of range for type}}
    %0 = constant <i8: 256> : !cuda_tile.tile<i8>
    return
  }
}

// -----
// Test integer out of bounds for i8 (negative overflow)

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{integer constant out of range for type}}
    %0 = constant <i8: -129> : !cuda_tile.tile<i8>
    return
  }
}

// -----
// Test integer out of bounds for i16 (positive overflow)

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{integer constant out of range for type}}
    %0 = constant <i16: 65536> : !cuda_tile.tile<i16>
    return
  }
}

// -----
// Test integer out of bounds for i16 (negative overflow)

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{integer constant out of range for type}}
    %0 = constant <i16: -32769> : !cuda_tile.tile<i16>
    return
  }
}

// -----

// Test f16 bitwidth mismatch - too many bytes with without quotes

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{float constant out of range for type}}
    %0 = constant <f16: 0x12345678> : !cuda_tile.tile<f16>
    return
  }
}

// -----

// Test f16 bitwidth mismatch - too many bytes with without quotes

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@+1 {{mismatch between the element type: 'f16' and the tile element type 'f32'}}
    %0 = constant <f16: 42.0> : !cuda_tile.tile<f32>
    return
  }
}

// -----

// Test f16 bitwidth mismatch - too many bytes with without quotes

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@below {{expect element type to be one of i1 or i8 or i16 or i32 or i64 or f16 or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 values, but got '<<NULL TYPE>>'}}
    // expected-error@below {{'cuda_tile.constant' unknown type: pluto}}
    %0 = constant <pluto : 42.0> : !cuda_tile.tile<f32>
    return
  }
}

// -----

// Test f16 bitwidth mismatch - too many bytes with without quotes

cuda_tile.module @kernels {
  entry @kernel() {
    // expected-error@below {{expect element type to be one of i1 or i8 or i16 or i32 or i64 or f16 or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 values, but got 'tensor<i32>'}}
    %0 = constant <tensor<i32> : 42.0> : tensor<i32>
    return
  }
}
