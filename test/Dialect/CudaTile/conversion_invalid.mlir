// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

cuda_tile.module @bitcast_different_shape {
  cuda_tile.entry @func() {
    %c0_i16 = cuda_tile.constant <i16: [1, 2, 3, 4]> : !cuda_tile.tile<4xi16>
    // expected-error @below{{op failed to verify that all of {source, result} have same shape}}
    %c1_i32 = cuda_tile.bitcast %c0_i16 : !cuda_tile.tile<4xi16> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @bitcast_different_width {
  cuda_tile.entry @func() {
    %c0_i32 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{op types must be equal width}}
    %c1_i16 = cuda_tile.bitcast %c0_i32 : !cuda_tile.tile<i32> -> !cuda_tile.tile<i16>
  }
}

// -----

cuda_tile.module @bitcast_int_to_pointer_invalid {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<i32>) {
    // expected-error @below{{operand #0 must be tile of i64 values, but got '!cuda_tile.tile<i32>'}}
    %c0_ptr = cuda_tile.int_to_ptr %arg0 : !cuda_tile.tile<i32> -> !cuda_tile.tile<!cuda_tile.ptr<i8>>
  }
}

// -----

cuda_tile.module @bitcast_pointer_to_int_invalid {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<!cuda_tile.ptr<i8>>) {
    // expected-error @below{{result #0 must be tile of i64 values, but got '!cuda_tile.tile<i32>'}}
    %c0_i32 = cuda_tile.ptr_to_int %arg0 : !cuda_tile.tile<!cuda_tile.ptr<i8>> -> !cuda_tile.tile<i32>
  }
}

// -----

cuda_tile.module @exti_invalid_noop {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: 1> : !cuda_tile.tile<i8>
    // expected-error @below{{extending to smaller or identical integer}}
    cuda_tile.exti %0 signed : !cuda_tile.tile<i8> -> !cuda_tile.tile<i8>
  }
}

// -----

cuda_tile.module @exti_invalid_truncate {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i16: [1, 2]> : !cuda_tile.tile<2xi16>
    // expected-error @below{{extending to smaller or identical integer}}
    cuda_tile.exti %0 signed : !cuda_tile.tile<2xi16> -> !cuda_tile.tile<2xi8>
  }
}

// -----

cuda_tile.module @exti_mismatched_shape {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: [1, 2]> : !cuda_tile.tile<2xi8>
    // expected-error @below{{failed to verify that all of {from, to} have same shape}}
    cuda_tile.exti %0 signed : !cuda_tile.tile<2xi8> -> !cuda_tile.tile<i16>
  }
}

// -----

cuda_tile.module @exti_no_signedness {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: [1, 2]> : !cuda_tile.tile<2xi8>
    // expected-error @below{{expected valid keyword}}
    // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
    cuda_tile.exti %0 : !cuda_tile.tile<2xi8> -> !cuda_tile.tile<2xi16>
  }
}


// -----

cuda_tile.module @ftof_mismatched_shape {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{failed to verify that all of {from, to} have same shape}}
    cuda_tile.ftof %0 : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<f32>
  }
}

// -----

cuda_tile.module @ftof_no_op {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{converting tiles must not be a no-op}}
    cuda_tile.ftof %0 : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<2xf16>
  }
}

// -----

cuda_tile.module @ftof_non_float_result {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error-re @below{{result #0 must be tile of f16 or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 {{(or f4E2M1FN )?}}values}}
    cuda_tile.ftof %0 : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @ftoi_mismatched_shape {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{failed to verify that all of {from, to} have same shape}}
    cuda_tile.ftoi %0 signed : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<i32>
  }
}

// -----

cuda_tile.module @ftoi_non_float_operand {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i16: [1, 2]> : !cuda_tile.tile<2xi16>
    // expected-error-re @below{{operand #0 must be tile of f16 or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 {{(or f4E2M1FN )?}}values}}
    cuda_tile.ftoi %0 signed : !cuda_tile.tile<2xi16> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @ftoi_no_signedness {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.0, 2.0]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{expected valid keyword}}
    // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
    cuda_tile.ftoi %0 : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<2xi32>
  }
}

// -----

cuda_tile.module @itof_mismatched_shape {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i16: [1, 2]> : !cuda_tile.tile<2xi16>
    // expected-error @below{{failed to verify that all of {from, to} have same shape}}
    cuda_tile.itof %0 signed : !cuda_tile.tile<2xi16> -> !cuda_tile.tile<f32>
  }
}

// -----

cuda_tile.module @itof_non_integer_operand {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <f16: [1.1, 2.2]> : !cuda_tile.tile<2xf16>
    // expected-error @below{{operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<2xf16>'}}
    cuda_tile.itof %0 signed : !cuda_tile.tile<2xf16> -> !cuda_tile.tile<2xf32>
  }
}

// -----

cuda_tile.module @itof_no_signedness {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: [1, 2]> : !cuda_tile.tile<2xi8>
    // expected-error @below{{expected valid keyword}}
    // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
    cuda_tile.itof %0 : !cuda_tile.tile<2xi8> -> !cuda_tile.tile<2xf16>
  }
}

// -----

cuda_tile.module @trunci_invalid_extend {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: [1, 2]> : !cuda_tile.tile<2xi8>
    // expected-error @below{{truncating to larger or identical integer}}
    cuda_tile.trunci %0 : !cuda_tile.tile<2xi8> -> !cuda_tile.tile<2xi16>
  }
}

// -----

cuda_tile.module @trunci_invalid_noop {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: 1> : !cuda_tile.tile<i8>
    // expected-error @below{{truncating to larger or identical integer}}
    cuda_tile.trunci %0 : !cuda_tile.tile<i8> -> !cuda_tile.tile<i8>
  }
}

// -----

cuda_tile.module @trunci_mismatched_shape {
  cuda_tile.entry @func() {
    %0 = cuda_tile.constant <i8: [1, 2]> : !cuda_tile.tile<2xi8>
    // expected-error @below{{failed to verify that all of {from, to} have same shape}}
    cuda_tile.trunci %0 : !cuda_tile.tile<2xi8> -> !cuda_tile.tile<i8>
  }
}

// -----

cuda_tile.module @iota_invalid_shape {
  cuda_tile.entry @func() {
    // expected-error @below{{expects result type to be 1-d tile}}
    cuda_tile.iota : !cuda_tile.tile<i64>
  }
}

// -----

cuda_tile.module @iota_mismatched_shape { 
  cuda_tile.entry @func() {
    // expected-error @below{{expects result type to be 1-d tile}}
    cuda_tile.iota : !cuda_tile.tile<32x64xi32>
  }
}

// -----

cuda_tile.module @iota_invalid_overflow {
  cuda_tile.entry @func() {
    // expected-error @below{{the number of elements 512 exceeds the maximum value of element type 'i8'}}
    cuda_tile.iota : !cuda_tile.tile<512xi8>
  }
}
