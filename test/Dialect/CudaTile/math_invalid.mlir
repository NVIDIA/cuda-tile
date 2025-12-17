// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// ****************** cuda_tile.absi ******************

cuda_tile.module @absi_invalid_fp_element {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<4x4xf32>) {
    // expected-error @below{{op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<4x4xf32>'}}
    %0 = cuda_tile.absi %arg0 : !cuda_tile.tile<4x4xf32>
  }
}

// -----

cuda_tile.module @absi_mismatched_type {       
  // expected-note @below{{prior use here}}
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<i32>) {
    // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
    %0 = cuda_tile.absi %arg0 : !cuda_tile.tile<1xi32>
  }
}

// -----

// ****************** cuda_tile.absf ******************
cuda_tile.module @absf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @absf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @absf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @absf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.absf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @absf_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.absf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.absf %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.ceil ******************
cuda_tile.module @ceil_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ceil %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @ceil_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ceil %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @ceil_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ceil %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @ceil_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.ceil' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.ceil %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @ceil_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.ceil' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.ceil %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.cos ******************
cuda_tile.module @cos_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.cos %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @cos_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.cos %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @cos_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.cos %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @cos_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.cos' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.cos %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @cos_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.cos' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.cos %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.cosh ******************
cuda_tile.module @cosh_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.cosh %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @cosh_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.cosh %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @cosh_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.cosh %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @cosh_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.cosh' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.cosh %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @cosh_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.cosh' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.cosh %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.exp2 ******************
cuda_tile.module @exp2_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.exp2 %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @exp2_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.exp2 %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @exp2_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.exp2 %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @exp2_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.exp2' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.exp2 %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @exp2_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.exp2' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.exp2 %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @exp2_invalid_ftz_dtype {
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{'cuda_tile.exp2' op flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
        %0 = exp2 %arg0 flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

// ****************** cuda_tile.exp ******************

cuda_tile.module @exp_different_element_type_type {// expected-note @below{{prior use here}}
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.exp %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @exp_different_shape {// expected-note @below{{prior use here}}
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.exp %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @exp_different_rank {// expected-note @below{{prior use here}}
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.exp %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @exp_invalid_type_i32 {
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.exp' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.exp %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

// ****************** cuda_tile.floor ******************
cuda_tile.module @floor_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.floor %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @floor_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.floor %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @floor_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.floor %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @floor_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.floor' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.floor %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @floor_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.floor' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.floor %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.log ******************
cuda_tile.module @log_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.log %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @log_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.log %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @log_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.log %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @log_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.log' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.log %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @log_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.log' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.log %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.log2 ******************
cuda_tile.module @log2_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.log2 %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @log2_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.log2 %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @log2_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.log2 %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @log2_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.log2' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.log2 %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @log2_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.log2' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.log2 %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.pow ******************
cuda_tile.module @pow_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @pow_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @pow_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @pow_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @pow_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @pow_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @pow_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.pow' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @pow_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.pow' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.pow %arg0, %arg1 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.rsqrt ******************
cuda_tile.module @rsqrt_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.rsqrt %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @rsqrt_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.rsqrt %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @rsqrt_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.rsqrt %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @rsqrt_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.rsqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.rsqrt %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @rsqrt_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.rsqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.rsqrt %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @rsqrt_invalid_f64_element {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<4xf64>) {
    // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f64'}}
    %0 = cuda_tile.rsqrt %arg0 flush_to_zero : !cuda_tile.tile<4xf64>
  }
}
// -----

// ****************** cuda_tile.sqrt ******************
cuda_tile.module @sqrt_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sqrt %arg0 rounding<approx> : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @sqrt_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sqrt %arg0 rounding<nearest_even> : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @sqrt_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sqrt %arg0 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @sqrt_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.sqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.sqrt %arg0 rounding<nearest_even> : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @sqrt_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.sqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.sqrt %arg0 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @sqrt_invalid_i16_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<4xi16>) {
    // expected-error @below{{'cuda_tile.sqrt' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<4xi16>'}}
    %0 = cuda_tile.sqrt %arg0 rounding<approx> : !cuda_tile.tile<4xi16>
  }
}

// -----

cuda_tile.module @sqrt_invalid_rounding_mode__f16_element {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<4xf16>) {
    // expected-error @below{{rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', 'approx'}}
    %0 = cuda_tile.sqrt %arg0 rounding<pippo> : !cuda_tile.tile<4xf16>
  }
}

// -----

cuda_tile.module @sqrt_invalid_approx_f16_element {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<4xf16>) {
    // expected-error @below{{approx modifier only supported for f32 data type, but got: 'f16'}}
    %0 = cuda_tile.sqrt %arg0 rounding<approx> : !cuda_tile.tile<4xf16>
  }
}

// -----

cuda_tile.module @sqrt_invalid_flush_to_zero_f16_element {
  cuda_tile.testing$func @func(%arg0 : !cuda_tile.tile<4xf16>) {
    // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
    %0 = cuda_tile.sqrt %arg0 rounding<approx> flush_to_zero : !cuda_tile.tile<4xf16>
  }
}

// -----

"builtin.module"() ({
  "cuda_tile.module"() <{sym_name = "sqrt_invalid_rnd_modifier"}> ({
    "cuda_tile.testing$func"() <{arg_attrs = [{}], function_type = (!cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
    ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>):
      // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf, approx]}}
      %0 = "cuda_tile.sqrt"(%arg0) <{rounding_mode = #cuda_tile.rounding<full>}> : (!cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
      "cuda_tile.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

// ****************** cuda_tile.sin ******************
cuda_tile.module @sin_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sin %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @sin_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sin %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @sin_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sin %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @sin_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.sin' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.sin %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @sin_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.sin' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.sin %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.sinh ******************
cuda_tile.module @sinh_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sinh %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @sinh_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sinh %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @sinh_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.sinh %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @sinh_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.sinh' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.sinh %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @sinh_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.sinh' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.sinh %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.tan ******************

cuda_tile.module @tan_different_element_type_type {// expected-note @below{{prior use here}}
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @tan_different_shape {// expected-note @below{{prior use here}}
    testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

// ****************** cuda_tile.tan ******************
cuda_tile.module @tan_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @tan_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @tan_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @tan_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.tan' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @tan_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.tan' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.tan %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

// ****************** cuda_tile.tanh ******************
cuda_tile.module @tanh_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tanh %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @tanh_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tanh %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @tanh_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.tanh %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @tanh_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.tanh' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.tanh %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @tanh_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.tanh' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.tanh %arg0 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
} 
