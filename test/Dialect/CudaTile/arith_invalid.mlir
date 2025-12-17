// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// ****************** cuda_tile.addi ******************
cuda_tile.module @addi_mismatching_rank_inputs {
    cuda_tile.entry @func() {
        %arg0 = "materialize_tensor"() : () -> !cuda_tile.tile<2x4x8xi32>
        // expected-note @below{{prior use here}}
        %arg1 = "materialize_tensor"() : () -> !cuda_tile.tile<1x2x4x8xi32>
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @addi_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @addi_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @addi_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @addi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @addi_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @addi_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.addi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @andi_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @andi_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @andi_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @andi_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @andi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @andi_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @andi_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.andi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.andi %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.addf ******************
cuda_tile.module @addf_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @addf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @addf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @addf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @addf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @addf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----


cuda_tile.module @addf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @addf_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.addf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @addf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.addf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @addf_invalid_ftz_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @addf_invalid_rnd_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'approx'}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<approx> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @addf_invalid_rnd_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'full'}}
        %0 = cuda_tile.addf %arg0, %arg1 rounding<full> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

"cuda_tile.module"() <{sym_name = "addf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.addf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<full>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----


"cuda_tile.module"() <{sym_name = "addf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.addf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<approx>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

// ****************** cuda_tile.cmpi ******************
// test: invalid predicate
cuda_tile.module @cmpi_invalid_predicate {
    cuda_tile.entry @func() {
        %c42 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        // expected-error @below{{'cuda_tile.cmpi' expected 'comparison_predicate' to be one of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', 'greater_than', 'greater_than_or_equal'}}
        cuda_tile.cmpi invalid_predicate %c42, %c42, invalid_sigdness : !cuda_tile.tile<i16> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: missing predicate
cuda_tile.module @cmpi_missing_predicate {
    cuda_tile.entry @func() {
        %c42 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        // expected-error @below{{custom op 'cuda_tile.cmpi' expected valid keyword}}
        // expected-error @below{{custom op 'cuda_tile.cmpi' expected 'comparison_predicate' to be one of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', 'greater_than', 'greater_than_or_equal'}}}
        cuda_tile.cmpi %c42, %c42, signed : !cuda_tile.tile<i16> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: non-integer operands
cuda_tile.module @cmpi_non_integer_operands {
    cuda_tile.entry @func() {
        %c42_f32 = cuda_tile.constant <f32: 42.0> : !cuda_tile.tile<f32>
        // expected-error @below{{'cuda_tile.cmpi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<f32>'}}
        cuda_tile.cmpi equal %c42_f32, %c42_f32, signed : !cuda_tile.tile<f32> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: mismatched operand types
cuda_tile.module @cmpi_mismatched_operand_types {
    cuda_tile.entry @func() {
        %c42_i16 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        %c42_i32 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
        // expected-error @below{{'cuda_tile.cmpi' op failed to verify that all of {lhs, rhs} have same type}}
        %x = "cuda_tile.cmpi"(%c42_i16, %c42_i32) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i16>, !cuda_tile.tile<i32>) -> !cuda_tile.tile<i1>
    }
}

// -----

// test: incorrect result shape
cuda_tile.module @cmpi_incorrect_result_shape {
    cuda_tile.entry @func() {
        %t0_2x2 = cuda_tile.constant <i32: [[1, 2], [3, 4]]> : !cuda_tile.tile<2x2xi32>
        // expected-error @below{{'cuda_tile.cmpi' op failed to verify that Result type has i1 element type and same shape as operands}}
        %x = "cuda_tile.cmpi"(%t0_2x2, %t0_2x2) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<2x2xi32>, !cuda_tile.tile<2x2xi32>) -> !cuda_tile.tile<i1>
    }
}

// -----

// test: incorrect result type
cuda_tile.module @cmpi_incorrect_result_type {
    cuda_tile.entry @func() {
        %c42 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        // expected-error @below{{'cuda_tile.cmpi' op result #0 must be tile of i1 values, but got '!cuda_tile.tile<i16>'}}
        %x = "cuda_tile.cmpi"(%c42, %c42) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i16>, !cuda_tile.tile<i16>) -> !cuda_tile.tile<i16>
    }
}

// -----

// test: float predicate used with integer operands
cuda_tile.module @cmpi_float_predicate {
    cuda_tile.entry @func() {
        %i1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
        %i2 = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
        // expected-error @below{{'cuda_tile.cmpi' expected signedness to be one of: {'signed', 'unsigned'}}}
        %x2 = cuda_tile.cmpi equal %i1, %i2, ordered : !cuda_tile.tile<i32> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: invalid predicate
cuda_tile.module @cmpi_invalid_predicate_standalone {
    cuda_tile.entry @func() {
        %c42 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        // expected-error @below{{'cuda_tile.cmpi' expected 'comparison_predicate' to be one of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', 'greater_than', 'greater_than_or_equal'}}}
        cuda_tile.cmpi invalid_predicate %c42, %c42, signed : !cuda_tile.tile<i16> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: missing predicate
cuda_tile.module @cmpi_missing_predicate_standalone {
    cuda_tile.entry @func() {
        %c42 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        // expected-error @below{{custom op 'cuda_tile.cmpi' expected valid keyword}}
        // expected-error @below{{custom op 'cuda_tile.cmpi' expected 'comparison_predicate' to be one of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', 'greater_than', 'greater_than_or_equal'}}}
        cuda_tile.cmpi %c42, %c42, signed : !cuda_tile.tile<i16> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: non-integer operands
cuda_tile.module @cmpi_non_integer_operands_standalone {
    cuda_tile.entry @func() {
        %c42_f32 = cuda_tile.constant <f32: 42.0> : !cuda_tile.tile<f32>
        // expected-error @below{{'cuda_tile.cmpi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<f32>'}}
        cuda_tile.cmpi equal %c42_f32, %c42_f32, signed : !cuda_tile.tile<f32> -> !cuda_tile.tile<i1>
    }
}

// -----

// test: mismatched operand types
cuda_tile.module @cmpi_mismatched_operand_types_standalone {
    cuda_tile.entry @func() {
        %c42_i16 = cuda_tile.constant <i16: 42> : !cuda_tile.tile<i16>
        %c42_i32 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
        // expected-error @below{{'cuda_tile.cmpi' op failed to verify that all of {lhs, rhs} have same type}}
        %x = "cuda_tile.cmpi"(%c42_i16, %c42_i32) {comparison_predicate = #cuda_tile.comparison_predicate<less_than>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<i16>, !cuda_tile.tile<i32>) -> !cuda_tile.tile<i1>
    }
}

// -----

// ****************** cuda_tile.cmpf ******************
// test: invalid predicate
cuda_tile.module @cmpf_invalid_predicate {
  cuda_tile.entry @func() {
    %c42 = cuda_tile.constant <f16: 42.0> : !cuda_tile.tile<f16>
    // expected-error @below{{'cuda_tile.cmpf' expected 'comparison_predicate' to be one of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', 'greater_than', 'greater_than_or_equal'}}}
    cuda_tile.cmpf invalid_predicate ordered %c42, %c42 : !cuda_tile.tile<f16> -> !cuda_tile.tile<i1>
  }
}

// -----

// test: invalid ordering
cuda_tile.module @cmpf_invalid_ordering {
  cuda_tile.entry @func() {
    %c42 = cuda_tile.constant <f16: 42.0> : !cuda_tile.tile<f16>
    // expected-error @below{{'cuda_tile.cmpf' expected 'comparison_ordering' to be one of: {'ordered', 'unordered'}}}
    cuda_tile.cmpf equal invalid_ordering %c42, %c42 : !cuda_tile.tile<f16> -> !cuda_tile.tile<i1>
  }
}

// -----

// test: missing predicate
cuda_tile.module @cmpf_missing_predicate {
  cuda_tile.entry @func() {
    %c42 = cuda_tile.constant <f16: 42.0> : !cuda_tile.tile<f16>
    // expected-error @below{{'cuda_tile.cmpf' expected 'comparison_predicate' to be one of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', 'greater_than', 'greater_than_or_equal'}}}
    cuda_tile.cmpf ordered %c42, %c42 : !cuda_tile.tile<f16> -> !cuda_tile.tile<i1>
  }
}

// -----

// test: non-float operands
cuda_tile.module @cmpf_non_float_operands {
  cuda_tile.entry @func() {
    %c42_i32 = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    // expected-error @below{{'cuda_tile.cmpf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i32>'}}
    cuda_tile.cmpf equal ordered %c42_i32, %c42_i32 : !cuda_tile.tile<i32> -> !cuda_tile.tile<i1>
  }
}

// -----

// test: mismatched operand types
cuda_tile.module @cmpf_mismatched_operand_types {
  cuda_tile.entry @func() {
    %c42_f16 = cuda_tile.constant <f16: 42.0> : !cuda_tile.tile<f16>
    %c42_f32 = cuda_tile.constant <f32: 42.0> : !cuda_tile.tile<f32>
    // expected-error @below{{'cuda_tile.cmpf' op failed to verify that all of {lhs, rhs} have same type}}
    %x = "cuda_tile.cmpf"(%c42_f16, %c42_f32) {comparison_predicate = #cuda_tile.comparison_predicate<greater_than>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<f16>, !cuda_tile.tile<f32>) -> !cuda_tile.tile<i1>
  }
}

// -----

// test: incorrect result shape
cuda_tile.module @cmpf_incorrect_result_shape {
  cuda_tile.entry @func() {
    %t0_2x2 = cuda_tile.constant <f32: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf32>
    // expected-error @below{{'cuda_tile.cmpf' op failed to verify that Result type has i1 element type and same shape as operands}}
    %x = "cuda_tile.cmpf"(%t0_2x2, %t0_2x2) {comparison_predicate = #cuda_tile.comparison_predicate<greater_than>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<i1>
  }
}

// -----

// test: incorrect result type
cuda_tile.module @cmpf_incorrect_result_type {
  cuda_tile.entry @func() {
    %c42 = cuda_tile.constant <f16: 42.0> : !cuda_tile.tile<f16>
    // expected-error @below{{'cuda_tile.cmpf' op result #0 must be tile of i1 values, but got '!cuda_tile.tile<f16>'}}
    %x = "cuda_tile.cmpf"(%c42, %c42) {comparison_predicate = #cuda_tile.comparison_predicate<greater_than>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<f16>, !cuda_tile.tile<f16>) -> !cuda_tile.tile<f16>
  }
}

// -----

// test: result shape doesn't match operand shape
cuda_tile.module @cmpf_result_shape_mismatch {
  cuda_tile.entry @func() {
    %a = cuda_tile.constant <f32: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf32>
    %b = cuda_tile.constant <f32: [[5.0, 6.0], [7.0, 8.0]]> : !cuda_tile.tile<2x2xf32>
    // expected-error @below{{'cuda_tile.cmpf' op failed to verify that Result type has i1 element type and same shape as operands}}
    %x = "cuda_tile.cmpf"(%a, %b) {comparison_predicate = #cuda_tile.comparison_predicate<greater_than>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<2x2xf32>, !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<4x1xi1>
  }
}

// -----

// test: result has correct element type (i1) but wrong rank
cuda_tile.module @cmpf_wrong_result_rank {
  cuda_tile.entry @func() {
    %a = cuda_tile.constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %b = cuda_tile.constant <f32: [3.0, 4.0]> : !cuda_tile.tile<2xf32>
    // expected-error @below{{'cuda_tile.cmpf' op failed to verify that Result type has i1 element type and same shape as operands}}
    %x = "cuda_tile.cmpf"(%a, %b) {comparison_predicate = #cuda_tile.comparison_predicate<greater_than>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<2xf32>, !cuda_tile.tile<2xf32>) -> !cuda_tile.tile<2x1xi1>
  }
}

// -----

// test: operands same type but different shapes
cuda_tile.module @cmpf_different_shapes {
  cuda_tile.entry @func() {
    %a = cuda_tile.constant <f32: [[1.0, 2.0]]> : !cuda_tile.tile<1x2xf32>
    // expected-note @below{{prior use here}}
    %b = cuda_tile.constant <f32: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf32>
    // expected-error @below{{use of value '%b' expects different type than prior uses: '!cuda_tile.tile<1x2xf32>' vs '!cuda_tile.tile<2x2xf32>'}}
    %x = cuda_tile.cmpf equal ordered %a, %b : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xi1>
  }
}

// -----

// test: result has same shape but wrong element type
cuda_tile.module @cmpi_wrong_result_type {
  cuda_tile.entry @func() {
    %a = cuda_tile.constant <i32: [1, 2]> : !cuda_tile.tile<2xi32>
    %b = cuda_tile.constant <i32: [3, 4]> : !cuda_tile.tile<2xi32>
    // expected-error @below{{'cuda_tile.cmpi' op result #0 must be tile of i1 values}}
    %x = "cuda_tile.cmpi"(%a, %b) {comparison_predicate = #cuda_tile.comparison_predicate<equal>, signedness = #cuda_tile.signedness<signed>} : (!cuda_tile.tile<2xi32>, !cuda_tile.tile<2xi32>) -> !cuda_tile.tile<2xi32>
  }
}

// -----

// test: operands have same shape but different element types
cuda_tile.module @cmpf_different_element_types {
  cuda_tile.entry @func() {
    %a = cuda_tile.constant <f32: [[1.0, 2.0]]> : !cuda_tile.tile<1x2xf32>
    // expected-note @below{{prior use here}}
    %b = cuda_tile.constant <f64: [[1.0, 2.0]]> : !cuda_tile.tile<1x2xf64>
    // expected-error @below{{use of value '%b' expects different type than prior uses: '!cuda_tile.tile<1x2xf32>' vs '!cuda_tile.tile<1x2xf64>'}}
    %x = cuda_tile.cmpf equal ordered %a, %b : !cuda_tile.tile<1x2xf32> -> !cuda_tile.tile<1x2xi1>
  }
}

// -----

// test: scalar operands but non-scalar result
cuda_tile.module @cmpf_scalar_operands_non_scalar_result {
  cuda_tile.entry @func() {
    %a = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    %b = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    // expected-error @below{{'cuda_tile.cmpf' op failed to verify that Result type has i1 element type and same shape as operands}}
    %x = "cuda_tile.cmpf"(%a, %b) {comparison_predicate = #cuda_tile.comparison_predicate<equal>, comparison_ordering = #cuda_tile.comparison_ordering<ordered>} : (!cuda_tile.tile<f32>, !cuda_tile.tile<f32>) -> !cuda_tile.tile<1xi1>
  }
}

// -----

// test: signed integer predicate used with float operands
cuda_tile.module @cmpf_invalid_predicate_type {
  cuda_tile.entry @func() {
    %f1 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    %f2 = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    // expected-error @below{{'cuda_tile.cmpf' expected 'comparison_ordering' to be one of: {'ordered', 'unordered'}}
    %x1 = cuda_tile.cmpf greater_than_or_equal signed %f1, %f2 : !cuda_tile.tile<f32> -> !cuda_tile.tile<i1>
  }
}

// -----

// ****************** cuda_tile.divi ******************

cuda_tile.module @divi_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.entry @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----
cuda_tile.module @floordivi_unsigned {
  cuda_tile.entry @func() {
    %s_i1 = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    // expected-error @below{{rounding mode 'negative_inf' is not allowed with 'unsigned' flag}}
    %floordivui_scalar_i1 = cuda_tile.divi %s_i1, %s_i1 unsigned rounding<negative_inf> : !cuda_tile.tile<i1>
  }
}

// -----

cuda_tile.module @divi_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @divi_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @divi_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @divi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @divi_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----


cuda_tile.module @divi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @divi_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.divi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.divi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @divi_no_signedness {
    cuda_tile.entry @func() {
        %i16 = cuda_tile.constant <i16: [1,2]> : !cuda_tile.tile<2xi16>
        // expected-error @below{{expected valid keyword}}
        // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
        %0 = cuda_tile.divi %i16, %i16 : !cuda_tile.tile<2xi16>
    }
}

// -----

// ****************** cuda_tile.divf ******************
cuda_tile.module @divf_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @divf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @divf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @divf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @divf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @divf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----


cuda_tile.module @divf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @divf_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.divf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @divf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.divf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @divf_invalid_flush_to_zero_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @divf_invalid_approx_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{approx modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @divf_invalid_full_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{full modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<full> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @divf_invalid_flush_to_zero_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xbf16>, %arg1: !cuda_tile.tile<2x4x8xbf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'bf16'}}
        %0 = cuda_tile.divf %arg0, %arg1 rounding<approx> flush_to_zero : !cuda_tile.tile<2x4x8xbf16>
    }
}

// -----

"cuda_tile.module"() <{sym_name = "divf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf, approx, full]}}
    %0 = "cuda_tile.divf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<nearest_int_to_zero>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

// ****************** cuda_tile.maxi ******************
cuda_tile.module @maxi_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @maxi_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @maxi_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @maxi_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @maxi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @maxi_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----


cuda_tile.module @maxi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @maxi_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.maxi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.maxi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @maxi_no_signedness {
    cuda_tile.entry @func() {
        %i16 = cuda_tile.constant <i16: [1,2]> : !cuda_tile.tile<2xi16>
        // expected-error @below{{expected valid keyword}}
        // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
        %0 = cuda_tile.maxi %i16, %i16 : !cuda_tile.tile<2xi16>
    }
}

// -----

// ****************** cuda_tile.maxf ******************
cuda_tile.module @maxf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @maxf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxf %arg0, %arg1 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @maxf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.maxf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @maxf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.maxf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @maxf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.maxf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @maxf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.maxf %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @maxf_invalid_unsigned_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ':'}}
        %0 = cuda_tile.maxf %arg0, %arg1 unsigned : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @maxf_invalid_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ':'}}
        %0 = cuda_tile.maxf %arg0, %arg1 invalid_modifier : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @maxf_invalid_ftz_bf16 {
    cuda_tile.testing$func @test(%arg0: !cuda_tile.tile<2x4xbf16>, %arg1: !cuda_tile.tile<2x4xbf16>) {
        // expected-error @below {{flush_to_zero modifier only supported for f32 data type, but got: 'bf16'}}
        %0 = cuda_tile.maxf %arg0, %arg1 flush_to_zero : !cuda_tile.tile<2x4xbf16>
    }
}

// -----


// ****************** cuda_tile.mini ******************
cuda_tile.module @mini_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mini_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @mini_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mini_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @mini_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mini_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----


cuda_tile.module @mini_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mini_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.mini' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.mini %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mini_no_signedness {
    cuda_tile.entry @func() {
        %i16 = cuda_tile.constant <i16: [1,2]> : !cuda_tile.tile<2xi16>
        // expected-error @below{{expected valid keyword}}
        // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
        %0 = cuda_tile.mini %i16, %i16 : !cuda_tile.tile<2xi16>
    }
}

// -----

// ****************** cuda_tile.minf ******************
cuda_tile.module @minf_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @minf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @minf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{#0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.minf %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @minf_invalid_unsigned_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ':'}}
        %0 = cuda_tile.minf %arg0, %arg1 unsigned : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_invalid_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ':'}}
        %0 = cuda_tile.minf %arg0, %arg1 invalid_modifier : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @minf_invalid_ftz_bf16 {
    cuda_tile.testing$func @test(%arg0: !cuda_tile.tile<2x4xbf16>, %arg1: !cuda_tile.tile<2x4xbf16>) {
        // expected-error @below {{flush_to_zero modifier only supported for f32 data type, but got: 'bf16'}}
        %0 = cuda_tile.minf %arg0, %arg1 flush_to_zero : !cuda_tile.tile<2x4xbf16>
    }
}

// -----

// ****************** cuda_tile.muli ******************
cuda_tile.module @muli_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.muli %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @muli_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.muli %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

// ****************** cuda_tile.mulf ******************
cuda_tile.module @mulf_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @mulf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @mulf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.mulf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @mulf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.mulf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mulf_invalid_ftz_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @mulf_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.mulf' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'invalid_mode'}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<invalid_mode> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.mulf' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'approx'}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @mulf_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.mulf' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'full'}}
        %0 = cuda_tile.mulf %arg0, %arg1 rounding<full> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

"cuda_tile.module"() <{sym_name = "mulf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.mulf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<full>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

"cuda_tile.module"() <{sym_name = "mulf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.mulf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<approx>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

// ****************** cuda_tile.fma ******************
cuda_tile.module @fma_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @fma_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @fma_mismatching_elementtype_third_operand {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg2' expects different type than prior uses}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>, %arg2: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.fma' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @fma_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>, %arg2: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.fma' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @fma_invalid_ftz_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>, %arg2: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @fma_invalid_ftz_modifier_bf16 {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xbf16>, %arg1: !cuda_tile.tile<2x4x8xbf16>, %arg2: !cuda_tile.tile<2x4x8xbf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'bf16'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<2x4x8xbf16>
    }
}

// -----

cuda_tile.module @fma_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.fma' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'invalid_mode'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<invalid_mode> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.fma' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'approx'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @fma_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>, %arg2: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.fma' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'full'}}
        %0 = cuda_tile.fma %arg0, %arg1, %arg2 rounding<full> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

"cuda_tile.module"() <{sym_name = "fma_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.fma"(%arg0, %arg1, %arg0) <{rounding_mode = #cuda_tile.rounding<full>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

"cuda_tile.module"() <{sym_name = "fma_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.fma"(%arg0, %arg1, %arg0) <{rounding_mode = #cuda_tile.rounding<approx>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

// ****************** cuda_tile.mulhii ******************
cuda_tile.module @mulhii_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mulhii_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @mulhii_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mulhii_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @mulhii_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mulhii_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @mulhii_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @mulhii_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.mulhii' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.mulhii %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.negf ******************
cuda_tile.module @negf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.negf %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @negf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.negf %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @negf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.negf %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @negf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.negf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.negf %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @negf_invalid_i1_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi1>) {
        // expected-error @below{{'cuda_tile.negf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi1>'}}
        %0 = cuda_tile.negf %arg0 : !cuda_tile.tile<2x4x8xi1>
    }
}

// -----

// ****************** cuda_tile.negi ******************

// -----

cuda_tile.module @negi_invalid_f16_element {
    cuda_tile.entry @func() {
        %f16 = cuda_tile.constant <f16: [1.0,2.0]> : !cuda_tile.tile<2xf16>
        // expected-error @below{{op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values, but got '!cuda_tile.tile<2xf16>'}}
        %x = cuda_tile.negi %f16 : !cuda_tile.tile<2xf16>
    }
}

// -----

// ****************** cuda_tile.ori ******************

cuda_tile.module @ori_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @ori_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @ori_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @ori_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @ori_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @ori_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @ori_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @ori_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{cuda_tile.ori' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.ori %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.remi ******************
cuda_tile.module @remi_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.remi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @remi_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.remi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @remi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.remi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @remi_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.remi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.remi %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @remi_no_signedness {
    cuda_tile.entry @func() {
        %i16 = cuda_tile.constant <i16: [1,2]> : !cuda_tile.tile<2xi16>
        // expected-error @below{{expected valid keyword}}
        // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
        %0 = cuda_tile.remi %i16, %i16 : !cuda_tile.tile<2xi16>
    }
}

// -----

// ****************** cuda_tile.remf ******************
cuda_tile.module @remf_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @remf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @remf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @remf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @remf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @remf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @remf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @remf_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.remf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @remf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.remf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.remf %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @remf_invalid_unsigned_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ':'}}
        %0 = cuda_tile.remf %arg0, %arg1 unsigned : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.select ******************
// Test missing condition type in type specification
cuda_tile.module @select_missing_condition_type {
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<2x4x8xi1>, %arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ','}}
        %0 = cuda_tile.select %cond, %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// Test missing result type in type specification
cuda_tile.module @select_missing_result_type {
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<2x4x8xi1>, %arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        %0 = cuda_tile.select %cond, %arg0, %arg1 : !cuda_tile.tile<2x4x8xi1>,
        // expected-error @below{{custom op 'cuda_tile.select' expected valid keyword}}
    }
}

// -----

// Test mismatched operand types
cuda_tile.module @select_mismatched_operand_types {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<2x4x8xi1>, %arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi64>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses: '!cuda_tile.tile<2x4x8xi32>' vs '!cuda_tile.tile<2x4x8xi64>'}}
        %0 = cuda_tile.select %cond, %arg0, %arg1 : !cuda_tile.tile<2x4x8xi1>, !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

// Test mismatched result type
cuda_tile.module @select_mismatched_result_type {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<2x4x8xi1>, %arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses: '!cuda_tile.tile<2x4x8xi64>' vs '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.select %cond, %arg0, %arg1 : !cuda_tile.tile<2x4x8xi1>, !cuda_tile.tile<2x4x8xi64>
    }
}

// -----

// Test invalid condition type
cuda_tile.module @select_invalid_condition_type {
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<2x4x8xi32>, %arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.select' op operand #0 must be tile of i1 values}}
        %0 = cuda_tile.select %cond, %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>, !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// Test mismatched condition shape
cuda_tile.module @select_mismatched_condition_shape {
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<1x2x4x8xi1>, %arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.select' op failed to verify that all of {cond, val_if_true, val_if_false, result} have same shape}}
        %0 = cuda_tile.select %cond, %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi1>, !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// Test missing operand
cuda_tile.module @select_missing_operand {
    cuda_tile.testing$func @func(%cond: !cuda_tile.tile<2x4x8xi1>, %arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{expected ','}}
        %0 = cuda_tile.select %cond, %arg0 : !cuda_tile.tile<2x4x8xi1>, !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.subi ******************
cuda_tile.module @subi_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.subi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @subi_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.subi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @subi_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.subi %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @subi_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.subi' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.subi %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.subf ******************
cuda_tile.module @subf_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<1x2x4x8xf32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @subf_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @subf_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @subf_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @subf_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @subf_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @subf_invalid_f8_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf8E5M2>, %arg1: !cuda_tile.tile<2x4x8xf8E5M2>) {
        // expected-error @below{{'cuda_tile.subf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xf8E5M2>'}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xf8E5M2>
    }
}

// -----

cuda_tile.module @subf_invalid_int_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.subf' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @subf_invalid_ftz_modifier {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf16>, %arg1: !cuda_tile.tile<2x4x8xf16>) {
        // expected-error @below{{flush_to_zero modifier only supported for f32 data type, but got: 'f16'}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<nearest_even> flush_to_zero : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @subf_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.subf' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'invalid_mode'}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<invalid_mode> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @subf_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.subf' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'approx'}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @subf_invalid_rounding_mode {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{custom op 'cuda_tile.subf' expected rounding mode to be one of: 'nearest_even', 'zero', 'negative_inf', 'positive_inf', got: 'full'}}
        %0 = cuda_tile.subf %arg0, %arg1 rounding<full> : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

"cuda_tile.module"() <{sym_name = "subf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.subf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<full>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----


"cuda_tile.module"() <{sym_name = "subf_invalid_rnd_modifier"}> ({
  "cuda_tile.testing$func"() <{arg_attrs = [{}, {}], function_type = (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> (), sym_name = "func"}> ({
  ^bb0(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>):
    // expected-error @below{{op invalid rounding error specified, expect one of [nearest_even, zero, negative_inf, positive_inf]}}
    %0 = "cuda_tile.subf"(%arg0, %arg1) <{rounding_mode = #cuda_tile.rounding<approx>}> : (!cuda_tile.tile<2x4x8xf32>, !cuda_tile.tile<2x4x8xf32>) -> !cuda_tile.tile<2x4x8xf32>
    "cuda_tile.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

// ****************** cuda_tile.shli ******************
cuda_tile.module @shli_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shli_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @shli_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shli_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @shli_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shli_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @shli_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shli_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.shli' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.shli %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

// ****************** cuda_tile.shri ******************
cuda_tile.module @shri_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shri_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @shri_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shri_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @shri_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shri_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @shri_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @shri_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.shri' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.shri %arg0, %arg1 signed : !cuda_tile.tile<2x4x8xf32>
    }
}

// -----

cuda_tile.module @shri_no_signedness {
    cuda_tile.entry @func() {
        %i16 = cuda_tile.constant <i16: [1,2]> : !cuda_tile.tile<2xi16>
        // expected-error @below{{expected valid keyword}}
        // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
        %0 = cuda_tile.shri %i16, %i16 : !cuda_tile.tile<2xi16>
    }
}

// -----

// ****************** cuda_tile.xori ******************

cuda_tile.module @xori_mismatching_rank_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<1x2x4x8xi32>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @xori_mismatching_rank_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<1x2x4x8xi32>
    }
}

// -----

cuda_tile.module @xori_mismatching_shape_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x8x4xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @xori_mismatching_shape_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<4x2x8xi32>
    }
}

// -----

cuda_tile.module @xori_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @xori_mismatching_elementtype_input_output {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi16>
    }
}

// -----

cuda_tile.module @xori_mismatching_elementtype_inputs {// expected-note @below{{prior use here}}
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>, %arg1: !cuda_tile.tile<2x4x8xi16>) {
        // expected-error @below{{use of value '%arg1' expects different type than prior uses}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

cuda_tile.module @xori_invalid_fp_element {
    cuda_tile.testing$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>, %arg1: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{'cuda_tile.xori' op operand #0 must be tile of i1 or i8 or i16 or i32 or i64 values}}
        %0 = cuda_tile.xori %arg0, %arg1 : !cuda_tile.tile<2x4x8xf32>
    }
}
