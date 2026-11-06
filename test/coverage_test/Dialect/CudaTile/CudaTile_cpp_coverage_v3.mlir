// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers parseAssumePredicate's unknown-name error path
// (parsed keyword doesn't match div_by/same_elements/bounded).

cuda_tile.module @kernels {
  testing$func @assume_unknown_predicate(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{unknown assume predicate attribute}}
    %0 = cuda_tile.assume foo_predicate, %arg0 : !cuda_tile.tile<i32>
  }
}

// -----

// Covers parseAssumePredicate's failure path inside DivByAttr::parse: the
// short-form keyword matched but the following content is malformed.
cuda_tile.module @kernels {
  testing$func @assume_div_by_malformed(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{expected '<'}}
    %0 = cuda_tile.assume div_by foo, %arg0 : !cuda_tile.tile<i32>
  }
}

// -----

// Covers parseAssumePredicate's failure path inside SameElementsAttr::parse.
cuda_tile.module @kernels {
  testing$func @assume_same_elements_malformed(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{expected '<'}}
    %0 = cuda_tile.assume same_elements, %arg0 : !cuda_tile.tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers AtomicCASTkoOp::verify error paths:
//   - non-float, non-integer element type rejected
//   - non-32/64 bitwidth rejected

// Covers AtomicCASTkoOp::verify wrong bitwidth path: i16 element type
// fails the 32-or-64-bit check.
cuda_tile.module @kernels {
  testing$func @atomic_cas_tko_bad_bitwidth(
      %p: !cuda_tile.tile<2x!cuda_tile.ptr<i16>>,
      %v: !cuda_tile.tile<2xi16>,
      %c: !cuda_tile.tile<2xi16>) {
    // expected-error @below{{expect only float or integer types with 32 or 64 bit}}
    %0, %t = cuda_tile.atomic_cas_tko relaxed device %p, %v, %c
        : !cuda_tile.tile<2x!cuda_tile.ptr<i16>>, !cuda_tile.tile<2xi16> -> !cuda_tile.tile<2xi16>, !cuda_tile.token
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers AtomicRedViewTkoOp::parse error paths:
//   - invalid memory_ordering_semantics keyword
//   - invalid memory_scope keyword
//   - unknown atomic RMW mode
//   - index types count mismatch

// Covers AtomicRedViewTkoOp::parse 'invalid memory_ordering_semantics'
// error path: the first keyword does not symbolize to a known semantics.
cuda_tile.module @kernels {
  testing$func @atomic_red_view_tko_bad_semantics(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{invalid memory_ordering_semantics: 'bogus_sem'}}
    %t = cuda_tile.atomic_red_view_tko bogus_sem device %view[%c0, %c0], add, %value
        : !cuda_tile.tile<2x2xi32>, !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Covers AtomicRedViewTkoOp::parse 'invalid memory_scope' error path:
// the optional scope keyword does not symbolize to a known scope.
cuda_tile.module @kernels {
  testing$func @atomic_red_view_tko_bad_scope(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{invalid memory_scope: 'bogus_scope'}}
    %t = cuda_tile.atomic_red_view_tko relaxed bogus_scope %view[%c0, %c0], add, %value
        : !cuda_tile.tile<2x2xi32>, !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Covers AtomicRedViewTkoOp::parse 'unknown atomic RMW mode' error path
// triggered when the mode keyword is not a recognized RMW mode.
cuda_tile.module @kernels {
  testing$func @atomic_red_view_tko_bad_mode(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{unknown atomic RMW mode: 'bogus_mode'}}
    %t = cuda_tile.atomic_red_view_tko relaxed device %view[%c0, %c0], bogus_mode, %value
        : !cuda_tile.tile<2x2xi32>, !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Covers AtomicRedViewTkoOp::parse "index types must be specified when
// indices are present" path: indices are non-empty but no index types follow
// the view type.
cuda_tile.module @kernels {
  testing$func @atomic_red_view_tko_no_index_types(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %t = cuda_tile.atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        : !cuda_tile.tile<2x2xi32>, !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>> -> !cuda_tile.token
  // expected-error @below{{index types must be specified when indices are present}}
  }
}

// -----

// Covers AtomicRedViewTkoOp::parse "expected N index types, got M" path:
// 2 indices declared but 3 index types provided.
cuda_tile.module @kernels {
  testing$func @atomic_red_view_tko_index_count_mismatch(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %t = cuda_tile.atomic_red_view_tko relaxed device %view[%c0, %c0], add, %value
        // expected-error @below{{expected 2 index types, got 3}}
        : !cuda_tile.tile<2x2xi32>, !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers AtomicRMWTkoOp::verify error paths:
//   - non-i32/i64/f32/f64 type rejected (e.g. i16 width)

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers CatOp::verify "expect lhs and rhs shapes to match at non-concat
// position" path and SelectOp::verify F4E2M1FN rejection path.

// Covers CatOp::verify "expect lhs and rhs shapes to match at non-concat
// position" path: lhs and rhs have the same rank but differ at the
// non-concat position (dim 0). Use rank-3 to avoid earlier checks firing.
cuda_tile.module @kernels {
  testing$func @cat_lhs_rhs_mismatch_non_concat(
      %a: !cuda_tile.tile<2x2x2xf32>,
      %b: !cuda_tile.tile<4x2x2xf32>) {
    // expected-error @below{{expect lhs and rhs shapes to match at non-concat position 0}}
    %0 = cuda_tile.cat %a, %b dim = 1
      : !cuda_tile.tile<2x2x2xf32>, !cuda_tile.tile<4x2x2xf32> -> !cuda_tile.tile<2x4x2xf32>
  }
}

// -----

// Covers SelectOp::verify "cannot operate on sub-byte type F4E2M1FN" path.
cuda_tile.module @kernels {
  testing$func @select_f4_rejected(
      %cond: !cuda_tile.tile<4xi1>,
      %t: !cuda_tile.tile<4xf4E2M1FN>,
      %f: !cuda_tile.tile<4xf4E2M1FN>) {
    // expected-error @below{{cannot operate on sub-byte type F4E2M1FN}}
    %0 = cuda_tile.select %cond, %t, %f
        : !cuda_tile.tile<4xi1>, !cuda_tile.tile<4xf4E2M1FN>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers parseDenseIntOrFPElementsAttr error paths:
//   - empty dense attribute
//   - missing colon between prefix type and values
//   - missing closing '>' after values
//   - element-type mismatch between prefix and tile

// Covers parseDenseIntOrFPElementsAttr "mismatch between element type" path.
// expected-error @below{{mismatch between the element type}}
%b = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i64>

// -----

// Covers parseDenseIntOrFPElementsAttr "inferred shape ... does not match" path
// for scalar result type (expected []).
// expected-error @below{{inferred shape of elements literal ([1]) does not match type ([])}}
%c = cuda_tile.constant <i32: [42]> : !cuda_tile.tile<i32>

// -----

// Covers parseDenseIntOrFPElementsAttr "ranks are not consistent between elements"
// path for jagged nested arrays.
// expected-error @below{{tensor literal is invalid; ranks are not consistent}}
%d = cuda_tile.constant <i32: [[1, 2], [3]]> : !cuda_tile.tile<2x2xi32>

// -----

// Covers parseDenseIntOrFPElementsAttr inferred-shape mismatch with non-scalar
// expected shape.
// expected-error @below{{inferred shape of elements literal ([3]) does not match type ([2])}}
%e = cuda_tile.constant <i32: [1, 2, 3]> : !cuda_tile.tile<2xi32>

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers cuda_tile::parseFunctionSignatureWithArguments helper paths
// (validateSSANameConsistency, parseSingleArgument, parseFunctionResultList):
//   - mixing SSA-named and unnamed function arguments
//   - parenthesized result list with multiple types and attributes
//   - empty parenthesized result list
//   - SSA names on entry arguments

// Covers validateSSANameConsistency "expected SSA identifier" path: the
// first argument has an SSA identifier but the second doesn't.
cuda_tile.module @kernels {
  // expected-error @below{{expected SSA identifier}}
  cuda_tile.entry @entry_inconsistent_named_unnamed(%arg0: !cuda_tile.tile<f32>, !cuda_tile.tile<f32>) {}
}

// -----

// Covers validateSSANameConsistency "expected type instead of SSA
// identifier" path: the first argument has no SSA identifier, but the
// second does.
cuda_tile.module @kernels {
  // expected-error @below{{expected type instead of SSA identifier}}
  cuda_tile.entry @entry_inconsistent_unnamed_named(!cuda_tile.tile<f32>, %arg0: !cuda_tile.tile<f32>) {}
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers enum-keyword custom-parsers error paths:
//   - parseSignedness symbolizeSignedness-failure
//   - parseComparisonOrdering symbolizeComparisonOrdering-failure
//   - parseComparisonPredicate symbolizeComparisonPredicate-failure
//   - parseDenseIntOrFPElementsAttr missing-colon-after-prefix-type

// Covers parseSignedness 'unknown signedness keyword' path via CmpIOp.
cuda_tile.module @kernels {
  testing$func @cmpi_bad_signedness(
      %a: !cuda_tile.tile<i32>,
      %b: !cuda_tile.tile<i32>) {
    // expected-error @below{{expected signedness to be one of: {'signed', 'unsigned'}}}
    %0 = cuda_tile.cmpi equal %a, %b, bogus_signedness : !cuda_tile.tile<i32> -> !cuda_tile.tile<i1>
  }
}

// -----

// Covers parseComparisonOrdering 'unknown ordering keyword' path via CmpFOp.
cuda_tile.module @kernels {
  testing$func @cmpf_bad_ordering(
      %a: !cuda_tile.tile<f32>,
      %b: !cuda_tile.tile<f32>) {
    // expected-error @below{{expected 'comparison_ordering' to be one}}
    %0 = cuda_tile.cmpf equal bogus_ordering %a, %b : !cuda_tile.tile<f32> -> !cuda_tile.tile<i1>
  }
}

// -----

// Covers parseComparisonPredicate 'unknown predicate keyword' path via CmpFOp.
cuda_tile.module @kernels {
  testing$func @cmpf_bad_predicate(
      %a: !cuda_tile.tile<f32>,
      %b: !cuda_tile.tile<f32>) {
    // expected-error @below{{expected 'comparison_predicate' to be one}}
    %0 = cuda_tile.cmpf bogus_predicate ordered %a, %b : !cuda_tile.tile<f32> -> !cuda_tile.tile<i1>
  }
}

// -----

// Covers parseDenseIntOrFPElementsAttr 'expect colon after prefix type'
// failure: prefix type i32 is followed by a value instead of ':'.
cuda_tile.module @kernels {
  // expected-error @below{{expected ':'}}
  %c = cuda_tile.constant <i32 1> : !cuda_tile.tile<i32>
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ExtractOp::verify and InsertOp::verify bounds-checking warning
// paths for negative and out-of-bounds constant indices (rare paths that
// emit warnings rather than errors).

// Covers ExtractOp::verify bounds-check warning for negative constant index.
cuda_tile.module @kernels {
  testing$func @extract_negative_index(%arg0: !cuda_tile.tile<8x8xf32>) {
    %neg = cuda_tile.constant <i32: -1> : !cuda_tile.tile<i32>
    %ok = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-warning @below{{extract index 0 has negative value -1}}
    %0 = cuda_tile.extract %arg0[%neg, %ok] : !cuda_tile.tile<8x8xf32> -> !cuda_tile.tile<4x4xf32>
  }
}

// -----

// Covers ExtractOp::verify bounds-check warning for out-of-bounds constant.
// Max valid index = (8/4) - 1 = 1. Index 5 is out of bounds.
cuda_tile.module @kernels {
  testing$func @extract_oob_index(%arg0: !cuda_tile.tile<8x8xf32>) {
    %big = cuda_tile.constant <i32: 5> : !cuda_tile.tile<i32>
    %ok = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-warning @below{{extract index 0 value 5 may be out of bounds}}
    %0 = cuda_tile.extract %arg0[%big, %ok] : !cuda_tile.tile<8x8xf32> -> !cuda_tile.tile<4x4xf32>
  }
}

// -----

// Covers InsertOp::verify bounds-check warning for negative constant index.
cuda_tile.module @kernels {
  testing$func @insert_negative_index(
      %src: !cuda_tile.tile<4x4xf32>,
      %dst: !cuda_tile.tile<8x8xf32>) {
    %neg = cuda_tile.constant <i32: -1> : !cuda_tile.tile<i32>
    %ok = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-warning @below{{insert index 0 has negative value -1}}
    %0 = cuda_tile.insert %src, %dst[%neg, %ok] : !cuda_tile.tile<4x4xf32>, !cuda_tile.tile<8x8xf32>
  }
}

// -----

// Covers InsertOp::verify bounds-check warning for out-of-bounds constant.
// Max valid index = (8/4) - 1 = 1. Index 5 is out of bounds.
cuda_tile.module @kernels {
  testing$func @insert_oob_index(
      %src: !cuda_tile.tile<4x4xf32>,
      %dst: !cuda_tile.tile<8x8xf32>) {
    %big = cuda_tile.constant <i32: 5> : !cuda_tile.tile<i32>
    %ok = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-warning @below{{insert index 0 value 5 may be out of bounds}}
    %0 = cuda_tile.insert %src, %dst[%big, %ok] : !cuda_tile.tile<4x4xf32>, !cuda_tile.tile<8x8xf32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers BreakOp::verify and ContinueOp::verify error paths:
//   - BreakOp outside of allowed loop (verifyEarlyExitOp non-IfOp ancestor path)
//   - BreakOp operand types do not match parent loop result types
//   - ContinueOp operand types do not match parent LoopOp iter_values
//   - ContinueOp operand types do not match parent ForOp result types
//   - ForOp::verifyRegions rejects `return` inside a for body

// Covers BreakOp::verify "can only be nested within ..." path: `break`
// nested inside `if` only (no enclosing loop). The ParentOneOf trait allows
// IfOp, so control reaches verifyEarlyExitOp which fails on the function
// ancestor.
cuda_tile.module @kernels {
  // expected-note @below{{see unexpected ancestor operation}}
  testing$func @break_in_if_no_loop(%arg0: !cuda_tile.tile<i1>, %arg1: !cuda_tile.tile<i32>) {
    cuda_tile.if %arg0 {
      // expected-error @below{{can only be nested within a ancestor chain of 'cuda_tile.loop', 'cuda_tile.if' operations}}
      cuda_tile.break %arg1 : !cuda_tile.tile<i32>
    }
  }
}

// -----

// Covers BreakOp::verify operand-type-mismatch path: parent loop has no
// results but the break carries an operand.
cuda_tile.module @kernels {
  testing$func @break_operand_type_mismatch(%arg0: !cuda_tile.tile<i32>) {
    cuda_tile.loop {
      // expected-error @below{{operand types must correspond to the parent loop result types}}
      cuda_tile.break %arg0 : !cuda_tile.tile<i32>
    }
  }
}

// -----

// Covers ContinueOp::verify "can only be nested within ..." path:
// `continue` placed inside an `if` only (no enclosing loop).
cuda_tile.module @kernels {
  // expected-note @below{{see unexpected ancestor operation}}
  testing$func @continue_in_if_no_loop(%arg0: !cuda_tile.tile<i1>, %arg1: !cuda_tile.tile<i32>) {
    cuda_tile.if %arg0 {
      // expected-error @below{{can only be nested within a ancestor chain of 'cuda_tile.for', 'cuda_tile.loop', 'cuda_tile.if' operations}}
      cuda_tile.continue %arg1 : !cuda_tile.tile<i32>
    }
  }
}

// -----

// Covers ContinueOp::verify operand-type-mismatch inside LoopOp:
// `continue` inside `loop` with no iter_values, but carries an i32 operand.
cuda_tile.module @kernels {
  testing$func @continue_loop_type_mismatch(%arg0: !cuda_tile.tile<i32>) {
    cuda_tile.loop {
      // expected-error @below{{`loop` is missing a valid terminator}}
      cuda_tile.continue %arg0 : !cuda_tile.tile<i32>
    }
  }
}

// -----

// Covers ContinueOp::verify operand-type-mismatch inside ForOp:
// `continue` operand types differ from `for` result types.
cuda_tile.module @kernels {
  testing$func @continue_for_type_mismatch(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i32>) {
    %lb = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %ub = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>
    %step = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    cuda_tile.for %i in (%lb to %ub, step %step) : !cuda_tile.tile<i32> {
      // expected-error @below{{`for` is missing a valid terminator}}
      cuda_tile.continue %arg0 : !cuda_tile.tile<i32>
    }
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers FToFOp::verify additional paths:
//   - no-op (source type == result type) rejected
//   - Rule 3 strict widening with invalid rounding mode

// Covers FToFOp::verify "converting tiles must not be a no-op" path: source
// type equals result type.
cuda_tile.module @kernels {
  testing$func @ftof_noop(%a: !cuda_tile.tile<f32>) {
    // expected-error @below{{converting tiles must not be a no-op}}
    %0 = cuda_tile.ftof %a : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers FToFOp::verify Rule 6 invalid-rounding-mode path: 'other narrowing
// conversions to standard types' permit only nearest_even and zero.

// Covers FToFOp Rule 6: f32 -> f16 (other narrowing to standard type) with
// disallowed 'negative_inf' rounding.
cuda_tile.module @ftof_f32_to_f16_invalid_rounding {
  cuda_tile.entry @kernel() {
    %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    // expected-error @below{{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
    %1 = cuda_tile.ftof %0 rounding<negative_inf> : !cuda_tile.tile<f32> -> !cuda_tile.tile<f16>
  }
}

// -----

// Covers FToFOp Rule 6 with f64 -> bf16 narrowing and 'positive_inf'.
cuda_tile.module @ftof_f64_to_bf16_invalid_rounding {
  cuda_tile.entry @kernel() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    // expected-error @below{{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
    %1 = cuda_tile.ftof %0 rounding<positive_inf> : !cuda_tile.tile<f64> -> !cuda_tile.tile<bf16>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers FToFOp::verify Rules 1, 2, 4, 5 invalid-rounding-mode paths.

// Covers FToFOp Rule 4: F64 -> F32 narrowing with non-IEEE-round-to-nearest
// non-zero, non-pos-inf, non-neg-inf rounding ('nearest_away') should fail.
cuda_tile.module @ftof_f64_to_f32_invalid_rounding {
  cuda_tile.entry @kernel() {
    %0 = cuda_tile.constant <f64: 1.0> : !cuda_tile.tile<f64>
    // expected-error @below{{invalid rounding mode specified for conversion from f64 to f32}}
    %1 = cuda_tile.ftof %0 rounding<nearest_away> : !cuda_tile.tile<f64> -> !cuda_tile.tile<f32>
  }
}

// -----

// Covers FToFOp Rule 5: F32 -> TF32 narrowing with invalid rounding mode.
cuda_tile.module @ftof_f32_to_tf32_invalid_rounding {
  cuda_tile.entry @kernel() {
    %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    // expected-error @below{{invalid rounding mode specified for conversion from f32 to tf32}}
    %1 = cuda_tile.ftof %0 rounding<positive_inf> : !cuda_tile.tile<f32> -> !cuda_tile.tile<tf32>
  }
}

// -----

// Covers FToFOp Rule 1: conversion to f8E8M0FNU with disallowed rounding
// (only ZERO and POSITIVE_INF are valid).
cuda_tile.module @ftof_to_f8e8m0fnu_invalid_rounding {
  cuda_tile.entry @kernel() {
    %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    // expected-error @below{{invalid rounding mode specified for conversion to f8E8M0FNU. Only 'zero' and 'positive_inf' are supported}}
    %1 = cuda_tile.ftof %0 rounding<nearest_even> : !cuda_tile.tile<f32> -> !cuda_tile.tile<f8E8M0FNU>
  }
}

// -----

// Covers FToFOp Rule 2: low-precision destination requires NEAREST_EVEN.
cuda_tile.module @ftof_to_low_precision_invalid_rounding {
  cuda_tile.entry @kernel() {
    %0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    // expected-error @below{{invalid rounding mode specified for conversion to low-precision type}}
    %1 = cuda_tile.ftof %0 rounding<zero> : !cuda_tile.tile<f32> -> !cuda_tile.tile<f8E4M3FN>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers verify error paths in:
//   - GetIndexSpaceShapeOp (mismatched number of results)
//   - GetTensorShapeOp (mismatched number of results)
//   - MakePartitionViewOp / MakeStridedViewOp / MakeGatherScatterViewOp
//     verify when the input tensor_view type doesn't match the embedded one

// Covers GetIndexSpaceShapeOp::verify "expected N results due to view index
// space rank" path.
cuda_tile.module @kernels {
  testing$func @get_index_space_shape_wrong_count(
      %pv: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>) {
    // The partition_view has view index rank 2; we declare only 1 result type
    // to force the count mismatch via the generic op form.
    // expected-error @below{{expected 2 results due to view index space rank, but got 1}}
    %0 = "cuda_tile.get_index_space_shape"(%pv) : (!cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>) -> !cuda_tile.tile<i32>
  }
}

// -----

// Covers GetTensorShapeOp::verify "expected N results due to tensor rank"
// path: tensor_view has rank 2, but only 1 result declared.
cuda_tile.module @kernels {
  testing$func @get_tensor_shape_wrong_count(
      %tv: !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>) {
    // expected-error @below{{expected 2 results due to tensor rank, but got 1}}
    %0 = "cuda_tile.get_tensor_shape"(%tv) : (!cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>) -> !cuda_tile.tile<i32>
  }
}

// -----

// Covers MakePartitionViewOp::verify "expected the type of the provided
// tensor_view value (...) to be the same as the view's tensor_view type"
// path.
cuda_tile.module @kernels {
  testing$func @make_partition_view_type_mismatch(
      %tv: !cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>) {
    // Provide a partition_view whose embedded tensor_view is 2x2xf32, while
    // the operand is 4x4xf32. The custom parser resolves the operand to the
    // partition's tensor_view type — use generic form to bypass that.
    // expected-error @below{{expected the type of the provided tensor_view value}}
    %0 = "cuda_tile.make_partition_view"(%tv) : (!cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>) -> !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>
  }
}

// -----

// Covers MakeStridedViewOp::verify "expected the type of the provided
// tensor_view value (...) to be the same as the view's tensor_view type"
// path.
cuda_tile.module @kernels {
  testing$func @make_strided_view_type_mismatch(
      %tv: !cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>) {
    // expected-error @below{{expected the type of the provided tensor_view value}}
    %0 = "cuda_tile.make_strided_view"(%tv) : (!cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>) -> !cuda_tile.strided_view<tile=(2x2), traversal_strides=[1,1], tensor_view<2x2xf32, strides=[2, 1]>>
  }
}

// -----

// Covers MakeGatherScatterViewOp::verify "expected the type of the provided
// tensor_view value (...) to be the same as the view's tensor_view type"
// path.
cuda_tile.module @kernels {
  testing$func @make_gather_scatter_view_type_mismatch(
      %tv: !cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>) {
    // expected-error @below{{expected the type of the provided tensor_view value}}
    %0 = "cuda_tile.make_gather_scatter_view"(%tv) : (!cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>) -> !cuda_tile.gather_scatter_view<tile=(2x2), tensor_view<2x2xf32, strides=[2, 1]>, sparse_dim=0>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers IfOp::verify error paths:
//   - has return type but then-branch yields nothing
//   - has no return type but then-branch yields something
//   - yield type does not match op result type
//   - has non-empty return type without an else branch

// Covers IfOp::verify "has return type ... but then branch does not yield
// anything" path.
cuda_tile.module @kernels {
  testing$func @if_then_no_yield(%c: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{then branch does not yield anything}}
    %r = cuda_tile.if %c -> (!cuda_tile.tile<i32>) {
    } else {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    }
    return %r : !cuda_tile.tile<i32>
  }
}

// -----

// Covers IfOp::verify "does not return a value, but then branch yields"
// path.
cuda_tile.module @kernels {
  testing$func @if_then_yields_no_result(%c: !cuda_tile.tile<i1>) {
    %a = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{does not return a value, but then branch yields}}
    cuda_tile.if %c {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    }
  }
}

// -----

// Covers IfOp::verify "type does not match yield type" path: then yields
// f32 but op declares i32.
cuda_tile.module @kernels {
  testing$func @if_yield_type_mismatch(%c: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
    // expected-error @below{{then branch yields '!cuda_tile.tile<f32>' but op result type is '!cuda_tile.tile<i32>'}}
    %r = cuda_tile.if %c -> (!cuda_tile.tile<i32>) {
      cuda_tile.yield %b : !cuda_tile.tile<f32>
    } else {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    }
    return %r : !cuda_tile.tile<i32>
  }
}

// -----

// Covers IfOp::verify "has non-empty return type, must define else branch"
// path.
cuda_tile.module @kernels {
  testing$func @if_no_else_with_result(%c: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{has non-empty return type, must define else branch}}
    %r = cuda_tile.if %c -> (!cuda_tile.tile<i32>) {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    }
    return %r : !cuda_tile.tile<i32>
  }
}

// -----

// Covers IfOp::verify "else branch yields" type mismatch path: then yields
// matching but else yields different type. The then check passes first;
// we hit the else-branch yield type check.
cuda_tile.module @kernels {
  testing$func @if_else_yield_type_mismatch(%c: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i32> {
    %a = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
    // expected-error @below{{else branch yields '!cuda_tile.tile<f32>' but op result type is '!cuda_tile.tile<i32>'}}
    %r = cuda_tile.if %c -> (!cuda_tile.tile<i32>) {
      cuda_tile.yield %a : !cuda_tile.tile<i32>
    } else {
      cuda_tile.yield %b : !cuda_tile.tile<f32>
    }
    return %r : !cuda_tile.tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers IotaOp::verify and PrintTkoOp::verify error paths.

// Covers IotaOp::verify "the number of elements ... exceeds the maximum
// value of element type" path. i8 can represent up to 256 values; 512
// elements requires more.
cuda_tile.module @kernels {
  testing$func @iota_overflow() {
    // expected-error @below{{the number of elements 512 exceeds the maximum value of element type 'i8'}}
    %0 = cuda_tile.iota : !cuda_tile.tile<512xi8>
  }
}

// -----

// Covers PrintTkoOp::verify "found unterminated format expression" path:
// trailing `%` with no conversion character.
cuda_tile.module @kernels {
  testing$func @print_tko_unterminated() {
    // expected-error @below{{found unterminated format expression}}
    %t = cuda_tile.print_tko "value=%" -> !cuda_tile.token
  }
}

// -----

// Covers PrintTkoOp::verify "incorrect number of operands" path: format
// has one `%d`, op has zero operands.
cuda_tile.module @kernels {
  testing$func @print_tko_missing_arg() {
    // expected-error @below{{incorrect number of operands: expected 1, found 0}}
    %t = cuda_tile.print_tko "%d" -> !cuda_tile.token
  }
}

// -----

// Covers PrintTkoOp::verify "incorrect number of operands" path: format
// has zero `%`, op has one operand (extras).
cuda_tile.module @kernels {
  testing$func @print_tko_extra_arg(%v: !cuda_tile.tile<i32>) {
    // expected-error @below{{incorrect number of operands: expected 0, found 1}}
    %t = cuda_tile.print_tko "nothing %% only escaped", %v : !cuda_tile.tile<i32> -> !cuda_tile.token
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakeTensorViewOp::verify dynamic-value-type-consistency paths.
// Reached via the generic op form because the custom parser binds all
// dynamic operands to a single splat type.

// Covers MakeTensorViewOp::verify "expected dynamic shape index ... to be of
// the same type as the other dynamic values" path.
cuda_tile.module @kernels {
  testing$func @mtv_mixed_dynamic_shape_types(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
      %a: !cuda_tile.tile<i64>,
      %b: !cuda_tile.tile<i32>) {
    // expected-error @below{{expected dynamic shape index 1 to be of the same type as the other dynamic values}}
    %tv = "cuda_tile.make_tensor_view"(%base, %a, %b)
        <{operandSegmentSizes = array<i32: 1, 2, 0>,
          static_shape = array<i64: -9223372036854775808, -9223372036854775808>,
          static_strides = array<i64: 32, 1>}>
        : (!cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<i64>, !cuda_tile.tile<i32>)
        -> !cuda_tile.tensor_view<?x?xf32, strides=[32, 1]>
  }
}

// -----

// Covers MakeTensorViewOp::verify "expected dynamic stride index ... to be of
// the same type as the other dynamic values" path. Dynamic shape is empty so
// the stride loop initializes the type from the first stride operand.
cuda_tile.module @kernels {
  testing$func @mtv_mixed_dynamic_stride_types(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
      %a: !cuda_tile.tile<i64>,
      %b: !cuda_tile.tile<i32>) {
    // expected-error @below{{expected dynamic stride index 1 to be of the same type as the other dynamic values}}
    %tv = "cuda_tile.make_tensor_view"(%base, %a, %b)
        <{operandSegmentSizes = array<i32: 1, 0, 2>,
          static_shape = array<i64: 32, 32>,
          static_strides = array<i64: -9223372036854775808, -9223372036854775808>}>
        : (!cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<i64>, !cuda_tile.tile<i32>)
        -> !cuda_tile.tensor_view<32x32xf32, strides=[?, ?]>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakeTensorViewOp::parse / verify error paths.

// Covers MakeTensorViewOp::parse "expected 'tensor_view' type, but got"
// path. The result type slot resolves to a non-tensor_view type.
cuda_tile.module @kernels {
  testing$func @mtv_wrong_result_type(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [4, 4], strides = [4, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<f32>
  // expected-error @below{{expected 'tensor_view' type, but got}}
  }
}

// -----

// Covers MakeTensorViewOp::parse "expected either integer or SSA value"
// path: dynamic-shape element neither an integer constant nor an SSA name.
cuda_tile.module @kernels {
  testing$func @mtv_bad_shape_element(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{expected either integer or SSA value}}
    %tv = cuda_tile.make_tensor_view %p, shape = [foo, 4], strides = [4, 1]
        : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>
  }
}

// -----

// Covers MakeTensorViewOp::verify "expected pointer to ELEM to build
// tensor_view of this type" path: base pointee is f64 but tensor_view
// element type is f32.
cuda_tile.module @kernels {
  testing$func @mtv_pointee_mismatch(%p: !cuda_tile.tile<!cuda_tile.ptr<f64>>) {
    // expected-error @below{{expected pointer to 'f32' to build tensor_view of this type, got 'f64'}}
    %tv = "cuda_tile.make_tensor_view"(%p) <{
        static_shape = array<i64: 4, 4>,
        static_strides = array<i64: 4, 1>,
        operandSegmentSizes = array<i32: 1, 0, 0>}>
        : (!cuda_tile.tile<!cuda_tile.ptr<f64>>) -> !cuda_tile.tensor_view<4x4xf32, strides=[4, 1]>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakePartitionViewOp::verify, MakeStridedViewOp::verify and
// MakeGatherScatterViewOp::verify type-mismatch error paths.
//
// These checks compare the type of the provided tensor_view operand with the
// inner tensor_view type embedded in the result view type. They are reachable
// only via the generic op form because the parser already binds the operand
// type to the view's inner tensor_view.

// Covers MakePartitionViewOp::verify "expected the type of the provided
// tensor_view value ... to be the same as the view's tensor_view type" path.
cuda_tile.module @kernels {
  testing$func @make_partition_view_mismatch(%tv: !cuda_tile.tensor_view<128xf32, strides=[1]>) {
    // expected-error @below{{expected the type of the provided tensor_view value}}
    %pv = "cuda_tile.make_partition_view"(%tv)
        : (!cuda_tile.tensor_view<128xf32, strides=[1]>)
        -> !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<256xf32, strides=[1]>>
  }
}

// -----

// Covers MakeStridedViewOp::verify type-mismatch path.
cuda_tile.module @kernels {
  testing$func @make_strided_view_mismatch(%tv: !cuda_tile.tensor_view<128xf32, strides=[1]>) {
    // expected-error @below{{expected the type of the provided tensor_view value}}
    %sv = "cuda_tile.make_strided_view"(%tv)
        : (!cuda_tile.tensor_view<128xf32, strides=[1]>)
        -> !cuda_tile.strided_view<tile=(8), traversal_strides=[1], !cuda_tile.tensor_view<256xf32, strides=[1]>>
  }
}

// -----

// Covers MakeGatherScatterViewOp::verify type-mismatch path.
cuda_tile.module @kernels {
  testing$func @make_gather_scatter_view_mismatch(%tv: !cuda_tile.tensor_view<128xf32, strides=[1]>) {
    // expected-error @below{{expected the type of the provided tensor_view value}}
    %gsv = "cuda_tile.make_gather_scatter_view"(%tv)
        : (!cuda_tile.tensor_view<128xf32, strides=[1]>)
        -> !cuda_tile.gather_scatter_view<tile=(8), !cuda_tile.tensor_view<256xf32, strides=[1]>, sparse_dim=0>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MmaFScaledOp::verify error paths:
//   - K dimension of lhs not divisible by lhs_scale K dimension
//   - f8 element type with vecSize != 32

// Covers MmaFScaledOp::verify "dim K of lhs_scale must be a divisor of dim K
// of lhs" error path. With lhs K = 128 and lhs_scale K = 256, 128 % 256 != 0.
cuda_tile.module @kernels {
  testing$func @mma_f_scaled_k_not_divisor(
      %lhs: !cuda_tile.tile<128x128xf8E5M2>,
      %rhs: !cuda_tile.tile<128x128xf8E5M2>,
      %acc: !cuda_tile.tile<128x128xf32>,
      %lhs_scale: !cuda_tile.tile<128x256xf8E8M0FNU>,
      %rhs_scale: !cuda_tile.tile<256x128xf8E8M0FNU>) {
    // expected-error @below{{shape error: dim 1 of lhs_scale (256) must be a divisor of dim 1 of lhs (128)}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale
        : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x256xf8E8M0FNU>, !cuda_tile.tile<256x128xf8E8M0FNU>
  }
}

// -----

// Covers MmaFScaledOp::verify "f8 element type requires block scale factor
// of lhs_scale and rhs_scale to be 32" path. With K=64, scale K=4, vecSize=16.
cuda_tile.module @kernels {
  testing$func @mma_f_scaled_f8_wrong_vecsize(
      %lhs: !cuda_tile.tile<128x64xf8E4M3FN>,
      %rhs: !cuda_tile.tile<64x128xf8E4M3FN>,
      %acc: !cuda_tile.tile<128x128xf32>,
      %lhs_scale: !cuda_tile.tile<128x4xf8E8M0FNU>,
      %rhs_scale: !cuda_tile.tile<4x128xf8E8M0FNU>) {
    // expected-error @below{{f8 element type requires block scale factor of lhs_scale and rhs_scale to be 32, but got 16}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale
        : !cuda_tile.tile<128x64xf8E4M3FN>, !cuda_tile.tile<64x128xf8E4M3FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }
}

// -----

// Covers MmaFScaledOp::verify "f4E2M1FN element type with f8E8M0FNU scales
// requires block scale factor ... to be 16 or 32" path with vecSize=8.
// With K=64, scale K=8, vecSize=8 which is not 16 or 32.
cuda_tile.module @kernels {
  testing$func @mma_f_scaled_f4_e8m0_wrong_vecsize(
      %lhs: !cuda_tile.tile<128x64xf4E2M1FN>,
      %rhs: !cuda_tile.tile<64x128xf4E2M1FN>,
      %acc: !cuda_tile.tile<128x128xf32>,
      %lhs_scale: !cuda_tile.tile<128x8xf8E8M0FNU>,
      %rhs_scale: !cuda_tile.tile<8x128xf8E8M0FNU>) {
    // expected-error @below{{f4E2M1FN element type with f8E8M0FNU scales requires block scale factor of lhs_scale and rhs_scale to be 16 or 32, but got 8}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale
        : !cuda_tile.tile<128x64xf4E2M1FN>, !cuda_tile.tile<64x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E8M0FNU>
  }
}

// -----

// Covers MmaFScaledOp::verify "f4E2M1FN element type with f8E4M3FN scales
// requires block scale factor ... to be 16" path with vecSize=8.
// With K=64 and scale K=8, vecSize=8 which is not 16.
cuda_tile.module @kernels {
  testing$func @mma_f_scaled_f4_e4m3_wrong_vecsize(
      %lhs: !cuda_tile.tile<128x64xf4E2M1FN>,
      %rhs: !cuda_tile.tile<64x128xf4E2M1FN>,
      %acc: !cuda_tile.tile<128x128xf32>,
      %lhs_scale: !cuda_tile.tile<128x8xf8E4M3FN>,
      %rhs_scale: !cuda_tile.tile<8x128xf8E4M3FN>) {
    // expected-error @below{{f4E2M1FN element type with f8E4M3FN scales requires block scale factor of lhs_scale and rhs_scale to be 16, but got 8}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale
        : !cuda_tile.tile<128x64xf4E2M1FN>, !cuda_tile.tile<64x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E4M3FN>, !cuda_tile.tile<8x128xf8E4M3FN>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MmaFOp::verify and MmaFScaledOp::verify rejection paths.

// Covers MmaFOp::verify "unsupported combination of element types" path:
// f16 x f16 input but f64 accumulator is not in the allowed set.
cuda_tile.module @kernels {
  testing$func @mma_f_bad_combo(
      %lhs: !cuda_tile.tile<16x16xf16>,
      %rhs: !cuda_tile.tile<16x16xf16>,
      %acc: !cuda_tile.tile<16x16xf64>) -> !cuda_tile.tile<16x16xf64> {
    // expected-error @below{{unsupported combination of element types}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc
        : !cuda_tile.tile<16x16xf16>, !cuda_tile.tile<16x16xf16>, !cuda_tile.tile<16x16xf64>
    return %0 : !cuda_tile.tile<16x16xf64>
  }
}

// -----

// Covers verifyMmaShapes batched dim 0 mismatch in lhs vs rhs (rank-3 inputs).
cuda_tile.module @kernels {
  testing$func @mma_f_batch_mismatch_rhs(
      %lhs: !cuda_tile.tile<2x16x16xf16>,
      %rhs: !cuda_tile.tile<4x16x16xf16>,
      %acc: !cuda_tile.tile<2x16x16xf32>) -> !cuda_tile.tile<2x16x16xf32> {
    // expected-error @below{{dim 0 of lhs (2) and dim 0 of rhs (4) must match}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc
        : !cuda_tile.tile<2x16x16xf16>, !cuda_tile.tile<4x16x16xf16>, !cuda_tile.tile<2x16x16xf32>
    return %0 : !cuda_tile.tile<2x16x16xf32>
  }
}

// -----

// Covers verifyMmaShapes batched dim 0 mismatch in lhs vs acc (rank-3 inputs).
cuda_tile.module @kernels {
  testing$func @mma_f_batch_mismatch_acc(
      %lhs: !cuda_tile.tile<2x16x16xf16>,
      %rhs: !cuda_tile.tile<2x16x16xf16>,
      %acc: !cuda_tile.tile<4x16x16xf32>) -> !cuda_tile.tile<4x16x16xf32> {
    // expected-error @below{{dim 0 of lhs (2) and dim 0 of acc (4) must match}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc
        : !cuda_tile.tile<2x16x16xf16>, !cuda_tile.tile<2x16x16xf16>, !cuda_tile.tile<4x16x16xf32>
    return %0 : !cuda_tile.tile<4x16x16xf32>
  }
}

// -----

// Covers verifyMmaShapes batched colDim mismatch (K dim): lhs colDim != rhs rowDim.
cuda_tile.module @kernels {
  testing$func @mma_f_k_mismatch_rhs_batched(
      %lhs: !cuda_tile.tile<2x16x32xf16>,
      %rhs: !cuda_tile.tile<2x16x16xf16>,
      %acc: !cuda_tile.tile<2x16x16xf32>) -> !cuda_tile.tile<2x16x16xf32> {
    // expected-error @below{{shape error: dim 2 of lhs (32) and dim 1 of rhs (16) must match}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc
        : !cuda_tile.tile<2x16x32xf16>, !cuda_tile.tile<2x16x16xf16>, !cuda_tile.tile<2x16x16xf32>
    return %0 : !cuda_tile.tile<2x16x16xf32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ModuleOp::verify, EntryOp::verify, and GetGlobalOp::verifySymbolUses
// error paths.

// Covers ModuleOp::verify "non-symbol operations are not allowed in a module
// body" path.
cuda_tile.module @kernels {
  // expected-error @below{{non-symbol operations are not allowed in a module body}}
  %c = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
}

// -----

// Covers EntryOp::verify "entry op must have scalar types (rank 0
// !cuda_tile.tile)" path.
cuda_tile.module @kernels {
  // expected-error @below{{entry op must have scalar types (rank 0 !cuda_tile.tile)}}
  cuda_tile.entry @bad_entry_arg(%arg0: !cuda_tile.tile<4xf32>) {}
}

// -----

// Covers GetGlobalOp::verifySymbolUses "does not reference a valid global"
// path: the @missing global is undefined.
cuda_tile.module @kernels {
  testing$func @missing_global() {
    // expected-error @below{{'missing' does not reference a valid global}}
    %0 = cuda_tile.get_global @missing : !cuda_tile.tile<!cuda_tile.ptr<f32>>
  }
}

// -----

// Covers GetGlobalOp::verifySymbolUses "pointee type of result type ... does
// not match type ... of the global @ ..." path.
cuda_tile.module @kernels {
  cuda_tile.global @g1 alignment = 4 <i32: [0]> : !cuda_tile.tile<1xi32>
  testing$func @global_type_mismatch() {
    // expected-error @below{{does not match type 'i32' of the global @g1}}
    %0 = cuda_tile.get_global @g1 : !cuda_tile.tile<!cuda_tile.ptr<f32>>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers reduce/scan code paths not yet exercised:
//   - BF16 operand / F16 identity (continue branch in verifyAggregateOp)
//   - F16 operand / BF16 identity (continue branch in verifyAggregateOp)
//   - ScanOp requires matching return shape: shape mismatch path

// Covers verifyAggregateOp "continue" branch for BF16 operand with F16
// identity (allowed mixed-precision pair). This is a successful parse.
cuda_tile.module @kernels {
  testing$func @reduce_bf16_f16_identity_ok(%arg0: !cuda_tile.tile<8xbf16>) {
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f16]
        : !cuda_tile.tile<8xbf16> -> !cuda_tile.tile<bf16>
        (%a : !cuda_tile.tile<bf16>, %b : !cuda_tile.tile<bf16>) {
      %s = cuda_tile.addf %a, %b : !cuda_tile.tile<bf16>
      cuda_tile.yield %s : !cuda_tile.tile<bf16>
    }
  }
}

// -----

// Covers verifyAggregateOp "continue" branch for F16 operand with BF16
// identity (allowed mixed-precision pair).
cuda_tile.module @kernels {
  testing$func @reduce_f16_bf16_identity_ok(%arg0: !cuda_tile.tile<8xf16>) {
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : bf16]
        : !cuda_tile.tile<8xf16> -> !cuda_tile.tile<f16>
        (%a : !cuda_tile.tile<f16>, %b : !cuda_tile.tile<f16>) {
      %s = cuda_tile.addf %a, %b : !cuda_tile.tile<f16>
      cuda_tile.yield %s : !cuda_tile.tile<f16>
    }
  }
}

// -----

// Covers verifyAggregateOp "expect same type for operand at index: 0 and
// result at index: 0" path in ScanOp (requiresMatchingReturnShape=true).
cuda_tile.module @kernels {
  testing$func @scan_shape_mismatch(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same type for operand at index: 0 and result at index: 0}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32]
        : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<4xf32>
        (%a : !cuda_tile.tile<f32>, %b : !cuda_tile.tile<f32>) {
      %s = cuda_tile.addf %a, %b : !cuda_tile.tile<f32>
      cuda_tile.yield %s : !cuda_tile.tile<f32>
    }
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers ReturnOp::verify EntryOp branch error path:
//   - return has operands (caught by enclosing function 0-result check).

// Covers ReturnOp::verify EntryOp branch "has 1 operands, but enclosing
// function (@...) returns 0" path: any return-with-operands from inside an
// entry op trips the same code path.
cuda_tile.module @kernels {
  cuda_tile.entry @entry_return_value(%a: !cuda_tile.tile<i32>) {
    // expected-error @below{{has 1 operands, but enclosing function (@entry_return_value) returns 0}}
    cuda_tile.return %a : !cuda_tile.tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers parseRoundingModeWithModes error paths:
//   - validator-failure path (FToI accepts only nearest_int_to_zero)
//   - unknown rounding mode keyword

// Covers the validator-failure branch in parseRoundingModeWithModes.
cuda_tile.module @kernels {
  testing$func @ftoi_disallowed_rounding(%arg0: !cuda_tile.tile<f32>) {
    // expected-error @below{{expected rounding mode to be one of: 'nearest_int_to_zero'}}
    %0 = cuda_tile.ftoi %arg0 signed rounding<nearest_even> : !cuda_tile.tile<f32> -> !cuda_tile.tile<i32>
  }
}

// -----

// Covers the symbolizeRoundingMode unknown-keyword branch in
// parseRoundingModeWithModes via FToI.
cuda_tile.module @kernels {
  testing$func @ftoi_unknown_rounding(%arg0: !cuda_tile.tile<f32>) {
    // expected-error @below{{expected rounding mode to be one of}}
    %0 = cuda_tile.ftoi %arg0 signed rounding<garbage_mode> : !cuda_tile.tile<f32> -> !cuda_tile.tile<i32>
  }
}

// -----

// Covers parseIEEERoundingMode validator-failure path via AddF: the IEEE
// rounding parser rejects 'approx' (a non-IEEE mode).
cuda_tile.module @kernels {
  testing$func @addf_non_ieee_rounding(
      %arg0: !cuda_tile.tile<f32>,
      %arg1: !cuda_tile.tile<f32>) {
    // expected-error @below{{expected rounding mode to be one of}}
    %0 = cuda_tile.addf %arg0, %arg1 rounding<approx> : !cuda_tile.tile<f32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers verifier paths:
//   - JoinTokensOp::verify single-token rejection
//   - GlobalOp::verify rank != 1
//   - BitcastOp::verify width mismatch
//   - ReshapeOp::verify num-elements mismatch
//   - ExtIOp::verify extending to smaller-or-equal int
//   - NegIOp::verify NUW rejected

// Covers JoinTokensOp::verify single-token rejection path.
cuda_tile.module @kernels {
  testing$func @join_tokens_too_few(%arg0: !cuda_tile.token) {
    // expected-error @below{{expect two or more tokens}}
    %t = cuda_tile.join_tokens %arg0 : !cuda_tile.token
  }
}

// -----

// Covers GlobalOp::verify rank != 1 path.
// expected-error @below{{type must have rank 1}}
cuda_tile.global @g0 <f32: [[1.0, 2.0], [3.0, 4.0]]> : !cuda_tile.tile<2x2xf32>

// -----

// Covers BitcastOp::verify width-mismatch path.
cuda_tile.module @kernels {
  testing$func @bitcast_width_mismatch(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{types must be equal width}}
    %0 = cuda_tile.bitcast %arg0 : !cuda_tile.tile<i32> -> !cuda_tile.tile<i64>
  }
}

// -----

// Covers ReshapeOp::verify mismatched-num-elements path.
cuda_tile.module @kernels {
  testing$func @reshape_count_mismatch(%arg0: !cuda_tile.tile<4x8xf32>) {
    // expected-error @below{{expected source tile and result tile to have the same number of elements}}
    %0 = cuda_tile.reshape %arg0 : !cuda_tile.tile<4x8xf32> -> !cuda_tile.tile<16xf32>
  }
}

// -----

// Covers ExtIOp::verify "extending to smaller or identical integer" path
// when extending from i32 to i16 (smaller).
cuda_tile.module @kernels {
  testing$func @exti_smaller(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{extending to smaller or identical integer}}
    %0 = cuda_tile.exti %arg0 unsigned : !cuda_tile.tile<i32> -> !cuda_tile.tile<i16>
  }
}

// -----

// Covers NegIOp::verify NUW rejection path with i32 element type.
cuda_tile.module @kernels {
  testing$func @negi_nuw_rejected(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{'no_unsigned_wrap' overflow flag is not supported}}
    %0 = cuda_tile.negi %arg0 overflow<no_unsigned_wrap> : !cuda_tile.tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers custom-parse error paths for view-creation ops and shape ops:
//   - MakePartitionViewOp::parse "expected 'partition_view' type" path
//   - MakeStridedViewOp::parse "expected 'strided_view' type" path
//   - MakeGatherScatterViewOp::parse "expected 'gather_scatter_view' type" path
//   - GetIndexSpaceShapeOp::parse "expected tile view" path
//   - GetTensorShapeOp::parse "expected tensor_view" path

// Covers MakePartitionViewOp::parse error when the result type is not a
// partition_view.
cuda_tile.module @kernels {
  testing$func @make_partition_view_wrong_type(%tv: !cuda_tile.tensor_view<128xf32, strides=[1]>) {
    // expected-error @below{{expected 'partition_view' type, but got}}
    %pv = cuda_tile.make_partition_view %tv : !cuda_tile.tensor_view<128xf32, strides=[1]>
  }
}

// -----

// Covers MakeStridedViewOp::parse error when the result type is not a
// strided_view.
cuda_tile.module @kernels {
  testing$func @make_strided_view_wrong_type(%tv: !cuda_tile.tensor_view<128xf32, strides=[1]>) {
    // expected-error @below{{expected 'strided_view' type, but got}}
    %sv = cuda_tile.make_strided_view %tv : !cuda_tile.tensor_view<128xf32, strides=[1]>
  }
}

// -----

// Covers MakeGatherScatterViewOp::parse error when the result type is not a
// gather_scatter_view.
cuda_tile.module @kernels {
  testing$func @make_gather_scatter_view_wrong_type(%tv: !cuda_tile.tensor_view<128xf32, strides=[1]>) {
    // expected-error @below{{expected 'gather_scatter_view' type, but got}}
    %gsv = cuda_tile.make_gather_scatter_view %tv : !cuda_tile.tensor_view<128xf32, strides=[1]>
  }
}

// -----

// Covers GetIndexSpaceShapeOp::parse error when the src type is not a tile
// view (any non-view type triggers it).
cuda_tile.module @kernels {
  testing$func @get_index_space_shape_not_view(%arg0: !cuda_tile.tile<128xf32>) {
    // expected-error @below{{expected tile view, got}}
    %0 = cuda_tile.get_index_space_shape %arg0 : !cuda_tile.tile<128xf32> -> !cuda_tile.tile<i32>
  }
}

// -----

// Covers GetTensorShapeOp::parse error when the src type is not a
// tensor_view.
cuda_tile.module @kernels {
  testing$func @get_tensor_shape_not_view(%arg0: !cuda_tile.tile<128xf32>) {
    // expected-error @below{{expected tensor_view, got}}
    %0 = cuda_tile.get_tensor_shape %arg0 : !cuda_tile.tile<128xf32> -> !cuda_tile.tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers parseViewTkoCommon error paths in load_view_tko/store_view_tko
// for the optional `inbounds = [...]` clause:
//   - wrong keyword inside inbounds list
//   - inbounds size mismatch (caught by op verify after parse)

// Covers parseViewTkoCommon "expected 'true' or 'false' in inbounds list"
// path: provide an invalid keyword inside the inbounds list.
cuda_tile.module @kernels {
  testing$func @load_view_tko_bad_inbounds(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 'true' or 'false' in inbounds list, got 'maybe'}}
    %v, %t = cuda_tile.load_view_tko weak %view[%c0, %c0] inbounds = [maybe, true]
        : !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>, !cuda_tile.tile<i32> -> !cuda_tile.tile<2x2xf32>, !cuda_tile.token
  }
}

// -----

// Covers LoadViewTkoOp::verify "inbounds size (...) must match the number
// of index dimensions" path. Build the op generically to bypass the parser's
// auto-sizing of `inbounds`.
cuda_tile.module @kernels {
  testing$func @load_view_tko_inbounds_size_mismatch(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{inbounds size (3) must match the number of index dimensions (2)}}
    %v, %t = "cuda_tile.load_view_tko"(%view, %c0, %c0) <{
        memory_ordering_semantics = 0 : i32,
        inbounds = array<i1: true, true, true>,
        operandSegmentSizes = array<i32: 1, 2, 0>}>
        : (!cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>) -> (!cuda_tile.tile<2x2xf32>, !cuda_tile.token)
  }
}

// -----

// Covers StoreViewTkoOp::verify "inbounds size (...) must match the number
// of index dimensions" path.
cuda_tile.module @kernels {
  testing$func @store_view_tko_inbounds_size_mismatch(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>,
      %val: !cuda_tile.tile<2x2xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{inbounds size (3) must match the number of index dimensions (2)}}
    %t = "cuda_tile.store_view_tko"(%val, %view, %c0, %c0) <{
        memory_ordering_semantics = 0 : i32,
        inbounds = array<i1: true, true, true>,
        operandSegmentSizes = array<i32: 1, 1, 2, 0>}>
        : (!cuda_tile.tile<2x2xf32>, !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xf32, strides=[2, 1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i32>) -> !cuda_tile.token
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// DivIOp, ExpOp, FToFOp, BitcastOp verifier error branches

// DivIOp: invalid rounding mode
cuda_tile.module @divi_invalid_rounding {
  testing$func @func() {
    %a = "materialize_tensor"() : () -> !cuda_tile.tile<4xi32>
    %b = "materialize_tensor"() : () -> !cuda_tile.tile<4xi32>
    // expected-error @below{{invalid rounding mode specified, expect one of [zero, negative_inf, positive_inf]}}
    %0 = cuda_tile.divi %a, %b signed rounding<nearest_even> : !cuda_tile.tile<4xi32>
  }
}

// -----

// ExpOp: invalid rounding mode
cuda_tile.module @exp_invalid_rounding {
  testing$func @func() {
    %a = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    // expected-error @below{{invalid rounding mode specified}}
    %0 = cuda_tile.exp %a rounding<nearest_even> : !cuda_tile.tile<4xf32>
  }
}

// -----

// FToFOp: no-op conversion
cuda_tile.module @ftof_noop {
  testing$func @func() {
    %a = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    // expected-error @below{{converting tiles must not be a no-op}}
    %0 = cuda_tile.ftof %a : !cuda_tile.tile<4xf32> -> !cuda_tile.tile<4xf32>
  }
}

// -----

// FToFOp: invalid rounding for f8E8M0FNU target
cuda_tile.module @ftof_bad_f8e8m0_rounding {
  testing$func @func() {
    %a = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    // expected-error @below{{invalid rounding mode specified for conversion to f8E8M0FNU}}
    %0 = cuda_tile.ftof %a rounding<nearest_even> : !cuda_tile.tile<4xf32> -> !cuda_tile.tile<4xf8E8M0FNU>
  }
}

// -----

// FToFOp: invalid rounding for narrowing to standard type (Rule 6)
cuda_tile.module @ftof_bad_rounding_non_e8m0 {
  testing$func @func() {
    %a = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    // expected-error @below{{invalid rounding mode specified. Only 'nearest_even' and 'zero' are supported for narrowing conversions to standard types}}
    %0 = cuda_tile.ftof %a rounding<negative_inf> : !cuda_tile.tile<4xf32> -> !cuda_tile.tile<4xf16>
  }
}

// -----

// BitcastOp: width mismatch
cuda_tile.module @bitcast_width_mismatch {
  testing$func @func() {
    %a = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    // expected-error @below{{types must be equal width}}
    %0 = cuda_tile.bitcast %a : !cuda_tile.tile<4xf32> -> !cuda_tile.tile<4xi16>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// parseAssumePredicate error paths

// Unknown assume predicate name
cuda_tile.module @assume_unknown {
  testing$func @func(%arg0: !cuda_tile.tile<i32>) {
    // expected-error @below{{unknown assume predicate attribute: foo_bar}}
    %0 = cuda_tile.assume foo_bar, %arg0 : tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers AtomicCASTkoOp::verify() - non-float/non-int type check (f16 has 16 bits)
// Covers AtomicCASTkoOp::verify() - pointee type mismatch

// AtomicCASTkoOp: non-float/non-int type check (f16 has 16 bits, not 32 or 64)
cuda_tile.module @atomic_cas_16bit {
  testing$func @func(
      %ptrs: !cuda_tile.tile<4xptr<f16>>,
      %val: !cuda_tile.tile<4xf16>,
      %cmp: !cuda_tile.tile<4xf16>) {
    // expected-error @below{{expect only float or integer types with 32 or 64 bit}}
    %result, %t = cuda_tile.atomic_cas_tko relaxed device %ptrs, %val, %cmp
        : tile<4xptr<f16>>, tile<4xf16> -> tile<4xf16>, token
  }
}

// -----

// AtomicCASTkoOp: pointee type vs val type mismatch
cuda_tile.module @atomic_cas_pointee_mismatch {
  testing$func @func(
      %ptrs: !cuda_tile.tile<4xptr<i32>>,
      %val: !cuda_tile.tile<4xf32>,
      %cmp: !cuda_tile.tile<4xf32>) {
    // expected-error @below{{expected pointee type}}
    %result, %t = cuda_tile.atomic_cas_tko relaxed device %ptrs, %val, %cmp
        : tile<4xptr<i32>>, tile<4xf32> -> tile<4xf32>, token
  }
}

// -----

// AtomicRedViewTkoOp: index count mismatch (too few indices)
cuda_tile.module @atomic_red_view_wrong_index_count {
  testing$func @func(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 index operand(s) for view with tile rank 2, got 1}}
    %t = cuda_tile.atomic_red_view_tko relaxed device %view[%c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// AtomicCASTkoOp::verify and AtomicRedViewTkoOp::verify error branches

// AtomicCASTkoOp: mask shape mismatch
cuda_tile.module @atomic_cas_mask_mismatch {
  testing$func @func(
      %ptrs: !cuda_tile.tile<4xptr<i32>>,
      %val: !cuda_tile.tile<4xi32>,
      %cmp: !cuda_tile.tile<4xi32>,
      %mask: !cuda_tile.tile<8xi1>) {
    // expected-error @below{{failed to verify that all of {pointers, val, cmp and mask} have same shape}}
    %result, %t = cuda_tile.atomic_cas_tko relaxed device %ptrs, %val, %cmp, %mask
        : tile<4xptr<i32>>, tile<4xi32>, tile<8xi1> -> tile<4xi32>, token
  }
}

// -----

// AtomicCASTkoOp: non-32/64bit type
cuda_tile.module @atomic_cas_bad_bitwidth {
  testing$func @func(
      %ptrs: !cuda_tile.tile<4xptr<i16>>,
      %val: !cuda_tile.tile<4xi16>,
      %cmp: !cuda_tile.tile<4xi16>) {
    // expected-error @below{{expect only float or integer types with 32 or 64 bit}}
    %result, %t = cuda_tile.atomic_cas_tko relaxed device %ptrs, %val, %cmp
        : tile<4xptr<i16>>, tile<4xi16> -> tile<4xi16>, token
  }
}

// -----

// AtomicRedViewTkoOp: XCHG mode not allowed
cuda_tile.module @atomic_red_view_xchg {
  testing$func @func(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{atomic_red_view_tko op cannot use xchg operation}}
    %t = cuda_tile.atomic_red_view_tko relaxed device %view[%c0, %c0], xchg, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
  }
}

// -----

// AtomicRedViewTkoOp: non-RELAXED memory ordering
cuda_tile.module @atomic_red_view_non_relaxed {
  testing$func @func(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{only 'relaxed' memory ordering is supported for view-based atomic reductions}}
    %t = cuda_tile.atomic_red_view_tko acquire device %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
  }
}

// -----

// AtomicRedViewTkoOp: sys memory scope
cuda_tile.module @atomic_red_view_sys_scope {
  testing$func @func(
      %view: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<2x2xi32, strides=[2, 1]>>,
      %value: !cuda_tile.tile<2x2xi32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{memory scope 'sys' is not supported for view-based atomic reductions}}
    %t = cuda_tile.atomic_red_view_tko relaxed sys %view[%c0, %c0], add, %value
        : tile<2x2xi32>, partition_view<tile=(2x2), tensor_view<2x2xi32, strides=[2,1]>>, tile<i32> -> token
  }
}

// -----

// Test CatOp::verify - dim out of range

cuda_tile.module @cat_dim_out_of_range {
    cuda_tile.testing$func @func() {
        %arg0 = "materialize_tensor"() : () -> !cuda_tile.tile<16x16xf32>
        %arg1 = "materialize_tensor"() : () -> !cuda_tile.tile<16x16xf32>
        // expected-error @below{{expect dim to be [0, 2)}}
        %result = cuda_tile.cat %arg0, %arg1 dim = 2 : !cuda_tile.tile<16x16xf32>, !cuda_tile.tile<16x16xf32> -> !cuda_tile.tile<16x32xf32>
    }
}

// -----

// Test CatOp::verify - non-concat dimension mismatch

cuda_tile.module @cat_non_concat_dim_mismatch {
    cuda_tile.testing$func @func() {
        %arg0 = "materialize_tensor"() : () -> !cuda_tile.tile<16x8xf32>
        %arg1 = "materialize_tensor"() : () -> !cuda_tile.tile<16x16xf32>
        // expected-error @below{{expect lhs and rhs shapes to match at non-concat position 1}}
        %result = cuda_tile.cat %arg0, %arg1 dim = 0 : !cuda_tile.tile<16x8xf32>, !cuda_tile.tile<16x16xf32> -> !cuda_tile.tile<32x8xf32>
    }
}

// -----

// Test CatOp::verify - result concat dimension size mismatch

cuda_tile.module @cat_result_size_mismatch {
    cuda_tile.testing$func @func() {
        %arg0 = "materialize_tensor"() : () -> !cuda_tile.tile<16x16xf32>
        %arg1 = "materialize_tensor"() : () -> !cuda_tile.tile<16x16xf32>
        // expected-error @below{{invalid concat at position 1}}
        %result = cuda_tile.cat %arg0, %arg1 dim = 1 : !cuda_tile.tile<16x16xf32>, !cuda_tile.tile<16x16xf32> -> !cuda_tile.tile<16x16xf32>
    }
}

// -----

// ****************** Integer Out of Range Tests ******************

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i1: -2> : !cuda_tile.tile<i1>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i1: 2> : !cuda_tile.tile<i1>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i8: -129> : !cuda_tile.tile<i8>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i8: 256> : !cuda_tile.tile<i8>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i16: -32769> : !cuda_tile.tile<i16>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i16: 65536> : !cuda_tile.tile<i16>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i32: -2147483649> : !cuda_tile.tile<i32>

// -----

// expected-error @below{{integer constant out of range for type}}
%0 = cuda_tile.constant <i1: [[0, 1], [1, 2]]> : !cuda_tile.tile<2x2xi1>

// -----

// ****************** Nested Array Shape Mismatch Tests ******************

// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <i32: [[1, 2], [3]]> : !cuda_tile.tile<2x2xi32>

// -----

// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <i32: [[1, 2, 4, 8], [4, 5, 6]]> : !cuda_tile.tile<2x4xi32>

// -----

// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <f32: [[1.0, 2.0], [3.0, 4.0, 5.0, 6.0]]> : !cuda_tile.tile<2x2xf32>

// -----

// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <i32: [[[1, 2], [3, 4]], [[5, 6], [7]]]> : !cuda_tile.tile<2x2x2xi32>

// -----

// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <i32: [[[1, 2]], [[3]]]> : !cuda_tile.tile<2x1x2xi32>

// -----

// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <f32: [[[1.0, 2.0, 3.0, 4.0]], [[4.0, 5.0]]]> : !cuda_tile.tile<2x1x4xf32>

// -----

// ****************** Boolean Values for Non-i1 Types ******************

// expected-error @below{{expected integer value}}
%0 = cuda_tile.constant <i32: true> : !cuda_tile.tile<i32>

// -----

// expected-error @below{{expected integer value}}
%0 = cuda_tile.constant <i16: false> : !cuda_tile.tile<i16>

// -----

// expected-error @below{{expected integer value}}
%0 = cuda_tile.constant <i64: [true, false, true, false, true, false, true, false]> : !cuda_tile.tile<8xi64>

// -----

// ****************** Float Parsing Failures ******************

// expected-error @below{{unexpected decimal integer literal for a floating point value}}
// expected-note @below{{add a trailing dot to make the literal a float}}
%0 = cuda_tile.constant <f32: 0xGGGG> : !cuda_tile.tile<f32>

// -----

// expected-error @below{{expected floating point literal}}
%0 = cuda_tile.constant <f16: invalid_float> : !cuda_tile.tile<f16>

// -----

// expected-error @below{{unexpected decimal integer literal for a floating point value}}
// expected-note @below{{add a trailing dot to make the literal a float}}
%0 = cuda_tile.constant <bf16: 0x> : !cuda_tile.tile<bf16>

// -----

// expected-error @below{{expected floating point literal}}
%0 = cuda_tile.constant <f64: [1.0, 2.0, invalid, 4.0, 5.0, 6.0, 7.0, 8.0]> : !cuda_tile.tile<8xf64>

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Dense element parse error paths

// Shape mismatch: array literal for scalar tile
// expected-error @below{{inferred shape of elements literal}}
%0 = cuda_tile.constant <i32: [1, 2]> : !cuda_tile.tile<i32>

// -----

// Shape mismatch: 1D array for 2D tile
// expected-error @below{{inferred shape of elements literal}}
%0 = cuda_tile.constant <i32: [1, 2]> : !cuda_tile.tile<2x2xi32>

// -----

// Element type mismatch: prefix f32 but tile i32
// expected-error @below{{mismatch between the element type}}
%0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<i32>

// -----

// Inconsistent rank in nested array
// expected-error @below{{tensor literal is invalid; ranks are not consistent between elements}}
%0 = cuda_tile.constant <i32: [[1, 2], [3]]> : !cuda_tile.tile<2x2xi32>

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// ExtractOp::verify warning branches for constant index bounds checking

// ExtractOp: constant index out-of-bounds
cuda_tile.module @extract_oob_index {
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<8xf32>
    %c4 = cuda_tile.constant <i32: 4> : !cuda_tile.tile<i32>
    // expected-warning @below{{extract index 0 value 4 may be out of bounds (max valid index: 1)}}
    %0 = cuda_tile.extract %src[%c4] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<4xf32>
  }
}

// -----

// ExtractOp: negative constant index
cuda_tile.module @extract_negative_index {
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<8xf32>
    %cn1 = cuda_tile.constant <i32: -1> : !cuda_tile.tile<i32>
    // expected-warning @below{{extract index 0 has negative value -1, which may cause undefined behavior}}
    %0 = cuda_tile.extract %src[%cn1] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<4xf32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers IfOp::verify() - result type is a tile view
// Covers IfOp::verify() - missing else with results
// Covers IfOp::verify() - then region empty check
// Covers IfOp::verify() - yield type mismatch checks

// IfOp: has non-empty return type but no else branch
cuda_tile.module @if_no_else_with_result {
  testing$func @func(%cond: !cuda_tile.tile<i1>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{has non-empty return type, must define else branch}}
    %r = cuda_tile.if %cond -> (tile<i32>) {
      cuda_tile.yield %c0 : tile<i32>
    }
  }
}

// -----

// IfOp: yield type mismatch in then branch
cuda_tile.module @if_yield_type_mismatch {
  testing$func @func(%cond: !cuda_tile.tile<i1>) {
    %ci = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %cf = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>
    // expected-error @below{{type does not match yield type}}
    %r = cuda_tile.if %cond -> (tile<i32>) {
      cuda_tile.yield %cf : tile<f32>
    } else {
      cuda_tile.yield %ci : tile<i32>
    }
  }
}

// -----

// IfOp: has return type but then branch does not yield
cuda_tile.module @if_then_no_yield {
  testing$func @func(%cond: !cuda_tile.tile<i1>) {
    %ci = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{has return type}}
    %r = cuda_tile.if %cond -> (tile<i32>) {
      cuda_tile.yield
    } else {
      cuda_tile.yield %ci : tile<i32>
    }
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// MmaFOp::verify and verifyMmaShapes error branches

// MmaFOp: unsupported input element type
// i8 is not in the allowed mmaf input types
cuda_tile.module @mma_unsupported_input {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x2xi16>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x2xi16>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<2x2xi32>
    // expected-error @below{{operand #0 must be mmaf operand tile type}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc : !cuda_tile.tile<2x2xi16>, !cuda_tile.tile<2x2xi16>, !cuda_tile.tile<2x2xi32>
  }
}

// -----

// MmaFOp: unsupported acc type for f16 input
cuda_tile.module @mma_bad_acc_type {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x2xf16>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x2xf16>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<2x2xf64>
    // expected-error @below{{unsupported combination of element types}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc : !cuda_tile.tile<2x2xf16>, !cuda_tile.tile<2x2xf16>, !cuda_tile.tile<2x2xf64>
  }
}

// -----

// verifyMmaShapes: 1D tiles not allowed
cuda_tile.module @mma_1d_tiles {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<4xf32>
    // expected-error @below{{operands must be 2D or 3D tiles}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc : !cuda_tile.tile<4xf32>, !cuda_tile.tile<4xf32>, !cuda_tile.tile<4xf32>
  }
}

// -----

// verifyMmaShapes: 3D batch dim mismatch lhs/rhs
cuda_tile.module @mma_batch_dim_mismatch_lhs_rhs {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x4x4xf32>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<4x4x4xf32>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<2x4x4xf32>
    // expected-error @below{{shape error: dim 0 of lhs}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc : !cuda_tile.tile<2x4x4xf32>, !cuda_tile.tile<4x4x4xf32>, !cuda_tile.tile<2x4x4xf32>
  }
}

// -----

// verifyMmaShapes: 3D batch dim mismatch lhs/acc
cuda_tile.module @mma_batch_dim_mismatch_lhs_acc {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x4x4xf32>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x4x4xf32>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<4x4x4xf32>
    // expected-error @below{{shape error: dim 0 of lhs}}
    %0 = cuda_tile.mmaf %lhs, %rhs, %acc : !cuda_tile.tile<2x4x4xf32>, !cuda_tile.tile<2x4x4xf32>, !cuda_tile.tile<4x4x4xf32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MmaFScaledOp::verify() - batched dim mismatch for rhs scale
// Covers MmaFScaledOp::verify() - scale element type mismatch (f8E8M0FNU with non-f8/f4 input)
// Covers MmaFScaledOp::verify() - K dimension reduction factor mismatch
// Covers MmaFScaledOp::verify() - f8 element type requires vecSize=32

// Batched: rhs batch dim mismatch with rhs_scale batch dim
cuda_tile.module @mmaf_scaled_batch_rhs_mismatch {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x128x128xf8E5M2>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<2x128x128xf8E5M2>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<2x128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<2x128x4xf8E8M0FNU>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<4x4x128xf8E8M0FNU>
    // expected-error @below{{shape error: dim 0 of rhs}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<2x128x128xf8E5M2>, !cuda_tile.tile<2x128x128xf8E5M2>, !cuda_tile.tile<2x128x128xf32>, !cuda_tile.tile<2x128x4xf8E8M0FNU>, !cuda_tile.tile<4x4x128xf8E8M0FNU>
  }
}

// -----

// Scale type f8E8M0FNU requires lhs to be f8E4M3FN, f8E5M2, or f4E2M1FN.
// Using i32 as lhs should trigger the error.
cuda_tile.module @mmaf_scaled_bad_scale_input_combo {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<128x4xf8E4M3FN>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<4x128xf8E4M3FN>
    // expected-error @below{{unsupported combination of element types. Scale type 'f8E4M3FN' expects lhs and rhs element types to be 'f4E2M1FN'}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E4M3FN>, !cuda_tile.tile<4x128xf8E4M3FN>
  }
}

// -----

// f8 element type requires block scale factor (vecSize) of 32
cuda_tile.module @mmaf_scaled_f8_wrong_vecsize {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<128x8xf8E8M0FNU>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<8x128xf8E8M0FNU>
    // expected-error @below{{f8 element type requires block scale factor}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E8M0FNU>
  }
}

// -----

// K dimension of lhs_scale and rhs_scale mismatch
cuda_tile.module @mmaf_scaled_k_dim_mismatch {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E4M3FN>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E4M3FN>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<128x4xf8E8M0FNU>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<8x128xf8E8M0FNU>
    // expected-error @below{{shape error: dim 1 of lhs_scale}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E4M3FN>, !cuda_tile.tile<128x128xf8E4M3FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E8M0FNU>
  }
}

// -----

// Row dim of lhs and lhs_scale mismatch (M dimension)
cuda_tile.module @mmaf_scaled_m_dim_mismatch {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<64x4xf8E8M0FNU>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<4x128xf8E8M0FNU>
    // expected-error @below{{shape error: dim 0 of lhs}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<64x4xf8E8M0FNU>, !cuda_tile.tile<4x128xf8E8M0FNU>
  }
}

// -----

// Col dim of rhs and rhs_scale mismatch (N dimension)
cuda_tile.module @mmaf_scaled_n_dim_mismatch {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf8E5M2>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<128x4xf8E8M0FNU>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<4x64xf8E8M0FNU>
    // expected-error @below{{shape error: dim 1 of rhs}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf8E5M2>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x4xf8E8M0FNU>, !cuda_tile.tile<4x64xf8E8M0FNU>
  }
}

// -----

// f4E2M1FN element type with f8E8M0FNU scales requires specific vecSize
cuda_tile.module @mmaf_scaled_f4_wrong_vecsize {
  testing$func @func() {
    %lhs = "materialize_tensor"() : () -> !cuda_tile.tile<128x64xf4E2M1FN>
    %rhs = "materialize_tensor"() : () -> !cuda_tile.tile<64x128xf4E2M1FN>
    %acc = "materialize_tensor"() : () -> !cuda_tile.tile<128x128xf32>
    %lhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<128x8xf8E8M0FNU>
    %rhs_scale = "materialize_tensor"() : () -> !cuda_tile.tile<8x128xf8E8M0FNU>
    // expected-error @below{{f4E2M1FN element type with f8E8M0FNU scales requires block scale factor}}
    %0 = cuda_tile.mmaf_scaled %lhs, %rhs, %acc, %lhs_scale, %rhs_scale : !cuda_tile.tile<128x64xf4E2M1FN>, !cuda_tile.tile<64x128xf4E2M1FN>, !cuda_tile.tile<128x128xf32>, !cuda_tile.tile<128x8xf8E8M0FNU>, !cuda_tile.tile<8x128xf8E8M0FNU>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers verifyPackUnpackTypes - same element width error
// Covers verifyPackUnpackTypes - size mismatch error

// PackOp: source and result have same element type width (both i8)
cuda_tile.module @pack_same_width {
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<8xi8>
    // expected-error @below{{expects source and result to have different element type widths}}
    %0 = cuda_tile.pack %src : !cuda_tile.tile<8xi8> -> !cuda_tile.tile<8xi8>
  }
}

// -----

// UnpackOp: size mismatch (16*8=128 bits source vs 2*32=64 bits result)
cuda_tile.module @unpack_size_mismatch {
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<16xi8>
    // expected-error @below{{expects source and result to have the same size in bytes}}
    %0 = cuda_tile.unpack %src : !cuda_tile.tile<16xi8> -> !cuda_tile.tile<2xi32>
  }
}

// -----

// UnpackOp: valid roundtrip (16*8=128 bits = 4*32=128 bits)
cuda_tile.module @unpack_valid {
  // CHECK: unpack
  testing$func @func() {
    %src = "materialize_tensor"() : () -> !cuda_tile.tile<16xi8>
    %0 = cuda_tile.unpack %src : !cuda_tile.tile<16xi8> -> !cuda_tile.tile<4xi32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// ReturnOp::verify error branches

// ReturnOp in entry with wrong operand count
cuda_tile.module @return_entry_with_operands {
  entry @func() {
    %a = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
    // expected-error @below{{has 1 operands, but enclosing function (@func) returns 0}}
    cuda_tile.return %a : tile<i32>
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers verifyAggregateOp - dim out of bounds
// Covers verifyAggregateOp - identity type mismatch with operand element type

// ScanOp: dim out of bounds
cuda_tile.module @scan_dim_oob {
  testing$func @func(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{dimension (1) is out of bound [0, 1)}}
    %0 = cuda_tile.scan %arg0 dim=1 reverse=false identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
      (%iter_arg : !cuda_tile.tile<f32>, %prev_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield %iter_arg : !cuda_tile.tile<f32>
    }
  }
}

// -----

// ScanOp: identity type mismatch with operand element type
cuda_tile.module @scan_identity_type_mismatch {
  testing$func @func(%arg0: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same type for operand at index: 0 and identity at index: 0 but got: 'i32' and 'f32'}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8xi32>
      (%iter_arg : !cuda_tile.tile<i32>, %prev_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %iter_arg : !cuda_tile.tile<i32>
    }
  }
}

// -----

// ReduceOp: dim out of bounds (high)
cuda_tile.module @reduce_dim_oob {
  testing$func @func(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{dimension (2) is out of bound [0, 1)}}
    %0 = cuda_tile.reduce %arg0 dim=2 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
      (%iter_arg : !cuda_tile.tile<f32>, %prev_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield %iter_arg : !cuda_tile.tile<f32>
    }
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/CudaTile.cpp
// Covers MakePartitionViewOp::parse() - non-partition_view type error
// Covers MakeStridedViewOp::parse() - non-strided_view type error
// Covers MakeGatherScatterViewOp::parse() - non-gather_scatter_view type error

// MakePartitionViewOp: wrong type (not a partition_view)
cuda_tile.module @partition_view_wrong_type {
  testing$func @func(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128, 64], strides = [64, 1] : !cuda_tile.tensor_view<128x64xf32, strides = [64, 1]>
    // expected-error @below{{expected 'partition_view' type, but got}}
    %pv = cuda_tile.make_partition_view %tv : tensor_view<128x64xf32, strides = [64, 1]>
  }
}

// -----

// MakeStridedViewOp: wrong type (not a strided_view)
cuda_tile.module @strided_view_wrong_type {
  testing$func @func(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128, 64], strides = [64, 1] : !cuda_tile.tensor_view<128x64xf32, strides = [64, 1]>
    // expected-error @below{{expected 'strided_view' type, but got}}
    %sv = cuda_tile.make_strided_view %tv : tensor_view<128x64xf32, strides = [64, 1]>
  }
}

// -----

// MakeGatherScatterViewOp: wrong type (not a gather_scatter_view)
cuda_tile.module @gs_view_wrong_type {
  testing$func @func(%p : !cuda_tile.tile<ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %p, shape = [128, 64], strides = [64, 1] : !cuda_tile.tensor_view<128x64xf32, strides = [64, 1]>
    // expected-error @below{{expected 'gather_scatter_view' type, but got}}
    %gs = cuda_tile.make_gather_scatter_view %tv : tensor_view<128x64xf32, strides = [64, 1]>
  }
}

