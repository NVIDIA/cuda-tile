// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/Types.cpp
// Covers GatherScatterViewType::parse failure where the comma between
// the tensor_view type and the 'sparse_dim' keyword is missing.
// This exercises the parseComma() arm of the
// 'parseComma() || parseKeyword("sparse_dim") || parseEqual()' chain
// that follows the tensor_view in the type syntax.
// expected-error @below {{expected ','}}
"use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(64x64), !cuda_tile.tensor_view<8192x128xf32, strides=[128,1]> sparse_dim=0>

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/Types.cpp
// Covers TensorViewType::parse error paths involving the optional
// ptr_attr, complementing the existing coverage_test that only exercises
// the no-attribute-after-comma branch.

// Covers TensorViewType::parse path where parseOptionalAttribute sees an
// attribute prefix '#' but the inner attribute body fails to parse,
// triggering the failed(optResult.value()) side of the `||` at the
// post-strides ptr_attr position (the non-0-D branch).
// expected-error @below {{expected '<'}}
// expected-error @below {{expected ptr_attr attribute}}
"use_type"() : () -> !cuda_tile.tensor_view<2xf32, strides=[1], #cuda_tile.ptr_attr>

// -----

// Covers TensorViewType::parse 0-D branch where parseOptionalAttribute
// sees a '#' prefix but the inner attribute body fails to parse —
// exercises the failed(optResult.value()) branch in the 0-D ptr_attr
// position (the if-block right after the optResult.has_value() check).
// expected-error @below {{expected '<'}}
"use_type"() : () -> !cuda_tile.tensor_view<f32, #cuda_tile.ptr_attr>

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/Types.cpp
// Covers GatherScatterViewType::verifyIndices error paths that the existing
// view_invalid.mlir tests do not exercise. These error paths are only
// reachable when the index operands violate per-view constraints, which the
// tablegen operand constraints (CudaTile_IntTileType) normally accept; we use
// the generic op form to provide the offending index types directly.

// Covers GatherScatterViewType::verifyIndices "expected index <i> to be a
// scalar tile for GatherScatterView" path: a non-sparse_dim index slot
// receives a non-scalar tile.
cuda_tile.module @kernels {
  testing$func @gather_scatter_view_non_scalar_other_dim(%v: !cuda_tile.gather_scatter_view<tile=(8x8), !cuda_tile.tensor_view<128x128xf32, strides=[128, 1]>, sparse_dim=0>, %gather_idx: !cuda_tile.tile<8xi32>, %other: !cuda_tile.tile<2xi32>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{expected index 1 to be a scalar tile for GatherScatterView}}
    %tile_out, %tok = "cuda_tile.load_view_tko"(%v, %gather_idx, %other, %t) <{memory_ordering_semantics = 0 : i32, inbounds = array<i1: false, false>, operandSegmentSizes = array<i32: 1, 2, 1>}> : (!cuda_tile.gather_scatter_view<tile=(8x8), !cuda_tile.tensor_view<128x128xf32, strides=[128, 1]>, sparse_dim=0>, !cuda_tile.tile<8xi32>, !cuda_tile.tile<2xi32>, !cuda_tile.token) -> (!cuda_tile.tile<8x8xf32>, !cuda_tile.token)
    return
  }
}

// -----

// Covers GatherScatterViewType::verifyIndices "expected scalar index <i> to
// have the same type as other scalar indices" path: with sparse_dim=0, the
// other two indices must agree on scalar element type.
cuda_tile.module @kernels {
  testing$func @gather_scatter_view_scalar_type_mismatch(%v: !cuda_tile.gather_scatter_view<tile=(8x8x8), !cuda_tile.tensor_view<128x128x128xf32, strides=[16384,128,1]>, sparse_dim=0>, %gather_idx: !cuda_tile.tile<8xi32>, %s1: !cuda_tile.tile<i32>, %s2: !cuda_tile.tile<i64>) {
    %t = make_token : !cuda_tile.token
    // expected-error @below{{expected scalar index 2 to have the same type as other scalar indices}}
    %tile_out, %tok = "cuda_tile.load_view_tko"(%v, %gather_idx, %s1, %s2, %t) <{memory_ordering_semantics = 0 : i32, inbounds = array<i1: false, false, false>, operandSegmentSizes = array<i32: 1, 3, 1>}> : (!cuda_tile.gather_scatter_view<tile=(8x8x8), !cuda_tile.tensor_view<128x128x128xf32, strides=[16384,128,1]>, sparse_dim=0>, !cuda_tile.tile<8xi32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i64>, !cuda_tile.token) -> (!cuda_tile.tile<8x8x8xf32>, !cuda_tile.token)
    return
  }
}

// -----

// Coverage target: tile_ir/lib/Dialect/CudaTile/IR/Types.cpp
// Covers additional parse-failure error paths for the cuda_tile view types
// that the existing coverage tests do not yet exercise.

// Covers TileType verification failure for a negative dimension.
// expected-error @below{{all dimensions must be positive constants, got -1}}
"use_type"() : () -> !cuda_tile.tile<-1xf32>

// -----

// Covers TensorViewType::parse failure when the element-type token is not a
// known type, exercising the parseCudaTileType() arm of the
// parseDimensionList / parseCudaTileType || chain inside TensorViewType::parse.
// expected-error @below{{unknown type: abc}}
"use_type"() : () -> !cuda_tile.tensor_view<abc xf32, strides=[1]>

// -----

// Covers GatherScatterViewType::parse failure when the leading 'tile' keyword
// is missing. Exercises the parseKeyword("tile") arm of the
// (parseKeyword("tile") || parseEqual()) check near the start of the parser.
// expected-error @below{{expected 'tile'}}
"use_type"() : () -> !cuda_tile.gather_scatter_view<size=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>, sparse_dim=0>

// -----

// Covers parseViewTileShape failure when the dimension list inside 'tile=(...)'
// is malformed (trailing 'x'). Hits the parseDimensionList() arm shared by
// PartitionViewType::parse and StridedViewType::parse.
// expected-error @below{{invalid dimension}}
"use_type"() : () -> !cuda_tile.partition_view<tile=(8x), !cuda_tile.tensor_view<128xf32, strides=[1]>>

// -----

// Same dimension-list failure as above but inside StridedViewType::parse.
// expected-error @below{{invalid dimension}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(8x), traversal_strides=[1], !cuda_tile.tensor_view<128xf32, strides=[1]>>

// -----

// Covers TileType::parse failure when the closing '>' is missing — exercises
// the parseGreater() arm of the parser's terminating || chain. A stray token
// after the element type causes the parser to bail at parseGreater().
// expected-error @below{{expected '>'}}
"use_type"() : () -> !cuda_tile.tile<8xf32 trailing>

// -----

// Covers TensorViewType::parse failure when the closing '>' is missing on
// a non-0-D tensor_view (stray token after the strides list). Exercises
// the parseGreater() failure path at the end of TensorViewType::parse.
// expected-error @below{{expected '>'}}
"use_type"() : () -> !cuda_tile.tensor_view<2xf32, strides=[1] trailing>

// -----

// Covers TensorViewType::parse 0-D failure when the closing '>' is missing
// (stray token after the element type with no comma). Exercises the
// parseGreater() failure path in the 0-D no-strides branch.
// expected-error @below{{expected '>'}}
"use_type"() : () -> !cuda_tile.tensor_view<f32 trailing>

// -----

// Target: Types.cpp — GatherScatterViewType parse/verify and verifyIndices (via load_view_tko).

// -----

// expected-error @below{{expected 'tensor_view' type, but got '!cuda_tile.tile<f32>'}}
%gs0 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tile<f32>, sparse_dim=0>

// -----

// expected-error @below{{expected tensor_view rank and tile rank to match, got tensor_view of rank 2 and tiles of rank 1}}
%gs1 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>, sparse_dim=0>

// -----

// expected-error @below{{tile shape dimensions must have power of two length but got [3, 4]}}
%gs2 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(3x4), !cuda_tile.tensor_view<12x16xf32, strides=[16,1]>, sparse_dim=0>

// -----

// expected-error @below{{gather dimension is outside of tensor view dimensions, expected strictly less than 2, got 2}}
%gs3 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4x4), !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>, sparse_dim=2>

// -----

// expected-error @below{{padding_value nan can only be used with floating point element types, got 'i32'}}
%gs4 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4x4), padding_value = nan, !cuda_tile.tensor_view<16x16xi32, strides=[16,1]>, sparse_dim=0>

// -----

cuda_tile.module @m {
  cuda_tile.testing$func @gs_gather_rank_not_1(
      %view: !cuda_tile.gather_scatter_view<tile=(8x8), !cuda_tile.tensor_view<64x64xf32, strides=[64,1]>, sparse_dim=0>) {
    %g = cuda_tile.constant <i32: 0> : !cuda_tile.tile<8x8xi32>
    %c = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected index at sparse_dim position 0 to be a 1D tensor for GatherScatterView, got rank 2}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %g, %c) <{inbounds = array<i1: false, false>, memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 2, 0>}> : (!cuda_tile.gather_scatter_view<tile=(8x8), tensor_view<64x64xf32, strides=[64,1]>, sparse_dim=0>, !cuda_tile.tile<8x8xi32>, !cuda_tile.tile<i32>) -> (!cuda_tile.tile<8x8xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @m {
  cuda_tile.testing$func @gs_gather_size_mismatch(
      %view: !cuda_tile.gather_scatter_view<tile=(8x8), !cuda_tile.tensor_view<64x64xf32, strides=[64,1]>, sparse_dim=0>) {
    %g = cuda_tile.constant <i32: 0> : !cuda_tile.tile<16xi32>
    %c = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected gather/scatter index size (16) to match tile shape at gather/scatter dimension 0 (8)}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %g, %c) <{inbounds = array<i1: false, false>, memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 2, 0>}> : (!cuda_tile.gather_scatter_view<tile=(8x8), tensor_view<64x64xf32, strides=[64,1]>, sparse_dim=0>, !cuda_tile.tile<16xi32>, !cuda_tile.tile<i32>) -> (!cuda_tile.tile<8x8xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @m {
  cuda_tile.testing$func @gs_scalar_index_not_scalar(
      %view: !cuda_tile.gather_scatter_view<tile=(8x8), !cuda_tile.tensor_view<64x64xf32, strides=[64,1]>, sparse_dim=0>) {
    %g = cuda_tile.constant <i32: 0> : !cuda_tile.tile<8xi32>
    %c = cuda_tile.constant <i32: 0> : !cuda_tile.tile<1xi32>
    // expected-error @below{{expected index 1 to be a scalar tile for GatherScatterView}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %g, %c) <{inbounds = array<i1: false, false>, memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 2, 0>}> : (!cuda_tile.gather_scatter_view<tile=(8x8), tensor_view<64x64xf32, strides=[64,1]>, sparse_dim=0>, !cuda_tile.tile<8xi32>, !cuda_tile.tile<1xi32>) -> (!cuda_tile.tile<8x8xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @m {
  cuda_tile.testing$func @gs_sparse_dim1_wrong_gather_rank(
      %view: !cuda_tile.gather_scatter_view<tile=(32x64), !cuda_tile.tensor_view<512x1024xf32, strides=[1024,1]>, sparse_dim=1>) {
    %row = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %bad = cuda_tile.constant <i32: 0> : !cuda_tile.tile<64x64xi32>
    // expected-error @below{{expected index at sparse_dim position 1 to be a 1D tensor for GatherScatterView, got rank 2}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %row, %bad) <{inbounds = array<i1: false, false>, memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 2, 0>}> : (!cuda_tile.gather_scatter_view<tile=(32x64), tensor_view<512x1024xf32, strides=[1024,1]>, sparse_dim=1>, !cuda_tile.tile<i32>, !cuda_tile.tile<64x64xi32>) -> (!cuda_tile.tile<32x64xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @m {
  cuda_tile.testing$func @gs_scalar_type_mismatch_3d(
      %view: !cuda_tile.gather_scatter_view<tile=(4x4x4), !cuda_tile.tensor_view<16x16x16xf32, strides=[256,16,1]>, sparse_dim=0>) {
    %g = cuda_tile.constant <i32: 0> : !cuda_tile.tile<4xi32>
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
    // expected-error @below{{'cuda_tile.load_view_tko' op expected scalar index 2 to have the same type as other scalar indices ('!cuda_tile.tile<i32>'), got '!cuda_tile.tile<i64>'}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %g, %a, %b) <{inbounds = array<i1: false, false, false>, memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 3, 0>}> : (!cuda_tile.gather_scatter_view<tile=(4x4x4), tensor_view<16x16x16xf32, strides=[256,16,1]>, sparse_dim=0>, !cuda_tile.tile<4xi32>, !cuda_tile.tile<i32>, !cuda_tile.tile<i64>) -> (!cuda_tile.tile<4x4x4xf32>, !cuda_tile.token)
  }
}

// -----

// Target: Types.cpp — GatherScatterViewType::parse error paths.

// -----

// GatherScatterViewType::parse — 'tile' keyword OK, but parseEqual fails.
// expected-error @below{{expected '='}}
%gs0 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile(4), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// GatherScatterViewType::parse — parseComma after tile paren fails.
// expected-error @below{{expected ','}}
%gs2 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4) !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// GatherScatterViewType::parse — padding present but comma missing after it.
// expected-error @below{{expected ','}}
%gs3 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), padding_value = zero !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// GatherScatterViewType::parse — parseCudaTileType fails.
// expected-error @below{{unknown type: TypesCovBadGsElem}}
%gs4 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), TypesCovBadGsElem, sparse_dim=0>

// -----

// GatherScatterViewType::parse — 'sparse_dim' keyword OK, but parseEqual fails.
// expected-error @below{{expected '='}}
%gs6 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim 0>

// -----

// GatherScatterViewType::parse — parseInteger(sparse_dim) fails.
// expected-error @below{{expected integer value}}
%gs7 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=x>

// -----

// GatherScatterViewType::parse — parseLParen fails (bracket instead of paren).
// expected-error @below{{expected '('}}
%gs9 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=[4], !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// GatherScatterViewType::parse — parseKeyword("sparse_dim") fails.
// expected-error @below{{expected 'sparse_dim'}}
%gs10 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tensor_view<16xf32, strides=[1]>, gather_dim=0>

// -----

// Target: Types.cpp — parse edge cases: parseLess/parseGreater, 0-D tensor_view, ptrAttr, paddingValue keyword.

// -----

// Covers TensorViewType::parse — 0-D tensor_view with comma followed by "strides" keyword.
// expected-error @below{{strides must not be provided for 0-d tiles}}
%tv0 = "use_type"() : () -> !cuda_tile.tensor_view<f32, strides=[1]>

// -----

// Covers TensorViewType::parse — 0-D tensor_view with comma then unexpected token (not strides, not ptrAttr).
// expected-error @below{{unexpected token after element type in 0-d tensor_view}}
%tv1 = "use_type"() : () -> !cuda_tile.tensor_view<f32, badtoken>

// -----

// Covers TileType::parse — parseGreater fails after element type parsed successfully.
// expected-error @below{{expected '>'}}
%tl0 = "use_type"() : () -> !cuda_tile.tile<4xf32,extra>

// -----

// Covers TensorViewType::parse — parseGreater fails after strides (no trailing comma).
// expected-error @below{{expected '>'}}
%tv2 = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[1] extra>

// -----

// Covers PartitionViewType::parse — parseLess fails (missing '<').
// expected-error @below{{expected '<'}}
%pv_less = "use_type"() : () -> !cuda_tile.partition_view

// -----

// Covers PartitionViewType::parse — parseGreater fails after tensor_view type.
// expected-error @below{{expected '>'}}
%pv0 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf32, strides=[1]> extra>

// -----

// Covers StridedViewType::parse — parseLess fails (missing '<').
// expected-error @below{{expected '<'}}
%sv_less = "use_type"() : () -> !cuda_tile.strided_view

// -----

// Covers StridedViewType::parse — parseGreater fails after tensor_view type.
// expected-error @below{{expected '>'}}
%sv0 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], !cuda_tile.tensor_view<16xf32, strides=[1]> extra>

// -----

// Covers GatherScatterViewType::parse — parseLess fails (missing '<').
// expected-error @below{{expected '<'}}
%gs_less = "use_type"() : () -> !cuda_tile.gather_scatter_view

// -----

// Covers GatherScatterViewType::parse — parseGreater fails after sparse_dim.
// expected-error @below{{expected '>'}}
%gs0 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0 extra>

// -----

// Covers parseOptionalPaddingValue — parseKeyword fails after "padding_value =" (number instead of keyword).
// expected-error @below{{expected valid keyword}}
%pv1 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value = 123, !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// Covers GatherScatterViewType::parse — dynamic dimension in tile shape (not allowed).
// expected-error @below{{expected static shape}}
%gs1 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(?), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// Covers TensorViewType::parse pointer attribute error: comma after strides,
// then an invalid attribute.
// expected-error @below{{expected ptr_attr attribute}}
%tv3 = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[1], notattr>

// -----

// Target: Types.cpp — parse error paths in parseOptionalPaddingValue, TensorViewType, PartitionViewType.

// -----

// parseOptionalPaddingValue — "padding_value" keyword matched but "=" is missing.
// expected-error @below{{expected '='}}
%pv0 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value nan, !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// PartitionViewType::parse — padding present but comma missing after it.
// expected-error @below{{expected ','}}
%pv1 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value = zero !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// PartitionViewType::parse — parseCudaTileType fails.
// expected-error @below{{unknown type: TypesCovBadPvElem}}
%pv3 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), TypesCovBadPvElem>

// -----

// parseOptionalViewDimMap — dim_map element is not an integer.
// expected-error @below{{expected integer value}}
%pv4 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf32, strides=[1]>, dim_map=[x]>

// -----

// parseStrideArray — non-integer, non-? value in strides (branch for no hint of integer).
// expected-error @below{{expected either 64-bit integer or question mark}}
%tv1 = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[abc]>

// -----

// TensorViewType::parse — parseCudaTileType fails in tensor_view parse.
// expected-error @below{{unknown type: TypesCovBadTvElem}}
%tv2 = "use_type"() : () -> !cuda_tile.tensor_view<4xTypesCovBadTvElem, strides=[1]>

// -----

// TensorViewType::parse — parseStrideArray failure due to keyword instead of strides.
// The parser sees 'strides', '=', then tries to parse the array but finds invalid content.
// expected-error @below{{expected either 64-bit integer or question mark}}
%tv3 = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[bad_stride]>

// -----

// Target: Types.cpp in CudaTile IR — parseOptionalPaddingValue and related type parse paths.

// -----

// expected-error @below{{invalid padding_value attribute specification. Got "bogus" but expect one of: zero, neg_zero, nan, pos_inf, neg_inf}}
%0 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value = bogus, !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// expected-error @below{{invalid padding_value attribute specification. Got "nope" but expect one of: zero, neg_zero, nan, pos_inf, neg_inf}}
%1 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), padding_value = nope, !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// Target: Types.cpp — verifyPartitionViewLike padding branches on partition/strided; ForOp parseCudaTileType comma-list (first type fails).

// -----

// PartitionViewType::verify → verifyPartitionViewLike (neg_zero + non-float element)
// expected-error @below{{padding_value neg_zero can only be used with floating point element types, got 'i32'}}
%p0 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value = neg_zero, !cuda_tile.tensor_view<16xi32, strides=[1]>>

// -----

// Same verifier, nan branch on partition_view (not only strided/gather).
// expected-error @below{{padding_value nan can only be used with floating point element types, got 'i32'}}
%p1 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value = nan, !cuda_tile.tensor_view<16xi32, strides=[1]>>

// -----

// StridedViewType::verify → verifyPartitionViewLike (neg_zero branch on strided path)
// expected-error @below{{padding_value neg_zero can only be used with floating point element types, got 'i32'}}
%s0 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], padding_value = neg_zero, !cuda_tile.tensor_view<16xi32, strides=[1]>>

// -----

// GatherScatterViewType::verify padding (neg_zero + integer element)
// expected-error @below{{padding_value neg_zero can only be used with floating point element types, got 'i32'}}
%g0 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), padding_value = neg_zero, !cuda_tile.tensor_view<16xi32, strides=[1]>, sparse_dim=0>

// -----

// ForOp::parse — parseCudaTileType(parser, result.types): first of two comma-separated types fails (Types.cpp comma-list lambda, first parseCudaTileType failure).
cuda_tile.module @cov_for_iter_comma_first_bad {
  cuda_tile.testing$func @f() {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    %c2 = cuda_tile.constant <i32: 2> : !cuda_tile.tile<i32>
    %cf = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>
    // expected-error @below{{unknown type: TypesCovForCommaFirstBad}}
    %r:2 = for %iv in (%c0 to %c2, step %c1) : tile<i32> iter_values(%x = %cf, %y = %c0) -> (TypesCovForCommaFirstBad, tile<i32>) {
      continue %x, %y : tile<f32>, tile<i32>
    }
  }
}

// -----

// Target: Types.cpp — PartitionViewType::parse error paths and parseViewTileShape branches.

// -----

// parseViewTileShape — 'tile' keyword parsed but '=' missing.
// expected-error @below{{expected '='}}
%pv0 = "use_type"() : () -> !cuda_tile.partition_view<tile(2), !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// PartitionViewType::parse — parseComma after tileShape fails.
// expected-error @below{{expected ','}}
%pv1 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2) !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// parseViewTileShape — parseLParen fails (bracket instead of paren).
// expected-error @below{{expected '('}}
%pv2 = "use_type"() : () -> !cuda_tile.partition_view<tile=[2], !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// parseOptionalViewDimMap — parseCommaSeparatedList with empty dim_map brackets passes parse
// but then the verify function catches the empty dim_map (wrong mapping count).
// expected-error @below{{expected dim_map to map exactly all 1 dimensions of the tile, got 0 mappings}}
%pv4 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf32, strides=[1]>, dim_map=[]>

// -----

// parseViewTileShape — tile dimension value is 0 (hits the <= 0 check in parseViewTileShape).
// expected-error @below{{tile dimension 0 exceeds i32 limitations (got 0, expected strictly positive and less than or equal to 2147483647)}}
%pv5 = "use_type"() : () -> !cuda_tile.partition_view<tile=(0), !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// parseViewTileShape — parseDimensionList with invalid syntax triggers parseRParen failure.
// expected-error @below{{expected ')'}}
%pv6 = "use_type"() : () -> !cuda_tile.partition_view<tile=(,), !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// Target: Types.cpp — StridedViewType::parse error paths.

// -----

// StridedViewType::parse — parseOptionalPaddingValue failure: "padding_value" with missing "=".
// expected-error @below{{expected '='}}
%sv0 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], padding_value nan, !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// StridedViewType::parse — padding present but comma missing after it.
// expected-error @below{{expected ','}}
%sv1 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], padding_value = zero !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// StridedViewType::parse — parseCudaTileType fails.
// expected-error @below{{unknown type: TypesCovBadSvElem}}
%sv2 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], TypesCovBadSvElem>

// -----

// parseViewTraversalStrides — 'traversal_strides' keyword parse OK, but parseEqual fails.
// expected-error @below{{expected '='}}
%sv4 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides[1], !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// StridedViewType::parse — parseComma after tileShape fails.
// expected-error @below{{expected ','}}
%sv5 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2) traversal_strides=[1], !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// StridedViewType::parse — parseComma after traversal_strides fails.
// expected-error @below{{expected ','}}
%sv6 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1] !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// parseViewTraversalStrides — parseInteger fails for stride element.
// expected-error @below{{expected integer value}}
%sv7 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[abc], !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// Target: Types.cpp — TensorViewType::verify (shape vs stride rank) and parse (strides keyword); StridedViewType::parse (traversal_strides keyword).

// -----

// TensorViewType::verify — empty strides vs non-empty shape.
// expected-error @below{{expected shape and stride to be of same rank but got shape of rank 2 and stride of rank 0}}
%tv0 = "use_type"() : () -> !cuda_tile.tensor_view<2x2xf32, strides=[]>

// -----

// TensorViewType::verify — more stride entries than shape ranks.
// expected-error @below{{expected shape and stride to be of same rank but got shape of rank 2 and stride of rank 3}}
%tv1 = "use_type"() : () -> !cuda_tile.tensor_view<2x2xf32, strides=[1,1,1]>

// -----

// TensorViewType::parse — comma present but keyword is not 'strides'.
// expected-error @below{{expected 'strides'}}
%tv2 = "use_type"() : () -> !cuda_tile.tensor_view<2xf32, not_strides=[1]>

// -----

// parseViewTraversalStrides — keyword must be 'traversal_strides'.
// expected-error @below{{expected 'traversal_strides'}}
%sv0 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), steps=[1], !cuda_tile.tensor_view<16xf32, strides=[1]>>

// -----

// Target: Types.cpp — TensorViewType::verify shape check with dynamic dims (PrintDynamic kDynamic path).

// -----

// TensorViewType::verify — shape with zero dimension triggers error, exercises PrintDynamic with ? and concrete.
// expected-error @below{{dimensions must have strictly positive constant sizes but got [?, 0]}}
%tv0 = "use_type"() : () -> !cuda_tile.tensor_view<?x0xf32, strides=[1,1]>

// -----

// TensorViewType::verify — shape with zero only, exercises PrintDynamic.
// expected-error @below{{dimensions must have strictly positive constant sizes but got [0, 0]}}
%tv1 = "use_type"() : () -> !cuda_tile.tensor_view<0x0xf32, strides=[1,1]>

// -----

// TensorViewType::verify — 3D shape with dynamic + zero, ensures kDynamic prints as ?.
// expected-error @below{{dimensions must have strictly positive constant sizes but got [?, ?, 0]}}
%tv2 = "use_type"() : () -> !cuda_tile.tensor_view<?x?x0xf32, strides=[1,1,1]>

// -----

// Target: Types.cpp — TensorViewType::parse strides keyword edge cases, TileType parse element type fail.

// -----

// TensorViewType::parse — parseEqual after 'strides' keyword fails.
// expected-error @below{{expected '='}}
%tv1 = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides[1]>

// -----

// TileType::parse — parseCudaTileType fails for element type.
// expected-error @below{{unknown type: BadTileElemR2}}
%tl0 = "use_type"() : () -> !cuda_tile.tile<4xBadTileElemR2>

// -----

// TensorViewType::parse — parseDimensionList fails (invalid dim syntax).
// expected-error @below{{expected valid keyword}}
%tv2 = "use_type"() : () -> !cuda_tile.tensor_view<-1xf32, strides=[1]>

// -----

// Target: Types.cpp — parseCudaTileType (unknown mnemonic + comma list), view parse/verify edges.

// -----

// parseCudaTileType via TileType::parse element type.
// expected-error @below{{unknown type: TypesCovUnkTileElem}}
%t0 = "use_type"() : () -> !cuda_tile.tile<2xTypesCovUnkTileElem>

// -----

// parseCudaTileType via TensorViewType::parse element type.
// expected-error @below{{unknown type: TypesCovUnkTvElem}}
%t1 = "use_type"() : () -> !cuda_tile.tensor_view<2xTypesCovUnkTvElem, strides=[1]>

// -----

// PartitionViewType::parse
// expected-error @below{{expected 'tensor_view' type, but got '!cuda_tile.tile<f32>'}}
%t2 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), !cuda_tile.tile<f32>>

// -----

// verifyPartitionViewLike padding float-only (strided path)
// expected-error @below{{padding_value pos_inf can only be used with floating point element types, got 'i32'}}
%t3 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], padding_value = pos_inf, !cuda_tile.tensor_view<16xi32, strides=[1]>>

// -----

// GatherScatterViewType::verify padding float-only (neg_inf branch)
// expected-error @below{{padding_value neg_inf can only be used with floating point element types, got 'i32'}}
%t4 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), padding_value = neg_inf, !cuda_tile.tensor_view<16xi32, strides=[1]>, sparse_dim=0>

// -----

// parseCudaTileType(AsmParser, SmallVectorImpl<Type>) — comma-separated result types (LoopOp::parse).
cuda_tile.module @cov_cuda_tile_type_comma_list {
  entry @e() {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{unknown type: TypesCovCommaListBad2}}
    %lr:2 = loop iter_values(%v = %c0) : tile<i32> -> tile<i32>, TypesCovCommaListBad2 {
      break %c0, %c0 : tile<i32>, tile<i32>
    }
  }
}

// -----

// parseCudaTileType on second function argument (CudaTile.cpp parseSingleArgument → Types.cpp).
cuda_tile.module @cov_cuda_tile_type_func_arg {
  // expected-error @below{{unknown type: TypesCovBadFuncArgTy}}
  testing$func @second_arg(%a: !cuda_tile.tile<i32>, %b: TypesCovBadFuncArgTy) -> !cuda_tile.tile<i32> {
  }
}

// -----

// Target: Types.cpp — parseOptionalViewDimMap (partition/strided) and GatherScatterViewType::parse sparse_dim keyword.

// -----

// parseOptionalViewDimMap — comma after tensor_view but keyword is not 'dim_map'.
// expected-error @below{{expected 'dim_map'}}
%p0 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf32, strides=[1]>, map=[0]>

// -----

// Same helper on StridedViewType::parse.
// expected-error @below{{expected 'dim_map'}}
%s0 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], !cuda_tile.tensor_view<16xf32, strides=[1]>, map=[0,1]>

// -----

// GatherScatterViewType::parse — after tensor_view, expect 'sparse_dim' before '='.
// expected-error @below{{expected 'sparse_dim'}}
%g0 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse=0>

// -----

// parseViewTileShape — partition_view must start with 'tile='.
// expected-error @below{{expected 'tile'}}
%p1 = "use_type"() : () -> !cuda_tile.partition_view<dims=(2), !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// parseOptionalViewDimMap — keyword 'dim_map' present but missing '=' before list.
// expected-error @below{{expected '='}}
%p2 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), !cuda_tile.tensor_view<4xf32, strides=[1]>, dim_map[0]>

// -----

// Target: Types.cpp — remaining GatherScatter / PartitionViewLike / StridedView parse+verify paths.

// -----

// expected-error @below{{0-dimension tile shape is not supported}}
%g0 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(), !cuda_tile.tensor_view<16xf32, strides=[1]>, sparse_dim=0>

// -----

// expected-error @below{{tile dimension 0 exceeds i32 limitations (got 1099511627776, expected strictly positive and less than or equal to 2147483647)}}
%g1 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(1099511627776x4), !cuda_tile.tensor_view<8192x16xf32, strides=[16,1]>, sparse_dim=0>

// -----

// GatherScatterViewType::parse tile dimension check (non-positive before i32 cast).
// expected-error @below{{tile dimension 0 exceeds i32 limitations (got 0, expected strictly positive and less than or equal to 2147483647)}}
%g2 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(0x4), !cuda_tile.tensor_view<16x64xf32, strides=[64,1]>, sparse_dim=0>

// -----

// expected-error @below{{F4E2M1FN tiles must have an even number of elements}}
%g3 = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(1), !cuda_tile.tensor_view<8xf4E2M1FN, strides=[1]>, sparse_dim=0>

// -----

// expected-error @below{{padding_value nan can only be used with floating point element types, got 'i32'}}
%p0 = "use_type"() : () -> !cuda_tile.partition_view<tile=(2), padding_value = nan, !cuda_tile.tensor_view<16xi32, strides=[1]>>

// -----

// expected-error @below{{expected 'tensor_view' type, but got '!cuda_tile.tile<f32>'}}
%s0 = "use_type"() : () -> !cuda_tile.strided_view<tile=(2), traversal_strides=[1], !cuda_tile.tile<f32>>

// -----

// Target: Types.cpp — additional reachable parse paths and verifyIndices
// happy-path branches not already exercised by Types_cpp_coverage.mlir.

// -----

// Covers parseStrideArray — `parseOptionalInteger<int64_t>` has a value but
// the value's status is failure() because the literal overflows int64.
// This hits the `if (failed(intParseResult.value())) return failure();` arm.
// expected-error @+1 {{integer value too large}}
%tv_overflow = "use_type"() : () -> !cuda_tile.tensor_view<4xf32, strides=[99999999999999999999]>

// -----

// Covers parseStrideArray — same overflow path but inside a longer strides
// list, ensuring the failure-on-first-element short-circuit kicks in.
// expected-error @+1 {{integer value too large}}
%tv_overflow2 = "use_type"() : () -> !cuda_tile.tensor_view<2x2xf32, strides=[99999999999999999999, 1]>

// -----

// Covers parseViewTileShape — `parseRParen` failure inside partition_view
// when `parseDimensionList` succeeds for one dim but the next token is not ')'.
// Using "2 3" (no x) means parseDimensionList stops after "2" and the next
// token is "3", not ')'.
// expected-error @+1 {{expected ')'}}
%pv_rparen = "use_type"() : () -> !cuda_tile.partition_view<tile=(2 3), !cuda_tile.tensor_view<4xf32, strides=[1]>>

// -----

// Covers parseViewTileShape — `parseRParen` failure inside strided_view
// (StridedViewType::parse calls the same parseViewTileShape helper).
// expected-error @+1 {{expected ')'}}
%sv_rparen = "use_type"() : () -> !cuda_tile.strided_view<tile=(2 3), traversal_strides=[1,1], !cuda_tile.tensor_view<4x4xf32, strides=[4,1]>>

// -----

// Covers GatherScatterViewType::parse — `parseRParen` failure after a
// non-')' token where parseDimensionList stopped early.
// expected-error @+1 {{expected ')'}}
%gs_rparen = "use_type"() : () -> !cuda_tile.gather_scatter_view<tile=(4 5), !cuda_tile.tensor_view<16x32xf32, strides=[32,1]>, sparse_dim=0>

// -----

// Covers GatherScatterViewType::verifyIndices — happy path with 3+ indices
// where all scalar (non-sparse_dim) tile types are equal. This exercises
// the `else if (indexTypes[i] != baseScalarType)` false branch (i.e. the
// equality case where verifyIndices falls through to `return success()`).
cuda_tile.module @gs_verifyindices_scalars_equal {
  cuda_tile.testing$func @f(
      %view: !cuda_tile.gather_scatter_view<tile=(4x4x4), !cuda_tile.tensor_view<16x16x16xf32, strides=[256,16,1]>, sparse_dim=0>) {
    %g = cuda_tile.constant <i32: [0, 1, 2, 3]> : !cuda_tile.tile<4xi32>
    %a = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %b = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %tile, %tok = load_view_tko weak %view[%g, %a, %b]
        : !cuda_tile.gather_scatter_view<tile=(4x4x4), tensor_view<16x16x16xf32, strides=[256,16,1]>, sparse_dim=0>, tile<4xi32>, tile<i32>, tile<i32>
        -> tile<4x4x4xf32>, token
  }
}

// -----

// Covers PartitionViewType::verifyIndices — happy path where every index is
// a 0-rank tile of the same type, exercising both the `dyn_cast` success
// branch and the `indexType != baseIndexType` false branch (the fall-through
// success path).
cuda_tile.module @pv_verifyindices_uniform {
  cuda_tile.testing$func @f(
      %view: !cuda_tile.partition_view<tile=(8x8), !cuda_tile.tensor_view<64x64xf32, strides=[64,1]>>) {
    %i0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %i1 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %tile, %tok = load_view_tko weak %view[%i0, %i1]
        : !cuda_tile.partition_view<tile=(8x8), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32>
        -> tile<8x8xf32>, token
  }
}

// -----

// Covers StridedViewType::verifyIndices — happy path equivalent to the
// PartitionView case above, with uniform 0-rank index tiles.
cuda_tile.module @sv_verifyindices_uniform {
  cuda_tile.testing$func @f(
      %view: !cuda_tile.strided_view<tile=(8x8), traversal_strides=[1,1], !cuda_tile.tensor_view<64x64xf32, strides=[64,1]>>) {
    %i0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %i1 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %tile, %tok = load_view_tko weak %view[%i0, %i1]
        : !cuda_tile.strided_view<tile=(8x8), traversal_strides=[1,1], tensor_view<64x64xf32, strides=[64,1]>>, tile<i32>
        -> tile<8x8xf32>, token
  }
}
