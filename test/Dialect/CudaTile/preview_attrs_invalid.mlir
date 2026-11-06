// RUN: cuda-tile-opt --preview=simt %s -verify-diagnostics -allow-unregistered-dialect \
// RUN:   -split-input-file

// Diagnostics for the public-preview attributes. Round-trip coverage lives in
// preview_attrs.mlir.

//===----------------------------------------------------------------------===//
// Symbolic-expression and layout attribute parsing
//===----------------------------------------------------------------------===//

// expected-error @+2 {{invalid kind of attribute specified}}
// expected-error @+1 {{failed to parse CudaTile_SymbolicExprAddAttr parameter 'lhs' which is to be a `::mlir::cuda_tile::SymbolicExprAttr`}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_add<42, #cuda_tile.preview$expr_int<1>> } {}

// -----

// expected-error @+2 {{invalid kind of attribute specified}}
// expected-error @+1 {{failed to parse CudaTile_SymbolicTupleAttr parameter 'values' which is to be a `::llvm::ArrayRef<::mlir::cuda_tile::SymbolicExprAttr>`}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[42]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// expected-error @+2 {{failed to parse compact layout syntax}}
// expected-error @+1 {{expected integer, identifier, or '('}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(M,):(1)> } {}

// -----

module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(M /+ 0):(1)> } {} // expected-error {{ceil division by zero is not allowed}}

// -----

// expected-error @+2 {{expected ','}}
// expected-error @+1 {{failed to parse compact layout syntax}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(M // 2):(1)> } {}

// -----

// expected-error @+1 {{shapes and strides must have the same rank, got 2 vs 1}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(M, 64):(1)> } {}

// -----
// expected-error @+1 {{shapes and strides must have the same rank, got 2 vs 1}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>, #cuda_tile.preview$expr_int<4>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// expected-error @+1 {{mode 0 is a tuple in shapes and an integer in strides}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<((2, 2), 4):(1, 4)> } {}

// -----

// expected-error @+1 {{mode 1 is a tuple in strides and an integer in shapes}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4, 4):(1, (1, 2))> } {}

// -----

// expected-error @+1 {{mode 0 is a tuple in shapes and an integer in strides}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<((2, 2), 4):(2 * 4, 4)> } {}

// -----

// expected-error @+1 {{mode 0 has rank 3 in shapes and rank 2 in strides}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<((2, 2, 2)):((1, 2))> } {}

// -----

// expected-error @+1 {{mode 1.0 is a tuple in shapes and an integer in strides}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(2, ((2, 2), 2)):(1, (2, (4, 8)))> } {}

// -----

// expected-error @+1 {{shape extents must be positive, but mode 1 is 0}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4, 0):(1, 4)> } {}

// -----

// expected-error @+1 {{shape extents must be positive, but mode 0 is -4}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(-4, 8):(1, 4)> } {}

// -----

// expected-error @+1 {{shape extents must be positive, but mode 1.0 is -2}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4, (-2, -2)):(1, (4, 8))> } {}

// -----

// expected-error @+1 {{shape extents must be positive, but mode 0 is 0}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4 - 4, 8):(1, 4)> } {}

// -----

// expected-error @+1 {{shape extents must be positive, but mode 0 is -9223372036854775808}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<-9223372036854775808>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// An expression literal is a signed 64-bit integer, so 2^63 has no
// representation and is rejected where it is spelled.
// expected-error @+2 {{integer literal does not fit in a signed 64-bit integer}}
// expected-error @+1 {{failed to parse compact layout syntax}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(9223372036854775808, 8):(1, 4)> } {}

// -----

// -9223372036854775808 is representable, but negating it is not, and the
// negation of a literal has to be folded into the literal to round-trip.
// expected-error @+1 {{negated integer literal does not fit in a signed 64-bit integer}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(-(-9223372036854775808), 8):(1, 4)> } {}

// -----

// The range check runs after the sign is applied, so the literal one step past
// the negative end of the range is rejected too.
// expected-error @+2 {{integer literal does not fit in a signed 64-bit integer}}
// expected-error @+1 {{failed to parse compact layout syntax}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(-9223372036854775809, 8):(1, 4)> } {}

// -----

// The bound belongs to the attribute, not to the compact grammar: the verbose
// spelling of a literal is an int64_t as well.
// expected-error @+2 {{integer value too large}}
// expected-error @+1 {{failed to parse CudaTile_SymbolicExprIntAttr parameter 'value'}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_int<9223372036854775808> } {}

// -----

// expected-error @+2 {{integer value too large}}
// expected-error @+1 {{failed to parse CudaTile_SymbolicExprIntAttr parameter 'value'}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_int<-9223372036854775809> } {}

// -----

// expected-error @+1 {{strides must be non-negative, but mode 1 is -4}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4, 8):(1, -4)> } {}

// -----

// expected-error @+1 {{strides must be non-negative, but mode 1.1 is -8}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4, (2, 2)):(1, (4, -8))> } {}

// -----

// expected-error @+1 {{strides must be non-negative, but mode 1 is -4}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(4, 8):(1, 4 - 8)> } {}

// -----

// expected-error @+1 {{strides must be non-negative, but mode 0 is -7}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(1, 8):(-7, 1)> } {}

// -----

// expected-error @+1 {{strides must be non-negative, but mode 0 is -9223372036854775808}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<-9223372036854775808>]>> } {}

// -----

module attributes { cuda_tile.bad = #cuda_tile.preview$const_param<name = "", type = !cuda_tile.preview$Datatype> } {} // expected-error {{constant parameter name cannot be empty}}

// -----

module attributes { cuda_tile.bad = #cuda_tile.preview$expr_var<"source.value"> } {} // expected-error {{expression variable name cannot contain '.'; dotted names are reserved for builtins}}

// -----

module attributes { cuda_tile.bad = #cuda_tile.preview$expr_builtin<"compiler.num_threads", index> } {} // expected-error {{builtin expression name must be in the 'context.' namespace}}

// -----

module attributes { cuda_tile.bad = #cuda_tile.preview$expr_builtin<"context.", index> } {} // expected-error {{builtin expression name must be in the 'context.' namespace}}

// -----

module attributes { cuda_tile.bad = #cuda_tile.preview$expr_builtin<"context.num_threads", i64> } {} // expected-error {{builtin expression result type must be index, got 'i64'}}

// -----

//===----------------------------------------------------------------------===//
// PreviewAttrs.cpp: parser and verifier error paths
//===----------------------------------------------------------------------===//

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers the early-return path inside LayoutAttr::parse when the leading '<'
// delimiter is missing.

//===----------------------------------------------------------------------===//
// LayoutAttr::parse — parser.parseLess() failure (missing leading '<').
//===----------------------------------------------------------------------===//

// Test: omit the leading '<' after `#cuda_tile.preview$layout`. The first
// `parser.parseLess()` call inside LayoutAttr::parse returns failure, taking
// the `return {};` early-out branch and emitting "expected '<'".
module attributes {
  // expected-error @+1 {{expected '<'}}
  cuda_tile.bad = #cuda_tile.preview$layout (M, 64):(64, 1)>
} {}

// -----

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers: verify methods for div-by-zero, empty var name, and empty const param.

// expected-error @+1 {{division by zero is not allowed}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<0>> } {}

// -----

// expected-error @+1 {{floor division by zero is not allowed}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<0>> } {}

// -----

// expected-error @+1 {{ceil division by zero is not allowed}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<0>> } {}

// -----

// expected-error @+1 {{expression variable name cannot be empty}}
module attributes { cuda_tile.bad = #cuda_tile.preview$expr_var<""> } {}

// -----

// expected-error @+1 {{constant parameter name cannot be empty}}
module attributes { cuda_tile.bad = #cuda_tile.preview$const_param<name = "", type = i32> } {}

// -----

// Verify expr_div with non-zero rhs (valid, no error) goes through verify successfully
module attributes { cuda_tile.valid = #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"X">, #cuda_tile.preview$expr_int<3>> } {}

// -----

// Verify expr_floor_div with non-zero rhs (valid, no error)
module attributes { cuda_tile.valid = #cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"X">, #cuda_tile.preview$expr_int<5>> } {}

// -----

// Verify expr_ceil_div with non-zero rhs (valid, no error)
module attributes { cuda_tile.valid = #cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"X">, #cuda_tile.preview$expr_int<7>> } {}

// -----

// Verify expr_div with variable rhs (valid, no error -- rhs is not int, skip check)
module attributes { cuda_tile.valid = #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"X">, #cuda_tile.preview$expr_var<"Y">> } {}

// -----

// Verify expr_floor_div with variable rhs (valid, rhs not int)
module attributes { cuda_tile.valid = #cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"X">, #cuda_tile.preview$expr_var<"Y">> } {}

// -----

// Verify expr_ceil_div with variable rhs (valid, rhs not int)
module attributes { cuda_tile.valid = #cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"X">, #cuda_tile.preview$expr_var<"Y">> } {}

// -----

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers LayoutAttr::parse() verbose-path error branches.

// LayoutAttr::parse verbose path -- missing "strides" keyword after shapes
// expected-error @+1 {{expected 'strides'}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, badkey = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// LayoutAttr::parse verbose path -- missing '>' after strides
// expected-error @+1 {{expected '>'}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]> extra> } {}

// -----

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers LayoutAttr::parse() compact-syntax error paths.

// LayoutAttr::parse -- compact syntax fails with missing colon between shapes and strides
// expected-error @+2 {{expected ':'}}
// expected-error @+1 {{failed to parse compact layout syntax}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<(M, 64) (64, 1)> } {}

// -----

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers additional partially-uncovered branches in LayoutAttr::parse():
//   - parseEqual failure after "shapes" / "strides"
//   - parseComma failure between shapes and strides
//   - parseAttribute failure on shapes / strides

// LayoutAttr::parse verbose path -- missing '=' after "shapes"
// expected-error @+1 {{expected '='}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// LayoutAttr::parse verbose path -- missing '=' after "strides"
// expected-error @+1 {{expected '='}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, strides #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// LayoutAttr::parse verbose path -- missing ',' between shapes-attr and strides
// expected-error @+1 {{expected ','}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]> strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>> } {}

// -----

// LayoutAttr::parse verbose path -- shapes attribute is missing (parseAttribute fails)
// expected-error @+1 {{expected attribute value}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = > } {}

// -----

// LayoutAttr::parse verbose path -- strides attribute is missing (parseAttribute fails)
// expected-error @+1 {{expected attribute value}}
module attributes { cuda_tile.bad = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, strides = > } {}

// -----

//===----------------------------------------------------------------------===//
// ScratchRequestAttr: range checks run on the simplified request
//===----------------------------------------------------------------------===//

// A request that folds to an invalid size or alignment is rejected however it
// was spelled, because the folded form is what a consumer of the attribute --
// and the compact printer -- sees.

// expected-error @+1 {{scratch size must be non-negative}}
module attributes { cuda_tile.bad = #cuda_tile.preview$scratch_request<smem.size = #cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_int<4>, #cuda_tile.preview$expr_int<12>>, smem.alignment = #cuda_tile.preview$expr_int<16>> } {}

// -----

// expected-error @+1 {{scratch alignment must be positive}}
module attributes { cuda_tile.bad = #cuda_tile.preview$scratch_request<smem.size = #cuda_tile.preview$expr_int<128>, smem.alignment = #cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_int<4>, #cuda_tile.preview$expr_int<4>>> } {}

// -----

// expected-error @+1 {{scratch alignment must be a power of two}}
module attributes { cuda_tile.bad = #cuda_tile.preview$scratch_request<smem.size = #cuda_tile.preview$expr_int<128>, smem.alignment = #cuda_tile.preview$expr_mul<#cuda_tile.preview$expr_int<3>, #cuda_tile.preview$expr_int<32>>> } {}
