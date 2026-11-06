// RUN: cuda-tile-opt --preview=simt %s | cuda-tile-opt --preview=simt | FileCheck %s

// Parse/print round-trip for the public-preview attributes, including the
// compact layout syntax and its verbose fallback. Diagnostics live in
// preview_attrs_invalid.mlir.

//===----------------------------------------------------------------------===//
// The public-preview attribute surface
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @preview_attr_surface attributes {
// CHECK-SAME: cuda_tile.dtype_param = #cuda_tile<preview$const_param<name = "T", type = !cuda_tile<preview$Datatype>>>
// CHECK-SAME: cuda_tile.expr_builtin = #cuda_tile<preview$expr_builtin<"context.num_threads", index>>
// CHECK-SAME: cuda_tile.expr_neg = #cuda_tile<preview$expr_neg<#cuda_tile<preview$expr_var<"M">>>>
// CHECK-SAME: cuda_tile.layout_context = #cuda_tile<preview$layout<(context.num_threads, 8):(8, 1)>>
// CHECK-SAME: cuda_tile.layout_context_expr = #cuda_tile<preview$layout<(context.num_threads * 4, -context.num_threads):(8, 1)>>
// CHECK-SAME: cuda_tile.layout_empty_nested = #cuda_tile<preview$layout<((), 16):((), 1)>>
// CHECK-SAME: cuda_tile.layout_expr = #cuda_tile<preview$layout<(M + 1, N * 4, K / 2, P /+ 8):(1, 4, 8, 16)>>
// CHECK-SAME: cuda_tile.layout_int_min = #cuda_tile<preview$layout<(M + -9223372036854775808, 1):(1, 0)>>
// CHECK-SAME: cuda_tile.layout_keyword_var = #cuda_tile<preview$layout<(shapes, strides):(1, 1)>>
// CHECK-SAME: cuda_tile.layout_neg_group = #cuda_tile<preview$layout<(-(M + N), -(M * N)):(1, 1)>>
// CHECK-SAME: cuda_tile.layout_neg_neg = #cuda_tile<preview$layout<(-(-M), -(-(-M))):(1, 1)>>
// CHECK-SAME: cuda_tile.layout_neg_operand = #cuda_tile<preview$layout<(M + -N, M - -N, M * -N):(1, 1, 1)>>
// CHECK-SAME: cuda_tile.layout_negative = #cuda_tile<preview$layout<(M + -1, 1):(1, 0)>>
// CHECK-SAME: cuda_tile.layout_nested = #cuda_tile<preview$layout<((M, 16), (1, M)):((16, 1), (M, 1))>>
// CHECK-SAME: cuda_tile.layout_rank0 = #cuda_tile<preview$layout<():()>>
// CHECK-SAME: cuda_tile.layout_simple = #cuda_tile<preview$layout<(M, 64):(64, 1)>>
// CHECK-SAME: cuda_tile.layout_sub = #cuda_tile<preview$layout<(M - N, -M):(1, 1)>>
// CHECK-SAME: cuda_tile.shape_param = #cuda_tile<preview$const_param<name = "S", type = !cuda_tile<preview$shape>>>
module @preview_attr_surface attributes {
  cuda_tile.dtype_param = #cuda_tile.preview$const_param<name = "T", type = !cuda_tile.preview$Datatype>,
  cuda_tile.expr_builtin = #cuda_tile.preview$expr_builtin<"context.num_threads", index>,
  cuda_tile.expr_neg = #cuda_tile.preview$expr_neg<#cuda_tile.preview$expr_var<"M">>,
  cuda_tile.shape_param = #cuda_tile.preview$const_param<name = "S", type = !cuda_tile.preview$shape>,
  cuda_tile.layout_simple = #cuda_tile.preview$layout<(M, 64):(64, 1)>,
  cuda_tile.layout_nested = #cuda_tile.preview$layout<((M, 16), (1, M)):((16, 1), (M, 1))>,
  cuda_tile.layout_expr = #cuda_tile.preview$layout<(M + 1, N * 4, K / 2, P /+ 8):(1, 4, 8, 16)>,
  cuda_tile.layout_sub = #cuda_tile.preview$layout<(M - N, -M):(1, 1)>,
  cuda_tile.layout_neg_neg = #cuda_tile.preview$layout<(-(-M), -(-(-M))):(1, 1)>,
  cuda_tile.layout_neg_group = #cuda_tile.preview$layout<(-(M + N), -(M * N)):(1, 1)>,
  cuda_tile.layout_neg_operand = #cuda_tile.preview$layout<(M + -N, M - -N, M * -N):(1, 1, 1)>,
  // Negative integer literals survive verification only inside a still
  // parametric expression, since concrete extents are positive and concrete
  // strides non-negative.
  cuda_tile.layout_negative = #cuda_tile.preview$layout<(M + -1, 1):(1, 0)>,
  cuda_tile.layout_int_min = #cuda_tile.preview$layout<(M + -9223372036854775808, 1):(1, 0)>,
  cuda_tile.layout_context = #cuda_tile.preview$layout<(context.num_threads, 8):(8, 1)>,
  cuda_tile.layout_context_expr = #cuda_tile.preview$layout<(context.num_threads * 4, -context.num_threads):(8, 1)>,
  cuda_tile.layout_keyword_var = #cuda_tile.preview$layout<(shapes, strides):(1, 1)>,
  cuda_tile.layout_rank0 = #cuda_tile.preview$layout<():()>,
  cuda_tile.layout_empty_nested = #cuda_tile.preview$layout<((), 16):((), 1)>
} {
}

//===----------------------------------------------------------------------===//
// Extern-related preview attributes (available in public builds)
//===----------------------------------------------------------------------===//

// Verify that the preview extern attributes are available in public builds
// and print with the public-preview dialect marker.
// CHECK-LABEL: module @preview_extern_attrs attributes {
// CHECK-DAG: test.access = #cuda_tile<preview$access<in>>
// CHECK-DAG: test.const_param = #cuda_tile<preview$const_param<name = "M", type = index>>
// CHECK-DAG: test.scratch_request = #cuda_tile<preview$scratch_request<smem.size = #cuda_tile<preview$expr_int<32>>, smem.alignment = #cuda_tile<preview$expr_int<16>>>>
module @preview_extern_attrs attributes {
  test.access = #cuda_tile.preview$access<in>,
  test.const_param = #cuda_tile.preview$const_param<name = "M", type = index>,
  test.scratch_request = #cuda_tile.preview$scratch_request<
    smem.size = #cuda_tile.preview$expr_int<32>,
    smem.alignment = #cuda_tile.preview$expr_int<16>>
} {}

//===----------------------------------------------------------------------===//
// PreviewAttrs.cpp: compact layout printing / expression round-trip
//===----------------------------------------------------------------------===//

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers compact layout printing/parsing and expression roundtrip.

// ---- Layout with subtraction expression ----
// Compact "M - 1" builds a preview$expr_sub and round-trips unchanged.
// CHECK-LABEL: module @test_sub_expr
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M - 1):(1)>>
module @test_sub_expr attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M - 1):(1)>
} {}

// ---- Layout with addition expression ----
// CHECK-LABEL: module @test_add_expr
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M + N):(1)>>
module @test_add_expr attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M + N):(1)>
} {}

// ---- Layout with multiplication expression ----
// CHECK-LABEL: module @test_mul_expr
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M * 4):(1)>>
module @test_mul_expr attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M * 4):(1)>
} {}

// ---- Layout with floor division expression ----
// CHECK-LABEL: module @test_floordiv_expr
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M / 4):(1)>>
module @test_floordiv_expr attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M / 4):(1)>
} {}

// ---- Layout with ceil division expression ----
// CHECK-LABEL: module @test_ceildiv_expr
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M /+ 4):(1)>>
module @test_ceildiv_expr attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M /+ 4):(1)>
} {}

// ---- Layout with nested tuple expressions ----
// CHECK-LABEL: module @test_nested_tuple
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<((M, N), (K, P)):((1, 2), (3, 4))>>
module @test_nested_tuple attributes {
  cuda_tile.l = #cuda_tile.preview$layout<((M, N), (K, P)):((1, 2), (3, 4))>
} {}

// ---- Layout with complex expression combining add + mul (tests precedence printing) ----
// CHECK-LABEL: module @test_precedence_add_mul
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M + N * 2):(1)>>
module @test_precedence_add_mul attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M + N * 2):(1)>
} {}

// ---- Layout with complex expression combining sub + mul (tests precedence) ----
// CHECK-LABEL: module @test_precedence_sub_mul
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M * 3 - N):(1)>>
module @test_precedence_sub_mul attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M * 3 - N):(1)>
} {}

// ---- Layout with right-associative expression via verbose syntax ----
// Verbose syntax is used here so the input carries a real preview$expr_sub,
// which is what exercises the printExprCompact needParens logic.
// CHECK-LABEL: module @test_right_assoc_sub
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M - (N - K)):(1)>>
module @test_right_assoc_sub attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_var<"K">>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- Layout with (M+N)*2 via verbose syntax (tests needParens for mul of add) ----
// CHECK-LABEL: module @test_parens_mul_add
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<((M + N) * 2):(1)>>
module @test_parens_mul_add attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_mul<#cuda_tile.preview$expr_add<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_var<"N">>, #cuda_tile.preview$expr_int<2>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- Layout with multiple expression types ----
// CHECK-LABEL: module @test_mixed_exprs
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M * 4 + N / 2, K /+ 8 - P):(1, 2)>>
module @test_mixed_exprs attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M * 4 + N / 2, K /+ 8 - P):(1, 2)>
} {}

// ---- Layout with negative integers ----
// Concrete extents are positive and concrete strides non-negative, so a
// negative literal only survives as an operand of a parametric expression.
// CHECK-LABEL: module @test_negative_int
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M * -3, 2):(N + -4, 1)>>
module @test_negative_int attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(M * -3, 2):(N + -4, 1)>
} {}

// ---- Layout with integer-only expressions ----
// CHECK-LABEL: module @test_int_only
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(16, 32, 64):(64, 2, 1)>>
module @test_int_only attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(16, 32, 64):(64, 2, 1)>
} {}

// ---- Layout with an unexpanded shape parameter ----
// `S` can expand into a tuple, so attribute is valid.
// CHECK-LABEL: module @test_expandable_shape_parameter
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(S, 1):((1, 2, 4), 0)>>
module @test_expandable_shape_parameter attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(S, 1):((1, 2, 4), 0)>
} {}

// ---- ConstParam roundtrip ----
// CHECK-LABEL: module @test_const_param
// CHECK-SAME: cuda_tile.cp = #cuda_tile<preview$const_param<name = "M", type = index>>
module @test_const_param attributes {
  cuda_tile.cp = #cuda_tile.preview$const_param<name = "M", type = index>
} {}

// ---- Verbose layout syntax parsing ----
// CHECK-LABEL: module @test_verbose_layout
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M, 64):(64, 1)>>
module @test_verbose_layout attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<64>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<64>, #cuda_tile.preview$expr_int<1>]>>
} {}

// ---- Layout with deeply nested expressions via verbose syntax ----
// CHECK-LABEL: module @test_deep_nesting
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<((M + 1) * (N - 2)):(1)>>
module @test_deep_nesting attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_mul<#cuda_tile.preview$expr_add<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<1>>, #cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- Layout with division in right operand via verbose syntax ----
// CHECK-LABEL: module @test_div_right_parens
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M / (N / K)):(1)>>
module @test_div_right_parens attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_var<"K">>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- Layout with ceildiv in right operand (tests isRight for ceildiv) ----
// CHECK-LABEL: module @test_ceildiv_right_parens
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(M /+ (N /+ K)):(1)>>
module @test_ceildiv_right_parens attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_var<"K">>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- Layout with expr_div (toward-zero) via verbose syntax ----
// This attr type is not covered by compact syntax; tests canPrintCompact returning false → verbose print
// CHECK-LABEL: module @test_div_verbose
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_int<4>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_div_verbose attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<4>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

//===----------------------------------------------------------------------===//
// PreviewAttrs.cpp: canPrintCompact() verbose-print fallbacks
//===----------------------------------------------------------------------===//

// Coverage target: lib/Dialect/CudaTile/IR/PreviewAttrs.cpp
// Covers branches in canPrintCompact() where the rhs sub-expression is NOT
// compact-printable (because it contains expr_div, which lacks a compact form),
// and the LayoutAttr::print branch where shapes are compact but strides are not.

// ---- LayoutAttr::print: shapes compact + strides NOT compact (contains expr_div)
// Triggers the verbose fallback because canPrintCompact(strides) returns false.
// CHECK-LABEL: module @test_strides_div_verbose
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<16>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_int<4>>>>]>>>>
module @test_strides_div_verbose attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_int<4>>]>>
} {}

// ---- canPrintCompact(SymbolicExprMulAttr): lhs OK, rhs NOT compact
// Covers the second-operand-false branch of canPrintCompact for expr_mul.
// CHECK-LABEL: module @test_mul_rhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_mul<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_mul_rhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_mul<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprAddAttr): lhs OK, rhs NOT compact
// CHECK-LABEL: module @test_add_rhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_add<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_add_rhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_add<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprSubAttr): lhs OK, rhs NOT compact
// CHECK-LABEL: module @test_sub_rhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_sub<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_sub_rhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprFloorDivAttr): lhs OK, rhs NOT compact
// CHECK-LABEL: module @test_floordiv_rhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_floor_div<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_floordiv_rhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprCeilDivAttr): lhs OK, rhs NOT compact
// CHECK-LABEL: module @test_ceildiv_rhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_ceil_div<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_ceildiv_rhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprNegAttr): returns false for a negated literal
// The compact grammar folds a leading '-' into the literal it precedes, so
// printing this node as `-4` would read back as the literal -4 rather than as a
// negation of 4. The negation is kept inside a parametric expression because a
// concrete extent has to be positive.
// CHECK-LABEL: module @test_neg_int_verbose
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_add<#cuda_tile<preview$expr_var<"M">>, #cuda_tile<preview$expr_neg<#cuda_tile<preview$expr_int<4>>>>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_neg_int_verbose attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_add<#cuda_tile.preview$expr_var<"M">, #cuda_tile.preview$expr_neg<#cuda_tile.preview$expr_int<4>>>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact: returns false for ConcreteTypeAttr (not in compact-printable set)
// Forces the verbose print fallback because canPrintCompact returns false for
// the wrapping tuple containing a ConcreteTypeAttr.
// CHECK-LABEL: module @test_concrete_type_verbose
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$concrete_type<f32>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_concrete_type_verbose attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$concrete_type<f32>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprMulAttr): lhs NOT compact (short-circuit)
// Covers the lhs-false short-circuit branch of canPrintCompact for expr_mul.
// CHECK-LABEL: module @test_mul_lhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_mul<#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>, #cuda_tile<preview$expr_var<"M">>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_mul_lhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_mul<#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>, #cuda_tile.preview$expr_var<"M">>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprAddAttr): lhs NOT compact (short-circuit)
// CHECK-LABEL: module @test_add_lhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_add<#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>, #cuda_tile<preview$expr_var<"M">>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_add_lhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_add<#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>, #cuda_tile.preview$expr_var<"M">>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprSubAttr): lhs NOT compact (short-circuit)
// CHECK-LABEL: module @test_sub_lhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_sub<#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>, #cuda_tile<preview$expr_var<"M">>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_sub_lhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_sub<#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>, #cuda_tile.preview$expr_var<"M">>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprFloorDivAttr): lhs NOT compact (short-circuit)
// CHECK-LABEL: module @test_floordiv_lhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_floor_div<#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>, #cuda_tile<preview$expr_var<"M">>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_floordiv_lhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_floor_div<#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>, #cuda_tile.preview$expr_var<"M">>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

// ---- canPrintCompact(SymbolicExprCeilDivAttr): lhs NOT compact (short-circuit)
// CHECK-LABEL: module @test_ceildiv_lhs_not_compact
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<shapes = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_ceil_div<#cuda_tile<preview$expr_div<#cuda_tile<preview$expr_var<"N">>, #cuda_tile<preview$expr_int<2>>>>, #cuda_tile<preview$expr_var<"M">>>>]>>, strides = #cuda_tile<preview$symbolic_tuple<[#cuda_tile<preview$expr_int<1>>]>>>>
module @test_ceildiv_lhs_not_compact attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_ceil_div<#cuda_tile.preview$expr_div<#cuda_tile.preview$expr_var<"N">, #cuda_tile.preview$expr_int<2>>, #cuda_tile.preview$expr_var<"M">>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>>
} {}

//===----------------------------------------------------------------------===//
// CompactLayoutParser.cpp: makeNeg / makeSub
//===----------------------------------------------------------------------===//

// Test negation of variables (triggers makeNeg) and subtraction expressions
// (triggers makeSub) in compact layout parsing. Both build the matching
// attribute -- preview$expr_neg and preview$expr_sub -- so each round-trips to
// the spelling it was written with.

// CHECK-LABEL: module @compact_layout_neg_sub attributes {
// CHECK-SAME: cuda_tile.layout_neg_var = #cuda_tile<preview$layout<(-M, 16):(0, 1)>>
// CHECK-SAME: cuda_tile.layout_sub_complex = #cuda_tile<preview$layout<(M - N * 2, 16):(0, 1)>>
// CHECK-SAME: cuda_tile.layout_sub_simple = #cuda_tile<preview$layout<(M - N, 16):(0, 1)>>
// CHECK-SAME: cuda_tile.layout_sub_strides = #cuda_tile<preview$layout<(M, 16):(K - 1, 1)>>
module @compact_layout_neg_sub attributes {
  cuda_tile.layout_neg_var = #cuda_tile.preview$layout<(-M, 16):(0, 1)>,
  cuda_tile.layout_sub_simple = #cuda_tile.preview$layout<(M - N, 16):(0, 1)>,
  cuda_tile.layout_sub_strides = #cuda_tile.preview$layout<(M, 16):(K - 1, 1)>,
  cuda_tile.layout_sub_complex = #cuda_tile.preview$layout<(M - N * 2, 16):(0, 1)>
} {
}

//===----------------------------------------------------------------------===//
// CompactLayoutParser: the sign of an integer literal
//===----------------------------------------------------------------------===//

// A '-' in front of a literal belongs to the literal, and a second one negates
// it again, so `--5` is `5`. A parenthesized negation is folded the same way:
// `-(4)` is the literal `-4`, which is also the only spelling the printer has
// for it.

// CHECK-LABEL: module @compact_int_literal_sign
// CHECK-SAME: cuda_tile.double_neg = #cuda_tile<preview$layout<(M + 5):(1)>>
// CHECK-SAME: cuda_tile.neg_group = #cuda_tile<preview$layout<(M + -4):(1)>>
// CHECK-SAME: cuda_tile.neg_int_min = #cuda_tile<preview$layout<(M + -9223372036854775808):(1)>>
module @compact_int_literal_sign attributes {
  cuda_tile.double_neg = #cuda_tile.preview$layout<(M + --5):(1)>,
  cuda_tile.neg_group = #cuda_tile.preview$layout<(M + -(4)):(1)>,
  cuda_tile.neg_int_min = #cuda_tile.preview$layout<(M + -9223372036854775808):(1)>
} {}

//===----------------------------------------------------------------------===//
// CompactLayoutParser: a parenthesized group in leading position
//===----------------------------------------------------------------------===//

// A '(' opening a shape/stride entry is ambiguous between a nested tuple and a
// grouped scalar sub-expression. These are written in compact form (rather than
// the verbose form the sibling cases use) so they test the parser directly and
// not just the printer's output.

// CHECK-LABEL: module @compact_leading_group
// CHECK-SAME: cuda_tile.mul_of_add = #cuda_tile<preview$layout<((M + N) * 2):(1)>>
// CHECK-SAME: cuda_tile.nested_tuple = #cuda_tile<preview$layout<((M, 16), (1, M)):((16, 1), (M, 1))>>
module @compact_leading_group attributes {
  cuda_tile.mul_of_add = #cuda_tile.preview$layout<((M + N) * 2):(1)>,
  cuda_tile.nested_tuple = #cuda_tile.preview$layout<((M, 16), (1, M)):((16, 1), (M, 1))>
} {}

// -----

// `(x)` groups a scalar, so it collapses; `(x, y)` stays a tuple. See the
// grammar comment on parseConstantArgument in
// tile_ir/include/cuda_tile/Dialect/CudaTile/IR/CompactLayoutParser.h.

// CHECK-LABEL: module @compact_group_vs_tuple
// CHECK-SAME: cuda_tile.grouped = #cuda_tile<preview$layout<(1, 16, 64):(0, 0, 1)>>
// CHECK-SAME: cuda_tile.tupled = #cuda_tile<preview$layout<(1, (16, 8), 64):(0, (8, 1), 1)>>
module @compact_group_vs_tuple attributes {
  cuda_tile.grouped = #cuda_tile.preview$layout<(1, (16), 64):(0, 0, 1)>,
  cuda_tile.tupled = #cuda_tile.preview$layout<(1, (16, 8), 64):(0, (8, 1), 1)>
} {}

//===----------------------------------------------------------------------===//
// CompactLayoutParser: a nested one-element tuple
//===----------------------------------------------------------------------===//

// Written verbosely so the INPUT does not depend on the printer. A nested
// one-element mode must keep its trailing comma, otherwise `(16)` would come
// back as a grouped scalar and the mode structure would be silently flattened.
// Both RUN lines matter here: the second one re-parses this file's own output.

// CHECK-LABEL: module @nested_singleton_tuple
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<((16,), 64):((1,), 16)>>
module @nested_singleton_tuple attributes {
  cuda_tile.l = #cuda_tile.preview$layout<shapes = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<16>]>, #cuda_tile.preview$expr_int<64>]>, strides = #cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$symbolic_tuple<[#cuda_tile.preview$expr_int<1>]>, #cuda_tile.preview$expr_int<16>]>>
} {}

// -----

// The outermost shapes/strides tuples are unambiguous -- their parens *are* the
// tuple -- so a one-element layout keeps printing without a trailing comma.

// CHECK-LABEL: module @top_level_singleton_tuple
// CHECK-SAME: cuda_tile.l = #cuda_tile<preview$layout<(16):(1)>>
module @top_level_singleton_tuple attributes {
  cuda_tile.l = #cuda_tile.preview$layout<(16):(1)>
} {}
