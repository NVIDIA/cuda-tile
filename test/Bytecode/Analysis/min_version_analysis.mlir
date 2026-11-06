// Tests for MinVersionAnalyzer utility.
// RUN: cuda-tile-translate -analyze-min-version -split-input-file %s | FileCheck %s
//
// Each section prints one version, separated by a blank line. The checks are
// chained (empty-line then next-line) so that a wrong version is reported
// against its own section; plain CHECK directives would match them as a
// subsequence and blame an unrelated one. Append new sections in that shape.
//
// arg_attrs (13.5) is covered end to end in encoding_stability/arg_attrs*.mlir.

// Base operations (constant, addf, return) are available in 13.1.
// CHECK: 13.1
cuda_tile.module @basic_13_1 {
  entry @kernel() {
    %c1 = constant <f32: 1.0> : tile<f32>
    %c2 = constant <f32: 2.0> : tile<f32>
    %r = addf %c1, %c2 : tile<f32>
    return
  }
}

// -----

// The atan2 operation was introduced in 13.2.
// CHECK-EMPTY:
// CHECK-NEXT: 13.2
cuda_tile.module @atan2_13_2 {
  entry @kernel() {
    %c1 = constant <f32: 1.0> : tile<f32>
    %c2 = constant <f32: 2.0> : tile<f32>
    %r = atan2 %c1, %c2 : tile<f32>
    return
  }
}

// -----

// The f8E8M0FNU type was introduced in 13.2.
// CHECK-EMPTY:
// CHECK-NEXT: 13.2
cuda_tile.module @f8e8m0fnu_13_2 {
  entry @kernel(%p: tile<ptr<f8E8M0FNU>>) {
    return
  }
}

// -----

// The alloca operation was introduced in 13.3.
// CHECK-EMPTY:
// CHECK-NEXT: 13.3
cuda_tile.module @alloca_13_3 {
  entry @kernel() {
    %p = alloca num_elem = 64, alignment = 128 : tile<ptr<f32>>
    return
  }
}

// -----

// The f4E2M1FN type was introduced in 13.3.
// CHECK-EMPTY:
// CHECK-NEXT: 13.3
cuda_tile.module @f4e2m1fn_13_3 {
  entry @kernel(%p: tile<ptr<f4E2M1FN>>) {
    return
  }
}

// -----

// The 'unsigned' attribute on for loop was added in 13.2 (op itself is 13.1).
// CHECK-EMPTY:
// CHECK-NEXT: 13.2
cuda_tile.module @for_unsignedCmp_13_2 {
  entry @kernel() {
    %c0 = constant <i32: 0> : tile<i32>
    %c10 = constant <i32: 10> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    for unsigned %iv in (%c0 to %c10, step %c1) : tile<i32> {
      continue
    }
    return
  }
}

// -----

// The result_token result was added in 250.1, but is not used here.
// CHECK-EMPTY:
// CHECK-NEXT: 250.0
cuda_tile.module @evolution_unused_result_250_0 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    return
  }
}

// -----

// The new_attr attribute was added in 250.1 (op itself is 250.0).
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_new_attr_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) new_attr = 42 -> !cuda_tile.token
    return
  }
}

// -----

// The optional_token operand was added in 250.1 (op itself is 250.0).
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_operand_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_in = make_token : !cuda_tile.token
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) token = %token_in : !cuda_tile.token -> !cuda_tile.token
    return
  }
}

// -----

// The 'normal' enum value is 250.0, so version is 250.0.
// CHECK-EMPTY:
// CHECK-NEXT: 250.0
cuda_tile.module @evolution_enum_250_0 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<normal> -> !cuda_tile.token
    return
  }
}

// -----

// The 'high' enum value was added in 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_enum_high_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<high> -> !cuda_tile.token
    return
  }
}

// -----

// The result_token is used here, so requires 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_used_result_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    %joined = join_tokens %token, %token : !cuda_tile.token
    return
  }
}

// -----

// BytecodeTestEvolvedAttr with both evolved params at their fallbacks: optional
// absent, dependentValue equal to baseValue. Neither has to be encoded.
// CHECK-EMPTY:
// CHECK-NEXT: 250.0
cuda_tile.module @evolution_attr_fallbacks_eq_250_0 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<123, 123> -> !cuda_tile.token
    return
  }
}

// -----

// BytecodeTestEvolvedAttr with optional param: param version 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_attr_with_optional_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<123, 123, 456> -> !cuda_tile.token
    return
  }
}

// -----

// BytecodeTestEvolvedAttr with dependentValue (required evolved param, added
// in 250.1) differing from its fallback $_param_baseValue: param version 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_attr_dependent_diff_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<42, 999> -> !cuda_tile.token
    return
  }
}

// -----

// The new_flag attribute was added in 250.1 (op itself is 250.0).
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @new_attribute_flag_250_1 {
  entry @kernel() {
    testing$bytecode_test_new_attribute new_flag
    return
  }
}

// -----

// The new_param attribute was added in 250.1 (op itself is 250.0).
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @new_attribute_param_250_1 {
  entry @kernel() {
    testing$bytecode_test_new_attribute new_param = 100
    return
  }
}

// -----

// BytecodeTestEvolvedType: newType (evolved 250.1) is explicitly set to its
// constant fallback i32. The parameter carries no information beyond the
// fallback, so it does not raise the minimum version.
// CHECK-EMPTY:
// CHECK-NEXT: 250.0
cuda_tile.module @evolution_type_constant_fallback_eq_250_0 {
  entry @kernel(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32>) {
    return
  }
}

// -----

// BytecodeTestEvolvedType: newType (evolved 250.1) differs from its constant
// fallback i32. Encoding the explicit value requires 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_type_constant_fallback_diff_250_1 {
  entry @kernel(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, f16>) {
    return
  }
}

// -----

// BytecodeTestEvolvedType with both evolved params at their fallbacks:
// dependentType equal to baseType, optionalType absent. `<f32, f32>` is what
// both fallbacks look like, so one spelling covers them.
// CHECK-EMPTY:
// CHECK-NEXT: 250.0
cuda_tile.module @evolution_type_fallbacks_eq_250_0 {
  entry @kernel(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32>) {
    return
  }
}

// -----

// BytecodeTestEvolvedType: dependentType (evolved 250.1) differs from its
// $_param_baseType fallback (baseType). Encoding the explicit value
// requires 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_type_dependent_fallback_diff_250_1 {
  entry @kernel(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, i64>) {
    return
  }
}

// -----

// BytecodeTestEvolvedType: optionalType (evolved 250.1, OptionalParameter) is
// present (f16) — differs from its implicit null fallback. Encoding the
// explicit value requires 250.1. Note: MLIR's contiguous-optional-group parsing
// forces optionalAttr to be set as well; either non-fallback parameter alone
// would suffice to bump the version, so this test does not isolate
// optionalType from optionalAttr.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_type_optional_fallback_diff_250_1 {
  entry @kernel(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32, zero, f16>) {
    return
  }
}

// -----

// A pointer type without a ptr_attr classification remains a 13.1 feature: the
// optional attribute is absent, so it must not raise the minimum version.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @plain_pointer_13_1 {
  entry @kernel(%ptr: tile<ptr<i32>>) {
    return
  }
}

// -----

// Pointer classifications (#cuda_tile.ptr_attr) were introduced in 13.4, but the
// unicast and multicast enum cases were added in 13.5. The classification is
// an attribute nested in the type, so it has to be scored. The other carrier and
// enum case are @multicast_tensor_view_13_5 below.
// CHECK-EMPTY:
// CHECK-NEXT: 13.5
cuda_tile.module @unicast_pointer_13_5 {
  entry @kernel(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<unicast>>>) {
    return
  }
}

// -----

// A concrete TileType uses its public fallback and remains a 13.1 type.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @concrete_tile_13_1 {
  entry @kernel() {
    %value = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    return
  }
}

// -----

// The other carrier and enum case: a classification nested in a tensor_view.
// CHECK-EMPTY:
// CHECK-NEXT: 13.5
cuda_tile.module @multicast_tensor_view_13_5 {
  entry @kernel(
      %view: tensor_view<16xf32, strides=[1], #cuda_tile.ptr_attr<multicast>>) {
    return
  }
}

// -----

// Integration case: the multimem op and the multicast classification it needs
// are both 13.5, so this pins a realistic module rather than the op alone.
// CHECK-EMPTY:
// CHECK-NEXT: 13.5
cuda_tile.module @multimem_store_13_5 {
  entry @kernel(%ptr: tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>) {
    %tv = make_tensor_view %ptr, shape = [8], strides = [1]
        : tile<ptr<i32, #cuda_tile.ptr_attr<multicast>>>
        -> tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>
    %view = make_partition_view %tv
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>
    %idx = constant <i32: 0> : tile<i32>
    %value = constant <i32: 1> : tile<4xi32>
    %token = multimem_store_view_tko weak %view, %value[%idx]
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<multicast>>>, tile<4xi32>, tile<i32>
        -> token
    return
  }
}

// -----

// A true load-view inbounds bit is information introduced in 13.4.
// CHECK-EMPTY:
// CHECK-NEXT: 13.4
cuda_tile.module @view_inbounds_13_4 {
  entry @kernel(%ptr: tile<ptr<i32>>) {
    %tv = make_tensor_view %ptr, shape = [8], strides = [1]
        : tensor_view<8xi32, strides=[1]>
    %view = make_partition_view %tv
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1]>>
    %idx = constant <i32: 0> : tile<i32>
    %value, %token = load_view_tko weak %view[%idx] inbounds = [true]
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1]>>, tile<i32>
        -> tile<4xi32>, token
    return
  }
}

// -----

// Complement: an all-`false` vector is what a pre-13.4 reader reconstructs by
// itself, so the writer leaves it out and the module stays at 13.1.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @view_inbounds_default_13_1 {
  entry @kernel(%ptr: tile<ptr<i32>>) {
    %tv = make_tensor_view %ptr, shape = [8], strides = [1]
        : tensor_view<8xi32, strides=[1]>
    %view = make_partition_view %tv
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1]>>
    %idx = constant <i32: 0> : tile<i32>
    %value, %token = load_view_tko weak %view[%idx] inbounds = [false]
        : partition_view<tile=(4), tensor_view<8xi32, strides=[1]>>, tile<i32>
        -> tile<4xi32>, token
    return
  }
}

// -----

// The `rounding_mode` attribute was added to tanh in 13.2 (op itself is 13.1).
// At its default the writer omits it, so it must not raise the minimum.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @tanh_default_rounding_13_1 {
  entry @kernel() {
    %c = constant <f32: 1.0> : tile<f32>
    %r = tanh %c : tile<f32>
    return
  }
}

// -----

// A non-default rounding mode has to be encoded, which needs 13.2.
// CHECK-EMPTY:
// CHECK-NEXT: 13.2
cuda_tile.module @tanh_approx_rounding_13_2 {
  entry @kernel() {
    %c = constant <f32: 1.0> : tile<f32>
    %r = tanh %c rounding<approx> : tile<f32>
    return
  }
}

// -----

// `symbol_visibility` on a global was added in 13.3, and `public` is what a
// pre-13.3 reader reconstructs, so a public global stays at 13.1.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @global_public_visibility_13_1 {
  global @g <i32: [42]> : tile<1xi32>
  entry @kernel() {
    return
  }
}

// -----

// Any other visibility has to be encoded, which needs 13.3; below that the
// writer refuses the module.
// CHECK-EMPTY:
// CHECK-NEXT: 13.3
cuda_tile.module @global_private_visibility_13_3 {
  global private @g <i32: [42]> : tile<1xi32>
  entry @kernel() {
    return
  }
}

// -----

// The `constant` flag shares the 13.3 global fields, and the writer likewise
// refuses to drop it, so it constrains the module the same way.
// CHECK-EMPTY:
// CHECK-NEXT: 13.3
cuda_tile.module @global_constant_13_3 {
  global constant @g <i32: [42]> : tile<1xi32>
  entry @kernel() {
    return
  }
}

// -----

// The new_param attribute (250.1) spelled at its default value: the writer
// elides it, so the operation stays at its own version.
// CHECK-EMPTY:
// CHECK-NEXT: 250.0
cuda_tile.module @new_attribute_default_param_250_0 {
  entry @kernel() {
    testing$bytecode_test_new_attribute new_param = 42
    return
  }
}

// -----

// A type reached only through an attribute parameter still has to be scored:
// dependentType differs from its fallback, which needs 250.1.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @type_nested_in_attr_250_1 {
  entry @kernel() {
    testing$bytecode_test_new_attribute opt_type =
        #cuda_tile.bytecode_test_optional_type<1
            elem = !cuda_tile.testing$bytecode_test_evolved<f32, i64>>
    return
  }
}

// -----

// Types nested in a serialized attribute do contribute. The entry argument is
// a 13.1 type; the only 250.1 content is below arg_attrs' ArrayAttr and
// DictionaryAttr wrappers, mirroring types nested below extern const_params.
// CHECK-EMPTY:
// CHECK-NEXT: 250.1
cuda_tile.module @evolution_nested_attr_type_250_1 {
  entry @kernel(%arg0: tile<i32> {
      cuda_tile.test_type = !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32, zero, f16>}) {
    return
  }
}

// -----

// Discardable operation attributes are not part of the CUDA Tile bytecode and
// therefore do not contribute to its minimum version.
// CHECK-EMPTY:
// CHECK-NEXT: 13.1
cuda_tile.module @discardable_type_attr_13_1 {
  entry @kernel() {
    %token = make_token {
      cuda_tile.test_type = !cuda_tile.testing$bytecode_test_evolved<f32, f32, i32, zero, f16>
    } : token
    return
  }
}

