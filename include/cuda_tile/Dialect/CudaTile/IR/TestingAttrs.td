//===- TestingAttrs.td - CUDA Tile Testing Attributes ------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These attributes are used for testing bytecode versioning.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// Testing Public-Preview Attribute
//===----------------------------------------------------------------------===//

def CudaTile_PreviewBytecodeTestAttr
    : CudaTilePreviewAttrDef<"PreviewBytecodeTest", "bytecode_test_attr",
                             CudaTile_TestOnlyPreviewFeature> {
  let summary = "Testing attribute for public-preview bytecode";
  let description = [{
    Exercises public-preview attribute serialization on a public-preview
    testing operation.
  }];
  let parameters = (ins
    CudaTilePreviewAttrParam<TypeParameter<"int64_t", "test value">>:$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Testing Enum Attribute - for enum versioning tests
//===----------------------------------------------------------------------===//

// Testing enum for bytecode versioning.
def CudaTile_BytecodeTestPriority
    : CudaTileI32EnumAttr<"BytecodeTestPriority", "test priority", "250.0",
        [CudaTileI32EnumAttrCase<"Low", "Low", 0, "250.0", "low">,
         CudaTileI32EnumAttrCase<"Normal", "Normal", 1, "250.0", "normal">,
         CudaTileI32EnumAttrCase<"High", "High", 2, "250.1", "high">]> {
  let cppNamespace = "::mlir::cuda_tile";
  let genSpecializedAttr = 0;
}

def CudaTile_BytecodeTestPriorityAttr
    : CudaTileEnumAttr<CudaTile_BytecodeTestPriority, "bytecode_test_priority"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Testing Evolved Attribute - Attr that evolved from 250.0 to 250.1
//===----------------------------------------------------------------------===//

// Mirrors BytecodeTestEvolvedType for types.
def CudaTile_BytecodeTestEvolvedAttr : CudaTileAttrDef<"BytecodeTestEvolved",
    "bytecode_test_evolved_attr", "250.0"> {
  let summary = "Testing attribute for parameter evolution (250.0 -> 250.1)";
  let description = [{
    Attribute that evolved across versions:
    - Version 250.0: Only has `baseValue` (int64_t)
    - Version 250.1: Added `optionalValue` (std::optional<int64_t>)
    - Version 250.1: Added `dependentValue` (int64_t with bytecodeFallbackValue
      referencing baseValue via $_param_ substitution)
  }];
  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"int64_t", "base value">, "250.0">:$baseValue,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>",
        "optional value added in 250.1">, "250.1">:$optionalValue,
    CudaTileAttrParam<TypeParameter<"int64_t", "dependent value">, "250.1",
        "$_param_baseValue">:$dependentValue);
  let assemblyFormat = "`<` $baseValue `,` $dependentValue (`,` $optionalValue^)? `>`";
}

//===----------------------------------------------------------------------===//
// Testing Self-Type Attribute - exercises a raw Type attribute parameter
//===----------------------------------------------------------------------===//

// Attribute carrying a self type (MLIR's AttributeSelfTypeParameter, used to
// implement TypedAttrInterface). The self type is serialized through the
// bytecode type table (Kind::Type). Mnemonic prints the scalar payload.
def CudaTile_BytecodeTestSelfTypeAttr : CudaTileAttrDef<"BytecodeTestSelfType",
    "bytecode_test_self_type", "250.0", [TypedAttrInterface]> {
  let summary = "Testing attribute with a self type (raw Type parameter)";
  let description = [{
    Attribute that stores a self type to exercise raw `mlir::Type` attribute
    parameter bytecode serialization:
    - `value`: an i64 scalar payload.
    - `type`: the attribute's self type, round-tripped via the type table.
  }];
  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"int64_t", "scalar payload">, "250.0">:$value,
    CudaTileAttrSelfTypeParam<"250.0">:$type);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Testing Optional-Type Attribute - exercises optional raw Type parameters
//===----------------------------------------------------------------------===//

// Attribute with optional raw Type parameters, exercising optional `mlir::Type`
// bytecode serialization (presence tracked by null-ness, not std::optional).
// `elem` is a bare `::mlir::Type` (generic read path); `int_type` is a concrete
// `::mlir::IntegerType` (narrowed via dyn_cast on read).
def CudaTile_BytecodeTestOptionalTypeAttr
    : CudaTileAttrDef<"BytecodeTestOptionalType", "bytecode_test_optional_type",
                      "250.0"> {
  let summary = "Testing attribute with optional raw Type parameters";
  let description = [{
    Attribute exercising optional `mlir::Type` parameter bytecode serialization:
    - `value`: an i64 payload.
    - `elem`: an optional bare `::mlir::Type` (generic Type read path).
    - `intType`: an optional concrete `::mlir::IntegerType` (narrowed on read).
  }];
  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"int64_t", "payload">, "250.0">:$value,
    CudaTileAttrParam<OptionalParameter<"::mlir::Type", "optional element type">, "250.0">:$elem,
    CudaTileAttrParam<OptionalParameter<"::mlir::IntegerType", "optional integer type">, "250.0">:$intType);
  let assemblyFormat = "`<` $value (`elem` `=` $elem^)? (`int_type` `=` $intType^)? `>`";
}

//===----------------------------------------------------------------------===//
// Testing Sandwiched Optional Attribute - evolved optional in between two
// non-evolved optionals.
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTestSandwichedOptionalAttr : CudaTileAttrDef<
    "BytecodeTestSandwichedOptional",
    "bytecode_test_sandwiched_optional_attr", "250.0"> {
  let summary = "Tests evolved optional sandwiched between non-evolved optionals";
  let description = [{
    Three optional parameters where the evolved one sits between two
    non-evolved ones:
    - Version 250.0: `before` (OptionalParameter, non-evolved)
    - Version 250.1: `middle` (OptionalParameter, evolved — added in 250.1)
    - Version 250.0: `after`  (OptionalParameter, non-evolved)

    Verifies that the bitfield encoding for OptionalParameter correctly
    interleaves evolved and non-evolved optionals across bytecode versions.
  }];
  let parameters = (ins
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>",
        "before">, "250.0">:$before,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>",
        "middle">, "250.1">:$middle,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>",
        "after">, "250.0">:$after);
  let assemblyFormat = "`<` struct($before, $middle, $after) `>`";
}

//===----------------------------------------------------------------------===//
// Testing Evolved ArrayRef Attribute - reproduces lifetime issues for evolved
// ArrayRef parameters.
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTestEvolvedArrayRefAttr : CudaTileAttrDef<
    "BytecodeTestEvolvedArrayRef",
    "bytecode_test_evolved_arrayref_attr", "250.0"> {
  let summary = "Testing attribute for evolved ArrayRef parameter lifetime";
  let description = [{
    Attribute with an evolved ArrayRef<Attribute> parameter followed by another
    ArrayRef<Attribute> parameter. The later parameter is intentionally used to
    pressure allocator reuse after deserializing the evolved parameter.
  }];
  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"int64_t", "base value">,
        "250.0">:$baseValue,
    CudaTileAttrParam<ArrayRefParameter<"::mlir::Attribute",
        "values added in 250.1">, "250.1",
        "::llvm::ArrayRef<::mlir::Attribute>()">:$values,
    CudaTileAttrParam<ArrayRefParameter<"::mlir::Attribute",
        "later values used to pressure allocator reuse">,
        "250.0">:$clobber);
  let assemblyFormat = "`<` $baseValue `,` `[` $values `]` `,` `[` $clobber `]` `>`";
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD
