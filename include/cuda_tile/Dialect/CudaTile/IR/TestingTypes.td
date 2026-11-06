//===- TestingTypes.td -------------------------------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_TESTINGTYPES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TESTINGTYPES_TD

include "mlir/IR/OpBase.td"
include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// Testing Types - Only available when TILE_IR_INCLUDE_TESTS is defined
//===----------------------------------------------------------------------===//

// A public-preview type and an ordinary stable container used to verify both
// direct and recursively nested preview-type bytecode policy.
def CudaTile_PreviewBytecodeTestType : CudaTilePreviewTypeDef<
    "PreviewBytecodeTest", "bytecode_test_type", "previewBytecodeTestType",
    CudaTile_TestOnlyPreviewFeature> {
  let summary = "Testing type for public-preview bytecode";
  let parameters = (ins
    CudaTilePreviewTypeParam<TypeParameter<"::mlir::Type", "nested type">>:$type);
  let assemblyFormat = "`<` $type `>`";
}

def CudaTile_BytecodeTestTypeContainer : CudaTileTypeDef<
    "BytecodeTestTypeContainer", "bytecode_test_type_container",
    "bytecodeTestTypeContainer", "13.5"> {
  let summary = "Stable testing container for a nested preview type";
  let parameters = (ins
    CudaTileTypeParam<TypeParameter<"::mlir::Type", "nested type">,
                      "13.5">:$nestedType);
  let assemblyFormat = "`<` $nestedType `>`";
}

// Testing type definition helper.
class CudaTileTestingTypeDef<string name, string mnemonic, string specName, 
                              string version = "250.0", list<Trait> traits = []>
  : CudaTileTypeDef<name, "testing$" # mnemonic, specName, version, traits>;

//===----------------------------------------------------------------------===//
// BytecodeTest_EvolvedType - Type that evolved from 250.0 to 250.1
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTest_EvolvedType : CudaTileTestingTypeDef<
    "BytecodeTestEvolved", "bytecode_test_evolved", "bytecode_test_evolved"> {
  let summary = "Testing type for parameter evolution (250.0 -> 250.1)";
  let description = [{
    Type that evolved across versions:
    - Version 250.0: Only has `baseType`
    - Version 250.1: Added `newType` (DefaultValuedParameter with explicit default i32)
    - Version 250.1: Added `optionalAttr` (OptionalParameter for attribute)
    - Version 250.1: Added `optionalType` (OptionalParameter for Type)
    - Version 250.1: Added `dependentType` (CudaTileEvolvedTypeParam with
      bytecodeFallbackValue referencing baseType via $_param_ substitution)
  }];

  let parameters = (ins
    CudaTileConstrainedTypeParam<CudaTile_NumberType, "250.0">:$baseType,
    CudaTileTypeParam<DefaultValuedParameter<"::mlir::Type", "::mlir::IntegerType::get($_ctxt, 32)", "new type defaults to i32">, "250.1">:$newType,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "optional attribute">, "250.1">:$optionalAttr,
    CudaTileTypeParam<OptionalParameter<"::mlir::Type", "optional type parameter">, "250.1">:$optionalType,
    CudaTileTypeParam<TypeParameter<"::mlir::Type", "dependent type">, "250.1", "$_param_baseType">:$dependentType
  );

  let mnemonic = "testing$bytecode_test_evolved";
  let assemblyFormat = "`<` $baseType `,` $dependentType (`,` $newType^)? (`,` $optionalAttr^)? (`,` $optionalType^)? `>`";
}

//===----------------------------------------------------------------------===//
// BytecodeTest_SandwichedOptionalType - evolved optional between stable optionals
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTest_SandwichedOptionalType : CudaTileTestingTypeDef<
    "BytecodeTestSandwichedOptional", "bytecode_test_sandwiched_optional",
    "bytecode_test_sandwiched_optional"> {
  let summary = "Testing type for sandwiched optional parameter evolution";
  let description = [{
    Type with three optional type parameters where the evolved one sits between
    two non-evolved ones:
    - Version 250.0: `before` (OptionalParameter)
    - Version 250.1: `middle` (OptionalParameter)
    - Version 250.0: `after`  (OptionalParameter)

    Verifies that the type bytecode bitfield encoding for OptionalParameter
    correctly interleaves evolved and non-evolved optionals across bytecode
    versions.
  }];

  let parameters = (ins
    CudaTileTypeParam<OptionalParameter<"::mlir::Type", "before">, "250.0">:$before,
    CudaTileTypeParam<OptionalParameter<"::mlir::Type", "middle">, "250.1">:$middle,
    CudaTileTypeParam<OptionalParameter<"::mlir::Type", "after">, "250.0">:$after
  );

  let mnemonic = "testing$bytecode_test_sandwiched_optional";
  let assemblyFormat = "`<` struct($before, $middle, $after) `>`";
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGTYPES_TD
