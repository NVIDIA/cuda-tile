//===- TestingOps.td - CUDA Tile Testing Operations --------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These operations are used for testing bytecode compatibility across versions.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_TESTINGOPS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TESTINGOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Types.td"
include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"
include "cuda_tile/Dialect/CudaTile/IR/TestingAttrs.td"

//===----------------------------------------------------------------------===//
// Testing Operations - Only available when TILE_IR_INCLUDE_TESTS is defined
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test_FuncOp
//===----------------------------------------------------------------------===//

def CudaTile_Test_FuncOp : CudaTileTestingOpDef<"func", "250.0", [
  IsolatedFromAbove, FunctionOpInterface, SingleBlock, OpAsmOpInterface,
  SingleBlockImplicitTerminator<"ReturnOp">, SymbolName, SymbolVisibility
]> {

  let arguments = (ins 
    CudaTileArg<SymbolNameAttr, "The name of the function.", "250.0">:$sym_name,
    CudaTileArg<TypeAttrOf<FunctionType>, "The type of the function.", "250.0">:$function_type,
    CudaTileArg<OptionalAttr<StrAttr>, "The visibility of the function.", "250.0">:$sym_visibility,
    CudaTileUnusedArg<OptionalAttr<DictArrayAttr>, "The argument attributes of the function.", "250.0">:$arg_attrs,
    CudaTileUnusedArg<OptionalAttr<DictArrayAttr>, "The result attributes of the function.", "250.0">:$res_attrs);
  
  let regions = (region SizedRegion<1>:$body);
  
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = CudaTile_DefaultDialect.classDecl # [{
    // FunctionOpInterface Methods

    /// Returns the region on the current operation
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

//===----------------------------------------------------------------------===//
// BytecodeTest_NewAttributeOp
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTest_NewAttributeOp : CudaTileTestingOpDef<"bytecode_test_new_attribute", "250.0"> {
  let summary = "Testing operation for bytecode new attribute versioning";
  let description = [{
    The :code:`bytecode_test_new_attribute` operation tests bytecode versioning when adding
    new attributes to existing operations.
  }];

  let arguments = (ins
    CudaTileArg<UnitAttr, "New UnitAttr flag added in version 250.1 for testing.", "250.1">:$new_flag,
    CudaTileArg<DefaultValuedAttr<I32Attr, "42">, "New parameter with default value added in version 250.1.", "250.1">:$new_param,
    CudaTileArg<OptionalAttr<CudaTile_BytecodeTestSelfTypeAttr>,
                "Self-type attr for exercising raw Type parameter bytecode.", "250.0">:$self_type_attr,
    CudaTileArg<OptionalAttr<CudaTile_BytecodeTestOptionalTypeAttr>,
                "Optional-Type attr for exercising optional raw Type parameter bytecode.", "250.0">:$opt_type_attr);

  let assemblyFormat = [{
    (`new_flag` $new_flag^)?
    (`new_param` `=` $new_param^)?
    (`self_type` `=` $self_type_attr^)?
    (`opt_type` `=` $opt_type_attr^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BytecodeTest_PreviewTypeOp
//===----------------------------------------------------------------------===//

// This is intentionally a stable testing operation: it verifies that direct
// and nested preview types independently trigger named preview policy.
def CudaTile_BytecodeTest_PreviewTypeOp
    : CudaTileTestingOpDef<"bytecode_test_preview_type", "13.5"> {
  let summary = "Testing operation carrying public-preview types";
  let results = (outs
    CudaTileArg<AnyType, "The type being exercised.", "13.5">:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// This intentionally stable operation verifies that an explicit preview enum
// attribute independently triggers named preview policy.
def CudaTile_BytecodeTest_MemorySpaceAttrOp
    : CudaTileTestingOpDef<"bytecode_test_memory_space_attr", "13.5"> {
  let summary = "Testing operation carrying a preview memory-space attribute";
  let arguments = (ins
    CudaTileArg<CudaTile_MemorySpaceAttr, "Preview memory space.",
                "13.5">:$memory_space);
}

//===----------------------------------------------------------------------===//
// BytecodeEvolutionTestOp
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTest_EvolutionOp :
    CudaTileTestingOpDef<"bytecode_test_evolution", "250.0", [AttrSizedOperandSegments]> {
  let summary = "Tests bytecode compatibility across operation evolution.";
  let description = [{
    The :code:`bytecode_evolution_test` operation tests bytecode versioning
    and backward compatibility when operations evolve by adding new optional
    operands, results, attributes, enum attrs, and standalone attrs across
    different bytecode versions.
  }];

  let arguments = (ins
      CudaTileArg<Variadic<CudaTile_TileType>, "Base input from version 250.0.", "250.0">:$inputs,
      CudaTileArg<OptionalAttr<I32Attr>, "Optional attribute added in 250.1.", "250.1">:$new_attr,
      CudaTileArg<Optional<CudaTile_TokenType>,
                  "Optional token added in version 250.1.", "250.1">:$optional_token,
      CudaTileArg<OptionalAttr<CudaTile_BytecodeTestPriorityAttr>,
                  "Priority enum attr added in 250.0.", "250.0">:$priority,
      CudaTileArg<OptionalAttr<CudaTile_BytecodeTestEvolvedAttr>,
                  "Evolved standalone attr available from 250.0.", "250.0">:$test_attr,
      CudaTileArg<OptionalAttr<CudaTile_BytecodeTestSandwichedOptionalAttr>,
                  "Sandwiched optional attr available from 250.0.", "250.0">:$sandwich_attr,
      CudaTileArg<OptionalAttr<CudaTile_BytecodeTestEvolvedArrayRefAttr>,
                  "Evolved ArrayRef attr available from 250.0.", "250.0">:$arrayref_attr);

  let results = (outs
      CudaTileArg<CudaTile_TokenType, "New token result added in version 250.1.", "250.1">:$result_token);

  let assemblyFormat = [{
    `(` $inputs `:` type($inputs) `)`
    (`new_attr` `=` $new_attr^)?
    (`token` `=` $optional_token^ `:` custom<CudaTileType>(type($optional_token)))?
    (`priority` `=` $priority^)?
    (`test_attr` `=` $test_attr^)?
    (`sandwich` `=` $sandwich_attr^)?
    (`arrayref` `=` $arrayref_attr^)?
    `->` custom<CudaTileType>(type($result_token))
    attr-dict
  }];
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGOPS_TD
