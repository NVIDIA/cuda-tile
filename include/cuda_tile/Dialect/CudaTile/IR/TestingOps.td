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

//===----------------------------------------------------------------------===//
// Testing Operations - Only available when TILE_IR_INCLUDE_TESTS is defined
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test_FuncOp
//===----------------------------------------------------------------------===//

def CudaTile_Test_FuncOp : CudaTileTestingOpDef<"func", "250.0", [
  IsolatedFromAbove, FunctionOpInterface, SingleBlock, OpAsmOpInterface, 
  SingleBlockImplicitTerminator<"ReturnOp">
]> {

  let arguments = (ins 
    CudaTileArg<SymbolNameAttr, "The name of the function.", "250.0">:$sym_name,
    CudaTileArg<TypeAttrOf<FunctionType>, "The type of the function.", "250.0">:$function_type,
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
    CudaTileArg<DefaultValuedAttr<I32Attr, "42">, "New parameter with default value added in version 250.1.", "250.1">:$new_param);

  let assemblyFormat = [{
    (`new_flag` $new_flag^)?
    (`new_param` `=` $new_param^)?
    attr-dict
  }];
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
    operands, results, and attributes across different bytecode versions.
  }];

  let arguments = (ins
      CudaTileArg<Variadic<CudaTile_TileType>, "Base input from version 250.0.", "250.0">:$inputs,
      CudaTileArg<OptionalAttr<I32Attr>, "Optional attribute added in 250.1 to test bit layout compatibility.", "250.1">:$new_attr,
      CudaTileArg<Optional<CudaTile_TokenType>,
                  "Optional token added in version 250.1.", "250.1">:$optional_token);

  let results = (outs
      CudaTileArg<CudaTile_TokenType, "New token result added in version 250.1.", "250.1">:$result_token);

  let assemblyFormat = [{
    `(` $inputs `:` type($inputs) `)`
    (`new_attr` `=` $new_attr^)?
    (`token` `=` $optional_token^ `:` custom<CudaTileType>(type($optional_token)))?
    `->` custom<CudaTileType>(type($result_token))
    attr-dict
  }];
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGOPS_TD
