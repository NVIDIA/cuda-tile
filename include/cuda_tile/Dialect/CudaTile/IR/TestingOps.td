//===- TestingOps.td - CUDA Tile Testing Operations --------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// These operations are used for testing bytecode compatibility across versions.
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

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGOPS_TD
