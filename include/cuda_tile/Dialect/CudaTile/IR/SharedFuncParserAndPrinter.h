//===- SharedFuncParserAndPrinter.h - CUDA Tile Printer/Parser --*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDFUNCPARSERANDPRINTER
#define CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDFUNCPARSERANDPRINTER

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"

namespace mlir::cuda_tile {

template <typename OpTy>
ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature using custom parsing that supports both
  // short form (tile<ptr<f32>>) and long form (!cuda_tile.tile<ptr<f32>>) types
  // within cuda_tile.module operations via OpAsmOpInterface default dialect
  // context.
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;

  // Use our custom parsing function instead of the standard MLIR
  // function_interface_impl to enable proper cuda_tile dialect type resolution
  // in function signatures.
  if (parseFunctionSignatureWithArguments(parser, /*allowVariadic=*/false,
                                          entryArgs, isVariadic, resultTypes,
                                          resultAttrs))
    return failure();

  SmallVector<Type> argTypes = llvm::to_vector(llvm::map_range(
      entryArgs, [](OpAsmParser::Argument arg) -> Type { return arg.type; }));
  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(OpTy::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  SmallVector<Attribute> argAttrs = llvm::to_vector(
      llvm::map_range(entryArgs, [](OpAsmParser::Argument arg) -> Attribute {
        return arg.attrs;
      }));
  result.addAttribute(OpTy::getArgAttrsAttrName(result.name),
                      ArrayAttr::get(parser.getContext(), argAttrs));

  // Parse the function body.
  Region *body = result.addRegion();
  ParseResult parseResult = parser.parseRegion(*body, entryArgs,
                                               /*enableNameShadowing=*/false);
  if (failed(parseResult))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  OpTy::ensureTerminator(*body, builder, result.location);

  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();

  return success();
}

template <typename OpTy>
void printFuncOp(OpTy op, OpAsmPrinter &printer) {
  // Print the operation and the function name.
  auto funcName =
      op.getOperation()
          ->template getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  printer << ' ';
  printer.printSymbolName(funcName);
  auto fnType = op.getFunctionType();
  printFunctionSignatureWithCudaTileTypes(printer, op, fnType.getInputs(),
                                          fnType.getResults());
  printer << ' ';
  printer.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true,
                      /*printEmptyBlock=*/false);
  printer.printOptionalAttrDict(
      op.getOperation()->getAttrs(),
      {op.getArgAttrsAttrName(), op.getFunctionTypeAttrName(),
       SymbolTable::getSymbolAttrName(), op.getResAttrsAttrName()});
}

} // end namespace mlir::cuda_tile.

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDFUNCPARSERANDPRINTER
