//===- SharedFuncParserAndPrinter.h - CUDA Tile Printer/Parser --*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
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

#include <type_traits>

namespace mlir::cuda_tile {

void printCompactConstParams(OpAsmPrinter &printer, Operation *op,
                             ArrayAttr params);
ParseResult parseCompactConstParams(OpAsmParser &parser, ArrayAttr &result);

void printCompactConstArgs(OpAsmPrinter &printer, Operation *op,
                           ArrayAttr args);
ParseResult parseCompactConstArgs(OpAsmParser &parser, ArrayAttr &result);
void printOptionalCompactConstArgs(OpAsmPrinter &printer, Operation *op,
                                   ArrayAttr args);
ParseResult parseOptionalCompactConstArgs(OpAsmParser &parser,
                                          ArrayAttr &result);

template <typename OpTy, typename = void>
struct HasConstParamsAttrName : std::false_type {};

// TODO: Once this helper only serves the extension function op, remove this
// detection trait and handle const_params directly. It exists today because
// the test FuncOp shares this parser/printer but does not define const_params.
template <typename OpTy>
struct HasConstParamsAttrName<
    OpTy,
    std::void_t<decltype(OpTy::getConstParamsAttrName(
        std::declval<::mlir::OperationName>()))>> : std::true_type {};

static void setPrivateVisibilityIfAbsent(OpAsmParser &parser,
                                         OperationState &result) {
  if (!result.attributes.get(
          SymbolOpInterface::getDefaultVisibilityAttrName())) {
    // These bodiless external function symbols are declarations, and MLIR
    // rejects public visibility on declarations. Keep the surface syntax
    // unchanged for now by defaulting the stored visibility to private.
    result.addAttribute(SymbolOpInterface::getDefaultVisibilityAttrName(),
                        parser.getBuilder().getStringAttr("private"));
  }
}

template <typename OpTy>
ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  bool isExtern = false;
  if (succeeded(parser.parseOptionalKeyword("extern"))) {
    isExtern = true;
    result.addAttribute(builder.getStringAttr("is_extern"),
                        builder.getUnitAttr());
  }

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, OpTy::getSymNameAttrName(result.name),
                             result.attributes)) {
    return failure();
  }

  if constexpr (HasConstParamsAttrName<OpTy>::value) {
    ArrayAttr constParams = builder.getArrayAttr({});
    if (isExtern && succeeded(parser.parseOptionalLess())) {
      if (parseCompactConstParams(parser, constParams) ||
          parser.parseGreater()) {
        return failure();
      }
    }
    result.addAttribute(OpTy::getConstParamsAttrName(result.name),
                        constParams);
  }

  // Parse the function signature using custom parsing that supports both
  // short form (tile<ptr<f32>>) and long form (!cuda_tile.tile<ptr<f32>>) types
  // within cuda_tile.module operations via OpAsmOpInterface default dialect
  // context.
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
  if (ArrayAttr normalizedArgAttrs =
          buildNormalizedArgAttrs(builder, argAttrs)) {
    result.addAttribute(OpTy::getArgAttrsAttrName(result.name),
                        normalizedArgAttrs);
  }

  SmallVector<Attribute> resultAttrStorage(resultAttrs.begin(),
                                           resultAttrs.end());
  if (ArrayAttr normalizedResultAttrs =
          buildNormalizedArgAttrs(builder, resultAttrStorage)) {
    result.addAttribute(OpTy::getResAttrsAttrName(result.name),
                        normalizedResultAttrs);
  }

  Region *body = result.addRegion();
  if (isExtern) {
    setPrivateVisibilityIfAbsent(parser, result);
    return success();
  }

  // Parse the function body.
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
  bool isExtern = op.getOperation()->hasAttr("is_extern");

  // Print the operation and the function name.
  auto funcName = op.getSymName();
  printer << ' ';
  if (isExtern) {
    printer << "extern ";
  }
  printer.printSymbolName(funcName);
  if constexpr (HasConstParamsAttrName<OpTy>::value) {
    if (isExtern) {
      ArrayAttr params = op.getConstParams();
      if (!params.empty()) {
        printer << "<";
        printCompactConstParams(printer, op.getOperation(), params);
        printer << ">";
      }
    }
  }
  auto fnType = op.getFunctionType();
  printFunctionSignatureWithCudaTileTypes(printer, op, fnType.getInputs(),
                                          fnType.getResults());
  printer << ' ';
  if (isExtern) {
    return;
  }
  printer.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true,
                      /*printEmptyBlock=*/false);
  SmallVector<StringRef> elidedAttrs = {
      op.getArgAttrsAttrName(), op.getFunctionTypeAttrName(),
      op.getSymNameAttrName(), op.getResAttrsAttrName()};
  if constexpr (HasConstParamsAttrName<OpTy>::value) {
    elidedAttrs.push_back(op.getConstParamsAttrName());
  }
  printer.printOptionalAttrDict(op.getOperation()->getAttrs(), elidedAttrs);
}

} // end namespace mlir::cuda_tile.

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDFUNCPARSERANDPRINTER
