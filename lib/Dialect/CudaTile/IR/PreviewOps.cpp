//===- PreviewOps.cpp - CUDA Tile public-preview operations -----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/ErrorHandling.h"

#include "cuda_tile/Dialect/CudaTile/IR/CompactLayoutParser.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedFuncParserAndPrinter.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h"

using namespace mlir;
using namespace mlir::cuda_tile;

constexpr llvm::StringLiteral kExternImplScratchKeyword = "scratch";
constexpr llvm::StringLiteral kScratchSmemSizeKey = "smem.size";
constexpr llvm::StringLiteral kScratchSmemAlignmentKey = "smem.alignment";
constexpr int64_t kDefaultScratchSmemAlignment = 128;

// The expression and tuple printers are mutually recursive: tuples contain
// expression elements, and expression positions can contain nested tuples.
static void
printCompactConstArg(OpAsmPrinter &printer, SymbolicExprAttr expr,
                     CompactExprPrecedence parentPrec =
                         CompactExprPrecedence::Additive,
                     bool isRight = false);

static CompactExprOperator getCompactConstArgOperator(SymbolicExprAttr expr) {
  if (llvm::isa<SymbolicExprAddAttr>(expr))
    return CompactExprOperator::Add;
  if (llvm::isa<SymbolicExprNegAttr>(expr))
    return CompactExprOperator::Neg;
  if (llvm::isa<SymbolicExprSubAttr>(expr))
    return CompactExprOperator::Sub;
  if (llvm::isa<SymbolicExprMulAttr>(expr))
    return CompactExprOperator::Mul;
  if (llvm::isa<SymbolicExprFloorDivAttr>(expr))
    return CompactExprOperator::FloorDiv;
  if (llvm::isa<SymbolicExprCeilDivAttr>(expr))
    return CompactExprOperator::CeilDiv;
  return CompactExprOperator::Primary;
}

static bool canPrintCompactConstArg(SymbolicExprAttr expr) {
  if (llvm::isa<SymbolicExprIntAttr, SymbolicExprConstStrAttr,
                SymbolicExprVarAttr, ConcreteTypeAttr>(expr))
    return true;
  if (auto tupleAttr = llvm::dyn_cast<SymbolicTupleAttr>(expr))
    return llvm::all_of(tupleAttr.getValues(), canPrintCompactConstArg);
  if (auto mulAttr = llvm::dyn_cast<SymbolicExprMulAttr>(expr))
    return canPrintCompactConstArg(mulAttr.getLhs()) &&
           canPrintCompactConstArg(mulAttr.getRhs());
  if (auto divAttr = llvm::dyn_cast<SymbolicExprFloorDivAttr>(expr))
    return canPrintCompactConstArg(divAttr.getLhs()) &&
           canPrintCompactConstArg(divAttr.getRhs());
  if (auto divAttr = llvm::dyn_cast<SymbolicExprCeilDivAttr>(expr))
    return canPrintCompactConstArg(divAttr.getLhs()) &&
           canPrintCompactConstArg(divAttr.getRhs());
  if (auto addAttr = llvm::dyn_cast<SymbolicExprAddAttr>(expr))
    return canPrintCompactConstArg(addAttr.getLhs()) &&
           canPrintCompactConstArg(addAttr.getRhs());
  if (auto negAttr = llvm::dyn_cast<SymbolicExprNegAttr>(expr))
    return canPrintCompactConstArg(negAttr.getOperand());
  if (auto subAttr = llvm::dyn_cast<SymbolicExprSubAttr>(expr))
    return canPrintCompactConstArg(subAttr.getLhs()) &&
           canPrintCompactConstArg(subAttr.getRhs());
  return false;
}

static void printCompactTupleConstArg(OpAsmPrinter &printer,
                                      SymbolicTupleAttr tupleAttr) {
  printer << "(";
  llvm::interleaveComma(
      tupleAttr.getValues(), printer.getStream(),
      [&](SymbolicExprAttr value) { printCompactConstArg(printer, value); });
  if (tupleAttr.getValues().size() == 1)
    printer << ",";
  printer << ")";
}

static void printCompactConstArg(OpAsmPrinter &printer, SymbolicExprAttr expr,
                                 CompactExprPrecedence parentPrec,
                                 bool isRight) {
  printCompactExprWithPrecedence(
      printer, getCompactConstArgOperator(expr), parentPrec, isRight,
      [&](CompactExprPrecedence prec) {
        if (auto tupleAttr = llvm::dyn_cast<SymbolicTupleAttr>(expr)) {
          printCompactTupleConstArg(printer, tupleAttr);
        } else if (auto intAttr = llvm::dyn_cast<SymbolicExprIntAttr>(expr)) {
          printer.getStream() << intAttr.getValue();
        } else if (auto varAttr = llvm::dyn_cast<SymbolicExprVarAttr>(expr)) {
          printer.getStream() << varAttr.getName();
        } else if (auto stringAttr =
                       llvm::dyn_cast<SymbolicExprConstStrAttr>(expr)) {
          printer << '"';
          llvm::printEscapedString(stringAttr.getValue(), printer.getStream());
          printer << '"';
        } else if (auto concreteAttr = llvm::dyn_cast<ConcreteTypeAttr>(expr)) {
          printer.printType(concreteAttr.getType().getValue());
        } else if (auto mulAttr = llvm::dyn_cast<SymbolicExprMulAttr>(expr)) {
          printCompactConstArg(printer, mulAttr.getLhs(), prec,
                               /*isRight=*/false);
          printer << "*";
          printCompactConstArg(printer, mulAttr.getRhs(), prec,
                               /*isRight=*/true);
        } else if (auto divAttr =
                       llvm::dyn_cast<SymbolicExprFloorDivAttr>(expr)) {
          printCompactConstArg(printer, divAttr.getLhs(), prec,
                               /*isRight=*/false);
          printer << "/";
          printCompactConstArg(printer, divAttr.getRhs(), prec,
                               /*isRight=*/true);
        } else if (auto divAttr =
                       llvm::dyn_cast<SymbolicExprCeilDivAttr>(expr)) {
          printCompactConstArg(printer, divAttr.getLhs(), prec,
                               /*isRight=*/false);
          printer << "/+";
          printCompactConstArg(printer, divAttr.getRhs(), prec,
                               /*isRight=*/true);
        } else if (auto addAttr = llvm::dyn_cast<SymbolicExprAddAttr>(expr)) {
          printCompactConstArg(printer, addAttr.getLhs(), prec,
                               /*isRight=*/false);
          printer << "+";
          printCompactConstArg(printer, addAttr.getRhs(), prec,
                               /*isRight=*/true);
        } else if (auto negAttr = llvm::dyn_cast<SymbolicExprNegAttr>(expr)) {
          printer << "-";
          printCompactConstArg(printer, negAttr.getOperand(), prec);
        } else if (auto subAttr = llvm::dyn_cast<SymbolicExprSubAttr>(expr)) {
          printCompactConstArg(printer, subAttr.getLhs(), prec,
                               /*isRight=*/false);
          printer << "-";
          printCompactConstArg(printer, subAttr.getRhs(), prec,
                               /*isRight=*/true);
        } else {
          llvm_unreachable("unsupported symbolic constant argument");
        }
      });
}

void mlir::cuda_tile::printCompactConstArgs(OpAsmPrinter &printer,
                                            Operation *op, ArrayAttr args) {
  if (!llvm::all_of(args, [](Attribute attr) {
        auto expr = llvm::dyn_cast<SymbolicExprAttr>(attr);
        return expr && canPrintCompactConstArg(expr);
      })) {
    printer.printAttribute(args);
    return;
  }

  printer << "[";
  llvm::interleaveComma(args, printer.getStream(), [&](Attribute attr) {
    printCompactConstArg(printer, llvm::cast<SymbolicExprAttr>(attr));
  });
  printer << "]";
}

ParseResult
mlir::cuda_tile::parseCompactConstArgs(OpAsmParser &parser, ArrayAttr &result) {
  auto startsVerboseList = [&]() {
    const char *curPtr = parser.getCurrentLocation().getPointer();
    if (!curPtr || *curPtr != '[')
      return false;
    for (++curPtr; *curPtr && (*curPtr == ' ' || *curPtr == '\t' ||
                               *curPtr == '\n' || *curPtr == '\r' ||
                               *curPtr == '\f');
         ++curPtr) {
    }
    return *curPtr == '#';
  };

  if (startsVerboseList()) {
    Attribute attr;
    SMLoc startLoc = parser.getCurrentLocation();
    if (parser.parseAttribute(attr))
      return failure();
    result = llvm::dyn_cast<ArrayAttr>(attr);
    if (!result) {
      return parser.emitError(startLoc,
                              "expected array attribute for constant arguments");
    }
    return success();
  }

  bool emittedDiagnostic = false;
  auto startLoc = parser.getCurrentLocation();
  CudaTileCompactLayoutParser compactParser(parser, parser.getContext(),
                                            startLoc, emittedDiagnostic);
  return compactParser.parseConstantArgumentList(result);
}

ParseResult mlir::cuda_tile::parseOptionalCompactConstArgs(
    OpAsmParser &parser, ArrayAttr &result) {
  result = parser.getBuilder().getArrayAttr({});
  if (failed(parser.parseOptionalLess()))
    return success();
  if (parseCompactConstArgs(parser, result) || parser.parseGreater())
    return failure();
  return success();
}

void mlir::cuda_tile::printOptionalCompactConstArgs(OpAsmPrinter &printer,
                                                    Operation *op,
                                                    ArrayAttr args) {
  if (args.empty())
    return;
  printer << "<";
  printCompactConstArgs(printer, op, args);
  printer << ">";
}

/// Classify a constant parameter for const_arg validation.
/// Returns true when a ConstParamAttr entry in an extern_impl's const_params
/// is a wildcard placeholder (name == "?"). Wildcards mark positions that
/// the impl does not specialize; the corresponding const_args entry still
/// binds the extern function's parameter but the impl declares no local
/// symbolic variable for it.
static bool isWildcardConstParam(ConstParamAttr cp) {
  return cp.getName() == "?";
}

/// Verify an extern_impl's const_args as bindings to the implemented extern
/// function.
///
/// SIMT implementations pass their own const_params one-to-one to the function
/// because monomorphization does not provide a separate implementation
/// evaluation context.
static LogicalResult verifyImplConstArgKinds(ExternImplOp op, FuncOp fn) {
  auto args = op.getConstArgs();
  auto fnParams = fn.getConstParams();
  if (!args) {
    return success();
  }

  MLIRContext *ctx = op.getContext();
  ArrayAttr implParams =
      op.getConstParams().value_or(ArrayAttr::get(ctx, {}));

  if (implParams.size() != fnParams.size()) {
    return op.emitOpError("impl const_param count (")
           << implParams.size()
           << ") must match the implemented function's const_param count ("
           << fnParams.size() << ")";
  }

  bool hadError = false;
  for (auto [i, fnParamAttr] : llvm::enumerate(fnParams)) {
    auto fnParam = llvm::cast<ConstParamAttr>(fnParamAttr);
    auto arg = (*args)[i];

    if (failed(
            verifyConstantArgumentKind(op.getOperation(), i, fnParam, arg))) {
      hadError = true;
      // Don't report a redundant identity error when the kind is already wrong.
      continue;
    }

    // SIMT implementations are monomorphized using the declaration's call-site
    // bindings. There is no separate implementation evaluation context, so a
    // local implementation parameter must preserve the corresponding
    // declaration parameter's name.
    auto implParam = llvm::cast<ConstParamAttr>(implParams[i]);
    if (isWildcardConstParam(implParam)) {
      continue;
    }

    if (implParam.getName() != fnParam.getName()) {
      op.emitOpError("const_params[")
          << i << "] name '" << implParam.getName()
          << "' must match implemented function const_param name '"
          << fnParam.getName() << "'";
      hadError = true;
      continue;
    }

    // Non-wildcard: enforce identity — the arg must be a plain variable
    // reference to the impl's own i-th const_param.
    auto varArg = dyn_cast<SymbolicExprVarAttr>(arg);
    if (!varArg || varArg.getName() != implParam.getName()) {
      op.emitOpError("const_args[")
          << i << "] must be an identity reference to impl param '"
          << implParam.getName()
          << "' (non-identity const_args are not supported)";
      hadError = true;
    }
  }
  return failure(hadError);
}

//===----------------------------------------------------------------------===//
// Helpers for extern implementation signatures
//===----------------------------------------------------------------------===//

static ParseResult parseExternAttrDict(OpAsmParser &parser,
                                       DictionaryAttr &result) {
  Builder &builder = parser.getBuilder();
  NamedAttrList attrs;

  if (failed(parser.parseOptionalLBrace())) {
    result = builder.getDictionaryAttr({});
    return success();
  }

  if (failed(parser.parseOptionalRBrace())) {
    do {
      std::string key;
      if (parser.parseKeywordOrString(&key) || parser.parseEqual()) {
        return failure();
      }

      Attribute value;
      if (parser.parseAttribute(value)) {
        return failure();
      }
      attrs.set(builder.getStringAttr(key), value);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
      return failure();
    }
  }

  result = attrs.getDictionary(parser.getContext());
  return success();
}

static void printExternAttrs(OpAsmPrinter &printer, DictionaryAttr attrs) {
  if (!attrs || attrs.empty()) {
    return;
  }

  printer << " {";
  llvm::interleaveComma(attrs.getValue(), printer, [&](NamedAttribute attr) {
    StringRef attrName = attr.getName();
    printer << attrName << " = ";
    printer.printAttribute(attr.getValue());
  });
  printer << "}";
}

static void printExternSignature(OpAsmPrinter &printer,
                                 FunctionOpInterface op) {
  auto funcType = cast<FunctionType>(op.getFunctionType());
  printFunctionSignatureWithCudaTileTypes(
      printer, funcType.getInputs(), op.getArgAttrsAttr(),
      funcType.getResults(), op.getResAttrsAttr(),
      [&](OpAsmPrinter &printer, unsigned i) { printer << "%arg" << i; },
      printExternAttrs);
}

static ParseResult parseExternSignature(OpAsmParser &parser,
                                        NamedAttrList &properties,
                                        StringRef functionTypeAttrName,
                                        StringRef argAttrsAttrName,
                                        StringRef resAttrsAttrName) {
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;
  if (cuda_tile::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, arguments, isVariadic, resultTypes,
          resultAttrs, parseExternAttrDict)) {
    return failure();
  }
  SmallVector<Type> argTypes = llvm::to_vector(llvm::map_range(
      arguments, [](const OpAsmParser::Argument &arg) { return arg.type; }));

  Builder &builder = parser.getBuilder();
  for (OpAsmParser::Argument &arg : arguments) {
    if (!arg.attrs) {
      arg.attrs = DictionaryAttr::get(parser.getContext());
    }
  }
  resultAttrs.resize(resultTypes.size(),
                     DictionaryAttr::get(parser.getContext()));
  for (DictionaryAttr &resultAttr : resultAttrs) {
    if (!resultAttr) {
      resultAttr = DictionaryAttr::get(parser.getContext());
    }
  }
  SmallVector<Attribute> argAttrStorage = llvm::to_vector(llvm::map_range(
      arguments,
      [](const OpAsmParser::Argument &arg) -> Attribute { return arg.attrs; }));
  SmallVector<Attribute> resultAttrStorage(resultAttrs.begin(),
                                           resultAttrs.end());
  properties.append(
      StringAttr::get(parser.getContext(), functionTypeAttrName),
      TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));
  if (ArrayAttr normalizedArgAttrs =
          cuda_tile::buildNormalizedArgAttrs(builder, argAttrStorage)) {
    properties.append(StringAttr::get(parser.getContext(), argAttrsAttrName),
                      normalizedArgAttrs);
  }
  properties.append(StringAttr::get(parser.getContext(), resAttrsAttrName),
                    builder.getArrayAttr(resultAttrStorage));
  return success();
}

//===----------------------------------------------------------------------===//
// Compact constant parameter / argument printers and parsers
//===----------------------------------------------------------------------===//

/// Print const_params compactly: [M: index, T:
/// !cuda_tile.preview$Datatype]
void mlir::cuda_tile::printCompactConstParams(OpAsmPrinter &printer,
                                              Operation *op,
                                              ArrayAttr params) {
  printer << "[";
  llvm::interleaveComma(params, printer.getStream(), [&](Attribute attr) {
    auto cp = llvm::cast<ConstParamAttr>(attr);
    if (isWildcardConstParam(cp)) {
      printer.getStream() << "?";
      return;
    }
    printer.getStream() << cp.getName() << ": ";
    printer.printType(cp.getType().getValue());
  });
  printer << "]";
}

/// Parse const_params in compact form: [M: index, ?, T:
/// !cuda_tile.preview$Datatype]
ParseResult mlir::cuda_tile::parseCompactConstParams(OpAsmParser &parser,
                                                     ArrayAttr &result) {
  if (parser.parseLSquare()) {
    return failure();
  }

  SmallVector<Attribute> params;
  if (succeeded(parser.parseOptionalRSquare())) {
    result = parser.getBuilder().getArrayAttr(params);
    return success();
  }

  do {
    if (succeeded(parser.parseOptionalQuestion())) {
      params.push_back(ConstParamAttr::get(
          parser.getContext(), "?",
          TypeAttr::get(NoneType::get(parser.getContext()))));
      continue;
    }
    StringRef name;
    Type type;
    if (parser.parseKeyword(&name) || parser.parseColon() ||
        parser.parseType(type)) {
      return failure();
    }
    auto param = ConstParamAttr::getChecked(
        [&] { return parser.emitError(parser.getCurrentLocation()); },
        parser.getContext(), name, TypeAttr::get(type));
    if (!param) {
      return failure();
    }
    params.push_back(param);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare()) {
    return failure();
  }
  result = parser.getBuilder().getArrayAttr(params);
  return success();
}

//===----------------------------------------------------------------------===//
// ExternImplOp
//===----------------------------------------------------------------------===//

static ParseResult parseScratchExpr(OpAsmParser &parser, StringRef key,
                                    SymbolicExprAttr &expr) {
  SMLoc valueLoc = parser.getCurrentLocation();
  Attribute attr;
  auto attrResult = parser.parseOptionalAttribute(attr);
  if (attrResult.has_value()) {
    if (failed(*attrResult)) {
      return failure();
    }
    expr = dyn_cast<SymbolicExprAttr>(attr);
    if (expr) {
      return success();
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      if (!intAttr.getValue().isSignedIntN(64)) {
        return parser.emitError(valueLoc)
               << "scratch field '" << key
               << "' does not fit in a signed 64-bit integer";
      }
      expr = SymbolicExprIntAttr::get(parser.getContext(),
                                      intAttr.getValue().getSExtValue());
      return success();
    }
    return parser.emitError(valueLoc)
           << "expected symbolic expression for scratch field '" << key << "'";
  }

  bool emittedDiagnostic = false;
  CudaTileCompactLayoutParser compactParser(parser, parser.getContext(),
                                            valueLoc, emittedDiagnostic);
  FailureOr<SymbolicExprAttr> result = compactParser.parseExpr();
  if (failed(result)) {
    if (emittedDiagnostic) {
      return failure();
    }
    return parser.emitError(valueLoc)
           << "expected symbolic expression for scratch field '" << key << "'";
  }
  expr = *result;
  return success();
}

static ParseResult parseExternImplScratchPayload(OpAsmParser &parser,
                                                 ScratchRequestAttr &scratch) {
  SMLoc scratchLoc = parser.getCurrentLocation();
  SymbolicExprAttr scratchSize;
  SymbolicExprAttr scratchAlignment;
  bool sawScratchSize = false;
  bool sawScratchAlignment = false;
  SMLoc scratchAlignmentLoc;

  auto parseDuplicateError = [&](StringRef key, SMLoc keyLoc) {
    return parser.emitError(keyLoc)
           << "duplicate scratch field '" << key << "'";
  };

  if (parser.parseLBrace()) {
    return failure();
  }
  if (succeeded(parser.parseOptionalRBrace())) {
    return parser.emitError(scratchLoc) << "scratch requires smem.size";
  }

  do {
    std::string keyStorage;
    SMLoc keyLoc = parser.getCurrentLocation();
    if (parser.parseKeywordOrString(&keyStorage) || parser.parseEqual()) {
      return failure();
    }
    StringRef key(keyStorage);

    if (key == kScratchSmemSizeKey) {
      if (sawScratchSize) {
        return parseDuplicateError(key, keyLoc);
      }
      if (parseScratchExpr(parser, key, scratchSize)) {
        return failure();
      }
      sawScratchSize = true;
      continue;
    }

    if (key == kScratchSmemAlignmentKey) {
      if (sawScratchAlignment) {
        return parseDuplicateError(key, keyLoc);
      }
      if (parseScratchExpr(parser, key, scratchAlignment)) {
        return failure();
      }
      sawScratchAlignment = true;
      scratchAlignmentLoc = keyLoc;
      continue;
    }

    return parser.emitError(keyLoc) << "unknown scratch field '" << key << "'";
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRBrace()) {
    return failure();
  }
  if (sawScratchAlignment && !sawScratchSize) {
    return parser.emitError(scratchAlignmentLoc)
           << "smem.alignment requires smem.size";
  }
  if (!sawScratchSize) {
    return parser.emitError(scratchLoc) << "scratch requires smem.size";
  }
  if (!scratchAlignment) {
    scratchAlignment = SymbolicExprIntAttr::get(
        parser.getContext(), kDefaultScratchSmemAlignment);
  }
  scratch = ScratchRequestAttr::getChecked(
      [&] { return parser.emitError(scratchLoc); }, parser.getContext(),
      scratchSize, scratchAlignment);
  if (!scratch) {
    return failure();
  }
  return success();
}

static void printExternImplScratchPayload(OpAsmPrinter &printer,
                                          ScratchRequestAttr scratch) {
  printer << kExternImplScratchKeyword << " {";
  printer << kScratchSmemSizeKey << " = ";
  printer.printAttribute(scratch.getSize());
  printer << ", " << kScratchSmemAlignmentKey << " = ";
  printer.printAttribute(scratch.getAlignment());
  printer << "}";
}

ParseResult ExternImplOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr)) {
    return failure();
  }
  NamedAttrList properties;
  properties.append(getSymNameAttrName(result.name), nameAttr);
  setPrivateVisibilityIfAbsent(parser, result);

  ArrayAttr constParams;
  if (succeeded(parser.parseOptionalLess())) {
    if (parseCompactConstParams(parser, constParams) || parser.parseGreater()) {
      return failure();
    }
    properties.append(getConstParamsAttrName(result.name), constParams);
  }

  FlatSymbolRefAttr implementsAttr;
  if (parser.parseKeyword("implements") ||
      parser.parseAttribute(implementsAttr)) {
    return failure();
  }
  properties.append(getImplementsAttrName(result.name), implementsAttr);

  ArrayAttr constArgs;
  if (succeeded(parser.parseOptionalLess())) {
    if (parseCompactConstArgs(parser, constArgs) || parser.parseGreater()) {
      return failure();
    }
    properties.append(getConstArgsAttrName(result.name), constArgs);
  }

  if (failed(parseExternSignature(parser, properties,
                                  getFunctionTypeAttrName(result.name),
                                  getArgAttrsAttrName(result.name),
                                  getResAttrsAttrName(result.name)))) {
    return failure();
  }
  result.addRegion();

  if (succeeded(parser.parseOptionalKeyword("synchronization"))) {
    SynchronizationAttr synchronization;
    if (parser.parseEqual() || parser.parseAttribute(synchronization)) {
      return failure();
    }
    properties.append(getSynchronizationAttrName(result.name), synchronization);
  }

  if (succeeded(parser.parseOptionalKeyword("arch"))) {
    if (parser.parseEqual()) {
      return failure();
    }
    Attribute attr;
    if (parser.parseAttribute(attr)) {
      return failure();
    }
    properties.append(getArchAttrName(result.name), attr);
  }

  if (succeeded(parser.parseOptionalKeyword(kExternImplScratchKeyword))) {
    ScratchRequestAttr scratch;
    if (parseExternImplScratchPayload(parser, scratch)) {
      return failure();
    }
    properties.append(getScratchSmemAttrName(result.name), scratch);
  }

  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }
  result.propertiesAttr = properties.getDictionary(parser.getContext());
  return success();
}

void ExternImplOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  if (auto params = getConstParams()) {
    printer << "<";
    printCompactConstParams(printer, getOperation(), *params);
    printer << ">";
  }
  printer << " implements ";
  printer.printAttributeWithoutType(getImplementsAttr());
  if (auto args = getConstArgs()) {
    printer << "<";
    printCompactConstArgs(printer, getOperation(), *args);
    printer << ">";
  }
  printer << ' ';
  printExternSignature(printer, *this);
  if (getSynchronizationAttr()) {
    printer << " synchronization = ";
    printer.printAttribute(getSynchronizationAttr());
  }
  if (getArch()) {
    printer << " arch = ";
    printer.printAttribute(getArchAttr());
  }
  if (auto scratch = getScratchSmem()) {
    printer << " ";
    printExternImplScratchPayload(printer, *scratch);
  }
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {getArgAttrsAttrName(), getFunctionTypeAttrName(),
       getSymNameAttrName(),
       SymbolOpInterface::getDefaultVisibilityAttrName(),
       getResAttrsAttrName(), getConstParamsAttrName(), getConstArgsAttrName(),
       getImplementsAttrName(), getScratchSmemAttrName(), getArchAttrName(),
       getSynchronizationAttrName()});
}

bool ExternImplOp::canDiscardOnUseEmpty() {
  Operation *symbolTableOp =
      getOperation()->getParentWithTrait<OpTrait::SymbolTable>();
  if (!symbolTableOp)
    return true;

  Operation *decl =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getImplementsAttr());
  if (!decl)
    return true;

  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(decl, symbolTableOp);
  if (!uses)
    return false;

  for (SymbolTable::SymbolUse use : *uses) {
    Operation *user = use.getUser();
    if (isa<ExternImplOp>(user))
      continue;
    return false;
  }
  return true;
}

static FailureOr<int64_t> concreteExprValue(SymbolicExprAttr expr) {
  auto intAttr = llvm::dyn_cast<SymbolicExprIntAttr>(expr.simplify());
  if (!intAttr) {
    return failure();
  }
  return intAttr.getValue();
}

/// Product of the simplified integer leaves of a (possibly nested) layout shape
/// mode, traversed positionally. Do NOT use Attribute::walk for this: it
/// deduplicates uniqued identical sub-attributes, so a mode like `(8, 8)` would
/// otherwise be counted as 8, not 64.
///
/// Fails on a leaf that does not simplify to an integer, or on a product that
/// overflows int64.
static FailureOr<int64_t> concreteModeSize(SymbolicExprAttr mode) {
  mode = mode.simplify();
  if (auto tuple = llvm::dyn_cast<SymbolicTupleAttr>(mode)) {
    int64_t product = 1;
    for (SymbolicExprAttr elem : tuple.getValues()) {
      FailureOr<int64_t> elemSize = concreteModeSize(elem);
      if (failed(elemSize)) {
        return failure();
      }
      std::optional<int64_t> next = llvm::checkedMul(product, *elemSize);
      if (!next) {
        return failure();
      }
      product = *next;
    }
    return product;
  }
  FailureOr<int64_t> value = concreteExprValue(mode);
  if (failed(value)) {
    return failure();
  }
  assert(*value > 0 && "expected the layout verifier to reject a non-positive "
                       "shape extent");
  return *value;
}

/// Product of the shape leaves whose corresponding stride is nonzero. This
/// mirrors `cute::filter_zeros`: stride-zero leaves describe broadcast copies
/// and therefore do not consume additional tile elements.
static FailureOr<int64_t> concreteInjectiveModeSize(SymbolicExprAttr shape,
                                                    SymbolicExprAttr stride) {
  shape = shape.simplify();
  stride = stride.simplify();
  auto shapeTuple = llvm::dyn_cast<SymbolicTupleAttr>(shape);
  auto strideTuple = llvm::dyn_cast<SymbolicTupleAttr>(stride);
  if (shapeTuple || strideTuple) {
    if (!shapeTuple || !strideTuple ||
        shapeTuple.getValues().size() != strideTuple.getValues().size()) {
      return failure();
    }
    int64_t product = 1;
    for (auto [shapeElem, strideElem] :
         llvm::zip(shapeTuple.getValues(), strideTuple.getValues())) {
      FailureOr<int64_t> elemSize =
          concreteInjectiveModeSize(shapeElem, strideElem);
      if (failed(elemSize)) {
        return failure();
      }
      std::optional<int64_t> next = llvm::checkedMul(product, *elemSize);
      if (!next) {
        return failure();
      }
      product = *next;
    }
    return product;
  }

  FailureOr<int64_t> shapeValue = concreteExprValue(shape);
  FailureOr<int64_t> strideValue = concreteExprValue(stride);
  if (failed(shapeValue) || failed(strideValue)) {
    return failure();
  }
  assert(*shapeValue > 0 && "expected the layout verifier to reject a "
                            "non-positive shape extent");
  return *strideValue == 0 ? 1 : *shapeValue;
}

/// Whether every non-unit leaf of a concrete mode has zero stride.
static FailureOr<bool> concreteModeHasZeroStride(SymbolicExprAttr shape,
                                                 SymbolicExprAttr stride) {
  shape = shape.simplify();
  stride = stride.simplify();
  auto shapeTuple = llvm::dyn_cast<SymbolicTupleAttr>(shape);
  auto strideTuple = llvm::dyn_cast<SymbolicTupleAttr>(stride);
  if (shapeTuple || strideTuple) {
    if (!shapeTuple || !strideTuple ||
        shapeTuple.getValues().size() != strideTuple.getValues().size()) {
      return failure();
    }
    for (auto [shapeElem, strideElem] :
         llvm::zip(shapeTuple.getValues(), strideTuple.getValues())) {
      FailureOr<bool> hasZeroStride =
          concreteModeHasZeroStride(shapeElem, strideElem);
      if (failed(hasZeroStride)) {
        return failure();
      }
      if (!*hasZeroStride) {
        return false;
      }
    }
    return true;
  }

  FailureOr<int64_t> shapeValue = concreteExprValue(shape);
  FailureOr<int64_t> strideValue = concreteExprValue(stride);
  if (failed(shapeValue) || failed(strideValue)) {
    return failure();
  }
  return *shapeValue == 1 || *strideValue == 0;
}

static LogicalResult verifyNativeImpl(ExternImplOp op, FunctionType funcTy,
                                      unsigned numInputs) {
  if (op.getConfig()) {
    return op.emitOpError("extern_impl must not carry 'config'");
  }
  if (funcTy.getNumResults() != 0) {
    return op.emitOpError(
               "extern_impl must not declare results; "
               "outputs are represented as 'out' or 'inout' fragment "
               "args, got ")
           << funcTy.getNumResults() << " result(s)";
  }

  // Every arg must carry cuda_tile.preview$access.
  for (size_t i = 0; i < numInputs; ++i) {
    if (!op.getArgAttrOfType<AccessAttr>(i, AccessAttr::name)) {
      return op.emitOpError("extern_impl argument ")
             << i << " must carry a 'cuda_tile.preview$access' arg attr";
    }
  }

  std::optional<unsigned> firstNonIn, lastNonIn;
  unsigned numNonIn = 0;
  for (size_t i = 0; i < numInputs; ++i) {
    auto accessAttr = op.getArgAttrOfType<AccessAttr>(i, AccessAttr::name);
    if (accessAttr.getValue() == Access::In) {
      continue;
    }
    if (!firstNonIn) {
      firstNonIn = i;
    }
    lastNonIn = i;
    ++numNonIn;
  }
  if (firstNonIn && (*lastNonIn - *firstNonIn + 1) != numNonIn) {
    return op.emitOpError("out/inout fragment args must form a single "
                          "contiguous run; found ")
           << numNonIn << " out/inout arg(s) spanning indices " << *firstNonIn
           << ".." << *lastNonIn << " with an 'in' arg in between";
  }

  // Collect symbolic variable names from const_params so we can verify below
  // that we don't use undeclared variables.
  llvm::StringSet<> paramNames;
  if (auto params = op.getConstParams()) {
    for (Attribute p : *params) {
      if (auto cp = llvm::dyn_cast<ConstParamAttr>(p)) {
        if (!isWildcardConstParam(cp)) {
          paramNames.insert(cp.getName());
        }
      }
    }
  }

  std::optional<int64_t> sharedParticipants;
  std::optional<unsigned> partRef;
  std::optional<int64_t> sharedInvocations;
  std::optional<unsigned> invRef;
  bool allFragmentsConcrete = true;

  for (size_t i = 0; i < numInputs; ++i) {
    auto frag = llvm::cast<FragmentType>(funcTy.getInput(i));

    // Ensure every variable in a fragment is defined in the const_params.
    if (!paramNames.empty() || !frag.isConcrete()) {
      auto checkVars = [&](Attribute attr, StringRef field) -> LogicalResult {
        StringRef unknownName;
        auto res = attr.walk([&](SymbolicExprVarAttr var) {
          if (!paramNames.contains(var.getName())) {
            unknownName = var.getName();
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (res.wasInterrupted()) {
          return op.emitOpError("argument ")
                 << i << " field " << field
                 << " references an undeclared symbolic variable "
                 << unknownName;
        }
        return success();
      };
      if (failed(checkVars(frag.getSymbolicShapeAttr(), "shape")) ||
          failed(checkVars(frag.getMemLayout(), "mem_layout")) ||
          failed(checkVars(frag.getFragmap(), "fragmap")) ||
          failed(checkVars(frag.getAlignment(), "alignment"))) {
        return failure();
      }
    }

    // Structural checks require concrete shape values; skip until
    // concretization.
    if (!frag.isConcrete()) {
      allFragmentsConcrete = false;
      continue;
    }

    // Fragmap must have exactly 3 flat shape entries
    // (invocation, participant, footprint).
    auto fragmapShapes = frag.getFragmap().getShapes().getValues();
    auto fragmapStrides = frag.getFragmap().getStrides().getValues();
    if (fragmapShapes.size() != 3) {
      return op.emitOpError("fragment arg ")
             << i
             << " fragmap must have 3 shape modes "
                "(invocation, participant, footprint), got "
             << fragmapShapes.size();
    }
    if (fragmapStrides.size() != 3) {
      return op.emitOpError("fragment arg ")
             << i
             << " fragmap must have 3 stride modes "
                "(invocation, participant, footprint), got "
             << fragmapStrides.size();
    }

    // A concrete fragment may still carry foldable constant expressions, and
    // the layout verifier admits a mode whose leaves multiply out of int64.
    FailureOr<int64_t> invSize = concreteModeSize(fragmapShapes[0]);
    FailureOr<int64_t> partSize = concreteModeSize(fragmapShapes[1]);
    FailureOr<int64_t> footSize = concreteModeSize(fragmapShapes[2]);
    if (failed(invSize) || failed(partSize) || failed(footSize)) {
      return op.emitOpError("fragment arg ")
             << i
             << " fragmap modes must contain signed 64-bit shape leaves with "
                "representable products";
    }

    // The fragment verifier already established that a concrete shape has
    // positive dimensions and a representable element count.
    int64_t numElems = frag.getNumElements();
    if (*footSize != numElems) {
      return op.emitOpError("fragment arg ")
             << i << " fragmap footprint mode size (" << *footSize
             << ") must equal the fragment element count (" << numElems << ")";
    }

    // SMEM is per-CTA: the P physical threads cooperating on an invocation
    // receive the same pointer into the CTA buffer. Injective/sub-CTA SMEM
    // participant placement is not modeled yet.
    MemorySpace memSpace = frag.getMemorySpace();
    if (memSpace == MemorySpace::SMEM) {
      FailureOr<bool> hasZeroStride =
          concreteModeHasZeroStride(fragmapShapes[1], fragmapStrides[1]);
      if (failed(hasZeroStride)) {
        return op.emitOpError("fragment arg ")
               << i
               << " smem fragmap participant mode must simplify to "
                  "signed 64-bit shape and stride integers";
      }
      if (!*hasZeroStride) {
        return op.emitOpError("fragment arg ")
               << i
               << " smem fragmap participant mode must have zero stride for "
                  "every non-unit submode";
      }
    }

    if (!sharedParticipants) {
      sharedParticipants = *partSize;
      partRef = i;
    } else if (*sharedParticipants != *partSize) {
      return op.emitOpError("fragment arg ")
             << i << " fragmap participant mode size (" << *partSize
             << ") must match arg " << *partRef << "'s (" << *sharedParticipants
             << "); all args of one invocation use the same participants";
    }

    if (!sharedInvocations) {
      sharedInvocations = *invSize;
      invRef = i;
    } else if (*sharedInvocations != *invSize) {
      return op.emitOpError("fragment arg ")
             << i << " fragmap invocation mode size (" << *invSize
             << ") must match arg " << *invRef << "'s (" << *sharedInvocations
             << ")";
    }
  }

  if (allFragmentsConcrete && sharedParticipants &&
      failed(verifySynchronizationParticipation(
          op.getSynchronization(), *sharedParticipants,
          [&]() { return op.emitOpError(); }))) {
    return failure();
  }

  return success();
}

bool ExternImplOp::isNativeSimtImplementation() {
  auto funcTy = llvm::cast<FunctionType>(getFunctionType());
  return !funcTy.getInputs().empty() &&
         llvm::all_of(funcTy.getInputs(),
                      [](Type type) { return llvm::isa<FragmentType>(type); });
}

LogicalResult ExternImplOp::verify() {
  // Verify const_params if present
  if (auto params = getConstParams()) {
    llvm::StringSet<> paramNames;
    for (size_t i = 0; i < params->size(); ++i) {
      auto param = llvm::dyn_cast<ConstParamAttr>((*params)[i]);
      if (!param) {
        return emitOpError("const_params[")
               << i << "] must be a ConstParamAttr";
      }

      // Wildcard slots (name == "?") are valid; skip name deduplication.
      if (isWildcardConstParam(param)) {
        continue;
      }

      // Check for duplicate parameter names
      auto [_, inserted] = paramNames.insert(param.getName());
      if (!inserted) {
        return emitOpError("duplicate constant parameter name: '")
               << param.getName() << "'";
      }
    }
  }

  // Verify constant arguments if present.
  if (auto args = getConstArgs()) {
    // Check that all args are symbolic expressions.
    for (size_t i = 0; i < args->size(); ++i) {
      if (!llvm::isa<SymbolicExprAttr>((*args)[i])) {
        return emitOpError("constant argument ")
               << i << " must be a symbolic expression attribute";
      }
    }

    // Arity and target-kind validation need the referenced target, so they are
    // handled in verifySymbolUses().
  }

  auto funcTy = llvm::dyn_cast<FunctionType>(getFunctionType());
  if (!funcTy) {
    return emitOpError("function_type must be a FunctionType");
  }

  unsigned numInputs = funcTy.getNumInputs();
  if (!isNativeSimtImplementation()) {
    return emitOpError(
        "function_type must have at least one input and all inputs must be "
        "fragment types");
  }

  if (auto archAttr = getArch()) {
    for (auto [i, attr] : llvm::enumerate(*archAttr)) {
      auto str = llvm::dyn_cast<StringAttr>(attr);
      if (!str || str.getValue().empty()) {
        return emitOpError("arch[") << i << "] must be a non-empty string";
      }
    }
  }

  return verifyNativeImpl(*this, funcTy, numInputs);
}

//===----------------------------------------------------------------------===//
// ExternImplOp - SymbolUserOpInterface
//===----------------------------------------------------------------------===//

LogicalResult
ExternImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *target =
      symbolTable.lookupNearestSymbolFrom(*this, getImplementsAttr());
  if (!target) {
    // Unresolved references are permitted. Declaration-dependent checks apply
    // only when the referenced symbol is present.
    return success();
  }

  auto fn = llvm::dyn_cast<FuncOp>(target);
  if (!fn || !fn.getIsExtern()) {
    return emitOpError("'")
           << getImplements()
           << "' does not reference a valid extern preview$func";
  }

  size_t fnParamCount = fn.getConstParams().size();

  size_t implArgCount = 0;
  if (auto args = getConstArgs()) {
    implArgCount = args->size();
  }

  if (implArgCount != fnParamCount) {
    return emitOpError("constant argument count (")
           << implArgCount << ") must match extern function constant "
           << "parameter count (" << fnParamCount << ")";
  }

  auto implFuncType = llvm::cast<FunctionType>(getFunctionType());
  auto fnFuncType = fn.getFunctionType();

  if (getConstArgs()) {
    if (failed(verifyImplConstArgKinds(*this, fn))) {
      return failure();
    }
  }

  // Implementation args correspond 1-to-1 with the extern func's args. The
  // declaration's out/inout results are represented as fragment arguments, so
  // only input counts need to match.
  if (implFuncType.getNumInputs() != fnFuncType.getNumInputs()) {
    return emitOpError("input count (")
           << implFuncType.getNumInputs()
           << ") must match extern function input count ("
           << fnFuncType.getNumInputs() << ")";
  }

  for (size_t i = 0; i < implFuncType.getNumInputs(); ++i) {
    auto implAccess = getArgAttrOfType<AccessAttr>(i, AccessAttr::name);
    auto fnAccess = fn.getArgAttrOfType<AccessAttr>(i, AccessAttr::name);
    if (!fnAccess) {
      return emitOpError("extern function argument ")
             << i << " is missing 'cuda_tile.preview$access' attr";
    }
    if (!implAccess || implAccess.getValue() != fnAccess.getValue()) {
      return emitOpError("simt-path impl argument ")
             << i << " access ("
             << (implAccess ? stringifyAccess(implAccess.getValue())
                            : "missing")
             << ") must match extern function argument access ("
             << stringifyAccess(fnAccess.getValue()) << ")";
    }

    auto fragment = llvm::cast<FragmentType>(implFuncType.getInput(i));
    auto tile = llvm::dyn_cast<TileType>(fnFuncType.getInput(i));
    if (!tile) {
      return emitOpError("simt-path extern function argument ")
             << i << " must be a cuda_tile.tile, got "
             << fnFuncType.getInput(i);
    }

    // Symbolic implementations are checked after specialization, when both the
    // fragment map and declaration tile have concrete extents.
    if (!fragment.isConcrete() || !tile.hasStaticShape()) {
      continue;
    }
    ArrayRef<SymbolicExprAttr> fragmapShapes =
        fragment.getFragmap().getShapes().getValues();
    ArrayRef<SymbolicExprAttr> fragmapStrides =
        fragment.getFragmap().getStrides().getValues();
    assert(fragmapShapes.size() == 3 && fragmapStrides.size() == 3 &&
           "verifySimtPath checks concrete fragmap rank");
    FailureOr<int64_t> injectiveInvocations =
        concreteInjectiveModeSize(fragmapShapes[0], fragmapStrides[0]);
    FailureOr<int64_t> injectiveParticipants =
        concreteInjectiveModeSize(fragmapShapes[1], fragmapStrides[1]);
    FailureOr<int64_t> footprint = concreteModeSize(fragmapShapes[2]);
    if (failed(injectiveInvocations) || failed(injectiveParticipants) ||
        failed(footprint)) {
      return emitOpError("simt-path argument ")
             << i
             << " fragmap modes must contain signed 64-bit shape and stride "
                "leaves with representable products";
    }
    // The tile verifier bounds a static shape to positive, representable
    // dimensions.
    int64_t tileElements = tile.getNumElements();
    std::optional<int64_t> coveredElements =
        llvm::checkedMul(*injectiveInvocations, *injectiveParticipants);
    if (coveredElements) {
      coveredElements = llvm::checkedMul(*coveredElements, *footprint);
    }
    if (!coveredElements || *coveredElements != tileElements) {
      return emitOpError("simt-path argument ")
             << i << " fragmap covers " << *injectiveInvocations
             << " distinct invocation(s) * " << *injectiveParticipants
             << " distinct participant(s) * " << *footprint
             << " footprint element(s), but the extern function tile has "
             << tileElements << " element(s)";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyConstParams(Operation *op, ArrayAttr params) {
  llvm::StringSet<> paramNames;
  for (auto [i, attr] : llvm::enumerate(params)) {
    auto param = llvm::dyn_cast<ConstParamAttr>(attr);
    if (!param)
      return op->emitOpError("const_params[")
             << i << "] must be a ConstParamAttr";

    if (param.getName() == "?")
      return op->emitOpError("const_params[")
             << i
             << "]: '?' wildcard is only valid in extern_impl const_params";

    auto [_, inserted] = paramNames.insert(param.getName());
    if (!inserted)
      return op->emitOpError("duplicate constant parameter name: '")
             << param.getName() << "'";
  }
  return success();
}

LogicalResult
mlir::cuda_tile::verifySignatureSymbolicScope(Operation *op, TypeRange inputs,
                                              TypeRange results,
                                              ArrayRef<Attribute> constParams) {
  llvm::StringSet<> paramNames;
  for (Attribute attr : constParams) {
    paramNames.insert(cast<ConstParamAttr>(attr).getName());
  }

  auto verifyType = [&](Type type, StringRef where) -> LogicalResult {
    StringRef unboundName;
    WalkResult result = type.walk([&](SymbolicExprVarAttr var) {
      if (!paramNames.contains(var.getName())) {
        unboundName = var.getName();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return op->emitOpError(where)
             << " references undeclared symbolic variable '" << unboundName
             << "'";
    return success();
  };

  for (Type type : inputs) {
    if (failed(verifyType(type, "input type")))
      return failure();
  }
  for (Type type : results) {
    if (failed(verifyType(type, "result type")))
      return failure();
  }

  return success();
}

static LogicalResult verifyFuncSymbolicScope(FuncOp func) {
  auto funcType = func.getFunctionType();
  return verifySignatureSymbolicScope(func.getOperation(), funcType.getInputs(),
                                      funcType.getResults(),
                                      func.getConstParams().getValue());
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFuncOp<FuncOp>(parser, result);
}

void FuncOp::print(OpAsmPrinter &printer) { printFuncOp(*this, printer); }

void FuncOp::ensureTerminator(Region &region, Builder &builder, Location loc) {
  // Declarations (extern functions) must not have a body, so an empty body
  // region is left untouched: never materialize a block/terminator for them.
  if (region.empty())
    return;

  // The callback must return a detached op: ensureRegionTerminator inserts it
  // into the block itself.
  ::mlir::impl::ensureRegionTerminator(
      region, builder, loc, [](OpBuilder &b, Location loc) -> Operation * {
        return Operation::create(
            loc, OperationName(ReturnOp::getOperationName(), b.getContext()),
            TypeRange{}, ValueRange{}, NamedAttrList{}, PropertyRef{},
            BlockRange{}, /*numRegions=*/0);
      });
}

LogicalResult FuncOp::verify() {
  auto funcTy = getFunctionType();
  bool isExternFunc = getIsExtern();

  if (isExternFunc && !getBody().empty())
    return emitOpError("extern functions must not have a body");

  if (!isExternFunc && getBody().empty())
    return emitOpError("non-extern functions must have a body");

  if (!isExternFunc && !getConstParams().empty())
    return emitOpError(
        "constant parameters are only valid on extern functions");

  if (isExternFunc && funcTy.getInputs().empty())
    return emitOpError("extern functions must declare at least one input");

  if (failed(verifyConstParams(getOperation(), getConstParams())))
    return failure();

  if (failed(verifyFuncSymbolicScope(*this)))
    return failure();

  auto verifyBoundaryType = [&](Type type, StringRef kind,
                                size_t index) -> LogicalResult {
    // A fragment is one invocation's view of a tile at the SIMT-interop ABI
    // boundary, so it is only meaningful in a preview$extern_impl signature.
    if (llvm::isa<FragmentType>(type)) {
      return emitOpError(kind)
             << " type " << index
             << " must not be a cuda_tile.preview$fragment; fragments are only "
                "valid in preview$extern_impl signatures, got "
             << type;
    }

    if (isExternFunc && !llvm::isa<TileType>(type))
      return emitOpError(kind)
             << " type " << index << " must be a cuda_tile.tile, got " << type;

    return success();
  };

  for (auto [index, type] : llvm::enumerate(funcTy.getInputs())) {
    if (failed(verifyBoundaryType(type, "input", index)))
      return failure();
    if (isExternFunc &&
        !getArgAttrOfType<AccessAttr>(index, AccessAttr::name))
      return emitOpError("extern function argument ")
             << index << " must have a 'cuda_tile.preview$access' arg attr";
  }
  for (auto [index, type] : llvm::enumerate(funcTy.getResults()))
    if (failed(verifyBoundaryType(type, "result", index)))
      return failure();

  // Bound outputs must match the leading result types. Additional results are
  // allowed after those bound outputs.
  if (isExternFunc) {
    unsigned numBoundInouts = 0;
    for (auto [index, type] : llvm::enumerate(funcTy.getInputs())) {
      auto access = getArgAttrOfType<AccessAttr>(index, AccessAttr::name);
      if (access.getValue() == Access::In)
        continue;
      if (numBoundInouts >= funcTy.getNumResults())
        return emitOpError(
            "number of bound inouts exceeded the number of results");
      if (type != funcTy.getResult(numBoundInouts))
        return emitOpError("extern function argument ")
               << index << " must match result " << numBoundInouts;
      ++numBoundInouts;
    }
  }

  if (failed(impl::verifyFuncDebugInfo(*this)))
    return failure();
  return success();
}

LogicalResult FuncOp::verifyRegions() {
  if (getIsExtern())
    return success();
  if (failed(impl::verifyFuncBodyDebugInfo(*this)))
    return failure();
  return success();
}

/// Calls and implementations bind values to a target's const_params. Convert
/// the parameter type into the symbolic-expression family legal for that
/// binding.
static FailureOr<SymbolicExprKind> getConstParamKind(Operation *op,
                                                     ConstParamAttr param) {
  Type type = param.getType().getValue();
  if (llvm::isa<IndexType>(type))
    return SymbolicExprKind::Int;
  if (llvm::isa<ShapeParamType>(type))
    return SymbolicExprKind::Tuple;
  if (llvm::isa<DTypeParamType>(type))
    return SymbolicExprKind::Datatype;
  if (llvm::isa<StringParamType>(type))
    return SymbolicExprKind::String;

  op->emitOpError("constant parameter '")
      << param.getName() << "' has unsupported type " << type;
  return failure();
}

static llvm::StringLiteral
getSymbolicExprKindDescription(SymbolicExprKind kind) {
  switch (kind) {
  case SymbolicExprKind::Int:
    return "integer expression";
  case SymbolicExprKind::Tuple:
    return "shape expression";
  case SymbolicExprKind::Datatype:
    return "datatype expression";
  case SymbolicExprKind::String:
    return "string expression";
  }
  llvm::report_fatal_error("unhandled SymbolicExprKind");
}

LogicalResult mlir::cuda_tile::verifyConstantArgumentKind(
    Operation *op, unsigned index, ConstParamAttr param, Attribute arg) {
  auto expr = llvm::dyn_cast<SymbolicExprAttr>(arg);
  if (!expr) {
    return op->emitOpError("constant argument #")
           << index << " for constant parameter '" << param.getName()
           << "' must be a symbolic expression attribute";
  }

  auto expectedKind = getConstParamKind(op, param);
  if (failed(expectedKind))
    return failure();

  return verifySymbolicExprKind(expr, *expectedKind, [&]() {
    return op->emitOpError("constant argument #")
           << index << " for constant parameter '" << param.getName()
           << "' must be a valid "
           << getSymbolicExprKindDescription(*expectedKind) << ": ";
  });
}

/// Calls instantiate extern declarations, so their constant arguments must be
/// resolvable at the call site. Declaration parameters are intentionally
/// symbolic, but cannot be forwarded as call arguments; compiler-provided
/// context builtins remain valid because they have a bounded definition.
static LogicalResult verifyCallConstArgIsStatic(Operation *op, unsigned index,
                                                ConstParamAttr param,
                                                SymbolicExprAttr expr) {
  if (auto tuple = llvm::dyn_cast<SymbolicTupleAttr>(expr);
      tuple && tuple.getValues().empty()) {
    return op->emitOpError("constant argument #")
           << index << " for constant parameter '" << param.getName()
           << "' must be statically known; empty tuple is not supported";
  }

  StringRef symbolicName;
  WalkResult result = expr.walk([&](SymbolicExprVarAttr var) {
    symbolicName = var.getName();
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted()) {
    return op->emitOpError("constant argument #")
           << index << " for constant parameter '" << param.getName()
           << "' must be statically known; references symbolic variable '"
           << symbolicName << "'";
  }
  return success();
}

static LogicalResult verifyCallConstArgs(CallOp op, Operation *decl) {
  ArrayAttr args = op.getConstArgs();
  ArrayAttr params = decl->getAttrOfType<ArrayAttr>("const_params");
  if (!params)
    params = ArrayAttr::get(op.getContext(), {});

  if (args.size() != params.size()) {
    return op.emitOpError("constant argument count (")
           << args.size()
           << ") must match extern function constant parameter count ("
           << params.size() << ")";
  }

  bool hadError = false;
  for (auto [i, paramAttr] : llvm::enumerate(params)) {
    auto param = llvm::cast<ConstParamAttr>(paramAttr);
    if (failed(verifyConstantArgumentKind(op.getOperation(), i, param,
                                          args[i])))
      hadError = true;
    if (failed(verifyCallConstArgIsStatic(op.getOperation(), i, param,
                                          llvm::cast<SymbolicExprAttr>(args[i]))))
      hadError = true;
  }
  return failure(hadError);
}

/// A declaration may state tiles with symbolic dimensions or a symbolic
/// element type; those are bound later, during parameter concretization.
/// Whatever the declaration already states concretely must agree with the call
/// site, so that e.g. a `tile<ptr<f32>>` cannot reach a `tile<f32>` parameter.
static bool isCompatibleWithDeclaredTile(TileType callSite, TileType declared) {
  if (declared.isConcreteTile())
    return callSite == declared;

  Type declaredElement = declared.getElementType();
  if (!llvm::isa<SymbolicType>(declaredElement) &&
      declaredElement != callSite.getElementType())
    return false;

  // Only enforce shape when the declaration already commits to a static one.
  return !declared.hasStaticShape() ||
         callSite.getShape() == declared.getShape();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  // Unresolved callees are permitted. There is no signature to validate until
  // the referenced symbol is present.
  // If a symbol is present but has the wrong kind, retain the diagnostic below.
  Operation *target = symbolTable.lookupNearestSymbolFrom(*this, fnAttr);
  if (!target)
    return success();

  if (!llvm::isa<FuncOp>(target)) {
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid preview$func";
  }

  auto fn = llvm::cast<FunctionOpInterface>(target);
  auto fnType = llvm::cast<FunctionType>(fn.getFunctionType());
  bool isExtern = target->hasAttr("is_extern");

  if (isExtern) {
    if (failed(verifyCallConstArgs(*this, target)))
      return failure();

    SmallVector<Type> dataOperands;
    SmallVector<Type> dataResults;
    auto collectDataTypes =
        [&](TypeRange types, StringRef kind,
            SmallVectorImpl<Type> &dataTypes) -> LogicalResult {
      for (auto [i, type] : llvm::enumerate(types)) {
        if (llvm::isa<TokenType>(type))
          continue;
        if (!llvm::isa<CudaTileDialectType>(type)) {
          return emitOpError(kind)
                 << " " << i
                 << " must be a cuda_tile type or token when calling an "
                    "extern preview$func, got "
                 << type;
        }
        auto tileType = llvm::dyn_cast<TileType>(type);
        if (!tileType || !tileType.isConcreteTile()) {
          return emitOpError(kind)
                 << " " << i
                 << " must be a concrete 'cuda_tile.tile' at call sites, got "
                 << type;
        }
        dataTypes.push_back(type);
      }
      return success();
    };

    if (failed(collectDataTypes(getOperandTypes(), "operand", dataOperands)))
      return failure();
    if (failed(collectDataTypes(getResultTypes(), "result", dataResults)))
      return failure();

    if (fnType.getNumInputs() != dataOperands.size()) {
      return emitOpError("input count (")
             << dataOperands.size()
             << ") must match extern preview$func input count ("
             << fnType.getNumInputs() << ")";
    }
    if (fnType.getNumResults() != dataResults.size()) {
      return emitOpError("result count (")
             << dataResults.size()
             << ") must match extern preview$func result count ("
             << fnType.getNumResults() << ")";
    }

    auto verifyCompatible = [&](TypeRange callTypes, TypeRange declTypes,
                                StringRef kind) -> LogicalResult {
      for (auto [i, callType, declType] :
           llvm::enumerate(callTypes, declTypes)) {
        auto callTile = llvm::cast<TileType>(callType);
        auto declTile = llvm::dyn_cast<TileType>(declType);
        if (!declTile || !isCompatibleWithDeclaredTile(callTile, declTile)) {
          auto diag = emitOpError(kind) << " type mismatch at index " << i;
          diag.attachNote() << "op " << kind << " types: " << callTypes;
          diag.attachNote() << "function " << kind << " types: " << declTypes;
          return diag;
        }
      }
      return success();
    };

    if (failed(verifyCompatible(dataOperands, fnType.getInputs(), "operand")))
      return failure();
    return verifyCompatible(dataResults, fnType.getResults(), "result");
  }

  if (!getConstArgs().empty()) {
    return emitOpError(
        "constant arguments are only valid when calling an extern "
        "preview$func");
  }

  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (getOperand(i).getType() != fnType.getInput(i)) {
      auto diag = emitOpError("operand type mismatch at index ") << i;
      diag.attachNote() << "op operand types: " << getOperandTypes();
      diag.attachNote() << "function operand types: " << fnType.getInputs();
      return diag;
    }
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }
  }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}
