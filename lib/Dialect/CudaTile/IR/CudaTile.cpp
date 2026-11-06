//===- CudaTile.cpp - CUDA Tile Dialect Op Verifiers ------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Interfaces.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

using namespace mlir;
using namespace mlir::cuda_tile;

/// Attribute name for `OptimizationHintsAttr` on ops that carry it in asm.
static constexpr char kOptimizationHintsAttr[] = "optimization_hints";

int64_t cuda_tile::getMaxSignedValueForBitwidth(int64_t n) {
  assert(n > 0 && n <= 64 && "invalid bitwidth");
  if (n == 64)
    return std::numeric_limits<int64_t>::max();
  return (static_cast<int64_t>(1) << (n - 1)) - 1;
}

int64_t cuda_tile::getMinSignedValueForBitwidth(int64_t n) {
  assert(n > 0 && n <= 64 && "invalid bitwidth");
  if (n == 64)
    return std::numeric_limits<int64_t>::min();
  return -(static_cast<int64_t>(1) << (n - 1));
}

uint64_t cuda_tile::getMaxUnsignedValueForBitwidth(int64_t n) {
  assert(n > 0 && n <= 64 && "invalid bitwidth");
  if (n == 64)
    return std::numeric_limits<uint64_t>::max();
  return (static_cast<int64_t>(1) << n) - 1;
}

cuda_tile::ModuleOp cuda_tile::extractCudaTileModuleOp(Operation *op) {
  // Try direct cast first
  if (auto directCudaTileModule = dyn_cast<cuda_tile::ModuleOp>(op))
    return directCudaTileModule;

  // Try nested case: look inside a regular ModuleOp
  if (auto moduleOp = dyn_cast<mlir::ModuleOp>(op)) {
    if (!moduleOp.getBody()->empty()) {
      if (auto nestedCudaTileModule =
              dyn_cast<cuda_tile::ModuleOp>(&moduleOp.getBody()->front()))
        return nestedCudaTileModule;
    }
  }

  // Not found
  return {};
}

namespace {

//===----------------------------------------------------------------------===//
// Custom Function Signature Parsing for CudaTile Operations
//===----------------------------------------------------------------------===//

// TODO: Leverage upstream changes to strip !cuda_tile. prefix.
/// Custom function signature parsing that uses parseCudaTileType to support
/// both short-form (tile<ptr<f32>>) and long-form
/// (!cuda_tile.tile<ptr<f32>>) types within OpAsmOpInterface default
/// dialect context.
///
/// Standard MLIR parseFunctionSignatureWithArguments() uses generic type
/// parsing that ignores OpAsmOpInterface::getDefaultDialect(), breaking
/// short-form type resolution within cuda_tile.module operations.

/// Validates consistent SSA name usage across function arguments.
static mlir::LogicalResult validateSSANameConsistency(
    mlir::OpAsmParser &parser, mlir::SMLoc loc, bool hasSSAName,
    llvm::ArrayRef<mlir::OpAsmParser::Argument> existingArgs) {
  if (existingArgs.empty())
    return mlir::success();

  bool prevHasSSAName = !existingArgs.back().ssaName.name.empty();
  if (hasSSAName != prevHasSSAName) {
    return parser.emitError(loc, hasSSAName
                                     ? "expected type instead of SSA identifier"
                                     : "expected SSA identifier");
  }
  return mlir::success();
}

static mlir::ParseResult parseOptionalSignatureAttrDict(
    mlir::OpAsmParser &parser, mlir::DictionaryAttr &attrs,
    cuda_tile::SignatureAttrDictParser parseAttrDict) {
  if (parseAttrDict) {
    return parseAttrDict(parser, attrs);
  }

  mlir::NamedAttrList attrList;
  if (parser.parseOptionalAttrDict(attrList)) {
    return mlir::failure();
  }
  attrs = attrList.empty() ? mlir::DictionaryAttr::get(parser.getContext())
                           : attrList.getDictionary(parser.getContext());
  return mlir::success();
}

/// Parses a single function argument with cuda_tile type support.
static mlir::ParseResult parseSingleArgument(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments,
    cuda_tile::SignatureAttrDictParser parseAttrDict) {
  mlir::OpAsmParser::Argument arg;
  arg.ssaName.location = parser.getCurrentLocation();

  // Parse optional SSA name
  auto ssaResult = parser.parseOptionalOperand(arg.ssaName);
  bool hasSSAName = ssaResult.has_value();

  if (hasSSAName) {
    if (mlir::failed(ssaResult.value()) || parser.parseColon())
      return mlir::failure();
  }

  // Validate consistent SSA name usage
  if (mlir::failed(validateSSANameConsistency(parser, arg.ssaName.location,
                                              hasSSAName, arguments)))
    return mlir::failure();

  // Parse type and attributes using cuda_tile-aware parser
  mlir::DictionaryAttr attrs;
  if (parseCudaTileType(parser, arg.type) ||
      parseOptionalSignatureAttrDict(parser, attrs, parseAttrDict) ||
      parser.parseOptionalLocationSpecifier(arg.sourceLoc)) {
    return mlir::failure();
  }

  arg.attrs = attrs;
  arguments.push_back(arg);
  return mlir::success();
}

/// Parses function argument list with variadic support.
static mlir::ParseResult parseFunctionArgumentList(
    mlir::OpAsmParser &parser, bool allowVariadic,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments,
    bool &isVariadic, cuda_tile::SignatureAttrDictParser parseAttrDict) {
  isVariadic = false;

  return parser.parseCommaSeparatedList(
      mlir::OpAsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
        if (isVariadic)
          return parser.emitError(
              parser.getCurrentLocation(),
              "variadic arguments must be at end of argument list");

        // Handle variadic ellipsis
        if (allowVariadic && mlir::succeeded(parser.parseOptionalEllipsis())) {
          isVariadic = true;
          return mlir::success();
        }

        return parseSingleArgument(parser, arguments, parseAttrDict);
      });
}

/// Parses type and attribute pairs for function results.
static mlir::ParseResult
parseTypeAndAttrList(mlir::OpAsmParser &parser,
                     llvm::SmallVectorImpl<mlir::Type> &types,
                     llvm::SmallVectorImpl<mlir::DictionaryAttr> &attrs,
                     cuda_tile::SignatureAttrDictParser parseAttrDict) {
  return parser.parseCommaSeparatedList([&]() -> mlir::ParseResult {
    types.emplace_back();
    attrs.emplace_back();
    if (parseCudaTileType(parser, types.back()) ||
        parseOptionalSignatureAttrDict(parser, attrs.back(), parseAttrDict)) {
      return mlir::failure();
    }
    return mlir::success();
  });
}

/// Parses function result list (single type or parenthesized type list).
static mlir::ParseResult parseFunctionResultList(
    mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Type> &resultTypes,
    llvm::SmallVectorImpl<mlir::DictionaryAttr> &resultAttrs,
    cuda_tile::SignatureAttrDictParser parseAttrDict) {
  if (mlir::failed(parser.parseOptionalLParen())) {
    // Single result type (no parentheses)
    mlir::Type resultType;
    if (parseCudaTileType(parser, resultType))
      return mlir::failure();
    resultTypes.push_back(resultType);
    resultAttrs.push_back(mlir::DictionaryAttr::get(parser.getContext()));
    // Keep bare single-result syntax unambiguous. A following `{` can start a
    // region or an op attr-dict, so single-result attrs are only supported in
    // the parenthesized form, e.g. `-> (T {attrs})`.
    return mlir::success();
  }

  // Parenthesized result list
  if (mlir::succeeded(parser.parseOptionalRParen()))
    return mlir::success(); // Empty result list

  if (parseTypeAndAttrList(parser, resultTypes, resultAttrs, parseAttrDict)) {
    return mlir::failure();
  }
  return parser.parseRParen();
}

} // namespace

/// Main function signature parser with cuda_tile dialect support.
mlir::ParseResult cuda_tile::parseFunctionSignatureWithArguments(
    mlir::OpAsmParser &parser, bool allowVariadic,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments,
    bool &isVariadic, llvm::SmallVectorImpl<mlir::Type> &resultTypes,
    llvm::SmallVectorImpl<mlir::DictionaryAttr> &resultAttrs,
    SignatureAttrDictParser parseAttrDict) {
  if (parseFunctionArgumentList(parser, allowVariadic, arguments, isVariadic,
                                parseAttrDict)) {
    return mlir::failure();
  }
  if (mlir::succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs,
                                   parseAttrDict);
  return mlir::success();
}

/// Print function signature with cuda_tile dialect type support.
static void printFunctionSignatureWithCudaTileTypes(
    OpAsmPrinter &printer, TypeRange argTypes, ArrayAttr argAttrs,
    bool isVariadic, TypeRange resultTypes, ArrayAttr resultAttrs,
    SignatureArgumentPrinter printArgument,
    SignatureAttrDictPrinter printAttrDict) {
  printer << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      printer << ", ";
    printArgument(printer, i);
    printer << ": ";
    printCudaTileType(printer, argTypes[i]);
    if (printAttrDict && argAttrs) {
      printAttrDict(printer, cast<DictionaryAttr>(argAttrs[i]));
    }
  }

  if (isVariadic) {
    if (!argTypes.empty())
      printer << ", ";
    printer << "...";
  }

  printer << ')';

  if (!resultTypes.empty()) {
    printer << " -> ";
    bool singleBareResult =
        resultTypes.size() == 1 &&
        (!resultAttrs || cast<DictionaryAttr>(resultAttrs[0]).empty());
    if (singleBareResult) {
      printCudaTileType(printer, resultTypes[0]);
    } else {
      printer << '(';
      for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
        if (i > 0) {
          printer << ", ";
        }
        printCudaTileType(printer, resultTypes[i]);
        if (printAttrDict && resultAttrs) {
          printAttrDict(printer, cast<DictionaryAttr>(resultAttrs[i]));
        }
      }
      printer << ')';
    }
  }
}

void cuda_tile::printFunctionSignatureWithCudaTileTypes(
    OpAsmPrinter &printer, TypeRange inputs, ArrayAttr argAttrs,
    TypeRange results, ArrayAttr resultAttrs,
    SignatureArgumentPrinter printArgument,
    SignatureAttrDictPrinter printAttrDict) {
  ::printFunctionSignatureWithCudaTileTypes(
      printer, inputs, argAttrs, /*isVariadic=*/false, results, resultAttrs,
      printArgument, printAttrDict);
}

/// Main function signature parser with cuda_tile dialect support, extracting
/// attributes and region from FunctionOpInterface.
void cuda_tile::printFunctionSignatureWithCudaTileTypes(OpAsmPrinter &printer,
                                                        Operation *op,
                                                        TypeRange inputs,
                                                        TypeRange results) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  Region &body = funcOp.getFunctionBody();
  // Declarations (e.g. extern functions) have no entry block, so there are no
  // block-argument values whose SSA names can be printed. Emit positional
  // `%argN` names in that case.
  bool hasEntryBlock = !body.empty();
  ::printFunctionSignatureWithCudaTileTypes(
      printer, inputs, funcOp.getArgAttrsAttr(), /*isVariadic=*/false, results,
      funcOp.getResAttrsAttr(),
      [&](OpAsmPrinter &printer, unsigned i) {
        if (hasEntryBlock) {
          printer.printOperand(body.getArgument(i));
        } else {
          printer << "%arg" << i;
        }
      },
      [](OpAsmPrinter &printer, DictionaryAttr dict) {
        if (!dict.empty()) {
          printer << ' ';
          printer.printAttribute(dict);
        }
      });
}

ArrayAttr
mlir::cuda_tile::buildNormalizedArgAttrs(Builder &builder,
                                         ArrayRef<Attribute> argAttrs) {
  auto isEmptyDict = [](Attribute attr) {
    auto dict = dyn_cast_or_null<DictionaryAttr>(attr);
    return !attr || (dict && dict.empty());
  };
  if (llvm::all_of(argAttrs, isEmptyDict))
    return {};

  SmallVector<Attribute> normalized;
  normalized.reserve(argAttrs.size());
  for (Attribute attr : argAttrs) {
    normalized.push_back(attr ? attr : builder.getDictionaryAttr({}));
  }
  return builder.getArrayAttr(normalized);
}

//===----------------------------------------------------------------------===//
// Custom DenseTypedElementsAttr Parsing
//===----------------------------------------------------------------------===//

// TODO: Leverage upstream changes to strip !cuda_tile. prefix.
static LogicalResult validateIntegerBounds(OpAsmParser &parser, int64_t intVal,
                                           Type elementType, SMLoc loc) {
  if (elementType.isInteger(1)) {
    // Union of signed [-1,1] and unsigned [0,1] = [-1,1]
    if (intVal < -1 || intVal > 1)
      return parser.emitError(loc, "integer constant out of range for type");
  } else if (elementType.isInteger(8)) {
    // Union of signed [-128,127] and unsigned [0,255] = [-128,255]
    if (intVal < -128 || intVal > 255)
      return parser.emitError(loc, "integer constant out of range for type");
  } else if (elementType.isInteger(16)) {
    // Union of signed [-32768,32767] and unsigned [0,65535] = [-32768,65535]
    if (intVal < -32768 || intVal > 65535)
      return parser.emitError(loc, "integer constant out of range for type");
  } else if (elementType.isInteger(32)) {
    // Union of signed [-2^31,2^31-1] and unsigned [0,2^32-1] = [-2^31,2^32-1]
    if (intVal < -2147483648LL || intVal > 4294967295LL)
      return parser.emitError(loc, "integer constant out of range for type");
  } else if (elementType.isInteger(64)) {
    // For i64, int64_t already covers the full signed range [-2^63,2^63-1]
    // The unsigned range [0,2^64-1] extends beyond int64_t, so we accept all
    // int64_t values negative values will be interpreted as large unsigned
    // values in two's complement
  }
  return success();
}

static bool isValidDenseElementType(Type elementType) {
  return elementType.isInteger(1) ||              // i1
         elementType.isInteger(8) ||              // i8
         elementType.isInteger(16) ||             // i16
         elementType.isInteger(32) ||             // i32
         elementType.isInteger(64) ||             // i64
         elementType.isF16() ||                   // f16
         elementType.isBF16() ||                  // bf16
         elementType.isF32() ||                   // f32
         elementType.isF64() ||                   // f64
         elementType.isTF32() ||                  // tf32
         isa<Float8E4M3FNType>(elementType) ||    // f8E4M3FN
         isa<Float8E5M2Type>(elementType) ||      // f8E5M2
         isa<Float8E8M0FNUType>(elementType) ||   // f8E8M0FNU
         isa<Float8E5M3FNUType>(elementType) || // f8E5M3FNU
         isa<Float4E2M1FNType>(elementType); // f4E2M1FN
}

// Decode `0x...` raw-byte storage into a `DenseTypedElementsAttr`.
// Mirrors upstream MLIR's `parseElementAttrHexValues` +
// `TensorLiteralParser::getHexAttr` (in
// mlir/lib/AsmParser/AttributeParser.cpp) so the dialect can round-trip
// the hex form its printer inherits from `DenseElementsAttr::print`
// past `-mlir-print-elementsattrs-with-hex-if-larger`.
static ParseResult parseDenseElementsHexAttr(OpAsmParser &parser, StringRef hex,
                                             Type prefixElementType,
                                             Type resultType,
                                             DenseTypedElementsAttr &attr,
                                             SMLoc hexLoc, SMLoc typeLoc) {
  // Same tile/element-type validation the numeric-literal branch does
  // at the bottom of `parseDenseTypedElementsAttr`; kept inline here
  // so this helper is self-contained and the existing branch is
  // untouched.
  auto tileType = dyn_cast<cuda_tile::TileType>(resultType);
  if (!tileType) {
    return parser.emitError(typeLoc)
           << "expected tile type, got " << resultType;
  }
  if (tileType.getElementType() != prefixElementType) {
    return parser.emitError(typeLoc)
           << "mismatch between the element type: " << prefixElementType
           << " and the tile element type " << tileType.getElementType();
  }

  // As in the literal form, a symbolic shape has no statically known element
  // count for the raw buffer to be validated against.
  if (!tileType.hasStaticShape()) {
    return parser.emitError(typeLoc)
           << "constant tile type must have a static shape, got " << tileType;
  }

  // Upstream's `Token::getHexStringValue` validates `0x` prefix and
  // hex-digit-ness in one shot; `OpAsmParser` doesn't expose `Token`,
  // so we inline it. Lower-case `0x` only, matching upstream.
  if (hex.size() < 2 || hex[0] != '0' || hex[1] != 'x' ||
      !llvm::all_of(hex.drop_front(2), llvm::isHexDigit)) {
    return parser.emitError(
        hexLoc, "expected string containing hex digits starting with `0x`");
  }

  SmallVector<char> rawData;
  llvm::append_range(rawData, llvm::fromHex(hex.drop_front(2)));

  // Big-endian hosts: the printer emits little-endian bytes (matching
  // on-disk bytecode); convert in-place before constructing the attr.
  if (llvm::endianness::native == llvm::endianness::big) {
    SmallVector<char> conv(rawData.size());
    DenseTypedElementsAttr::convertEndianOfArrayRefForBEmachine(
        rawData, MutableArrayRef<char>(conv), tileType);
    rawData = std::move(conv);
  }

  ArrayRef<char> buffer(rawData);
  if (!DenseElementsAttr::isValidRawBuffer(tileType, buffer)) {
    return parser.emitError(typeLoc)
           << "elements hex data size is invalid for provided type: "
           << tileType;
  }

  attr = llvm::cast<DenseTypedElementsAttr>(
      DenseElementsAttr::getFromRawBuffer(tileType, buffer));
  return success();
}

// Parse format: constant <f32: 0x7F800000> : tile<f32>
static ParseResult parseDenseTypedElementsAttr(OpAsmParser &parser,
                                               DenseTypedElementsAttr &attr,
                                               Type &resultType) {
  if (parser.parseLess())
    return failure();

  // We use the prefix element type to understand how to parse the dense values.
  Type prefixElementType;
  if (parseCudaTileType(parser, prefixElementType))
    return parser.emitError(parser.getCurrentLocation())
           << "expect element type to be one of i1 or i8 or i16 or i32 or i64 "
              "or f16 "
              "or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 or "
              "f8E8M0FNU or f4E2M1FN"
              " or f8E5M3FNU"
              " values, but got "
           << prefixElementType;

  // Validate that prefixElementType is one of the allowed types
  if (!isValidDenseElementType(prefixElementType)) {
    return parser.emitError(parser.getCurrentLocation())
           << "expect element type to be one of i1 or i8 or i16 or i32 or i64 "
              "or f16 "
              "or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 or "
              "f8E8M0FNU or f4E2M1FN"
              " or f8E5M3FNU"
              " values, but got "
           << prefixElementType;
  }

  bool isInteger = prefixElementType.isIntOrIndex();

  if (parser.parseColon())
    return failure();

  // Hex blob form: `<elem: "0x...."> : tile<...>`. Upstream MLIR's
  // `DenseTypedElementsAttr` printer switches to this form once the
  // element count exceeds `-mlir-print-elementsattrs-with-hex-if-larger`
  // (default 100). Accept the same shape so print/parse is a fixed
  // point.
  std::string hexValueStorage;
  SMLoc hexLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalString(&hexValueStorage))) {
    if (parser.parseGreater() || parser.parseColon()) {
      return failure();
    }
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parseCudaTileType(parser, resultType)) {
      return failure();
    }
    return parseDenseElementsHexAttr(parser, hexValueStorage, prefixElementType,
                                     resultType, attr, hexLoc, typeLoc);
  }

  SmallVector<APFloat> floatValues;
  SmallVector<int64_t> integerValues;
  SmallVector<int64_t> inferredShape;

  //===----------------------------------------------------------------------===//
  // Helper Functions for Enhanced Dense Parsing
  //===----------------------------------------------------------------------===//

  // Parse a single numeric value (integer or float, positive or negative)
  auto parseNumericValue = [&]() -> ParseResult {
    SMLoc loc = parser.getCurrentLocation();

    if (isInteger) {
      // Error when true or false passed to an int that is not an i1
      if (prefixElementType.getIntOrFloatBitWidth() != 1 &&
          (succeeded(parser.parseOptionalKeyword("true")) ||
           succeeded(parser.parseOptionalKeyword("false"))))
        return parser.emitError(loc, "expected integer value");

      int64_t intVal;
      if (parser.parseInteger(intVal))
        return parser.emitError(loc, "expected integer value");

      // Validate the integer fits in the target type
      if (failed(validateIntegerBounds(parser, intVal, prefixElementType, loc)))
        return failure();

      integerValues.push_back(intVal);
      return success();
    }

    assert(!isInteger && "expect integer but parsing a float");
    const llvm::fltSemantics *targetSemantics = &APFloat::IEEEdouble();
    APFloat floatValue(APFloat::IEEEdouble());
    if (auto floatType = dyn_cast<FloatType>(prefixElementType))
      targetSemantics = &floatType.getFloatSemantics();

    if (succeeded(parser.parseFloat(*targetSemantics, floatValue))) {
      floatValues.push_back(floatValue);
      return success();
    }
    return failure();
  };

  //===----------------------------------------------------------------------===//
  // Main Parsing Logic - Recursive Array Structure with Shape Tracking
  //===----------------------------------------------------------------------===//

  // Parse nested array structure or single scalar with shape tracking
  std::function<ParseResult(SmallVectorImpl<int64_t> &)>
      parseNestedArrayWithShape =
          [&](SmallVectorImpl<int64_t> &currentShape) -> ParseResult {
    // Parse array structure with brackets
    if (succeeded(parser.parseOptionalLSquare())) {
      SmallVector<int64_t> elementShape;
      int64_t elementCount = 0;

      // Parse each element in the array
      auto parseArrayElement = [&]() -> ParseResult {
        elementCount++;

        // Handle nested arrays (recursive case)
        if (succeeded(parser.parseOptionalLSquare())) {
          SmallVector<int64_t> firstElementShape;
          int64_t nestedElementCount = 0;
          bool isFirstElement = true;

          // Parse comma-separated nested elements
          if (failed(parser.parseCommaSeparatedList([&]() -> ParseResult {
                nestedElementCount++;
                SmallVector<int64_t> currentElementShape;
                if (failed(parseNestedArrayWithShape(currentElementShape)))
                  return failure();

                // Capture shape from first element for consistency checking
                if (isFirstElement) {
                  firstElementShape = currentElementShape;
                  isFirstElement = false;
                } else {
                  // Validate shape consistency across all elements
                  if (currentElementShape != firstElementShape) {
                    return parser.emitError(
                        parser.getCurrentLocation(),
                        "tensor literal is invalid; ranks are not consistent "
                        "between elements");
                  }
                }
                return success();
              })))
            return failure();

          if (failed(parser.parseRSquare()))
            return failure();

          // Build shape for this nested array: [count] + [first_element_shape]
          SmallVector<int64_t> thisNestedShape;
          thisNestedShape.push_back(nestedElementCount);
          thisNestedShape.append(firstElementShape.begin(),
                                 firstElementShape.end());

          // Use first element's shape as template for remaining elements
          if (elementShape.empty()) {
            elementShape = thisNestedShape;
          } else {
            // Validate consistency with previous elements
            if (thisNestedShape != elementShape) {
              return parser.emitError(
                  parser.getCurrentLocation(),
                  "tensor literal is invalid; ranks are not consistent "
                  "between elements");
            }
          }
          return success();
        }
        return parseNumericValue();
      };

      // Parse all elements in the array
      if (failed(parser.parseCommaSeparatedList(parseArrayElement)))
        return failure();

      if (failed(parser.parseRSquare()))
        return failure();

      // Build final shape: [element_count] + [element_shape]
      currentShape.push_back(elementCount);
      currentShape.append(elementShape.begin(), elementShape.end());
      return success();
    }
    return parseNumericValue();
  };

  // Parse the value (can be scalar or nested array)
  if (failed(parseNestedArrayWithShape(inferredShape)))
    return failure();

  if (parser.parseGreater())
    return failure();

  // Parse colon and then the type to determine how to interpret values
  if (parser.parseColon())
    return failure();

  SMLoc typeLoc = parser.getCurrentLocation();
  if (parseCudaTileType(parser, resultType))
    return failure();

  // Create dense attribute with the tile type
  auto tileType = dyn_cast<cuda_tile::TileType>(resultType);
  if (!tileType) {
    return parser.emitError(typeLoc)
           << "result #0 must be tile of i1 or i8 or i16 or i32 or i64 or f16 "
              "or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 or "
              "f8E8M0FNU or f4E2M1FN values, but got "
           << resultType;
  }
  auto elementType = tileType.getElementType();
  if (prefixElementType != elementType)
    return parser.emitError(typeLoc)
           << "mismatch between the element type: " << prefixElementType
           << " and the tile element type " << elementType;

  // A dense value needs a statically known element count, which a symbolic
  // shape does not provide. Reject it here: building the attribute for such a
  // type asserts inside DenseElementsAttr.
  if (!tileType.hasStaticShape()) {
    return parser.emitError(typeLoc)
           << "constant tile type must have a static shape, got " << tileType;
  }

  // Verify shape consistency
  ArrayRef<int64_t> expectedShape = tileType.getShape();

  // Format a shape array as a string for error messages: [1,2,3]
  auto formatShapeForError = [](ArrayRef<int64_t> shape) -> std::string {
    std::string shapeStr;
    llvm::raw_string_ostream os(shapeStr);
    os << "[";
    llvm::interleaveComma(shape, os);
    os << "]";
    return os.str();
  };

  // For scalar tiles, we should have a single value with no shape dimensions
  if (expectedShape.empty()) {
    if (!inferredShape.empty()) {
      // Format inferred shape for error message using helper
      return parser.emitError(typeLoc) << "inferred shape of elements literal ("
                                       << formatShapeForError(inferredShape)
                                       << ") does not match type ([])";
    }
  } else {
    // Allow scalar (empty inferred shape) to match any expected shape (splat
    // behavior) Only validate shape if we have a non-scalar input
    if (!inferredShape.empty() && inferredShape != expectedShape) {
      // Format both shapes for error message using helper
      return parser.emitError(typeLoc)
             << "inferred shape of elements literal ("
             << formatShapeForError(inferredShape) << ") does not match type ("
             << formatShapeForError(expectedShape) << ")";
    }
  }

  if (integerValues.empty() && floatValues.empty())
    return parser.emitError(parser.getCurrentLocation(),
                            "dense attribute cannot be empty");

  // Determine if we should interpret as float or integer based on element type
  if (elementType.isIntOrIndex()) {
    if (elementType.isInteger(1)) {
      SmallVector<bool> boolValues;
      for (int64_t val : integerValues)
        boolValues.push_back(val != 0);
      attr = llvm::cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<bool>(boolValues)));
    } else if (elementType.isInteger(8)) {
      SmallVector<int8_t> i8Values;
      for (int64_t val : integerValues)
        i8Values.push_back(static_cast<int8_t>(val));
      attr = llvm::cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int8_t>(i8Values)));
    } else if (elementType.isInteger(16)) {
      SmallVector<int16_t> i16Values;
      for (int64_t val : integerValues)
        i16Values.push_back(static_cast<int16_t>(val));
      attr = llvm::cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int16_t>(i16Values)));
    } else if (elementType.isInteger(32)) {
      SmallVector<int32_t> i32Values;
      for (int64_t val : integerValues)
        i32Values.push_back(static_cast<int32_t>(val));
      attr = llvm::cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int32_t>(i32Values)));
    } else if (elementType.isInteger(64)) {
      attr = llvm::cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int64_t>(integerValues)));
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unsupported integer type");
    }
  } else { // Handle floating point numerical values.
    assert(isa_and_nonnull<FloatType>(elementType) && "expect a float type");
    attr = llvm::cast<DenseTypedElementsAttr>(
        DenseElementsAttr::get(tileType, ArrayRef<APFloat>(floatValues)));
  }
  return success();
}

// constant <f32: 42.0> : tile<f32>
static void printDenseTypedElementsAttr(OpAsmPrinter &p, Operation *op,
                                        DenseTypedElementsAttr attr,
                                        Type resultType) {
  // Print the dense values part (everything before the colon)
  std::string attrStr;
  llvm::raw_string_ostream attrStream(attrStr);
  attr.print(attrStream);
  attrStream.flush();
  auto tileType = cast<cuda_tile::TileType>(resultType);

  // Find the colon separator
  size_t colonPos = attrStr.find(" : ");
  if (colonPos != std::string::npos) {
    p << "<";
    p << tileType.getElementType();
    p << ": ";
    // Print everything before the colon, but skip the first 6 characaters:
    // dense<
    p << attrStr.substr(6, colonPos - 6);
    // Print the colon and space
    p << " : ";
    // Print the type using custom printer to omit cuda_tile prefix
    printCudaTileType(p, resultType);
  } else {
    // Fallback to default printing if something goes wrong
    p.printAttribute(attr);
  }
}

static ParseResult
parseDenseTypedElementsAttrNoResult(OpAsmParser &parser,
                                    DenseTypedElementsAttr &attr) {
  Type resultType;
  return parseDenseTypedElementsAttr(parser, attr, resultType);
}

static void printDenseTypedElementsAttrNoResult(OpAsmPrinter &p, Operation *op,
                                                DenseTypedElementsAttr attr) {
  printDenseTypedElementsAttr(p, op, attr, attr.getType());
}

//===----------------------------------------------------------------------===//
// Signedness parsing
//===----------------------------------------------------------------------===//

static ParseResult parseSignedness(OpAsmParser &parser, SignednessAttr &attr) {
  StringRef enumKeyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&enumKeyword)))
    return parser.emitError(loc)
           << "expected signedness to be one of: {'signed', 'unsigned'}";
  auto maybeEnum = symbolizeSignedness(enumKeyword);
  if (!maybeEnum)
    return parser.emitError(loc)
           << "expected signedness to be one of: {'signed', 'unsigned'}";
  attr = SignednessAttr::get(parser.getContext(), *maybeEnum);
  return success();
}

static void printSignedness(OpAsmPrinter &p, Operation *op,
                            SignednessAttr attr) {
  p << stringifySignedness(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Comparison Predicate parsing
//===----------------------------------------------------------------------===//

static ParseResult parseComparisonPredicate(OpAsmParser &parser,
                                            ComparisonPredicateAttr &attr) {
  StringRef enumKeyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&enumKeyword)))
    return parser.emitError(loc)
           << "expected 'comparison_predicate' to be one "
              "of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', "
              "'greater_than', 'greater_than_or_equal'}";
  auto maybeEnum = symbolizeComparisonPredicate(enumKeyword);
  if (!maybeEnum)
    return parser.emitError(loc)
           << "expected 'comparison_predicate' to be one "
              "of: {'equal', 'not_equal', 'less_than', 'less_than_or_equal', "
              "'greater_than', 'greater_than_or_equal'}";
  attr = ComparisonPredicateAttr::get(parser.getContext(), *maybeEnum);
  return success();
}

static void printComparisonPredicate(OpAsmPrinter &p, Operation *op,
                                     ComparisonPredicateAttr attr) {
  p << stringifyComparisonPredicate(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Comparison Ordering parsing
//===----------------------------------------------------------------------===//

static ParseResult parseComparisonOrdering(OpAsmParser &parser,
                                           ComparisonOrderingAttr &attr) {
  StringRef enumKeyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&enumKeyword)))
    return parser.emitError(loc) << "expected 'comparison_ordering' to be one "
                                    "of: {'ordered', 'unordered'}";
  auto maybeEnum = symbolizeComparisonOrdering(enumKeyword);
  if (!maybeEnum)
    return parser.emitError(loc) << "expected 'comparison_ordering' to be one "
                                    "of: {'ordered', 'unordered'}";
  attr = ComparisonOrderingAttr::get(parser.getContext(), *maybeEnum);
  return success();
}

static void printComparisonOrdering(OpAsmPrinter &p, Operation *op,
                                    ComparisonOrderingAttr attr) {
  p << stringifyComparisonOrdering(attr.getValue());
}

//===----------------------------------------------------------------------===//
// Rounding Mode parsing
//===----------------------------------------------------------------------===//

static void printRoundingModeIfNotRN(OpAsmPrinter &p, Operation *op,
                                     RoundingModeAttr attr) {
  if (attr.getValue() == RoundingMode::NEAREST_EVEN)
    return;
  p << "rounding<";
  p << stringifyRoundingMode(attr.getValue());
  p << ">";
}

static ParseResult parseRoundingModeWithModes(
    OpAsmParser &parser, RoundingModeAttr &attr,
    ArrayRef<StringRef> allowedModes,
    std::function<ParseResult(OpAsmParser &, RoundingMode, StringRef)>
        validator = nullptr,
    RoundingMode defaultMode = RoundingMode::NEAREST_EVEN) {
  // Try to parse the optional "rounding" keyword
  if (succeeded(parser.parseOptionalKeyword("rounding"))) {
    // If "rounding" keyword is found, we must parse the full syntax:
    // rounding<mode>
    if (parser.parseLess())
      return failure();

    // Parse the rounding mode string
    StringRef roundingModeStr;
    if (parser.parseKeyword(&roundingModeStr))
      return failure();

    if (parser.parseGreater())
      return failure();

    // Convert string to RoundingMode enum
    auto roundingMode = symbolizeRoundingMode(roundingModeStr);
    if (!roundingMode.has_value()) {
      auto diag = parser.emitError(parser.getCurrentLocation())
                  << "expected rounding mode to be one of: ";
      llvm::interleaveComma(allowedModes, diag, [&](StringRef mode) {
        diag << "'" << mode << "'";
      });
      return diag;
    }

    // Apply custom validation if provided
    if (validator) {
      if (failed(validator(parser, roundingMode.value(), roundingModeStr)))
        return failure();
    }

    attr = RoundingModeAttr::get(parser.getContext(), roundingMode.value());
  } else {
    // No "rounding" keyword found, use the specified default rounding mode
    attr = RoundingModeAttr::get(parser.getContext(), defaultMode);
  }
  return success();
}

static ParseResult parseDivFOpRoundingMode(OpAsmParser &parser,
                                           RoundingModeAttr &attr) {
  static const StringRef allowedModes[] = {
      "nearest_even", "zero", "negative_inf", "positive_inf", "approx", "full"};
  return parseRoundingModeWithModes(parser, attr, allowedModes);
}

static void printDivFOpRoundingMode(OpAsmPrinter &p, Operation *op,
                                    RoundingModeAttr attr) {
  printRoundingModeIfNotRN(p, op, attr);
}

static ParseResult parseSqrtOpRoundingMode(OpAsmParser &parser,
                                           RoundingModeAttr &attr) {
  static const StringRef allowedModes[] = {
      "nearest_even", "zero", "negative_inf", "positive_inf", "approx"};
  return parseRoundingModeWithModes(parser, attr, allowedModes);
}

static void printSqrtOpRoundingMode(OpAsmPrinter &p, Operation *op,
                                    RoundingModeAttr attr) {
  printRoundingModeIfNotRN(p, op, attr);
}

static ParseResult parseTanHOpRoundingMode(OpAsmParser &parser,
                                           RoundingModeAttr &attr) {
  static const StringRef allowedModes[] = {"approx", "full"};
  return parseRoundingModeWithModes(parser, attr, allowedModes, nullptr,
                                    RoundingMode::FULL);
}

static void printTanHOpRoundingMode(OpAsmPrinter &p, Operation *op,
                                    RoundingModeAttr attr) {
  if (attr.getValue() == RoundingMode::FULL)
    return;
  p << "rounding<";
  p << stringifyRoundingMode(attr.getValue());
  p << ">";
}

static ParseResult parseExpOpRoundingMode(OpAsmParser &parser,
                                          RoundingModeAttr &attr) {
  static const StringRef allowedModes[] = {"approx", "full"};
  return parseRoundingModeWithModes(parser, attr, allowedModes, nullptr,
                                    RoundingMode::FULL);
}

static void printExpOpRoundingMode(OpAsmPrinter &p, Operation *op,
                                   RoundingModeAttr attr) {
  if (attr.getValue() == RoundingMode::FULL)
    return;
  p << "rounding<";
  p << stringifyRoundingMode(attr.getValue());
  p << ">";
}

static void printIEEERoundingMode(OpAsmPrinter &p, Operation *op,
                                  RoundingModeAttr attr) {
  printRoundingModeIfNotRN(p, op, attr);
}

static ParseResult parseRoundingModeWithModes(
    OpAsmParser &parser, RoundingModeAttr &attr,
    ArrayRef<StringRef> allowedModes, RoundingMode defaultMode,
    std::function<ParseResult(OpAsmParser &, RoundingMode, StringRef)>
        validator = nullptr) {
  // Try to parse the optional "rounding" keyword
  if (succeeded(parser.parseOptionalKeyword("rounding"))) {
    // If "rounding" keyword is found, we must parse the full syntax:
    // rounding<mode>
    if (parser.parseLess())
      return failure();

    // Parse the rounding mode string
    StringRef roundingModeStr;
    if (parser.parseKeyword(&roundingModeStr))
      return failure();

    if (parser.parseGreater())
      return failure();

    // Convert string to RoundingMode enum
    auto roundingMode = symbolizeRoundingMode(roundingModeStr);
    if (!roundingMode.has_value()) {
      auto diag = parser.emitError(parser.getCurrentLocation())
                  << "expected rounding mode to be one of: ";
      llvm::interleaveComma(allowedModes, diag, [&](StringRef mode) {
        diag << "'" << mode << "'";
      });
      return diag << ", got: "
                  << "'" << roundingModeStr << "'";
    }

    // Apply custom validation if provided
    if (validator) {
      if (failed(validator(parser, roundingMode.value(), roundingModeStr))) {
        auto diag = parser.emitError(parser.getCurrentLocation())
                    << "expected rounding mode to be one of: ";
        llvm::interleaveComma(allowedModes, diag, [&](StringRef mode) {
          diag << "'" << mode << "'";
        });
        return diag << ", got: "
                    << "'" << roundingModeStr << "'";
      }
    }

    attr = RoundingModeAttr::get(parser.getContext(), roundingMode.value());
  } else {
    // No "rounding" keyword found, use the specified default rounding mode
    attr = RoundingModeAttr::get(parser.getContext(), defaultMode);
  }
  return success();
}

static ParseResult parseIntegerRoundingMode(OpAsmParser &parser,
                                            RoundingModeAttr &attr) {
  static const StringRef allowedModes[] = {"nearest_int_to_zero"};

  auto intgerValidator = [](OpAsmParser &parser, RoundingMode roundingMode,
                            StringRef roundingModeStr) -> ParseResult {
    // Only allow integer rounding modes
    if (roundingMode != RoundingMode::NEAREST_INT_TO_ZERO)
      return failure();
    return success();
  };

  return parseRoundingModeWithModes(parser, attr, allowedModes,
                                    RoundingMode::NEAREST_INT_TO_ZERO,
                                    intgerValidator);
}

static void printIntegerRoundingMode(OpAsmPrinter &printer, Operation *op,
                                     RoundingModeAttr attr) {
  if (attr.getValue() == RoundingMode::NEAREST_INT_TO_ZERO)
    return;
  printer << "rounding<";
  printer << stringifyRoundingMode(attr.getValue());
  printer << ">";
}

static ParseResult parseIEEERoundingMode(OpAsmParser &parser,
                                         RoundingModeAttr &attr) {
  static const StringRef allowedModes[] = {
      "nearest_even", "zero", "negative_inf", "positive_inf", "nearest_away"};

  auto ieeeValidator = [](OpAsmParser &parser, RoundingMode roundingMode,
                          StringRef roundingModeStr) -> ParseResult {
    // Only allow IEEE rounding modes
    if (roundingMode != RoundingMode::NEAREST_EVEN &&
        roundingMode != RoundingMode::ZERO &&
        roundingMode != RoundingMode::NEGATIVE_INF &&
        roundingMode != RoundingMode::POSITIVE_INF &&
        roundingMode != RoundingMode::NEAREST_AWAY) {
      return failure();
    }
    return success();
  };

  return parseRoundingModeWithModes(parser, attr, allowedModes,
                                    RoundingMode::NEAREST_EVEN, ieeeValidator);
}

//===----------------------------------------------------------------------===//
// Assume Predicate parsing (allows attributes without # and cuda_tile prefix)
//===----------------------------------------------------------------------===//

static ParseResult parseAssumePredicate(OpAsmParser &parser,
                                        AssumePredicateAttrInterface &attr) {
  SMLoc loc = parser.getCurrentLocation();

  // Try parsing full attribute syntax first (#cuda_tile.div_by<...>)
  Attribute parsedAttr;
  auto parseResult = parser.parseOptionalAttribute(parsedAttr);
  if (parseResult.has_value()) {
    if (succeeded(*parseResult)) {
      if (auto assumeAttr =
              dyn_cast<AssumePredicateAttrInterface>(parsedAttr)) {
        attr = assumeAttr;
        return success();
      }
      return parser.emitError(loc) << "expected assume predicate attribute";
    }
    return *parseResult;
  }

  // Try parsing shortened syntax (div_by<...> or same_elements<...>)
  StringRef attrName;
  if (failed(parser.parseKeyword(&attrName)))
    return parser.emitError(loc) << "expected attribute name";

  if (attrName == "div_by") {
    // Reuse existing DivByAttr::parse method
    if (auto parsedAttr = DivByAttr::parse(parser, Type{})) {
      attr = static_cast<AssumePredicateAttrInterface>(parsedAttr);
      return success();
    }
    return failure();

  } else if (attrName == "same_elements") {
    // Reuse existing SameElementsAttr::parse method
    if (auto parsedAttr = SameElementsAttr::parse(parser, Type{})) {
      attr = static_cast<AssumePredicateAttrInterface>(parsedAttr);
      return success();
    }
    return failure();

  } else if (attrName == "bounded") {
    // Parse bounded predicate (no parameters needed)
    if (auto parsedAttr = BoundedAttr::parse(parser, Type{})) {
      attr = static_cast<AssumePredicateAttrInterface>(parsedAttr);
      return success();
    }
    return failure();

  } else {
    return parser.emitError(loc)
           << "unknown assume predicate attribute: " << attrName
           << " (expected 'div_by', 'same_elements', or 'bounded')";
  }
}

static void printAssumePredicate(OpAsmPrinter &p, Operation *op,
                                 AssumePredicateAttrInterface attr) {
  // Print the attribute to a string stream to get the full representation
  std::string attrStr;
  llvm::raw_string_ostream attrStream(attrStr);
  attr.print(attrStream);
  attrStream.flush();

  // Remove the #cuda_tile. prefix if present
  const std::string prefix = "#cuda_tile.";
  if (StringRef(attrStr).starts_with(prefix)) {
    // Print without the prefix
    p << StringRef(attrStr).drop_front(prefix.size());
  } else {
    // Fallback to default printing if prefix not found
    p.printAttribute(attr);
  }
}

//===----------------------------------------------------------------------===//
// Control Flow Op Utilies
//===----------------------------------------------------------------------===//

template <typename OpT>
static ParseResult
parseControlFlowRegion(OpAsmParser &p, Region &region,
                       ArrayRef<OpAsmParser::Argument> arguments = {}) {
  if (failed(p.parseRegion(region, arguments)))
    return failure();
  OpT::ensureTerminator(region, p.getBuilder(),
                        p.getEncodedSourceLoc(p.getNameLoc()));
  return success();
}
static ParseResult parseIfOpRegion(OpAsmParser &p, Region &region) {
  return parseControlFlowRegion<IfOp>(p, region);
}

template <typename ImplicitTerminatorOpT, typename OpT>
static void printControlFlowRegion(OpAsmPrinter &p, OpT op, Region &region) {
  // We do not print the terminator if it is implicit and has no operands.
  bool printBlockTerminators =
      region.front().getTerminator()->getNumOperands() != 0 ||
      !isa<ImplicitTerminatorOpT>(region.front().getTerminator());
  p.printRegion(region, /*printEntryBlockArgs=*/false, printBlockTerminators);
}
static void printIfOpRegion(OpAsmPrinter &p, IfOp op, Region &region) {
  printControlFlowRegion<YieldOp>(p, op, region);
}

//===----------------------------------------------------------------------===//
// Custom Region Parsing/Printing
//===----------------------------------------------------------------------===//

ParseResult parseArgumentRegion(OpAsmParser &parser, Region &region) {
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic;
  if (parseFunctionArgumentList(parser, /*allowVariadic=*/false, arguments,
                                isVariadic, /*parseAttrDict=*/{})) {
    return failure();
  }
  return parser.parseRegion(region, arguments);
}

template <typename OpT>
void printArgumentRegion(OpAsmPrinter &p, OpT op, Region &region) {
  p.printNewline();
  printFunctionSignatureWithCudaTileTypes(
      p, region.getArgumentTypes(), /*argAttrs=*/{}, /*resultTypes=*/{},
      /*resultAttrs=*/{},
      [&](OpAsmPrinter &printer, unsigned i) {
        printer.printOperand(region.getArgument(i));
      },
      /*printAttrDict=*/{});
  p << ' ';
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// View Load and Store Utilities
//===----------------------------------------------------------------------===//

// Parses memory ordering semantics and scope attributes for token-ordered
// operations
static ParseResult
parseMemoryAttributes(OpAsmParser &parser,
                      MemoryOrderingSemanticsAttr &memoryOrderingSemantics,
                      MemoryScopeAttr &memoryScopeAttr) {
  // Step 1. Parse memory ordering semantics.
  SMLoc loc = parser.getCurrentLocation();
  StringRef memorySem;
  (void)parser.parseKeyword(&memorySem);
  auto attrOptional = symbolizeMemoryOrderingSemantics(memorySem);
  if (!attrOptional) {
    return parser.emitError(loc) << "invalid memory_ordering_semantics "
                                    "attribute specification. Got \""
                                 << memorySem
                                 << "\" but expect one of: weak, relaxed, "
                                    "acquire, release, acq_rel";
  }
  memoryOrderingSemantics = MemoryOrderingSemanticsAttr::get(
      parser.getBuilder().getContext(), *attrOptional);

  // Step 2. Parse memory scope (only specific valid keywords).
  loc = parser.getCurrentLocation();
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword))) {
    // We succeeded to parse an optional keyword. Make sure it is not
    // conflicting with "weak".
    if (attrOptional.value() == cuda_tile::MemoryOrderingSemantics::WEAK) {
      return parser.emitError(loc)
             << "operation specifies weak memory ordering semantics, but then "
                "provides \""
             << keyword << "\" scope, expected no memory scope.";
    }

    auto attr = symbolizeMemoryScope(keyword);
    if (!attr) {
      return parser.emitError(loc)
             << "invalid memory_scope attribute specification. Got \""
             << keyword << "\" but expect one of: tl_blk, device, sys";
    }
    memoryScopeAttr =
        MemoryScopeAttr::get(parser.getBuilder().getContext(), *attr);
  }
  return success();
}

static void
printMemoryAttributes(OpAsmPrinter &printer, Operation *,
                      MemoryOrderingSemanticsAttr memoryOrderingSemantics,
                      MemoryScopeAttr memoryScopeAttr) {
  printer << ' ' << memoryOrderingSemantics.getValue();
  if (memoryScopeAttr)
    printer << ' ' << memoryScopeAttr.getValue();
}

//===----------------------------------------------------------------------===//
// Debuginfo Verifier
//===----------------------------------------------------------------------===//

/// Verifies that the debug info for a given function and its ops is valid.
/// Rules:
/// Rule 1: If a function has scope, it must have subprogram scope.
/// Rule 2: If a function has subprogram scope, the function name must match
/// the subprogram scope linkage name.
/// Rule 3: If a function does not have scope, its operations must not have
/// scope.
/// Rule 4: Operation scope must match function scope.
/// Rule 5: Global variables must not have scope.
/// Rule 6: Function location must not be a CallSiteLoc.
class DebugInfoVerifier {
public:
  /// Verify the debug info for a CudaTile function.
  static LogicalResult verifyFunc(FunctionOpInterface func) {
    // Rule 6: Function location must not be a CallSiteLoc.
    if (isa<CallSiteLoc>(func.getLoc()))
      return func.emitOpError()
             << "invalid function debug info location: " << func.getLoc()
             << ". Function location must not be a CallSiteLoc.";

    // We only need to verify DILocAttr location types.
    if (auto diLoc = getDILoc(func.getLoc())) {
      // Rule 1: If a function has scope, it must have subprogram scope.
      auto subprogram = dyn_cast<DISubprogramAttr>(diLoc.getScope());
      if (!subprogram)
        return func.emitOpError()
               << "invalid function debug info scope: " << diLoc.getScope()
               << ". Function location must have cuda_tile.di_subprogram "
                  "debug info scope.";
      // Rule 2: If a function has subprogram scope, the function name must
      // match the subprogram scope linkage name.
      if (subprogram.getLinkageName() != func.getName())
        return func.emitOpError()
               << "invalid function debug info scope: " << subprogram
               << ". Function name \"" << func.getName()
               << "\" does not match subprogram scope linkage name "
               << subprogram.getLinkageName() << ".";
    }
    return success();
  }

  /// Verify the debug info for all ops in a CudaTile function.
  static LogicalResult verifyFuncBody(FunctionOpInterface func) {
    DISubprogramAttr fnSubprogram;
    if (auto diLoc = getDILoc(func.getLoc()))
      fnSubprogram = getSubprogram(diLoc.getScope());

    // Walk through all operations in the function, including those within
    // control flow regions.
    LogicalResult result = success();
    func.walk([&](Operation *op) {
      DISubprogramAttr opSubprogram;
      if (auto diLoc = getDILoc(op->getLoc()))
        opSubprogram = getSubprogram(diLoc.getScope());

      if (opSubprogram) {
        // Rule 3: If a function does not have scope, its operations must not
        // have scope.
        if (!fnSubprogram) {
          result = op->emitOpError()
                   << "invalid operation debug info scope: " << opSubprogram
                   << ". Operation has debug info scope, but function debug "
                      "info scope is undefined.";
          return WalkResult::interrupt();
        }
        // Rule 4: Operation scope must match function scope.
        if (fnSubprogram != opSubprogram) {
          result = op->emitError()
                   << "invalid operation debug info scope: " << opSubprogram
                   << ". Operation debug info scope does not match function "
                      "debug info scope: "
                   << fnSubprogram << ".";
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return result;
  }

  /// Verify the debug info for a CudaTile module.
  static LogicalResult verifyModule(cuda_tile::ModuleOp module) {
    for (auto &op : module.getOps())
      if (!isa<FunctionOpInterface>(op))
        // Rule 5: Global variables must not have scope.
        if (auto diLoc = getDILoc(op.getLoc()))
          return op.emitOpError()
                 << "invalid operation debug info scope: " << diLoc
                 << ". Global variables must not have scope.";
    return success();
  }

private:
  /// Returns a subprogram attribute for a given local scope attribute.
  static DISubprogramAttr getSubprogram(DILocalScopeAttr scope) {
    return TypeSwitch<DILocalScopeAttr, DISubprogramAttr>(scope)
        .Case([](DISubprogramAttr subprogram) { return subprogram; })
        .Case([](DILexicalBlockAttr block) {
          return getSubprogram(block.getScope());
        })
        .Default([](DILocalScopeAttr) { return DISubprogramAttr(); });
  }

  /// Returns a CudaTile location for a given location attribute.
  static DILocAttr getDILoc(LocationAttr loc) {
    return TypeSwitch<LocationAttr, DILocAttr>(loc)
        .Case([](DILocAttr diLoc) { return diLoc; })
        .Case([](CallSiteLoc callSiteLoc) {
          return getDILoc(callSiteLoc.getCaller());
        })
        .Case([](FusedLoc fusedLoc) {
          for (auto subloc : fusedLoc.getLocations())
            if (auto diLoc = getDILoc(subloc))
              return diLoc;
          return DILocAttr();
        })
        .Case([](NameLoc nameLoc) { return getDILoc(nameLoc.getChildLoc()); })
        .Case([](OpaqueLoc opaqueLoc) {
          return getDILoc(opaqueLoc.getFallbackLocation());
        })
        .Default([](LocationAttr) { return DILocAttr(); });
  }
};

LogicalResult cuda_tile::impl::verifyFuncDebugInfo(FunctionOpInterface funcOp) {
  return DebugInfoVerifier::verifyFunc(funcOp);
}

LogicalResult
cuda_tile::impl::verifyFuncBodyDebugInfo(FunctionOpInterface funcOp) {
  return DebugInfoVerifier::verifyFuncBody(funcOp);
}

//===----------------------------------------------------------------------===//
// Custom parsers/printers for operations
//===----------------------------------------------------------------------===//

static ParseResult
parseSymbolVisibility(OpAsmParser &parser,
                      cuda_tile::SymbolVisibilityAttr &attr) {
  // Only parse valid visibility keywords using restricted list.
  // This ensures we don't consume other keywords like "constant" or
  // "alignment".
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword, {"public", "private"}))) {
    auto visibility = cuda_tile::symbolizeSymbolVisibility(keyword);
    attr =
        cuda_tile::SymbolVisibilityAttr::get(parser.getContext(), *visibility);
    return success();
  }

  // Default to public for backward compatibility if no visibility keyword found
  attr = cuda_tile::SymbolVisibilityAttr::get(
      parser.getContext(), cuda_tile::SymbolVisibility::Public);
  return success();
}

static void printSymbolVisibility(OpAsmPrinter &printer, Operation *op,
                                  cuda_tile::SymbolVisibilityAttr attr) {
  // Omit 'public' since it's the default value
  if (attr.getValue() != cuda_tile::SymbolVisibility::Public)
    printer << stringifySymbolVisibility(attr.getValue()) << " ";
}

// `inbounds = [true, false, ...]` parser for use as a custom directive in
// declarative assembly formats. When the clause is absent, synthesize the
// required all-false attribute from the already-parsed index rank.
static ParseResult
parseInbounds(OpAsmParser &parser, DenseBoolArrayAttr &inbounds,
              ArrayRef<OpAsmParser::UnresolvedOperand> indices) {
  if (failed(parser.parseOptionalKeyword("inbounds"))) {
    inbounds = parser.getBuilder().getDenseBoolArrayAttr(
        SmallVector<bool>(indices.size(), /*value=*/false));
    return success();
  }
  if (parser.parseEqual()) {
    return failure();
  }

  SmallVector<bool> values;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
            StringRef tok;
            auto loc = parser.getCurrentLocation();
            if (parser.parseKeyword(&tok)) {
              return failure();
            }
            if (tok == "true") {
              values.push_back(true);
            } else if (tok == "false") {
              values.push_back(false);
            } else {
              return parser.emitError(loc)
                     << "expected 'true' or 'false' in inbounds list, got '"
                     << tok << "'";
            }
            return success();
          })) {
    return failure();
  }

  inbounds = parser.getBuilder().getDenseBoolArrayAttr(values);
  return success();
}

static void printInboundsList(OpAsmPrinter &printer, ArrayRef<bool> inbounds) {
  printer << "inbounds = [";
  llvm::interleaveComma(inbounds, printer,
                        [&](bool v) { printer << (v ? "true" : "false"); });
  printer << "]";
}

// Companion printer for declarative assembly-format users. Elide the all-false
// value; that is the canonical textual spelling for the parser-synthesized
// conservative default.
static void printInbounds(OpAsmPrinter &printer, Operation * /*op*/,
                          DenseBoolArrayAttr inbounds, ValueRange /*indices*/) {
  if (!inbounds ||
      !llvm::any_of(inbounds.asArrayRef(), [](bool v) { return v; })) {
    return;
  }
  printInboundsList(printer, inbounds.asArrayRef());
}

static void printInbounds(OpAsmPrinter &printer, ArrayRef<bool> inbounds,
                          bool elideAllFalse = false) {
  if (elideAllFalse && !llvm::any_of(inbounds, [](bool v) { return v; })) {
    return;
  }
  printer << " ";
  printInboundsList(printer, inbounds);
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.cpp.inc"

namespace mlir::cuda_tile {
ParseResult parseOptionalCompactConstArgs(OpAsmParser &parser,
                                          ArrayAttr &result);
void printOptionalCompactConstArgs(OpAsmPrinter &printer, Operation *op,
                                   ArrayAttr args);
} // namespace mlir::cuda_tile

#define GET_OP_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/Enums.cpp.inc"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// Common helpers for canonicalization
//===----------------------------------------------------------------------===//

/// Try to get constant bool defined by given Value
/// tile<i1> or tile<...xi1> is expected for defining ConstantOp
static std::optional<bool> getConstantBoolValue(Value value) {
  auto cond = value.getDefiningOp<ConstantOp>();
  if (!cond)
    return std::nullopt;
  auto type = cond.getType().getElementType();
  auto intType = llvm::dyn_cast<IntegerType>(type);
  if (!intType || intType.getWidth() != 1)
    return std::nullopt;
  DenseTypedElementsAttr cstAttr = cond.getValue();
  if (cstAttr.isSplat() || cstAttr.size() == 1)
    return *cstAttr.getValues<bool>().begin();
  return std::nullopt;
}

static inline bool isConstantTrueVal(Value value) {
  auto val = getConstantBoolValue(value);
  return val && *val;
}

static inline bool isConstantFalseVal(Value value) {
  auto val = getConstantBoolValue(value);
  return val && !(*val);
}

/// Check whether all elements of a constant integer value match |pred|.
static bool
isConstantIntMatching(Value value,
                      llvm::function_ref<bool(const APInt &)> pred) {
  auto constVal = value.getDefiningOp<cuda_tile::ConstantOp>();
  if (!constVal)
    return false;
  auto intType = dyn_cast<IntegerType>(constVal.getType().getElementType());
  if (!intType)
    return false;
  DenseTypedElementsAttr cstAttr = constVal.getValue();
  if (cstAttr.isSplat()) {
    return pred(cstAttr.getSplatValue<APInt>());
  }
  return llvm::all_of(cstAttr.getValues<APInt>(), pred);
}

static bool isConstantOnesValue(Value value) {
  return isConstantIntMatching(value, [](const APInt &v) { return v == 1; });
}

static bool isConstantZeroValue(Value value) {
  return isConstantIntMatching(value,
                               [](const APInt &v) { return v.isZero(); });
}

// Helper function to insert SelectOp for given cond & values
static inline Value createSelectOpByType(PatternRewriter &rewriter,
                                         Location loc, Value cond,
                                         Value trueVal, Value falseVal) {
  Type ty = trueVal.getType();
  // We should call this function only for TileType
  // TokenType is handled in IfOp canonicalization patterns
  // and TensorView & TileView types are not supported as IfOp yield types
  assert(isa<TileType>(ty) && "Only TileType is supported by SelectOp");

  auto tileType = llvm::cast<TileType>(ty);
  auto shape = tileType.getShape();
  if (shape.empty())
    return SelectOp::create(rewriter, loc, cond, trueVal, falseVal);

  auto condType = llvm::cast<TileType>(cond.getType());
  auto reshape = ReshapeOp::create(
      rewriter, loc, reshapeTileTypeToRank(condType, tileType.getRank()), cond);
  auto broadcast =
      BroadcastOp::create(rewriter, loc, getI1SameShape(tileType), reshape);
  return SelectOp::create(rewriter, loc, broadcast, trueVal, falseVal);
}

// Helper function to insert XOrIOp with tile of ones
static inline Value createXOrForValue(PatternRewriter &rewriter, Location loc,
                                      Value cond) {
  auto condType = llvm::cast<TileType>(cond.getType());
  TileType constType = getI1SameShape(condType);
  llvm::APInt val(1, 1);
  auto constAttr = DenseIntElementsAttr::get(constType, val);
  auto constOp = ConstantOp::create(rewriter, loc, constType, constAttr);
  return XOrIOp::create(rewriter, loc, cond, constOp);
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "OpsCanonicalization.inc"
} // namespace

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static inline LogicalResult verifyIEEERoundingModes(OpTy op) {
  auto rounding = op.getRoundingMode();
  if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                           RoundingMode::NEGATIVE_INF,
                           RoundingMode::POSITIVE_INF},
                          rounding)) {
    return op.emitOpError("invalid rounding mode specified, expect "
                          "one of [nearest_even, zero, negative_inf, "
                          "positive_inf]");
  }
  return success();
}

LogicalResult AddFOp::verify() {
  if (failed(verifyIEEERoundingModes(*this)))
    return failure();
  return verifyFtz(*this, getFlushToZero());
}

// Canonicalize add operations to put multiply operations on the LHS
// This enables FMA fusion patterns to work more reliably

LogicalResult canonicalizeAddOperands(AddFOp op, PatternRewriter &rewriter) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();

  // Check if RHS is a multiply and LHS is not
  bool rhsIsMul = isa_and_nonnull<MulFOp>(rhs.getDefiningOp());
  bool lhsIsMul = isa_and_nonnull<MulFOp>(lhs.getDefiningOp());

  // If RHS is multiply but LHS is not, swap them
  if (rhsIsMul && !lhsIsMul) {
    rewriter.replaceOpWithNewOp<AddFOp>(op, rhs, lhs, op.getRoundingModeAttr(),
                                        op.getFlushToZeroAttr());
    return success();
  }
  return failure();
}

LogicalResult AddFOp::canonicalize(AddFOp op, PatternRewriter &rewriter) {
  return canonicalizeAddOperands(op, rewriter);
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

LogicalResult AllocaOp::verify() {
  return verifyAlloca<AllocaOp, PointerType>(*this);
}

//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

LogicalResult AssumeOp::verify() {
  return getPredicate().verifyWithAssumeOp(getOperation());
}

void AssumeOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  Operation *producer = getValue().getDefiningOp();
  if (producer) {
    if (auto opAsmOpIface = dyn_cast<OpAsmOpInterface>(producer)) {
      std::string name = "assume_";
      opAsmOpIface.getAsmResultNames([&](Value v, StringRef valueName) {
        if (v == getValue())
          name += valueName;
      });
      setNameFn(getResult(), name);
      return;
    }
  }
  setNameFn(getResult(), "assume");
}

OpFoldResult AssumeOp::fold(FoldAdaptor adaptor) {
  if (auto producerOp = this->getValue().getDefiningOp<AssumeOp>()) {
    if (producerOp.getPredicate() == this->getPredicate())
      return producerOp.getResult();
  }
  return {};
}

namespace {

LogicalResult verifyMultimemSupportedElementType(Operation *op,
                                                 Type elemTy) {
  if (elemTy.isIntOrIndex()) {
    const int bits = static_cast<int>(elemTy.getIntOrFloatBitWidth());
    if (bits != 32 && bits != 64)
      return op->emitOpError(
          "multimem requires 32- or 64-bit integer/index element types; got ")
             << bits << " bits";
    return success();
  }
  if (elemTy.isFloat()) {
    if (!elemTy.isF16() && !elemTy.isBF16() && !elemTy.isF32() &&
        !elemTy.isF64())
      return op->emitOpError(
          "multimem supports only f16, bf16, f32, and f64 floating-point "
          "element types");
    return success();
  }
  return op->emitOpError(
      "multimem expects integer, index, or standard floating-point tensor "
      "elements");
}

LogicalResult verifyMultimemReductionElementType(Operation *op,
                                                 MultimemReductionKind kind,
                                                 Type elemTy) {
  switch (kind) {
  case MultimemReductionKind::AND:
  case MultimemReductionKind::OR:
  case MultimemReductionKind::XOR:
  case MultimemReductionKind::UMIN:
  case MultimemReductionKind::UMAX:
    if (elemTy.isFloat())
      return op->emitOpError(
          "and/or/xor/umin/umax reductions require integer element types");
    break;
  case MultimemReductionKind::MIN:
  case MultimemReductionKind::MAX:
    // multimem.reduce accepts only ADD for floating-point elements, whereas
    // multimem.load_reduce also supports MIN/MAX for f16 and bf16.
    if (isa<MultimemReduceViewTkoOp>(op) && elemTy.isFloat()) {
      return op->emitOpError(
          "supports only add for floating-point element types");
    }
    if (elemTy.isFloat() && !elemTy.isF16() && !elemTy.isBF16())
      return op->emitOpError(
                 "min/max supports f16 and bf16 float types only; got ")
             << elemTy;
    break;
  default:
    break;
  }
  return success();
}

LogicalResult verifyMultimemViewAccessContract(Operation *op, Type viewType) {
  auto partitionView = dyn_cast<PartitionViewType>(viewType);
  if (!partitionView)
    return op->emitOpError("multimem view operations support only "
                           "partition_view; got ")
           << viewType;

  TensorViewType tensorView = partitionView.getTensorView();
  Type elementType = tensorView.getElementType();
  if (!isa<Float16Type, BFloat16Type>(elementType))
    return success();

  size_t unitStrideCount = llvm::count(tensorView.getStrides(), int64_t{1});

  if (unitStrideCount != 1)
    return op->emitOpError()
           << "f16/bf16 multimem views must have exactly one element stride "
              "statically known to equal 1; got "
           << unitStrideCount;

  return success();
}

/// Verifies that the total tile size for a multimem op is a multiple of 4
/// bytes for float element types.  PTX multimem instructions require a minimum
/// 32-bit (4-byte) transaction granularity; tiles smaller than 4 bytes total
/// cannot be lowered correctly.  Integer types are excluded because they use
/// scalar (num_rep=1) paths regardless of tile size.
LogicalResult verifyMultimemFloatTileSize(Operation *op, TileType tileTy) {
  Type elemTy = tileTy.getElementType();
  // Only enforce for float types; integer/index types use scalar PTX paths.
  if (!isa<FloatType>(elemTy))
    return success();
  unsigned bitsPerElt = elemTy.getIntOrFloatBitWidth();
  int64_t totalElements = tileTy.getNumElements();
  int64_t totalBits = totalElements * static_cast<int64_t>(bitsPerElt);
  if (totalBits % 32 != 0)
    return op->emitOpError(
               "multimem op requires tile total size to be a multiple of 4 "
               "bytes for float element types; got ")
           << totalElements << " element(s) of " << bitsPerElt
           << " bits (total " << totalBits << " bits)";
  return success();
}

/// When the view traces to make_tensor_view, require multicast pointer base.
LogicalResult verifyMultimemTensorViewMulticastBase(Operation *op,
                                                             Value viewVal) {
  llvm::DenseSet<Value> visited;
  Value cur = viewVal;
  while (Operation *defOp = cur.getDefiningOp()) {
    if (!visited.insert(cur).second)
      return op->emitOpError(
          "cyclic view use-def chain while resolving multimem tensor_view base");
    if (auto mtv = dyn_cast<MakeTensorViewOp>(defOp)) {
      auto baseTy = dyn_cast<TileType>(mtv.getBase().getType());
      if (!baseTy)
        return success();
      Type elemTy = baseTy.getElementType();
      auto ptrTy = dyn_cast<PointerType>(elemTy);
      if (!ptrTy) {
        return op->emitOpError(
            "multimem requires a make_tensor_view base of scalar tile "
            "`!cuda_tile.tile<!cuda_tile.ptr<..., #cuda_tile.ptr_attr<multicast>>>`, "
            "got element type ")
            << elemTy;
      }
      auto ptrAttr = ptrTy.getPtrAttr();
      if (!ptrAttr || ptrAttr.getValue() != PtrAttr::MULTICAST) {
        return op->emitOpError("expected a multicast pointer, but got ")
               << (ptrAttr ? stringifyPtrAttr(ptrAttr.getValue()) : "none");
      }
      return success();
    }
    if (auto mpv = dyn_cast<MakePartitionViewOp>(defOp)) {
      cur = mpv.getTensorView();
      continue;
    }
    if (auto msv = dyn_cast<MakeStridedViewOp>(defOp)) {
      cur = msv.getTensorView();
      continue;
    }
    if (auto mgsv = dyn_cast<MakeGatherScatterViewOp>(defOp)) {
      cur = mgsv.getTensorView();
      continue;
    }
    return success();
  }
  return success();
}

} // namespace

// Verify that an explicit `inbounds` annotation has one entry per index
// dimension. Used by the view ops that accept an `inbounds` annotation.
template <typename OpTy>
static LogicalResult verifyInboundsSize(OpTy op) {
  std::optional<ArrayRef<bool>> inbounds = op.getInbounds();
  if (!inbounds) {
    return success();
  }
  if (inbounds->size() != op.getIndex().size()) {
    return op.emitOpError("inbounds size (")
           << inbounds->size()
           << ") must match the number of index dimensions ("
           << op.getIndex().size() << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MultimemLoadReduceViewTkoOp
//===----------------------------------------------------------------------===//

LogicalResult MultimemLoadReduceViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(getOperation(), getView(), getIndex(),
                                       getReducedValues())))
    return failure();
  auto reducedTileTy = dyn_cast<TileType>(getReducedValues().getType());
  if (!reducedTileTy)
    return emitOpError("expected reduced values to be a cuda_tile tile type");
  Type elemTy = reducedTileTy.getElementType();
  if (failed(verifyMultimemSupportedElementType(getOperation(), elemTy)))
    return failure();
  if (failed(verifyMultimemFloatTileSize(getOperation(), reducedTileTy)))
    return failure();
  if (failed(verifyMultimemReductionElementType(getOperation(), getReduction(),
                                                elemTy))) {
    return failure();
  }
  if (failed(verifyMultimemTensorViewMulticastBase(getOperation(), getView())))
    return failure();
  if (failed(verifyMultimemPtrAttr(*this, getView())))
    return failure();
  if (failed(verifyMultimemViewAccessContract(getOperation(),
                                              getView().getType())))
    return failure();
  if (failed(verifyInboundsSize(*this))) {
    return failure();
  }
  return impl::verifyMemoryModelLoad(*this, getMemoryOrderingSemantics(),
                                     getMemoryScope());
}

//===----------------------------------------------------------------------===//
// MultimemReduceViewTkoOp
//===----------------------------------------------------------------------===//

LogicalResult MultimemReduceViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(getOperation(), getView(), getIndex(),
                                       getTile())))
    return failure();
  auto valuesTileTy = dyn_cast<TileType>(getTile().getType());
  if (!valuesTileTy)
    return emitOpError("expected tile operand to be a cuda_tile tile type");
  Type elemTy = valuesTileTy.getElementType();
  if (failed(verifyMultimemSupportedElementType(getOperation(), elemTy)))
    return failure();
  if (failed(verifyMultimemFloatTileSize(getOperation(), valuesTileTy)))
    return failure();
  if (failed(verifyMultimemReductionElementType(getOperation(), getReduction(),
                                                elemTy))) {
    return failure();
  }
  if (failed(verifyMultimemTensorViewMulticastBase(getOperation(), getView())))
    return failure();
  if (failed(verifyMultimemPtrAttr(*this, getView())))
    return failure();
  if (failed(verifyMultimemViewAccessContract(getOperation(),
                                              getView().getType())))
    return failure();
  if (failed(verifyInboundsSize(*this))) {
    return failure();
  }
  auto sem = getMemoryOrderingSemantics();
  if (sem != MemoryOrderingSemantics::RELAXED &&
      sem != MemoryOrderingSemantics::RELEASE) {
    return emitOpError("memory ordering semantics must be one of: "
                       "relaxed, release");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MultimemStoreViewTkoOp
//===----------------------------------------------------------------------===//

LogicalResult MultimemStoreViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(getOperation(), getView(), getIndex(),
                                       getTile())))
    return failure();
  auto valuesTileTy = dyn_cast<TileType>(getTile().getType());
  if (!valuesTileTy)
    return emitOpError("expected tile operand to be a cuda_tile tile type");
  if (failed(verifyMultimemSupportedElementType(
          getOperation(), valuesTileTy.getElementType())))
    return failure();
  if (failed(verifyMultimemFloatTileSize(getOperation(), valuesTileTy)))
    return failure();
  if (failed(verifyMultimemTensorViewMulticastBase(getOperation(), getView())))
    return failure();
  if (failed(verifyMultimemPtrAttr(*this, getView())))
    return failure();
  if (failed(verifyMultimemViewAccessContract(getOperation(),
                                              getView().getType())))
    return failure();
  if (failed(verifyInboundsSize(*this))) {
    return failure();
  }
  return impl::verifyMemoryModelStore(*this, getMemoryOrderingSemantics(),
                                      getMemoryScope());
}

//===----------------------------------------------------------------------===//
// AtomicRMWTkoOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicRMWTkoOp::verify() {
  Type ptrElemTy = getPointers().getType().getElementType();
  auto ptrType = dyn_cast<cuda_tile::PointerType>(ptrElemTy);
  if (!ptrType)
    return emitOpError("expected pointer element type, got ") << ptrElemTy;
  // Unicast (and default) pointers are valid for atomic RMW; multicast is not.
  if (auto ptrAttr = ptrType.getPtrAttr();
      ptrAttr && ptrAttr.getValue() == PtrAttr::MULTICAST) {
    return emitOpError("multicast pointer is not supported for atomic_rmw");
  }
  Type pointeeType = ptrType.getPointeeType();
  Type argElType = getArg().getType().getElementType();
  if (pointeeType != argElType)
    return emitOpError("expected pointee type (")
           << pointeeType << ") to match element type of 'arg' (" << argElType
           << ")";

  // We cannot add to AllShapesMatch since it is an optional argument.
  auto mask = getMask();
  if (mask && cast<ShapedType>(mask.getType()).getShape() !=
                  cast<ShapedType>(getArg().getType()).getShape())
    return emitOpError(
        "failed to verify that all of {pointers, arg, mask} have same shape");

  // Check compatibility of RMW mode with element type.
  if (failed(verifyAtomicRMWMode(*this, getMode(), argElType)))
    return failure();

  // Verify memory ordering semantics.
  if (failed(verifyAtomicMemoryOrdering(*this, getMemoryOrderingSemantics())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// AtomicRedViewTkoOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicRedViewTkoOp::verify() {
  auto viewType = dyn_cast<cuda_tile::TileView>(getView().getType());
  if (!viewType)
    return emitOpError("view operand must be a TileView type");

  if (isa<cuda_tile::GatherScatterViewType>(getView().getType())) {
    return emitOpError(
        "gather_scatter_view is not supported; use partition_view instead");
  }

  auto viewTileType = cast<cuda_tile::TileType>(viewType.getViewTileType());
  Type viewElemType = viewTileType.getElementType();
  Type valueElType = getValue().getType().getElementType();

  if (viewElemType != valueElType)
    return emitOpError("view element type (")
           << viewElemType << ") must match value element type (" << valueElType
           << ")";

  // For element-wise atomics, view tile shape must match value shape
  ArrayRef<int64_t> viewShape = viewTileType.getShape();
  ArrayRef<int64_t> valueShape =
      cast<ShapedType>(getValue().getType()).getShape();

  if (viewShape != valueShape)
    return emitOpError("view tile shape ")
           << viewShape << " must match value shape " << valueShape
           << " for element-wise atomic operations";

  if (getMode() == AtomicRMWMode::XCHG) {
    return emitOpError("atomic_red_view_tko op cannot use xchg operation");
  }

  // Check compatibility of RMW mode with element type.
  if (failed(verifyAtomicRMWMode(*this, getMode(), valueElType)))
    return failure();

  // Verify `inbounds` length matches the number of tile coordinates.
  if (failed(verifyInboundsSize(*this))) {
    return failure();
  }

  // Memory ordering is restricted to RELAXED only.
  // RELAXED maps to PTX .relaxed.gpu for the atomic reduction to
  // GMEM, while implicit loads are weak (value not returned). This enables TMA
  // REDG (cp.reduce.async.bulk) optimization on Hopper+ GPUs.
  if (getMemoryOrderingSemantics() != MemoryOrderingSemantics::RELAXED) {
    return emitOpError(
        "only 'relaxed' memory ordering is supported for view-based "
        "atomic reductions");
  }

  // For TMA compatibility (cp.reduce.async.bulk), system scope is not
  // supported. Only tl_blk and device scopes are allowed.
  if (getMemoryScope() == MemoryScope::SYS)
    return emitOpError("memory scope 'sys' is not supported for view-based "
                       "atomic reductions; use 'tl_blk' or 'device' for TMA "
                       "compatibility");

  // Verify index count matches view tile rank.
  size_t expectedRank = viewTileType.getShape().size();
  if (getIndex().size() != expectedRank) {
    return emitOpError("expected ")
           << expectedRank << " index operand(s) for view with tile rank "
           << expectedRank << ", got " << getIndex().size();
  }

  // Verify each index is a scalar (rank-0) integer tile.
  for (auto [i, idx] : llvm::enumerate(getIndex())) {
    auto idxTy = cast<cuda_tile::TileType>(idx.getType());
    if (!idxTy.getShape().empty()) {
      return emitOpError("index operand ")
             << i << " must be a scalar tile (rank 0), got shape "
             << idxTy.getShape();
    }
    if (!isa<mlir::IntegerType>(idxTy.getElementType())) {
      return emitOpError("index operand ")
             << i << " must be a scalar integer tile, got element type "
             << idxTy.getElementType();
    }
  }

  if (failed(verifyUnrestrictedViewPtrAttr(*this, getView()))) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AtomicCASTkoOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicCASTkoOp::verify() {
  Type ptrElemTy = getPointers().getType().getElementType();
  auto ptrType = dyn_cast<cuda_tile::PointerType>(ptrElemTy);
  if (!ptrType)
    return emitOpError("expected pointer element type, got ") << ptrElemTy;
  // Unicast (and default) pointers are valid for atomic CAS; multicast is not.
  if (auto ptrAttr = ptrType.getPtrAttr();
      ptrAttr && ptrAttr.getValue() == PtrAttr::MULTICAST) {
    return emitOpError("multicast pointer is not supported for atomic_cas");
  }
  Type pointeeType = ptrType.getPointeeType();
  Type valElType = getVal().getType().getElementType();
  if (pointeeType != valElType)
    return emitOpError("expected pointee type (")
           << pointeeType << ") to match element type of 'val' (" << valElType
           << ")";
  if (!isa<FloatType>(valElType) && !isa<IntegerType>(valElType))
    return emitOpError("expect only float or integer types with 32 or 64 bit");
  unsigned bitWidth = valElType.getIntOrFloatBitWidth();
  if (bitWidth != 32 && bitWidth != 64)
    return emitOpError("expect only float or integer types with 32 or 64 bit");

  // We cannot add to AllShapesMatch since it is an optional argument.
  auto mask = getMask();
  if (mask && cast<ShapedType>(mask.getType()).getShape() !=
                  cast<ShapedType>(getVal().getType()).getShape())
    return emitOpError("failed to verify that all of {pointers, val, cmp and "
                       "mask} have same shape");

  // Verify memory ordering semantics.
  if (failed(verifyAtomicMemoryOrdering(*this, getMemoryOrderingSemantics())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastOp::verify() {
  auto srcType = getSource().getType().getElementType();
  auto resType = getResult().getType().getElementType();

  // All numeric conversions are allowed if bitwidths match
  if (srcType.getIntOrFloatBitWidth() == resType.getIntOrFloatBitWidth())
    return success();

  return emitOpError("types must be equal width")
         << ", cannot convert " << getSource().getType() << " of width "
         << srcType.getIntOrFloatBitWidth() << " to type "
         << getResult().getType() << " of width "
         << resType.getIntOrFloatBitWidth();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  auto srcTy = getSource().getType();
  auto resultTy = getResult().getType();

  for (auto [srcDim, resultDim] :
       llvm::zip_equal(srcTy.getShape(), resultTy.getShape()))
    if (srcDim != resultDim && srcDim != 1)
      return emitOpError("expects the shape of source tile to be compatible "
                         "with that of the result tile")
             << ", but got: " << srcTy.getShape() << " and "
             << resultTy.getShape();
  return success();
}

void BroadcastOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "bcast");
}

//===----------------------------------------------------------------------===//
// CatOp
//===----------------------------------------------------------------------===//

LogicalResult CatOp::verify() {
  int64_t dim = getDim();
  auto lhsTileType = cast<cuda_tile::TileType>(getLhs().getType());
  auto rhsTileType = cast<cuda_tile::TileType>(getRhs().getType());
  auto resultTileType = cast<cuda_tile::TileType>(getResult().getType());
  // lhs and rhs have the same rank.
  int64_t rank = lhsTileType.getRank();
  if (dim < 0 || dim >= rank)
    return emitOpError("expect dim to be [0, ")
           << rank << "), but got: " << dim;

  ArrayRef<int64_t> lhsShape = lhsTileType.getShape();
  ArrayRef<int64_t> rhsShape = rhsTileType.getShape();
  assert(lhsShape.size() == rhsShape.size() && "expect same rank");
  // Verify for the result dimensions
  ArrayRef<int64_t> resultShape = resultTileType.getShape();
  if (lhsShape[dim] + rhsShape[dim] != resultShape[dim]) {
    return emitOpError("invalid concat at position ")
           << dim << ", expected: " << lhsShape[dim] + rhsShape[dim]
           << " but got: " << resultShape[dim];
  }
  for (int64_t idx = 0; idx < rank; idx++) {
    if (idx != dim) {
      if (lhsShape[idx] != rhsShape[idx]) {
        return emitOpError("expect lhs and rhs shapes to match at "
                           "non-concat position ")
               << idx << ", expected: " << lhsShape[idx]
               << " but got: " << rhsShape[idx];
      }
      if (lhsShape[idx] != resultShape[idx]) {
        return emitOpError("expect lhs and result shapes to match at "
                           "non-concat position ")
               << idx << ", expected: " << lhsShape[idx]
               << " but got: " << resultShape[idx];
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

void ConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  Type type = getType().getElementType();
  DenseTypedElementsAttr cstAttr = getValue();
  auto intType = llvm::dyn_cast<IntegerType>(type);

  // Sugar i1 constants with 'true' and 'false'.
  if (intType && intType.getWidth() == 1) {
    if (llvm::all_of(cstAttr.getValues<bool>(), [](bool v) { return v; }))
      return setNameFn(getResult(), "true");
    if (llvm::all_of(cstAttr.getValues<bool>(), [](bool v) { return !v; }))
      return setNameFn(getResult(), "false");
    return setNameFn(getResult(), "cst_i1");
  }

  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "cst";
  if (cstAttr.isSplat() || cstAttr.size() == 1) {
    auto intData = cstAttr.tryGetValues<APInt>();
    if (succeeded(intData))
      specialName << "_" << *intData->begin();
    else {
      auto floatData = cstAttr.tryGetValues<APFloat>();
      if (succeeded(floatData)) {
        APFloat fElt = *floatData->begin();
        if (fElt.isNaN()) {
          specialName << "_NaN";
        } else {
          llvm::APFloat::integerPart parts[2] = {0, 0}; // enough for 128 bits
          bool exact;
          fElt.convertToInteger(llvm::MutableArrayRef(parts),
                                /*Width=*/64,
                                /*IsSigned=*/false, llvm::APFloat::rmTowardZero,
                                &exact);
          uint64_t val = parts[0];
          if (exact)
            specialName << "_" << val;
        }
      }
    }
  }
  specialName << "_" << type;
  return setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

/// Utility verifier that checks that the given early exit operation is nested
/// within an allowed loop.
template <typename... AllowedLoopOpsT>
static LogicalResult verifyEarlyExitOp(Operation *earlyExitOp) {
  // Find the ancestor loop operation.
  Region *parentRegion = nullptr;
  Operation *parentLoop = earlyExitOp;
  while (true) {
    parentRegion = parentLoop->getParentRegion();
    parentLoop = parentRegion->getParentOp();
    if (isa<AllowedLoopOpsT...>(parentLoop))
      break;
    if (!isa<IfOp>(parentLoop)) {
      InFlightDiagnostic diag = earlyExitOp->emitOpError(
          "can only be nested within a ancestor chain of '");
      llvm::interleave(
          ArrayRef<StringRef>(
              {AllowedLoopOpsT::getOperationName()..., "cuda_tile.if"}),
          diag, "', '");
      diag << "' operations";
      return diag.attachNote(parentLoop->getLoc())
             << "see unexpected ancestor operation";
    }
  }
  return success();
}

LogicalResult BreakOp::verify() {
  auto res = verifyEarlyExitOp<LoopOp>(*this);
  if (failed(res))
    return res;

  // Verify that the operand types match the parent loop results types.
  auto parentLoop = this->getOperation()->getParentOfType<cuda_tile::LoopOp>();
  assert((parentLoop != nullptr) && "break has no enclosing LoopOp");
  if (parentLoop->getResultTypes() != this->getOperandTypes()) {
    return emitOpError("operand types must correspond to the parent loop "
                       "result types: ")
           << "(" << this->getOperandTypes() << ") vs ("
           << parentLoop->getResultTypes() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

LogicalResult ContinueOp::verify() {
  auto res = verifyEarlyExitOp<ForOp, LoopOp>(*this);
  if (failed(res))
    return res;

  // Find the nearest ancestor loop (can be LoopOp or ForOp)
  Region *parentRegion = nullptr;
  Operation *parentLoop = this->getOperation();
  while (true) {
    parentRegion = parentLoop->getParentRegion();
    parentLoop = parentRegion->getParentOp();
    if (isa<cuda_tile::LoopOp, cuda_tile::ForOp>(parentLoop))
      break;
    assert((parentLoop != nullptr) &&
           "continue op has no enclosing LoopOp or ForOp");
  }
  // Verify that the operand types match the parent loop types
  if (isa<LoopOp>(parentLoop)) {
    // Continue inside Loop yields to next iteration, must match iter_values
    if (parentLoop->getOperandTypes() != this->getOperandTypes()) {
      return emitError(
                 "`loop` is missing a valid terminator. `continue` op should ")
             << "have operand types that match the parent loop iter_values: ("
             << parentLoop->getOperandTypes() << "), but found: ("
             << this->getOperandTypes() << ")";
    }
  } else if (parentLoop->getResultTypes() != this->getOperandTypes()) { // ForOp
    return emitError(
               "`for` is missing a valid terminator. `continue` op should ")
           << "have operand types that match the parent loop return types: ("
           << parentLoop->getResultTypes() << "), but found: ("
           << this->getOperandTypes() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetIndexSpaceShapeOp
//===----------------------------------------------------------------------===//

LogicalResult GetIndexSpaceShapeOp::verify() {
  TileView srcType = getSrc().getType();

  auto results = getResultTypes();
  if (results.size() != srcType.getViewIndexRank())
    return emitOpError("expected ")
           << srcType.getViewIndexRank()
           << " results due to view index space rank, but got "
           << results.size();

  return success();
}

void GetIndexSpaceShapeOp::print(OpAsmPrinter &p) {
  p << " " << getSrc() << " : ";
  printCudaTileType(p, getSrc().getType());
  p << " -> ";
  printCudaTileType(p, getResultTypes()[0]);
}

ParseResult GetIndexSpaceShapeOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  if (parser.parseOperand(src) || parser.parseColon())
    return failure();

  Type resultType;
  Type srcType;
  SMLoc srcTypeLoc = parser.getCurrentLocation();
  if (parseCudaTileType(parser, srcType) || parser.parseArrow() ||
      parseCudaTileType(parser, resultType))
    return failure();

  TileView srcTensorViewType = llvm::dyn_cast<TileView>(srcType);
  if (!srcTensorViewType)
    return parser.emitError(srcTypeLoc, "expected tile view, got ") << srcType;

  if (failed(parser.resolveOperand(src, srcType, result.operands)))
    return failure();

  size_t rank = srcTensorViewType.getViewIndexRank();
  for (size_t i = 0; i < rank; i++)
    result.addTypes(resultType);

  return success();
}

//===----------------------------------------------------------------------===//
// GetTensorShapeOp
//===----------------------------------------------------------------------===//

LogicalResult GetTensorShapeOp::verify() {
  TensorViewType srcType = getSrc().getType();

  auto results = getResultTypes();
  if (results.size() != srcType.getShape().size())
    return emitOpError("expected ")
           << srcType.getShape().size()
           << " results due to tensor rank, but got " << results.size();

  return success();
}

void GetTensorShapeOp::print(OpAsmPrinter &p) {
  p << " " << getSrc() << " : ";
  printCudaTileType(p, getSrc().getType());
  p << " -> ";
  printCudaTileType(p, getResultTypes()[0]);
}

ParseResult GetTensorShapeOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  if (parser.parseOperand(src) || parser.parseColon())
    return failure();

  Type resultType;
  Type srcType;
  SMLoc srcTypeLoc = parser.getCurrentLocation();
  if (parseCudaTileType(parser, srcType) || parser.parseArrow() ||
      parseCudaTileType(parser, resultType))
    return failure();

  TensorViewType srcTensorViewType = llvm::dyn_cast<TensorViewType>(srcType);
  if (!srcTensorViewType)
    return parser.emitError(srcTypeLoc, "expected tensor_view, got ")
           << srcType;

  if (failed(parser.resolveOperand(src, srcType, result.operands)))
    return failure();

  size_t rank = srcTensorViewType.getShape().size();
  for (size_t i = 0; i < rank; i++)
    result.addTypes(resultType);

  return success();
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

LogicalResult DivFOp::verify() {
  auto rounding = getRoundingMode();
  if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                           RoundingMode::NEGATIVE_INF,
                           RoundingMode::POSITIVE_INF, RoundingMode::APPROX,
                           RoundingMode::FULL},
                          rounding)) {
    return emitOpError("invalid rounding mode specified, expect "
                       "one of [nearest_even, zero, negative_inf, "
                       "positive_inf, approx, full]");
  }
  bool hasApprox = rounding == RoundingMode::APPROX;
  bool hasFull = rounding == RoundingMode::FULL;
  bool hasIEEERounding = !hasApprox && !hasFull;
  return verifyDivFPModifiers(*this, hasIEEERounding, hasApprox, hasFull,
                              getFlushToZero());
}

//===----------------------------------------------------------------------===//
// DivIOp
//===----------------------------------------------------------------------===//

LogicalResult DivIOp::verify() {
  auto rounding = getRounding();
  if (!llvm::is_contained({RoundingMode::ZERO, RoundingMode::NEGATIVE_INF,
                           RoundingMode::POSITIVE_INF},
                          rounding)) {
    return emitOpError("invalid rounding mode specified, expect "
                       "one of [zero, negative_inf, positive_inf]");
  }
  if (rounding == RoundingMode::NEGATIVE_INF &&
      getSignedness() == Signedness::Unsigned) {
    return emitOpError("rounding mode 'negative_inf' is not allowed with "
                       "'unsigned' flag");
  }
  return success();
}

Speculation::Speculatability DivIOp::getSpeculatability() {
  auto cst = getRhs().getDefiningOp<ConstantOp>();
  if (!cst) {
    return Speculation::NotSpeculatable;
  }
  auto attr = dyn_cast<DenseIntElementsAttr>(cst.getValue());
  if (!attr) {
    return Speculation::NotSpeculatable;
  }
  bool isSigned = getSignedness() == Signedness::Signed;
  for (const APInt &v : attr.getValues<APInt>()) {
    // X / 0 => UB
    if (v.isZero()) {
      return Speculation::NotSpeculatable;
    }
    // isAllOnes() is -1 in two's complement.
    // INT_MIN / -1 overflows and is UB for signed division.
    // We conservatively assume the dividend (lhs) may be INT_MIN;
    // proving otherwise would require range analysis.
    if (isSigned && v.isAllOnes()) {
      return Speculation::NotSpeculatable;
    }
  }
  return Speculation::Speculatable;
}

Speculation::Speculatability RemIOp::getSpeculatability() {
  // X % 0 => UB (for both signed and unsigned).
  // X % -1 is always 0 and well-defined, no need to check for -1.
  auto cst = getRhs().getDefiningOp<ConstantOp>();
  if (!cst) {
    return Speculation::NotSpeculatable;
  }
  auto attr = dyn_cast<DenseIntElementsAttr>(cst.getValue());
  if (!attr) {
    return Speculation::NotSpeculatable;
  }
  for (const APInt &v : attr.getValues<APInt>()) {
    if (v.isZero()) {
      return Speculation::NotSpeculatable;
    }
  }
  return Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// ExtIOp
//===----------------------------------------------------------------------===//

LogicalResult ExtIOp::verify() {
  IntegerType from = cast<IntegerType>(getFrom().getType().getElementType());
  IntegerType to = cast<IntegerType>(getTo().getType().getElementType());

  if (to.getWidth() <= from.getWidth())
    return emitOpError("extending to smaller or identical integer");

  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  cuda_tile::TileType sourceType = getSource().getType();
  cuda_tile::TileType resultType = getResult().getType();

  // Check 1: Element type compatibility
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result element type do not match");

  // Check 2: Rank matching
  if (sourceType.getRank() != resultType.getRank())
    return emitOpError("source and result must have the same rank");

  // Check 3: Even division requirement
  for (int i = 0, e = static_cast<int>(sourceType.getRank()); i < e; ++i) {
    int64_t sourceDim = sourceType.getDimSize(i);
    int64_t resultDim = resultType.getDimSize(i);
    if (sourceDim % resultDim != 0)
      return emitOpError("source dimension ")
             << i << " size (" << sourceDim
             << ") must be evenly divisible by result dimension " << i
             << " size (" << resultDim << ")";
  }

  // Check 4: Index count validation
  if (static_cast<int64_t>(getIndices().size()) != sourceType.getRank())
    return emitOpError("expected ")
           << sourceType.getRank() << " indices, but got "
           << getIndices().size();

  // Check 5: Index type validation and bounds checking
  for (int i = 0, e = static_cast<int>(sourceType.getRank()); i < e; ++i) {
    Value index = getIndices()[i];
    auto indexType = dyn_cast<cuda_tile::TileType>(index.getType());

    if (!indexType)
      return emitOpError("index ") << i << " must be a Tile type";

    // Check if index is scalar
    if (indexType.getRank() != 0)
      return emitOpError("index ")
             << i << " must be a scalar tile, got rank " << indexType.getRank();

    // Check if index has i32 element type
    auto indexElemType = indexType.getElementType();
    if (!indexElemType.isInteger(32))
      return emitOpError("index ") << i << " must have i32 element type";

    // Bounds checking for constant indices (warnings only to allow compilation)
    if (auto constantOp = index.getDefiningOp<cuda_tile::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseIntElementsAttr>(constantOp.getValue())) {
        if (attr.isSplat()) {
          APInt apValue = attr.getSplatValue<APInt>();
          int64_t indexValue = apValue.getSExtValue();
          int64_t sourceDimSize = sourceType.getDimSize(i);
          int64_t resultDimSize = resultType.getDimSize(i);
          int64_t maxValidIndex = (sourceDimSize / resultDimSize) - 1;

          // Warn for negative indices
          if (indexValue < 0) {
            mlir::emitWarning(getLoc())
                << "extract index " << i << " has negative value " << indexValue
                << ", which may cause undefined behavior";
          }

          // Warn for out-of-bounds indices
          if (indexValue > maxValidIndex) {
            mlir::emitWarning(getLoc())
                << "extract index " << i << " value " << indexValue
                << " may be out of bounds (max valid index: " << maxValidIndex
                << ")";
          }
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

LogicalResult InsertOp::verify() {
  cuda_tile::TileType sourceType = getSource().getType();
  cuda_tile::TileType resultType = getResult().getType();

  // Check 1: Element type compatibility
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("source and result element type do not match");
  }

  // Check 2: Rank matching
  if (sourceType.getRank() != resultType.getRank()) {
    return emitOpError("source and result must have the same rank");
  }

  // Check 3: Even division requirement
  for (int64_t i = 0, e = sourceType.getRank(); i < e; ++i) {
    int64_t sourceDim = sourceType.getDimSize(static_cast<unsigned>(i));
    int64_t resultDim = resultType.getDimSize(static_cast<unsigned>(i));
    if (resultDim % sourceDim != 0) {
      return emitOpError("result dimension ")
             << i << " size (" << resultDim
             << ") must be evenly divisible by source dimension " << i
             << " size (" << sourceDim << ")";
    }
  }

  // Check 4: Index count validation
  if (static_cast<int64_t>(getIndices().size()) != sourceType.getRank()) {
    return emitOpError("expected ")
           << sourceType.getRank() << " indices, but got "
           << getIndices().size();
  }

  // Check 5: Bounds checking for constant indices.
  // Index type and scalar-ness are enforced by the ODS operand constraint.
  for (int64_t i = 0, e = sourceType.getRank(); i < e; ++i) {
    Value index = getIndices()[i];

    // Bounds checking for constant indices (warnings only to allow compilation)
    if (auto constantOp = index.getDefiningOp<cuda_tile::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseIntElementsAttr>(constantOp.getValue())) {
        if (attr.isSplat()) {
          APInt apValue = attr.getSplatValue<APInt>();
          int64_t indexValue = apValue.getSExtValue();
          int64_t sourceDimSize = sourceType.getDimSize(i);
          int64_t resultDimSize = resultType.getDimSize(i);
          int64_t maxValidIndex = (resultDimSize / sourceDimSize) - 1;

          // Warn for negative indices
          if (indexValue < 0) {
            mlir::emitWarning(getLoc())
                << "insert index " << i << " has negative value " << indexValue
                << ", which may cause undefined behavior";
          }

          // Warn for out-of-bounds indices
          if (indexValue > maxValidIndex) {
            mlir::emitWarning(getLoc())
                << "insert index " << i << " value " << indexValue
                << " may be out of bounds (max valid index: " << maxValidIndex
                << ")";
          }
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IToFOp
//===----------------------------------------------------------------------===//

LogicalResult IToFOp::verify() {
  auto rounding = getRoundingMode();
  if (isa<Float8E8M0FNUType>(getResult().getType().getElementType())) {
    return emitOpError("integer to f8E8M0FNU conversion is not supported, "
                       "please first convert to another float type");
  }
  if (rounding != RoundingMode::NEAREST_EVEN) {
    return emitOpError(
        "invalid rounding mode specified. Only 'nearest_even' is supported");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MmaFOp
//===----------------------------------------------------------------------===//

template <typename MmaOpT>
LogicalResult verifyMmaShapes(MmaOpT op) {
  cuda_tile::TileType lhsType = op.getLhs().getType();
  cuda_tile::TileType rhsType = op.getRhs().getType();
  cuda_tile::TileType accType = op.getAcc().getType();

  // Check shapes. Tablegen has AllRanksMatch constraint.
  if (lhsType.getRank() != 2 && lhsType.getRank() != 3)
    return op.emitOpError("operands must be 2D or 3D tiles");

  int batched = static_cast<int>(lhsType.getRank() == 3);
  if (batched) {
    if (lhsType.getShape()[0] != rhsType.getShape()[0])
      return op.emitOpError("shape error: dim 0 of lhs (")
             << lhsType.getShape()[0] << ") and dim 0 of rhs ("
             << rhsType.getShape()[0] << ") must match, but got lhs shape ("
             << lhsType.getShape() << ") and rhs shape (" << rhsType.getShape()
             << ")";
    if (lhsType.getShape()[0] != accType.getShape()[0])
      return op.emitOpError("shape error: dim 0 of lhs (")
             << lhsType.getShape()[0] << ") and dim 0 of acc ("
             << accType.getShape()[0] << ") must match, but got lhs shape ("
             << lhsType.getShape() << ") and acc shape (" << accType.getShape()
             << ")";
  }
  int rowDim = batched + 0;
  int colDim = batched + 1;
  if (lhsType.getShape()[colDim] != rhsType.getShape()[rowDim])
    return op.emitOpError(" shape error: dim ")
           << colDim << " of lhs (" << lhsType.getShape()[colDim]
           << ") and dim " << rowDim << " of rhs ("
           << rhsType.getShape()[rowDim] << ") must match, but got lhs shape ("
           << lhsType.getShape() << ") and rhs shape (" << rhsType.getShape()
           << ")";
  if (lhsType.getShape()[rowDim] != accType.getShape()[rowDim])
    return op.emitOpError(" shape error: dim ")
           << rowDim << " of lhs (" << lhsType.getShape()[rowDim]
           << ") and dim " << rowDim << " of acc ("
           << accType.getShape()[rowDim] << ") must match, but got lhs shape ("
           << lhsType.getShape() << ") and acc shape (" << accType.getShape()
           << ")";
  if (rhsType.getShape()[colDim] != accType.getShape()[colDim])
    return op.emitOpError(" shape error: dim ")
           << colDim << " of rhs (" << rhsType.getShape()[colDim]
           << ") and dim " << colDim << " of acc ("
           << accType.getShape()[colDim] << ") must match, but got rhs shape ("
           << rhsType.getShape() << ") and acc shape (" << accType.getShape()
           << ")";
  return success();
}

LogicalResult MmaFOp::verify() {
  if (failed(verifyMmaShapes(*this)))
    return failure();

  cuda_tile::TileType lhsType = getLhs().getType();
  cuda_tile::TileType accType = getAcc().getType();

  // Check element types. Tablegen has AllTypesMatch on lhs and rhs.
  struct AllowedMMAType {
    Type inputType;
    SmallVector<Type> allowedOutputTypes;
  };

  auto ctx = getContext();
  AllowedMMAType allowedMMATypes[] = {
      // Types must be created with context, so array can't be static
      {Float4E2M1FNType::get(ctx),
       {Float16Type::get(ctx), Float32Type::get(ctx)}},
      {Float8E4M3FNType::get(ctx),
       {Float16Type::get(ctx), Float32Type::get(ctx)}},
      // f8 (e5m2) x f8 (e5m2) -> {f16,f32}
      {Float8E5M2Type::get(ctx),
       {Float16Type::get(ctx), Float32Type::get(ctx)}},
      // f16 x f16 -> {f16,f32}
      {Float16Type::get(ctx), {Float16Type::get(ctx), Float32Type::get(ctx)}},
      // bf16 x bf16 -> f32
      {BFloat16Type::get(ctx), {Float32Type::get(ctx)}},
      // tf32 x tf32 -> f32
      {FloatTF32Type::get(ctx), {Float32Type::get(ctx)}},
      // f32 x f32 -> f32
      {Float32Type::get(ctx), {Float32Type::get(ctx)}},
      // f64 x f64 -> f64
      {Float64Type::get(ctx), {Float64Type::get(ctx)}},
  };

  bool checked = false;
  for (const auto &allowedMMAType : allowedMMATypes) {
    if (allowedMMAType.inputType == lhsType.getElementType()) {
      if (!llvm::is_contained(allowedMMAType.allowedOutputTypes,
                              accType.getElementType()))
        return emitOpError(
                   "unsupported combination of element types. Input type ")
               << lhsType.getElementType()
               << " expects accumulator/result type to be one of {"
               << allowedMMAType.allowedOutputTypes << "}, but got "
               << accType.getElementType();
      checked = true;
      break;
    }
  }

  if (!checked) {
    return emitOpError("unsupported input element type: ")
           << lhsType.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MmaFScaledOp
//===----------------------------------------------------------------------===//

LogicalResult MmaFScaledOp::verify() {
  if (failed(verifyMmaShapes(*this)))
    return failure();

  cuda_tile::TileType lhsType = getLhs().getType();
  cuda_tile::TileType rhsType = getRhs().getType();
  cuda_tile::TileType lhsScaleType = getLhsScale().getType();
  cuda_tile::TileType rhsScaleType = getRhsScale().getType();

  // Check shapes.
  int batched = static_cast<int>(lhsType.getRank() == 3);
  int rowDim = batched + 0;
  int colDim = batched + 1;

  // Must always match in the batch dimension.
  if (batched) {
    if (lhsType.getShape()[0] != lhsScaleType.getShape()[0])
      return emitOpError("shape error: dim 0 of lhs (")
             << lhsType.getShape()[0] << ") and dim 0 of lhs_scale ("
             << lhsScaleType.getShape()[0]
             << ") must match, but got lhs shape (" << lhsType.getShape()
             << ") and lhs_scale shape (" << lhsScaleType.getShape() << ")";
    if (rhsType.getShape()[0] != rhsScaleType.getShape()[0])
      return emitOpError("shape error: dim 0 of rhs (")
             << rhsType.getShape()[0] << ") and dim 0 of rhs_scale ("
             << rhsScaleType.getShape()[0]
             << ") must match, but got rhs shape (" << rhsType.getShape()
             << ") and rhs_scale shape (" << rhsScaleType.getShape() << ")";
  }

  // N and M dimension must match, K can be different.
  if (lhsType.getShape()[rowDim] != lhsScaleType.getShape()[rowDim])
    return emitOpError("shape error: dim ")
           << rowDim << " of lhs (" << lhsType.getShape()[rowDim]
           << ") and dim " << rowDim << " of lhs_scale ("
           << lhsScaleType.getShape()[rowDim]
           << ") must match, but got lhs shape (" << lhsType.getShape()
           << ") and lhs_scale shape (" << lhsScaleType.getShape() << ")";

  if (rhsType.getShape()[colDim] != rhsScaleType.getShape()[colDim])
    return emitOpError("shape error: dim ")
           << colDim << " of rhs (" << rhsType.getShape()[colDim]
           << ") and dim " << colDim << " of rhs_scale ("
           << rhsScaleType.getShape()[colDim]
           << ") must match, but got rhs shape (" << rhsType.getShape()
           << ") and rhs_scale shape (" << rhsScaleType.getShape() << ")";

  // Make sure K dimension of lhs_scale and rhs_scale match.
  if (lhsScaleType.getShape()[colDim] != rhsScaleType.getShape()[rowDim]) {
    return emitOpError("shape error: dim ")
           << colDim << " of lhs_scale (" << lhsScaleType.getShape()[colDim]
           << ") and dim " << rowDim << " of rhs_scale ("
           << rhsScaleType.getShape()[rowDim]
           << ") must match, but got lhs_scale shape ("
           << lhsScaleType.getShape() << ") and rhs_scale shape ("
           << rhsScaleType.getShape() << ")";
  }

  // Check element types.
  if (isa<Float8E5M3FNUType>(lhsScaleType.getElementType())) {
    if (!isa<Float4E2M1FNType>(lhsType.getElementType())) {
      return emitOpError(
                 "unsupported combination of element types. Scale type ")
             << lhsScaleType.getElementType()
             << " expects lhs and rhs element types to be "
             << Float4E2M1FNType::get(getContext()) << ", but got "
             << lhsType.getElementType();
    }
  }
  if (isa<Float8E4M3FNType>(lhsScaleType.getElementType())) {
    if (!isa<Float4E2M1FNType>(lhsType.getElementType()))
      return emitOpError(
                 "unsupported combination of element types. Scale type ")
             << lhsScaleType.getElementType()
             << " expects lhs and rhs element types to be "
             << Float4E2M1FNType::get(getContext()) << ", but got "
             << lhsType.getElementType();
  }
  if (isa<Float8E8M0FNUType>(lhsScaleType.getElementType())) {
    if (!isa<Float8E4M3FNType, Float8E5M2Type, Float4E2M1FNType>(
            lhsType.getElementType()))
      return emitOpError(
                 "unsupported combination of element types. Scale type ")
             << lhsScaleType.getElementType()
             << " expects lhs and rhs element types to be one of {"
             << Float8E4M3FNType::get(getContext()) << ", "
             << Float8E5M2Type::get(getContext()) << ", "
             << Float4E2M1FNType::get(getContext()) << "}, but got "
             << lhsType.getElementType();
  }

  // Check K dimension reduction factor ("vecSize" or "blockSize").
  int64_t dimKSize = lhsType.getShape()[colDim];
  int64_t scaleDimKSize = lhsScaleType.getShape()[colDim];
  if (dimKSize % scaleDimKSize != 0) {
    return emitOpError("shape error: dim ")
            << colDim << " of lhs_scale (" << scaleDimKSize
            << ") must be a divisor of dim " << colDim << " of lhs (" << dimKSize
            << "), but got lhs_scale shape (" << lhsScaleType.getShape()
            << ") and lhs shape (" << lhsType.getShape() << ")";
  }
  int64_t lhsScaleVecSize = dimKSize / scaleDimKSize;

  // F8 element types requires vecSize=32 (QMMA K=32, 1x).
  if (isa<Float8E4M3FNType, Float8E5M2Type>(lhsType.getElementType())) {
    if (lhsScaleVecSize != 32) {
      return emitOpError()
              << "shape error: f8 element type requires block scale factor of "
                "lhs_scale and rhs_scale to be 32, but got "
              << lhsScaleVecSize;
    }
  }
  // F4 element type can have multiple vecSize values, depending on scale.
  if (isa<Float4E2M1FNType>(lhsType.getElementType())) {
    // F4 elements with e4m3 scales must have vecSize=16 (OMMA K=64, 4x).
    if (isa<Float8E4M3FNType>(lhsScaleType.getElementType())) {
      if (lhsScaleVecSize != 16) {
        return emitOpError()
                << "shape error: f4E2M1FN element type with f8E4M3FN scales "
                  "requires block scale factor of lhs_scale and rhs_scale "
                  "to be 16, but got "
                << lhsScaleVecSize;
      }
    }
    // F4 with e8m0 scales can have vecSize=16 or 32 (OMMA K=64, 4x or 2x).
    if (isa<Float8E8M0FNUType>(lhsScaleType.getElementType())) {
      if (lhsScaleVecSize != 16 && lhsScaleVecSize != 32) {
        return emitOpError()
                << "shape error: f4E2M1FN element type with f8E8M0FNU scales "
                  "requires block scale factor of lhs_scale and rhs_scale "
                  "to be 16 or 32, but got "
                << lhsScaleVecSize;
      }
    }
    // F4 with UE5M3 scales can have vecSize=16 or 32 on SM107.
    if (isa<Float8E5M3FNUType>(lhsScaleType.getElementType())) {
      if (lhsScaleVecSize != 16 && lhsScaleVecSize != 32) {
        return emitOpError()
               << "shape error: f4E2M1FN element type with f8E5M3FNU scales "
                  "requires block scale factor of lhs_scale and rhs_scale "
                  "to be 16 or 32, but got "
               << lhsScaleVecSize;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MmaIOp
//===----------------------------------------------------------------------===//

LogicalResult MmaIOp::verify() {
  // Only need to verify shapes, as tablegen enforces element types
  return verifyMmaShapes(*this);
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

LogicalResult ExpOp::verify() {
  auto roundingMode = getRoundingMode();
  if (!llvm::is_contained({RoundingMode::FULL, RoundingMode::APPROX},
                          roundingMode)) {
    return emitOpError("invalid rounding mode specified, expect one of "
                       "[approx, full]");
  }
  bool hasApprox = roundingMode == RoundingMode::APPROX;
  return verifyApprox(*this, hasApprox);
}

//===----------------------------------------------------------------------===//
// Exp2Op
//===----------------------------------------------------------------------===//

LogicalResult Exp2Op::verify() { return verifyFtz(*this, getFlushToZero()); }

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

LogicalResult FmaOp::verify() {
  if (failed(verifyIEEERoundingModes(*this)))
    return failure();
  return verifyFtz(*this, getFlushToZero());
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

/// Verifies that the initial iterator values of the given loop match the
/// region arguments.
template <typename LoopOpT>
static LogicalResult verifyLoopIterValues(LoopOpT op, ResultRange results,
                                          ValueRange iterVals) {
  auto loopInits = op.getInitValues();
  if (iterVals.size() != loopInits.size()) {
    return op.emitOpError("mismatch in number of region iterator values and "
                          "loop iterator inits: ")
           << iterVals.size() << " vs " << loopInits.size();
  }

  for (auto [index, initArg, iterArg] : llvm::enumerate(loopInits, iterVals)) {
    if (isa<TensorViewType>(iterArg.getType()))
      return op.emitOpError() << "loop-carried value " << index
                              << " is a tensor_view, "
                                 "which is not supported";

    if (isa<TileView>(iterArg.getType()))
      return op.emitOpError() << "loop-carried value " << index
                              << " is a tile view, "
                                 "which is not supported";

    if (initArg.getType() != iterArg.getType())
      return op.emitOpError()
             << "init value " << index << " and region iter_value " << index
             << " have different type: " << initArg.getType()
             << " != " << iterArg.getType();
  }

  // Verify that results are not tensor_view or tile_view.
  for (auto [index, result] : llvm::enumerate(results)) {
    if (isa<TensorViewType>(result.getType()))
      return op.emitOpError() << "result type " << index
                              << " is a tensor_view, "
                                 "which is not supported";

    if (isa<TileView>(result.getType()))
      return op.emitOpError() << "result type " << index
                              << " is a tile view, "
                                 "which is not supported";
  }

  return success();
}

/// Prints the iterator values for a loop operation.
static void printLoopIteratorValues(OpAsmPrinter &p, OperandRange initVals,
                                    Block::BlockArgListType regionIterValues) {
  // Prints the initialization list in the form of
  //   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
  // where 'inner' values are assumed to be region arguments and 'outer'
  // values are regular SSA values.
  p << "iter_values(";
  llvm::interleaveComma(llvm::zip(regionIterValues, initVals), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ") -> (";
  printCudaTileType(p, initVals.getTypes());
  p << ") ";
}

void ForOp::build(
    OpBuilder &builder, OperationState &result, Value lb, Value ub, Value step,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder,
    bool unsignedCmp) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands({lb, ub, step});
  result.addOperands(initArgs);
  if (unsignedCmp)
    result.addAttribute(getUnsignedCmpAttrName(result.name),
                        builder.getUnitAttr());
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(lb.getType(), result.location);
  for (Value v : initArgs) {
    result.addTypes(v.getType());
    bodyBlock->addArgument(v.getType(), v.getLoc());
  }
  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (initArgs.empty() && !bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0),
                bodyBlock->getArguments().drop_front());
  }
}

LogicalResult ForOp::verifyRegions() {
  // First block argument must be the induction variable.
  if (getNumRegionArgs() == 0)
    return emitOpError(
        "expected at least one block argument for induction variable");
  Value indVar = getInductionVar(), lowerBound = getLowerBound();
  if (indVar.getType() != lowerBound.getType()) {
    return emitOpError("expected induction variable to be same type as bounds "
                       "and step: ")
           << indVar.getType() << " vs " << lowerBound.getType();
  }

  // Verify that for loops do not contain return ops (even nested in regions)
  auto result = getBody()->walk(
      [&](ReturnOp returnOp) { return WalkResult::interrupt(); });
  if (result.wasInterrupted())
    return emitOpError("'for' loop cannot contain 'return' operations");

  return verifyLoopIterValues(*this, getResults(), getRegionIterValues());
}

void ForOp::print(OpAsmPrinter &p) {
  Value inductionVar = getInductionVar();
  if (getUnsignedCmp())
    p << " unsigned";
  p << " " << inductionVar << " in (" << getLowerBound() << " to "
    << getUpperBound() << ", step " << getStep() << ") : ";
  printCudaTileType(p, inductionVar.getType());
  p << " ";
  if (OperandRange initVals = getInitValues(); !initVals.empty())
    printLoopIteratorValues(p, initVals, getRegionIterValues());

  printControlFlowRegion<ContinueOp>(p, *this, getRegion());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getUnsignedCmpAttrName()});
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the optional 'unsigned' keyword.
  auto attrName = getUnsignedCmpAttrName(result.name);
  if (succeeded(parser.parseOptionalKeyword("unsigned")))
    result.addAttribute(attrName, parser.getBuilder().getUnitAttr());

  // Parse the induction variable followed by '='.
  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> iterOperands;

  if (parser.parseOperand(inductionVariable.ssaName) ||
      parser.parseKeyword("in") ||
      // Parse loop bounds.
      parser.parseLParen() || parser.parseOperand(lb) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.parseComma() || parser.parseKeyword("step") ||
      parser.parseOperand(step) || parser.parseRParen() ||
      parser.parseColon() || parseCudaTileType(parser, inductionVariable.type))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs(1, inductionVariable);
  if (succeeded(parser.parseOptionalKeyword("iter_values"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, iterOperands) ||
        parser.parseArrow() || parser.parseLParen() ||
        parseCudaTileType(parser, result.types) || parser.parseRParen())
      return failure();
    if (iterOperands.size() != result.types.size()) {
      return parser.emitError(
          parser.getNameLoc(),
          "mismatch in number of loop-carried values and defined values");
    }

    // Set region iter_arg types.
    auto parsedRegionArgs = llvm::drop_begin(regionArgs, 1);
    for (auto [regionArg, type] :
         llvm::zip_equal(parsedRegionArgs, result.types)) {
      regionArg.type = type;
    }
  }

  // Parse the body region.
  if (parseControlFlowRegion<ForOp>(parser, *result.addRegion(), regionArgs) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve operands.
  if (parser.resolveOperands({lb, ub, step}, inductionVariable.type,
                             result.operands) ||
      parser.resolveOperands(iterOperands, result.types, parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

void ForOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getNumResults())
    setNameFn(getResult(0), "for");
}

void ForOp::getAsmBlockArgumentNames(Region &region,
                                     OpAsmSetValueNameFn setNameFn) {
  setNameFn(getInductionVar(), "loopIdx");
  for (auto [index, arg] :
       llvm::enumerate(llvm::drop_begin(region.getArguments())))
    setNameFn(arg, "iterArg" + std::to_string(index));
}

//===----------------------------------------------------------------------===//
// FToIOp
//===----------------------------------------------------------------------===//

LogicalResult FToIOp::verify() {
  auto rounding = getRoundingMode();
  if (rounding != RoundingMode::NEAREST_INT_TO_ZERO)
    return emitOpError("invalid rounding mode specified. Only "
                       "'nearest_int_to_zero' is supported");
  return success();
}

//===----------------------------------------------------------------------===//
// FToFOp
//===----------------------------------------------------------------------===//

/// Returns true for low-precision float types (anything narrower than the
/// smallest standard IEEE float, f16). Covers FP4, FP6, and FP8 formats.
static bool isLowPrecisionFloatType(Type ty) {
  return cast<FloatType>(ty).getWidth() < 16;
}

/// Returns true if every value in srcTy can be exactly represented in dstTy:
/// at least as much precision and an exponent range that covers the source.
static bool isStrictWidening(Type srcTy, Type dstTy) {
  const llvm::fltSemantics &src = cast<FloatType>(srcTy).getFloatSemantics();
  const llvm::fltSemantics &dst = cast<FloatType>(dstTy).getFloatSemantics();
  return llvm::APFloat::semanticsPrecision(dst) >=
             llvm::APFloat::semanticsPrecision(src) &&
         llvm::APFloat::semanticsMaxExponent(dst) >=
             llvm::APFloat::semanticsMaxExponent(src) &&
         llvm::APFloat::semanticsMinExponent(dst) <=
             llvm::APFloat::semanticsMinExponent(src);
}

LogicalResult FToFOp::verify() {
  if (getFrom().getType() == getTo().getType()) {
    return emitOpError("converting tiles must not be a no-op");
  }
  auto rounding = getRoundingMode();
  auto srcElemTy = getFrom().getType().getElementType();
  auto dstElemTy = getTo().getType().getElementType();

  // Enforce the float rounding rules in order. See the FToFOp table in
  // tile_ir/include/cuda_tile/Dialect/CudaTile/IR/Ops.td. The first matching
  // rule is applied.

  // Rule 1: Conversion to f8E8M0FNU — only ZERO and POSITIVE_INF.
  if (isa<Float8E8M0FNUType>(dstElemTy)) {
    if (!llvm::is_contained({RoundingMode::ZERO, RoundingMode::POSITIVE_INF},
                            rounding)) {
      return emitOpError(
          "invalid rounding mode specified for conversion to f8E8M0FNU. Only "
          "'zero' and 'positive_inf' are supported");
    }
    return success();
  }

  // Rule 2: Conversion to other low-precision types — only NEAREST_EVEN.
  if (isLowPrecisionFloatType(dstElemTy)) {
    if (rounding != RoundingMode::NEAREST_EVEN) {
      return emitOpError(
          "invalid rounding mode specified for conversion to low-precision "
          "type. Only 'nearest_even' is supported");
    }
    return success();
  }

  // Rule 3: Strict widenings — any IEEE float rounding mode is valid
  // (no rounding actually occurs since source is exactly representable).
  if (isStrictWidening(srcElemTy, dstElemTy)) {
    if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                             RoundingMode::NEGATIVE_INF,
                             RoundingMode::POSITIVE_INF,
                             RoundingMode::NEAREST_AWAY},
                            rounding)) {
      return emitOpError("invalid rounding mode specified for widening "
                         "conversion. Only IEEE float rounding modes are "
                         "supported: 'nearest_even', 'zero', 'negative_inf', "
                         "'positive_inf', 'nearest_away'");
    }
    return success();
  }

  // Rule 4: F64 → F32 narrowing.
  if (srcElemTy.isF64() && dstElemTy.isF32()) {
    if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                             RoundingMode::NEGATIVE_INF,
                             RoundingMode::POSITIVE_INF},
                            rounding)) {
      return emitOpError(
          "invalid rounding mode specified for conversion from f64 to f32. "
          "Only 'nearest_even', 'zero', 'negative_inf', and 'positive_inf' "
          "are supported");
    }
    return success();
  }

  // Rule 5: F32 → TF32 narrowing.
  if (srcElemTy.isF32() && dstElemTy.isTF32()) {
    if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                             RoundingMode::NEAREST_AWAY},
                            rounding)) {
      return emitOpError(
          "invalid rounding mode specified for conversion from f32 to tf32. "
          "Only 'nearest_even', 'zero', and 'nearest_away' are supported");
    }
    return success();
  }

  // Rule 6: Other narrowing to standard types — NEAREST_EVEN and ZERO.
  // (Low-precision destinations were handled by rules 1 and 2.)
  if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO},
                          rounding)) {
    return emitOpError(
        "invalid rounding mode specified. Only 'nearest_even' and "
        "'zero' are supported for narrowing conversions to standard types");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EntryOp
//===----------------------------------------------------------------------===//

ParseResult EntryOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes)) {
    return failure();
  }

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
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  SmallVector<Attribute> argAttrs = llvm::to_vector(
      llvm::map_range(entryArgs, [](OpAsmParser::Argument arg) -> Attribute {
        return arg.attrs;
      }));
  if (ArrayAttr normalizedArgAttrs =
          buildNormalizedArgAttrs(builder, argAttrs)) {
    result.addAttribute(getArgAttrsAttrName(result.name), normalizedArgAttrs);
  }

  // Parse OptimizationHints attribute. An explicit empty form
  // is semantically identical to having no attribute at all.
  // Normalize it away here so the in-memory IR has a single canonical
  // shape and bytecode round-trip is stable. The printer already elides the
  // empty form.
  if (succeeded(parser.parseOptionalKeyword(kOptimizationHintsAttr))) {
    if (parser.parseEqual())
      return failure();
    auto opt_hint = dyn_cast_or_null<OptimizationHintsAttr>(
        OptimizationHintsAttr::parse(parser, Type{}));
    if (!opt_hint) {
      return failure();
    }
    if (!opt_hint.getValue().empty()) {
      result.addAttribute(getOptimizationHintsAttrName(result.name), opt_hint);
    }
  }

  // Parse the function body.
  Region *body = result.addRegion();
  ParseResult parseResult = parser.parseRegion(*body, entryArgs,
                                               /*enableNameShadowing=*/false);
  if (failed(parseResult))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  ensureTerminator(*body, builder, result.location);

  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();

  return success();
}

void EntryOp::print(OpAsmPrinter &printer) {
  // Print the operation and the function name.
  auto funcName = getSymName();
  printer << ' ';
  printer.printSymbolName(funcName);
  auto fnType = getFunctionType();
  printFunctionSignatureWithCudaTileTypes(printer, *this, fnType.getInputs(),
                                          fnType.getResults());
  if (getOptimizationHints() && !getOptimizationHints()->getValue().empty()) {
    printer << " " << kOptimizationHintsAttr << "=";
    getOptimizationHintsAttr().print(printer);
  }
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true,
                      /*printEmptyBlock=*/false);
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {getArgAttrsAttrName(), getFunctionTypeAttrName(), getSymNameAttrName(),
       getResAttrsAttrName(), kOptimizationHintsAttr});
}

LogicalResult EntryOp::verify() {
  if (getNumResults() != 0)
    return emitOpError("entry op must not return values");

  // A kernel is launched from the host, so it is always externally referenced.
  if (getVisibility() != SymbolTable::Visibility::Public)
    return emitOpError("entry op must have public symbol visibility");

  for (Type operandTy : getArgumentTypes()) {
    auto tileTy = dyn_cast<cuda_tile::TileType>(operandTy);
    if (tileTy && tileTy.getRank() != 0)
      return emitOpError("entry op must have "
                         "scalar types (rank 0 !cuda_tile.tile)");
  }

  // A kernel entry point declares no constant parameters, so its signature has
  // no scope in which a symbolic variable could be bound.
  if (failed(verifySignatureSymbolicScope(getOperation(), getArgumentTypes(),
                                          getResultTypes(),
                                          /*constParams=*/{}))) {
    return failure();
  }

  if (failed(impl::verifyFuncDebugInfo(*this)))
    return failure();
  return verifyOptHintsCommon(this);
}

LogicalResult EntryOp::verifyRegions() {
  if (failed(impl::verifyFuncBodyDebugInfo(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verify() {
  if (getValue().getType().getRank() != 1)
    return emitOpError("type must have rank 1");
  // Visibility is modeled by `symbol_visibility`; the builtin attribute is not
  // serialized and would be silently dropped.
  if ((*this)->hasAttr(SymbolOpInterface::getDefaultVisibilityAttrName()))
    return emitOpError("expected visibility to be specified with "
                       "`symbol_visibility`, not `sym_visibility`");
  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getNameAttr());
  if (!global)
    return emitOpError("'")
           << getName() << "' does not reference a valid global";

  Type globalElType = global.getValue().getType().getElementType();
  auto resultType = cast<cuda_tile::PointerType>(
      cast<cuda_tile::TileType>(getResult().getType()).getElementType());
  if (globalElType != resultType.getPointeeType())
    return emitOpError("pointee type of result type ")
           << resultType << " does not match type " << globalElType
           << " of the global @" << getName();

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  if (getRegions().empty())
    return emitOpError("must define then branch");

  bool hasResults = getNumResults() != 0;
  auto retTypes = getResultTypes();

  auto checkRegionYieldTypes = [this, hasResults,
                                retTypes](Region &region,
                                          std::string name) -> LogicalResult {
    auto yield = dyn_cast<YieldOp>(region.front().back());
    if (yield) {
      auto yieldTypes = yield->getOperandTypes();
      if (hasResults && yieldTypes.empty())
        return emitOpError("has return type of ")
               << retTypes << " but " << name
               << " branch does not yield anything";
      if (!hasResults && !yieldTypes.empty())
        return emitOpError("does not return a value, but ")
               << name << " branch yields " << yieldTypes;
      if (yieldTypes != retTypes)
        return emitOpError("type does not match yield type, ")
               << name << " branch yields " << yieldTypes
               << " but op result type is " << retTypes;
    }
    return success();
  };

  for (const auto &[i, retType] : llvm::enumerate(retTypes)) {
    if (isa<TensorViewType>(retType))
      return emitOpError("result type ")
             << i << " is a tensor_view, which is not supported";
    if (isa<TileView>(retType))
      return emitOpError("result type ")
             << i << " is a tile view, which is not supported";
  }

  Region &thenRegion = getThenRegion();
  if (thenRegion.empty())
    return emitOpError("must define then branch");
  LogicalResult thenCheck = checkRegionYieldTypes(thenRegion, "then");
  if (failed(thenCheck))
    return thenCheck;

  Region &elseRegion = getElseRegion();
  if (elseRegion.empty()) {
    if (hasResults)
      return emitOpError("has non-empty return type, must define else branch");

    // empty else block with no expected yield, nothing to check
    return verifyOptHintsCommon(this);
  }
  LogicalResult elseCheck = checkRegionYieldTypes(elseRegion, "else");
  if (failed(elseCheck))
    return elseCheck;
  return verifyOptHintsCommon(this);
}

Block *IfOp::getThenBlock() { return &getThenRegion().back(); }
Operation *IfOp::getThenTerminator() { return getThenBlock()->getTerminator(); }

Block *IfOp::getElseBlock() {
  Region &r = getElseRegion();
  return r.empty() ? nullptr : &r.back();
}
Operation *IfOp::getElseTerminator() {
  Block *elseBlock = getElseBlock();
  return elseBlock ? elseBlock->getTerminator() : nullptr;
}

/// Return True if Terminator is ContinueOp/ReturnOp/BreakOp,
/// so no operation from parent region will be executed after it
/// Return False if Terminator is YieldOp or null
static inline bool isTerminatorForParent(Operation *op) {
  return op && llvm::isa<ReturnOp, ContinueOp, BreakOp>(op);
}

/// Erase rest of block below given uop
/// Needed when region, that replaced the operation, contains terminator
static void eraseRestOfBlockFrom(Operation *start, PatternRewriter &rewriter) {
  Block *block = start->getBlock();
  for (Operation &op :
       llvm::make_early_inc_range(llvm::iterator_range<Block::iterator>(
           start->getIterator(), block->end()))) {
    op.dropAllUses();
    rewriter.eraseOp(&op);
  }
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static LogicalResult replaceOpWithRegion(PatternRewriter &rewriter,
                                         Operation *op, Region &region,
                                         ValueRange blockArgs = {}) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  // Region ends with YieldOp - just redirect uses
  if (auto y = dyn_cast<YieldOp>(terminator)) {
    rewriter.replaceOp(op, y->getOperands());
    rewriter.eraseOp(y);
    return success();
  }

  // If the chosen branch ends in Continue/Break/Return, then all operations
  // from the original IfOp onward in the parent block are unreachable.
  if (isTerminatorForParent(terminator)) {
    // Erase the IfOp and everything after it in the parent block.
    eraseRestOfBlockFrom(op, rewriter);
    return success();
  }

  // Unknown terminator kind: conservatively bail.
  return failure();
}

/// Porting of SCF::IfOp fold
/// m_One() matching for XorIOp's Rhs is replaced
LogicalResult IfOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  // if (!c) then A() else B() -> if c then B() else A()
  if (getElseRegion().empty())
    return failure();

  XOrIOp xorStmt = getCondition().getDefiningOp<XOrIOp>();
  if (!xorStmt)
    return failure();

  if (!isConstantTrueVal(xorStmt.getRhs()))
    return failure();

  getConditionMutable().assign(xorStmt.getLhs());
  Block *thenBlock = &getThenRegion().front();
  // It would be nicer to use iplist::swap, but that has no implemented
  // callbacks See: https://llvm.org/doxygen/ilist_8h_source.html#l00224
  getThenRegion().getBlocks().splice(getThenRegion().getBlocks().begin(),
                                     getElseRegion().getBlocks());
  getElseRegion().getBlocks().splice(getElseRegion().getBlocks().begin(),
                                     getThenRegion().getBlocks(), thenBlock);
  return success();
}

/// Perform canonicalization for IfOp with static True/False condition,
/// similar to SCF::IfOp but with additional support for cuda_tile::ConstantOp
/// as defining op and cuda_tile::ContinueOp, cuda_tile::BreakOp,
/// cuda_tile::ReturnOp as terminator inside IfOp
struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Get condition value from ConstantOp
    auto condition = getConstantBoolValue(op.getCondition());
    if (!condition)
      return failure();

    if (condition.value())
      return replaceOpWithRegion(rewriter, op, op.getThenRegion());
    if (!op.getElseRegion().empty())
      return replaceOpWithRegion(rewriter, op, op.getElseRegion());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Porting of SCF::IfOp::ConvertTrivialIfToSelect
/// Additional support for ContinueOp/BreakOp/ReturnOp terminators
/// in one of the regions - in this case we always yield the same value
/// When both regions end without YieldOp - nothing to do

struct ConvertToSelect : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return failure();

    auto cond = op.getCondition();
    auto thenTerminator = op.getThenTerminator();
    auto elseTerminator = op.getElseTerminator();
    bool thenYield = llvm::isa<YieldOp>(thenTerminator);
    bool elseYield = llvm::isa<YieldOp>(elseTerminator);
    // If there is no YieldOp at all - nothing to do
    if (!thenYield && !elseYield)
      return failure();

    // If branch has non-YieldOp - take the same yield args both for then & else
    auto thenYieldArgs = thenYield ? thenTerminator->getOperands()
                                   : elseTerminator->getOperands();
    auto elseYieldArgs = elseYield ? elseTerminator->getOperands()
                                   : thenTerminator->getOperands();
    auto thenRegion = thenYield ? &op.getThenRegion() : &op.getElseRegion();
    auto elseRegion = elseYield ? &op.getElseRegion() : &op.getThenRegion();

    // Check if all yielded value types are TileType
    // As yielded types should match IfOp's result types
    // there is no need to check thenYieldArgs & elseYieldArgs separately
    if (!llvm::all_of(op->getResultTypes(), [](Type ty) {
          auto tileType = llvm::dyn_cast<TileType>(ty);
          return tileType;
        }))
      return failure();

    SmallVector<Type> nonHoistable;
    for (auto [trueVal, falseVal] : llvm::zip(thenYieldArgs, elseYieldArgs))
      if (thenRegion == trueVal.getParentRegion() ||
          elseRegion == falseVal.getParentRegion())
        nonHoistable.push_back(trueVal.getType());
    // Early exit if there aren't any yielded values we can
    // hoist outside the if.
    if (nonHoistable.size() == op->getNumResults())
      return failure();

    IfOp replacement = IfOp::create(rewriter, op.getLoc(), nonHoistable, cond);
    replacement.getThenRegion().takeBody(op.getThenRegion());
    replacement.getElseRegion().takeBody(op.getElseRegion());

    SmallVector<Value> results(op->getNumResults());
    assert(thenYieldArgs.size() == results.size());
    assert(elseYieldArgs.size() == results.size());

    auto replacementThenRegion =
        thenYield ? &replacement.getThenRegion() : &replacement.getElseRegion();
    auto replacementElseRegion =
        elseYield ? &replacement.getElseRegion() : &replacement.getThenRegion();
    SmallVector<Value> trueYields;
    SmallVector<Value> falseYields;
    rewriter.setInsertionPoint(replacement);
    for (const auto &it :
         llvm::enumerate(llvm::zip(thenYieldArgs, elseYieldArgs))) {
      Value trueVal = std::get<0>(it.value());
      Value falseVal = std::get<1>(it.value());
      if (replacementThenRegion == trueVal.getParentRegion() ||
          replacementElseRegion == falseVal.getParentRegion()) {
        results[it.index()] = replacement.getResult(trueYields.size());
        trueYields.push_back(trueVal);
        falseYields.push_back(falseVal);
      } else if (trueVal == falseVal)
        results[it.index()] = trueVal;
      else
        results[it.index()] = createSelectOpByType(rewriter, op.getLoc(), cond,
                                                   trueVal, falseVal);
    }

    if (thenYield) {
      rewriter.setInsertionPointToEnd(replacement.getThenBlock());
      rewriter.replaceOpWithNewOp<YieldOp>(replacement.getThenTerminator(),
                                           trueYields);
    }

    if (elseYield) {
      rewriter.setInsertionPointToEnd(replacement.getElseBlock());
      rewriter.replaceOpWithNewOp<YieldOp>(replacement.getElseTerminator(),
                                           falseYields);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Porting of SCF::IfOp::RemoveUnusedResults::transferBody
/// Additonal support for handling non-YieldOp terminator
struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults,
                    PatternRewriter &rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Replace the yield op by one that returns only the used values.
    if (auto yieldOp = dyn_cast<YieldOp>(dest->getTerminator())) {
      SmallVector<Value, 4> usedOperands;
      llvm::transform(usedResults, std::back_inserter(usedOperands),
                      [&](OpResult result) {
                        return yieldOp.getOperand(result.getResultNumber());
                      });
      rewriter.modifyOpInPlace(yieldOp,
                               [&]() { yieldOp->setOperands(usedOperands); });
    }
  }

  /// Porting of SCF::IfOp::RemoveUnusedResults
  /// Additional support for non-YieldOp terminator inside transferBody()
  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0)
      return failure();

    auto thenTerminator = op.getThenTerminator();
    auto elseTerminator = op.getElseTerminator();
    bool thenYield = llvm::isa<YieldOp>(thenTerminator);
    bool elseYield = llvm::isa<YieldOp>(elseTerminator);
    // If there is no YieldOp at all - nothing to do
    if (!thenYield && !elseYield)
      return failure();

    // Compute the list of used results.
    SmallVector<OpResult, 4> usedResults;
    llvm::copy_if(op.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    // Replace the operation if only a subset of its results have uses.
    if (usedResults.size() == op.getNumResults())
      return failure();

    // Compute the result types of the replacement operation.
    SmallVector<Type, 4> newTypes;
    llvm::transform(usedResults, std::back_inserter(newTypes),
                    [](OpResult result) { return result.getType(); });

    // Create a replacement operation with empty then and else regions.
    auto newOp =
        IfOp::create(rewriter, op.getLoc(), newTypes, op.getCondition());
    rewriter.createBlock(&newOp.getThenRegion());
    rewriter.createBlock(&newOp.getElseRegion());

    // Move the bodies and replace the terminators (note there is a then and
    // an else region since the operation returns results).
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);

    return success();
  }
};

/// Porting of SCF::ReplaceIfYieldWithConditionOrValue
/// ContinueOp/BreakOp/ReturnOp terminators are not supported
struct ReplaceYieldWithValue : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there are no results that could be replaced.
    if (op.getNumResults() == 0)
      return failure();

    auto thenTerminator = op.getThenTerminator();
    auto elseTerminator = op.getElseTerminator();
    bool thenYield = llvm::isa<YieldOp>(thenTerminator);
    bool elseYield = llvm::isa<YieldOp>(elseTerminator);
    // IF there is non-YieldOp terminator - this case is not supported here
    // and suitable YieldOp + ReturnOp patterns are handled inside
    // canonicalizeIfOpConvertToSelect
    if (!thenYield || !elseYield)
      return failure();

    auto thenYieldArgs = thenTerminator->getOperands();
    auto elseYieldArgs = elseTerminator->getOperands();

    rewriter.setInsertionPoint(op->getBlock(),
                               op.getOperation()->getIterator());
    bool changed = false;
    for (auto [trueResult, falseResult, opResult] :
         llvm::zip(thenYieldArgs, elseYieldArgs, op.getResults())) {
      if (trueResult == falseResult) {
        if (!opResult.use_empty()) {
          opResult.replaceAllUsesWith(trueResult);
          changed = true;
        }
        continue;
      }

      auto trueVal = getConstantBoolValue(trueResult);
      auto falseVal = getConstantBoolValue(falseResult);
      if (!trueVal || !falseVal)
        continue;
      if (!*trueVal && *falseVal) {
        if (!opResult.use_empty()) {
          Value notCond =
              createXOrForValue(rewriter, op.getLoc(), op.getCondition());
          opResult.replaceAllUsesWith(notCond);
          changed = true;
        }
      }
      if (*trueVal && !*falseVal) {
        if (!opResult.use_empty()) {
          opResult.replaceAllUsesWith(op.getCondition());
          changed = true;
        }
      }
    }
    return success(changed);
  }
};

/// Porting of SCF::IfOp::CombineIfs
/// Added additional support for ContinueOp/BreakOp/ReturnOp terminators
struct CombineIfs : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp nextIf,
                                PatternRewriter &rewriter) const override {
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevIf = dyn_cast<IfOp>(nextIf->getPrevNode());
    if (!prevIf)
      return failure();

    // Determine the logical then/else blocks when prevIf's
    // condition is used. Null means the block does not exist
    // in that case (e.g. empty else). If neither of these
    // are set, the two conditions cannot be compared.
    Block *nextThen = nullptr;
    Block *nextElse = nullptr;
    if (nextIf.getCondition() == prevIf.getCondition()) {
      nextThen = nextIf.getThenBlock();
      if (!nextIf.getElseRegion().empty())
        nextElse = nextIf.getElseBlock();
    }
    if (XOrIOp notv = nextIf.getCondition().getDefiningOp<XOrIOp>()) {
      if (isConstantTrueVal(notv.getRhs()) &&
          notv.getLhs() == prevIf.getCondition()) {
        nextElse = nextIf.getThenBlock();
        if (!nextIf.getElseRegion().empty())
          nextThen = nextIf.getElseBlock();
      }
    }
    if (XOrIOp notv = prevIf.getCondition().getDefiningOp<XOrIOp>()) {
      if (isConstantTrueVal(notv.getRhs()) &&
          notv.getLhs() == nextIf.getCondition()) {
        nextElse = nextIf.getThenBlock();
        if (!nextIf.getElseRegion().empty())
          nextThen = nextIf.getElseBlock();
      }
    }

    // First If ends with ReturnOp/ContinueOp/BreakOp
    // no need to take next block from nextIf
    if (isTerminatorForParent(prevIf.getThenTerminator()))
      nextThen = nullptr;
    if (isTerminatorForParent(prevIf.getElseTerminator()))
      nextElse = nullptr;

    if (!nextThen && !nextElse)
      return failure();

    // Initialize prevThenYielded & prevElseYielded with
    // prevIf.getResults(), so that llvm::zip() below will not be
    // truncated. It is safe as corresponding values are used inside
    // only when nextThen/nextElse are true (so when be properly initialized)
    SmallVector<Value> prevThenYielded = prevIf.getResults();
    SmallVector<Value> prevElseYielded = prevIf.getResults();
    if (nextThen && !prevIf.getThenRegion().empty())
      prevThenYielded = prevIf.getThenTerminator()->getOperands();
    if (nextElse && !prevIf.getElseRegion().empty())
      prevElseYielded = prevIf.getElseTerminator()->getOperands();
    // Replace all uses of return values of op within nextIf with the
    // corresponding yields
    for (auto it :
         llvm::zip(prevIf.getResults(), prevThenYielded, prevElseYielded))
      for (OpOperand &use :
           llvm::make_early_inc_range(std::get<0>(it).getUses())) {
        if (nextThen && nextThen->getParent()->isAncestor(
                            use.getOwner()->getParentRegion())) {
          rewriter.startOpModification(use.getOwner());
          use.set(std::get<1>(it));
          rewriter.finalizeOpModification(use.getOwner());
        } else if (nextElse && nextElse->getParent()->isAncestor(
                                   use.getOwner()->getParentRegion())) {
          rewriter.startOpModification(use.getOwner());
          use.set(std::get<2>(it));
          rewriter.finalizeOpModification(use.getOwner());
        }
      }

    SmallVector<Type> mergedTypes(prevIf.getResultTypes());
    llvm::append_range(mergedTypes, nextIf.getResultTypes());

    IfOp combinedIf = IfOp::create(rewriter, nextIf.getLoc(), mergedTypes,
                                   prevIf.getCondition());

    rewriter.inlineRegionBefore(prevIf.getThenRegion(),
                                combinedIf.getThenRegion(),
                                combinedIf.getThenRegion().begin());

    if (nextThen) {
      YieldOp thenYield = dyn_cast<YieldOp>(combinedIf.getThenTerminator());
      YieldOp thenYield2 = dyn_cast<YieldOp>(nextThen->getTerminator());
      rewriter.mergeBlocks(nextThen, combinedIf.getThenBlock());
      rewriter.setInsertionPointToEnd(combinedIf.getThenBlock());

      if (thenYield && thenYield2) {
        SmallVector<Value> mergedYields(thenYield.getOperands());
        llvm::append_range(mergedYields, thenYield2.getOperands());
        YieldOp::create(rewriter, thenYield2.getLoc(), mergedYields);
      }
      if (thenYield)
        rewriter.eraseOp(thenYield);
      if (thenYield2)
        rewriter.eraseOp(thenYield2);
    }

    rewriter.inlineRegionBefore(prevIf.getElseRegion(),
                                combinedIf.getElseRegion(),
                                combinedIf.getElseRegion().begin());

    if (nextElse) {
      if (combinedIf.getElseRegion().empty()) {
        rewriter.inlineRegionBefore(*nextElse->getParent(),
                                    combinedIf.getElseRegion(),
                                    combinedIf.getElseRegion().begin());
      } else {
        YieldOp elseYield = dyn_cast<YieldOp>(combinedIf.getElseTerminator());
        YieldOp elseYield2 = dyn_cast<YieldOp>(nextElse->getTerminator());
        rewriter.mergeBlocks(nextElse, combinedIf.getElseBlock());

        rewriter.setInsertionPointToEnd(combinedIf.getElseBlock());

        if (elseYield && elseYield2) {
          SmallVector<Value> mergedElseYields(elseYield.getOperands());
          llvm::append_range(mergedElseYields, elseYield2.getOperands());

          YieldOp::create(rewriter, elseYield2.getLoc(), mergedElseYields);
        }
        if (elseYield)
          rewriter.eraseOp(elseYield);
        if (elseYield2)
          rewriter.eraseOp(elseYield2);
      }
    }

    SmallVector<Value> prevValues;
    SmallVector<Value> nextValues;
    for (const auto &pair : llvm::enumerate(combinedIf.getResults()))
      if (pair.index() < prevIf.getNumResults())
        prevValues.push_back(pair.value());
      else
        nextValues.push_back(pair.value());
    rewriter.replaceOp(prevIf, prevValues);
    rewriter.replaceOp(nextIf, nextValues);
    return success();
  }
};

/// Porting of SCF::IfOp::RemoveEmptyElseBranch
struct RemoveEmptyElseBranch : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Cannot remove else region when there are operation results.
    if (ifOp.getNumResults())
      return failure();
    Block *elseBlock = ifOp.getElseBlock();
    if (!elseBlock || !llvm::hasSingleElement(*elseBlock))
      return failure();
    // Cannot remove else region with not-yield terminator
    if (isTerminatorForParent(ifOp.getElseTerminator()))
      return failure();
    auto newIfOp = rewriter.cloneWithoutRegions(ifOp);
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.eraseOp(ifOp);
    return success();
  }
};

/// Porting of SCF::IfOp::CombineNestedIfs
/// Added additional support for ContinueOp/BreakOp/ReturnOp terminators
struct CombineNestedIfs : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto nestedOps = op.getThenBlock()->without_terminator();
    // Nested `if` must be the only op in block.
    if (!llvm::hasSingleElement(nestedOps))
      return failure();

    // If there is an else block, it can only yield
    if (op.getElseBlock() && !llvm::hasSingleElement(*op.getElseBlock()))
      return failure();

    auto nestedIf = dyn_cast<IfOp>(*nestedOps.begin());
    if (!nestedIf)
      return failure();

    if (nestedIf.getElseBlock() &&
        !llvm::hasSingleElement(*nestedIf.getElseBlock()))
      return failure();

    // Support only YieldOp as terminator except for nestedIf's then-block
    if (isTerminatorForParent(op.getThenTerminator()) ||
        isTerminatorForParent(op.getElseTerminator()) ||
        isTerminatorForParent(nestedIf.getElseTerminator()))
      return failure();

    // Support ReturnOp/ContinueOp/BreakOp only inside nestedIf
    // and only in the absence of else-blocks
    if (isTerminatorForParent(nestedIf.getThenTerminator()) &&
        (op.getElseBlock() || nestedIf.getElseBlock()))
      return failure();

    SmallVector<Value> thenYield(op.getThenTerminator()->getOperands());
    SmallVector<Value> elseYield;
    if (op.getElseBlock())
      llvm::append_range(elseYield, op.getElseTerminator()->getOperands());

    // A list of indices for which we should upgrade the value yielded
    // in the else to a select.
    SmallVector<unsigned> elseYieldsToUpgradeToSelect;

    // If the outer scf.if yields a value produced by the inner scf.if,
    // only permit combining if the value yielded when the condition
    // is false in the outer scf.if is the same value yielded when the
    // inner scf.if condition is false.
    // Note that the array access to elseYield will not go out of bounds
    // since it must have the same length as thenYield, since they both
    // come from the same scf.if.
    for (const auto &tup : llvm::enumerate(thenYield)) {
      if (tup.value().getDefiningOp() == nestedIf) {
        auto nestedIdx = llvm::cast<OpResult>(tup.value()).getResultNumber();
        if (nestedIf.getElseTerminator()->getOperand(nestedIdx) !=
            elseYield[tup.index()]) {
          return failure();
        }
        // If the correctness test passes, we will yield
        // corresponding value from the inner scf.if
        thenYield[tup.index()] =
            nestedIf.getThenTerminator()->getOperand(nestedIdx);
        continue;
      }

      // Otherwise, we need to ensure the else block of the combined
      // condition still returns the same value when the outer condition is
      // true and the inner condition is false. This can be accomplished if
      // the then value is defined outside the outer scf.if and we replace the
      // value with a select that considers just the outer condition. Since
      // the else region contains just the yield, its yielded value is
      // defined outside the scf.if, by definition.

      // If the then value is defined within the scf.if, bail.
      if (tup.value().getParentRegion() == &op.getThenRegion())
        return failure();
      // SelectOp can't be inserted for non-TileType value
      auto tileType = llvm::dyn_cast<TileType>(tup.value().getType());
      if (!tileType)
        return failure();
      elseYieldsToUpgradeToSelect.push_back(tup.index());
    }

    Location loc = op.getLoc();
    Value newCondition = AndIOp::create(rewriter, loc, op.getCondition(),
                                        nestedIf.getCondition());
    auto newIf = IfOp::create(rewriter, loc, op.getResultTypes(), newCondition);
    Block *newIfBlock = rewriter.createBlock(&newIf.getThenRegion());

    SmallVector<Value> results;
    llvm::append_range(results, newIf.getResults());
    rewriter.setInsertionPoint(newIf);

    for (auto idx : elseYieldsToUpgradeToSelect)
      results[idx] =
          createSelectOpByType(rewriter, op.getLoc(), op.getCondition(),
                               thenYield[idx], elseYield[idx]);

    rewriter.mergeBlocks(nestedIf.getThenBlock(), newIfBlock);
    rewriter.setInsertionPointToEnd(newIf.getThenBlock());
    auto newTerminator = newIf.getThenTerminator();
    if (llvm::isa<YieldOp>(newTerminator))
      rewriter.replaceOpWithNewOp<YieldOp>(newTerminator, thenYield);
    if (!elseYield.empty()) {
      rewriter.createBlock(&newIf.getElseRegion());
      rewriter.setInsertionPointToEnd(newIf.getElseBlock());
      YieldOp::create(rewriter, loc, elseYield);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Perform canonicalization for IfOp with two ReturnOp/ContinueOp/BreakOp
/// Move Else-Region to Parent
/// replaceOpWithRegion will clear out unreachable operations
struct MoveTerminatorToParent : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getElseRegion().empty())
      return failure();
    if (!isTerminatorForParent(op.getThenTerminator()) ||
        !isTerminatorForParent(op.getElseTerminator()))
      return failure();

    auto newIfOp = IfOp::create(rewriter, op.getLoc(), SmallVector<Type>(),
                                op.getCondition());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());

    return replaceOpWithRegion(rewriter, op, op.getElseRegion());
  }
};

void IfOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                       ::mlir::MLIRContext *context) {
  results.add<RemoveUnusedResults, ReplaceYieldWithValue, RemoveStaticCondition,
              ConvertToSelect, RemoveEmptyElseBranch, CombineIfs,
              CombineNestedIfs, MoveTerminatorToParent>(context);
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  auto resultType = getResult().getType();
  auto shape = resultType.getShape();
  auto elemType = resultType.getElementType();

  if (shape.size() != 1)
    return emitOpError("expects result type to be 1-d tile");

  uint64_t numElems = shape[0];
  uint32_t bitwidth = elemType.getIntOrFloatBitWidth();

  // The result of ((uint64_t)1) << 64 is 1 (overflow).
  // We don't need to check for i64 since `numElems` cannot exceed 1^64.
  if (bitwidth < 64) {
    uint64_t maxValue = ((uint64_t)1) << bitwidth;
    if (numElems > maxValue)
      return emitOpError("the number of elements ")
             << numElems << " exceeds the maximum value of element type "
             << elemType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// JoinTokensOp
//===----------------------------------------------------------------------===//

LogicalResult JoinTokensOp::verify() {
  size_t numTokens = getTokens().size();
  if (numTokens < 2)
    return emitOpError("expect two or more tokens");
  return success();
}

//===----------------------------------------------------------------------===//
// Memory Semantics Parsing Utilities
//===----------------------------------------------------------------------===//

LogicalResult
cuda_tile::impl::verifyMemoryModelLoad(Operation *op,
                                       MemoryOrderingSemantics memoryOrdering,
                                       std::optional<MemoryScope> scope) {
  // First validate the memory ordering is supported
  switch (memoryOrdering) {
  case MemoryOrderingSemantics::WEAK:
  case MemoryOrderingSemantics::RELAXED:
  case MemoryOrderingSemantics::ACQUIRE:
    break; // Valid orderings
  default:
    return op->emitOpError(
               "expect one of: weak, relaxed, or acquire, but got: ")
           << stringifyMemoryOrderingSemantics(memoryOrdering);
  }

  // Then validate scope requirements based on ordering
  if (memoryOrdering == MemoryOrderingSemantics::WEAK) {
    if (scope.has_value())
      return op->emitOpError("weak load must not have memory scope");
  } else {
    // RELAXED or ACQUIRE require scope
    if (!scope.has_value())
      return op->emitOpError("memory scope is required for ")
             << stringifyMemoryOrderingSemantics(memoryOrdering) << " load";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ViewLoadStore Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Parse the trailing `attributes { ... }` keyword-attr dict and reject any
/// attribute whose name appears in `reservedKeys`. Reserved keys are those the
/// corresponding printer emits explicitly (and elides from the trailing dict);
/// allowing them through the trailing dict would let mixed spellings duplicate
/// or override the canonical values once the parser merges attrs onto the op.
ParseResult parseAttrDictRejectingReservedKeys(
    OpAsmParser &parser, NamedAttrList &attributes,
    ArrayRef<StringRef> reservedKeys) {
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(attributes))
    return failure();
  for (StringRef key : reservedKeys) {
    if (attributes.get(key)) {
      return parser.emitError(loc)
             << "attribute '" << key
             << "' is reserved and may not appear in the trailing "
                "`attributes { ... }` dict; it is printed explicitly and "
                "must be set via the canonical syntax instead";
    }
  }
  return success();
}

/// Parse common elements for LoadViewTkoOp and StoreViewTkoOp:
/// - Optional token
/// - Optional optimization_hints
/// - Optional inbounds = [true, false, ...]
/// - Optional attribute dict
ParseResult parseViewTkoCommon(OpAsmParser &parser, NamedAttrList &attributes,
                               OptimizationHintsAttr &optHints, bool &hasToken,
                               OpAsmParser::UnresolvedOperand &token,
                               DenseBoolArrayAttr &inbounds,
                               SMLoc &boundAttributeLoc) {
  // Parse optional token
  hasToken = succeeded(parser.parseOptionalKeyword("token"));
  if (hasToken && (parser.parseEqual() || parser.parseOperand(token)))
    return failure();

  // Parse optional optimization_hints
  if (succeeded(parser.parseOptionalKeyword("optimization_hints"))) {
    if (parser.parseEqual())
      return failure();
    Attribute parsedAttr = OptimizationHintsAttr::parse(parser, Type{});
    if (!parsedAttr)
      return failure();
    optHints = cast<OptimizationHintsAttr>(parsedAttr);
  }

  // Parse optional `inbounds = [true, false, ...]`.
  if (succeeded(parser.parseOptionalKeyword("inbounds"))) {
    if (parser.parseEqual()) {
      return failure();
    }
    boundAttributeLoc = parser.getCurrentLocation();
    SmallVector<bool> values;
    if (parser.parseCommaSeparatedList(
            OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
              StringRef tok;
              auto loc = parser.getCurrentLocation();
              if (parser.parseKeyword(&tok)) {
                return failure();
              }
              if (tok == "true") {
                values.push_back(true);
              } else if (tok == "false") {
                values.push_back(false);
              } else {
                return parser.emitError(loc)
                       << "expected 'true' or 'false' in inbounds list, "
                          "got '"
                       << tok << "'";
              }
              return success();
            })) {
      return failure();
    }
    inbounds = parser.getBuilder().getDenseBoolArrayAttr(values);
  }

  // Parse attributes. Reject reserved keys that the printer emits explicitly
  // (see printViewTkoOptionals) so they cannot be smuggled through the trailing
  // `attributes { ... }` dict and override the canonical fields.
  if (parseAttrDictRejectingReservedKeys(
          parser, attributes,
          {"memory_ordering_semantics", "memory_scope", "optimization_hints",
           "operandSegmentSizes", "inbounds"}))
    return failure();

  return success();
}

/// Parse index types that may be either:
/// - A single type (splat) applied to all indices
/// - Individual types for each index
/// Returns the parsed index types.
ParseResult parseIndexTypes(OpAsmParser &parser,
                            ArrayRef<OpAsmParser::UnresolvedOperand> indices,
                            SmallVectorImpl<Type> &indexTypes, SMLoc &typeLoc) {
  // Parse index types (may be one type for all, or individual types)
  if (succeeded(parser.parseOptionalComma())) {
    // Comma present means index types must follow
    typeLoc = parser.getCurrentLocation();
    Type maybeIndexType;
    if (failed(parseCudaTileType(parser, maybeIndexType)))
      return parser.emitError(typeLoc) << "expected index type after comma";

    // Check if this is followed by comma (multiple types) or arrow (splat)
    if (succeeded(parser.parseOptionalComma())) {
      // Multiple types - backtrack and parse all
      indexTypes.push_back(maybeIndexType);
      while (true) {
        Type idxType;
        if (parseCudaTileType(parser, idxType))
          return failure();
        indexTypes.push_back(idxType);
        if (failed(parser.parseOptionalComma()))
          break;
      }
    } else {
      // Splat type - use same type for all indices
      indexTypes.resize(indices.size(), maybeIndexType);
    }
  }
  return success();
}

/// Bundle of parser state shared between `LoadViewTkoOp::parse` /
/// `StoreViewTkoOp::parse`.
struct ViewTkoParseState {
  // Operands.
  ArrayRef<OpAsmParser::UnresolvedOperand> indices;
  ArrayRef<Type> indexTypes;
  bool hasToken = false;
  OpAsmParser::UnresolvedOperand token;

  // Memory-model attributes. `memSemantics` is required; `memScope` is set
  // only for non-`weak` orderings.
  MemoryOrderingSemanticsAttr memSemantics;
  MemoryScopeAttr memScope;

  // Other optional attributes parsed by `parseViewTkoCommon`.
  OptimizationHintsAttr optHints;
  DenseBoolArrayAttr inbounds;

  // Trailing optional attribute dict of the form `attributes { ... }`.
  NamedAttrList attributes;

  // operandSegmentSizes payload for the op's AttrSizedOperandSegments.
  // Owned so callers can pass a brace-enclosed list without lifetime worries.
  SmallVector<int32_t, 4> segmentSizes;

  // Source locations used for diagnostics about index-type / inbounds
  // mismatches.
  SMLoc typeLoc;
  SMLoc boundAttributeLoc;
  SMLoc indicesLoc;
};

/// Finalize parsing by resolving operands and setting attributes.
ParseResult finalizeViewTkoParse(OpAsmParser &parser, OperationState &result,
                                 const ViewTkoParseState &parsed) {
  // Ensure we have the right number of index types
  if (!parsed.indices.empty() && parsed.indexTypes.empty()) {
    return parser.emitError(parsed.indicesLoc)
           << "index types must be specified when indices are present";
  }

  if (!parsed.indexTypes.empty() &&
      parsed.indexTypes.size() != parsed.indices.size()) {
    return parser.emitError(parsed.typeLoc)
           << "expected " << parsed.indices.size() << " index types, got "
           << parsed.indexTypes.size();
  }

  if (parsed.inbounds &&
      static_cast<size_t>(parsed.inbounds.size()) != parsed.indices.size()) {
    return parser.emitError(parsed.boundAttributeLoc)
           << "inbounds size (" << parsed.inbounds.size()
           << ") must match the number of index dimensions ("
           << parsed.indices.size() << ")";
  }

  if (parser.resolveOperands(parsed.indices, parsed.indexTypes,
                             parser.getCurrentLocation(), result.operands)) {
    return failure();
  }

  if (parsed.hasToken &&
      parser.resolveOperand(parsed.token,
                            cuda_tile::TokenType::get(parser.getContext()),
                            result.operands)) {
    return failure();
  }

  // Set attributes
  result.addAttribute("memory_ordering_semantics", parsed.memSemantics);
  if (parsed.memScope) {
    result.addAttribute("memory_scope", parsed.memScope);
  }
  if (parsed.optHints) {
    result.addAttribute("optimization_hints", parsed.optHints);
  }
  // The `inbounds` attribute is required; if the user omitted the textual
  // `inbounds = [...]` clause, synthesize an all-`false` vector sized to the
  // index rank (every dimension is conservatively out-of-bounds).
  DenseBoolArrayAttr inbounds = parsed.inbounds;
  if (!inbounds) {
    inbounds = parser.getBuilder().getDenseBoolArrayAttr(
        SmallVector<bool>(parsed.indices.size(), /*value=*/false));
  }
  result.addAttribute("inbounds", inbounds);
  result.addAttributes(parsed.attributes);

  // Set operand segment sizes
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(parsed.segmentSizes));

  return success();
}

/// Print common elements for LoadViewTkoOp and StoreViewTkoOp.
template <typename OpType>
void printViewTkoCommon(OpAsmPrinter &p, OpType op) {
  // Print memory attributes
  printMemoryAttributes(p, op, op.getMemoryOrderingSemanticsAttr(),
                        op.getMemoryScopeAttr());
}

/// Print optional token for LoadViewTkoOp and StoreViewTkoOp.
template <typename OpType>
void printViewTkoOptionals(OpAsmPrinter &p, OpType op) {
  // Print optional token
  if (op.getToken()) {
    p << " token = ";
    p.printOperand(op.getToken());
  }

  // Print optional optimization_hints
  if (op.getOptimizationHintsAttr()) {
    p << " optimization_hints = ";
    op.getOptimizationHintsAttr().print(p);
  }

  // Print inbounds in `[true, false, ...]` form. The attribute is required
  // and always present in the IR (synthesized by the parser as all-`false`
  // when the textual `inbounds = [...]` clause is omitted), and the
  // bytecode writer always encodes it. In the textual form, however, we
  // elide the clause when every element is `false`: that is the value the
  // parser synthesizes for an omitted clause, so a printed-then-reparsed IR
  // is preserved unchanged.
  ArrayRef<bool> inbounds = op.getInbounds();
  if (llvm::any_of(inbounds, [](bool v) { return v; })) {
    p << " inbounds = [";
    llvm::interleaveComma(inbounds, p,
                          [&](bool v) { p << (v ? "true" : "false"); });
    p << "]";
  }

  // Print attributes (eliding the ones we printed explicitly above)
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {"memory_ordering_semantics",
                                      "memory_scope", "optimization_hints",
                                      "operandSegmentSizes", "inbounds"});
}

/// Print index types (splat or individual).
void printIndexTypes(OpAsmPrinter &p, ValueRange indices) {
  if (!indices.empty()) {
    // Check if all index types are the same (splat)
    bool allSame = llvm::all_equal(indices.getTypes());
    if (allSame) {
      // Print single type for all indices
      p << ", ";
      printCudaTileType(p, indices.front().getType());
    } else {
      // Print all index types individually
      for (Value idx : indices) {
        p << ", ";
        printCudaTileType(p, idx.getType());
      }
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// AtomicRedViewTkoOp: custom parse/print (placed after file-scope helpers)
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::AtomicRedViewTkoOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  // Parse memory ordering semantics
  SMLoc semLoc = parser.getCurrentLocation();
  StringRef semStr;
  if (parser.parseKeyword(&semStr)) {
    return failure();
  }
  auto semOpt = symbolizeMemoryOrderingSemantics(semStr);
  if (!semOpt) {
    return parser.emitError(semLoc)
           << "invalid memory_ordering_semantics: '" << semStr
           << "'; expected one of: weak, relaxed, acquire, release, acq_rel";
  }
  MemoryOrderingSemanticsAttr memSemantics =
      MemoryOrderingSemanticsAttr::get(parser.getContext(), *semOpt);

  // Parse optional memory scope (verify() enforces RELAXED + valid scope)
  MemoryScopeAttr memScope;
  SMLoc scopeLoc = parser.getCurrentLocation();
  StringRef scopeStr;
  if (succeeded(parser.parseOptionalKeyword(&scopeStr))) {
    auto scopeOpt = symbolizeMemoryScope(scopeStr);
    if (!scopeOpt) {
      return parser.emitError(scopeLoc)
             << "invalid memory_scope: '" << scopeStr
             << "'; expected one of: tl_blk, device, sys";
    }
    memScope = MemoryScopeAttr::get(parser.getContext(), *scopeOpt);
  }

  // Parse %view[%idx0, %idx1, ...]
  OpAsmParser::UnresolvedOperand view;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  if (parser.parseOperand(view) || parser.parseLSquare() ||
      parser.parseOperandList(indices) || parser.parseRSquare()) {
    return failure();
  }

  // Parse , mode , %value
  SMLoc modeLoc = parser.getCurrentLocation();
  StringRef modeStr;
  OpAsmParser::UnresolvedOperand value;
  if (parser.parseComma() || parser.parseKeyword(&modeStr) ||
      parser.parseComma() || parser.parseOperand(value)) {
    return failure();
  }
  auto modeOpt = symbolizeAtomicRMWMode(modeStr);
  if (!modeOpt) {
    return parser.emitError(modeLoc)
           << "unknown atomic RMW mode: '" << modeStr
           << "'; expected one of: and, or, xor, add, addf, max, min, "
              "umax, umin";
  }

  // Parse optional 'token = %tok'
  OpAsmParser::UnresolvedOperand token;
  bool hasToken = succeeded(parser.parseOptionalKeyword("token"));
  if (hasToken && (parser.parseEqual() || parser.parseOperand(token))) {
    return failure();
  }

  // Parse optional `inbounds = [true, false, ...]` clause. The attribute is
  // required on the op (synthesized below as all-`false` when the textual
  // clause is omitted), matching the load/store view_tko pattern.
  DenseBoolArrayAttr inbounds;
  SMLoc boundAttributeLoc = parser.getCurrentLocation();
  if (parseInbounds(parser, inbounds, indices)) {
    return failure();
  }

  // Parse optional attr-dict. Reject reserved keys that the printer emits
  // explicitly (see AtomicRedViewTkoOp::print) so they cannot be smuggled
  // through the trailing `attributes { ... }` dict and override the canonical
  // fields.
  NamedAttrList attributes;
  if (parseAttrDictRejectingReservedKeys(parser, attributes,
                                         {"memory_ordering_semantics",
                                          "memory_scope", "mode",
                                          "operandSegmentSizes", "inbounds"})) {
    return failure();
  }

  // Parse types: : value_type, view_type, [index_types] -> token_type
  // Following store_view_tko pattern: value type first, then view type, then
  // optional index types. Placing value type before the index types allows
  // parseIndexTypes to correctly detect the end of index types via '->'.
  if (parser.parseColon()) {
    return failure();
  }
  Type viewType;
  SmallVector<Type> indexTypes;
  SMLoc typeLoc = parser.getCurrentLocation();
  Type valueType, tokenType;
  if (parseCudaTileType(parser, valueType) || parser.parseComma() ||
      parseCudaTileType(parser, viewType)) {
    return failure();
  }
  if (parseIndexTypes(parser, indices, indexTypes, typeLoc)) {
    return failure();
  }
  if (parser.parseArrow() || parseCudaTileType(parser, tokenType)) {
    return failure();
  }

  // Resolve operands: view, index..., value, token?
  if (parser.resolveOperand(view, viewType, result.operands)) {
    return failure();
  }
  if (!indices.empty() && indexTypes.empty()) {
    return parser.emitError(parser.getCurrentLocation())
           << "index types must be specified when indices are present";
  }
  if (!indexTypes.empty() && indexTypes.size() != indices.size()) {
    return parser.emitError(typeLoc)
           << "expected " << indices.size() << " index types, got "
           << indexTypes.size();
  }
  if (parser.resolveOperands(indices, indexTypes, typeLoc, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(value, valueType, result.operands)) {
    return failure();
  }
  if (hasToken && parser.resolveOperand(
                      token, cuda_tile::TokenType::get(parser.getContext()),
                      result.operands)) {
    return failure();
  }

  // Set attributes
  result.addAttribute("memory_ordering_semantics", memSemantics);
  if (memScope) {
    result.addAttribute("memory_scope", memScope);
  }
  result.addAttribute("mode",
                      AtomicRMWModeAttr::get(parser.getContext(), *modeOpt));
  // Size-check explicit inbounds against parsed index count; synthesize an
  // all-`false` vector when the textual `inbounds = [...]` clause is omitted
  // so the attribute is always present in the IR.
  if (inbounds && static_cast<size_t>(inbounds.size()) != indices.size()) {
    return parser.emitError(boundAttributeLoc)
           << "inbounds size (" << inbounds.size()
           << ") must match the number of index dimensions (" << indices.size()
           << ")";
  }
  if (!inbounds) {
    inbounds = parser.getBuilder().getDenseBoolArrayAttr(
        SmallVector<bool>(indices.size(), /*value=*/false));
  }
  result.addAttribute("inbounds", inbounds);
  result.addAttributes(attributes);
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(indices.size()), 1, hasToken ? 1 : 0}));

  result.types.push_back(tokenType);
  return success();
}

void cuda_tile::AtomicRedViewTkoOp::print(OpAsmPrinter &p) {
  // Print memory attributes (e.g. "relaxed device")
  printViewTkoCommon(p, *this);

  // Print %view[%idx0, %idx1, ...], mode, %value
  p << " ";
  p.printOperand(getView());
  p << "[";
  llvm::interleaveComma(getIndex(), p, [&](Value idx) { p.printOperand(idx); });
  p << "]";
  p << ", " << stringifyAtomicRMWMode(getMode()) << ", ";
  p.printOperand(getValue());

  // Print optional 'token = %tok'
  if (getToken()) {
    p << " token = ";
    p.printOperand(getToken());
  }

  // The attribute is always present in IR; elide the parser-synthesized
  // all-false default from the textual form.
  printInbounds(p, getInbounds(), /*elideAllFalse=*/true);

  // Print attr-dict (exclude explicitly-printed attrs)
  p.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(), {"memory_ordering_semantics", "memory_scope",
                                   "mode", "operandSegmentSizes", "inbounds"});

  // Print types: value_type, view_type, [index_types] -> token_type
  // Matches store_view_tko: value type first so parseIndexTypes can use '->'
  // to delimit the end of optional index types.
  p << " : ";
  printCudaTileType(p, getValue().getType());
  p << ", ";
  printCudaTileType(p, getView().getType());
  printIndexTypes(p, getIndex());
  p << " -> ";
  printCudaTileType(p, getResultToken().getType());
}

//===----------------------------------------------------------------------===//
// LoadViewTkoOp
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::LoadViewTkoOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  OpAsmParser::UnresolvedOperand view;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  OpAsmParser::UnresolvedOperand token;
  NamedAttrList attributes;
  OptimizationHintsAttr optHints;
  DenseBoolArrayAttr inbounds;
  bool hasToken = false;
  SMLoc typeLoc;
  SMLoc boundAttributeLoc;
  SMLoc indicesLoc;

  // Parse memory attributes
  MemoryOrderingSemanticsAttr memSemantics;
  MemoryScopeAttr memScope;
  if (parseMemoryAttributes(parser, memSemantics, memScope))
    return failure();

  // Parse view and indices
  if (parser.parseOperand(view) || parser.parseLSquare()) {
    return failure();
  }
  indicesLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indices) || parser.parseRSquare()) {
    return failure();
  }

  // Parse common optional elements
  if (parseViewTkoCommon(parser, attributes, optHints, hasToken, token,
                         inbounds, boundAttributeLoc)) {
    return failure();
  }

  // Parse types: : view_type, index_types... -> result_type, token_type
  if (parser.parseColon())
    return failure();

  Type viewType;
  SmallVector<Type> indexTypes;
  Type resultType, tokenType;

  if (parseCudaTileType(parser, viewType))
    return failure();

  // Parse index types
  if (parseIndexTypes(parser, indices, indexTypes, typeLoc))
    return failure();

  // Parse arrow and result types
  if (parser.parseArrow() || parseCudaTileType(parser, resultType) ||
      parser.parseComma() || parseCudaTileType(parser, tokenType))
    return failure();

  // Resolve view operand
  if (parser.resolveOperand(view, viewType, result.operands))
    return failure();

  // Finalize parsing
  ViewTkoParseState parsed{
      /*indices=*/indices,
      /*indexTypes=*/indexTypes,
      /*hasToken=*/hasToken,
      /*token=*/token,
      /*memSemantics=*/memSemantics,
      /*memScope=*/memScope,
      /*optHints=*/optHints,
      /*inbounds=*/inbounds,
      /*attributes=*/std::move(attributes),
      /*segmentSizes=*/
      {1, static_cast<int32_t>(indices.size()), hasToken ? 1 : 0},
      /*typeLoc=*/typeLoc,
      /*boundAttributeLoc=*/boundAttributeLoc,
      /*indicesLoc=*/indicesLoc,
  };
  if (finalizeViewTkoParse(parser, result, parsed)) {
    return failure();
  }

  // Set result types
  result.types.push_back(resultType);
  result.types.push_back(tokenType);

  return success();
}

void cuda_tile::LoadViewTkoOp::print(OpAsmPrinter &p) {
  printViewTkoCommon(p, *this);

  // Print view and indices
  p << " ";
  p.printOperand(getView());
  p << "[";
  llvm::interleaveComma(getIndex(), p, [&](Value idx) { p.printOperand(idx); });
  p << "]";

  printViewTkoOptionals(p, *this);

  // Print types
  p << " : ";
  printCudaTileType(p, getView().getType());
  printIndexTypes(p, getIndex());

  // Print result types
  p << " -> ";
  printCudaTileType(p, getTile().getType());
  p << ", ";
  printCudaTileType(p, getResultToken().getType());
}

LogicalResult LoadViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(this)))
    return failure();
  if (failed(verifyUnrestrictedViewPtrAttr(*this, getView()))) {
    return failure();
  }

  ArrayRef<bool> inbounds = getInbounds();
  if (inbounds.size() != getIndex().size()) {
    return emitOpError("inbounds size (")
           << inbounds.size() << ") must match the number of index dimensions ("
           << getIndex().size() << ")";
  }

  return impl::verifyMemoryModelLoad(*this, getMemoryOrderingSemantics(),
                                     getMemoryScope());
}

//===----------------------------------------------------------------------===//
// LoadPtrTkoOp
//===----------------------------------------------------------------------===//

LogicalResult LoadPtrTkoOp::verify() {
  if (failed(verifyOptHintsCommon(this)))
    return failure();
  auto elemTy = getSource().getType().getElementType();
  if (auto ptrTy = dyn_cast<PointerType>(elemTy)) {
    if (auto ptrAttr = ptrTy.getPtrAttr();
        ptrAttr && ptrAttr.getValue() == PtrAttr::MULTICAST) {
      return emitOpError("multicast pointer is not allowed");
    }
  }
  return impl::verifyMemoryModelLoad(*this, getMemoryOrderingSemantics(),
                                     getMemoryScope());
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

LogicalResult LoopOp::verifyRegions() {
  return verifyLoopIterValues(*this, getResults(), getRegionIterValues());
}

void LoopOp::print(OpAsmPrinter &p) {
  p << " ";
  bool hasIters = !getInitValues().empty();
  bool hasReturn = !getResultTypes().empty();

  if (hasIters) {
    p << "iter_values(";
    llvm::interleaveComma(
        llvm::zip(getRegionIterValues(), getInitValues()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ") ";
  }
  if (hasIters || hasReturn)
    p << ": ";
  if (hasIters) {
    printCudaTileType(p, getInitValues().getTypes());
    p << " ";
    if (hasReturn)
      p << "-> ";
  }
  if (hasReturn) {
    printCudaTileType(p, getResultTypes());
    p << " ";
  }

  printControlFlowRegion<ContinueOp>(p, *this, getRegion());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> iterOperands;
  SmallVector<Type, 4> iterTypes;

  if (failed(parser.parseOptionalKeyword("iter_values"))) {
    // no iter_values, but can still have a return type
    if (succeeded(parser.parseOptionalColon()))
      if (parseCudaTileType(parser, result.types))
        return failure();
  } else {
    // iter_values are present and must have colon followed by types
    if (parser.parseAssignmentList(regionArgs, iterOperands) ||
        parser.parseColon() || parseCudaTileType(parser, iterTypes))
      return failure();
    if (regionArgs.size() != iterTypes.size())
      return parser.emitError(
          parser.getCurrentLocation(),
          "found different number of iter_values and types");
    // check for optional result type(s)
    if (succeeded(parser.parseOptionalArrow()))
      if (parseCudaTileType(parser, result.types))
        return failure();
    // Set region argument types for loop body
    for (auto [regionArg, type] : llvm::zip_equal(regionArgs, iterTypes))
      regionArg.type = type;
  }

  // Parse region and attr dict.
  if (parseControlFlowRegion<LoopOp>(parser, *result.addRegion(), regionArgs) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve operands.
  if (parser.resolveOperands(iterOperands, iterTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MakeTensorViewOp
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand basePtrOperand;
  if (parser.parseOperand(basePtrOperand) || parser.parseComma())
    return ParseResult::failure();

  SmallVector<std::tuple<int64_t, SMLoc>> opSideShape;
  SmallVector<std::tuple<int64_t, SMLoc>> opSideStrides;
  SmallVector<Value> shapeDynamicValues;
  SmallVector<Value> strideDynamicValues;

  auto parseConstOrValue =
      [&](SmallVectorImpl<std::tuple<int64_t, SMLoc>> &elements,
          SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicValues)
      -> ParseResult {
    SMLoc location = parser.getCurrentLocation();

    int64_t constant = 0;
    OptionalParseResult intParseResult = parser.parseOptionalInteger(constant);
    if (intParseResult.has_value()) {
      if (failed(intParseResult.value()))
        return ParseResult::failure();
      elements.push_back({constant, location});
      return ParseResult::success();
    }

    OpAsmParser::UnresolvedOperand operand;
    OptionalParseResult valueParseResult = parser.parseOptionalOperand(operand);
    if (!valueParseResult.has_value())
      return parser.emitError(location, "expected either integer or SSA value");

    if (failed(valueParseResult.value()))
      return ParseResult::failure();
    dynamicValues.push_back(operand);

    elements.push_back({cuda_tile::TensorViewType::kDynamic, location});

    // Make sure dynamic elements remain int32_t-addressable.
    if (dynamicValues.size() >
        static_cast<size_t>(std::numeric_limits<int32_t>::max()))
      return parser.emitError(location,
                              "too many dynamic operands of a particular kind "
                              "(must be fewer than ")
             << std::numeric_limits<int32_t>::max() << ")";

    return ParseResult::success();
  };

  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand>
      unresolvedDynShapeOperands;
  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand>
      unresolvedDynStrideOperands;
  auto shapeElemParser = [&]() -> ParseResult {
    return parseConstOrValue(opSideShape, unresolvedDynShapeOperands);
  };
  auto strideElemParser = [&]() -> ParseResult {
    return parseConstOrValue(opSideStrides, unresolvedDynStrideOperands);
  };

  if (parser.parseKeyword("shape") || parser.parseEqual())
    return ParseResult::failure();

  SMLoc shapeDeclLoc = parser.getCurrentLocation();
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                     shapeElemParser) ||
      parser.parseComma() || parser.parseKeyword("strides") ||
      parser.parseEqual())
    return ParseResult::failure();

  SMLoc strideDeclLoc = parser.getCurrentLocation();
  NamedAttrList attributes;
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                     strideElemParser) ||
      parser.parseOptionalAttrDict(attributes) || parser.parseColon())
    return ParseResult::failure();

  // Parse the type section after ':'.
  //
  // Pointer base with an explicit dynamic-index type:
  //   `: type($base) , type($index) -> type($result)`
  //
  // Pointer base without dynamic index type printed (only when the
  // tensor_view has no `?` shape or stride dimensions):
  //   `: type($base) -> type($result)`
  //
  // Otherwise the first type is the dynamic index tile or the tensor_view
  // alone:
  //   `: type(index) -> type($result)` or `: type($result)`
  SMLoc firstTypeLoc = parser.getCurrentLocation();
  Type firstType;
  if (parseCudaTileType(parser, firstType)) {
    return ParseResult::failure();
  }

  Type basePtrType;
  SMLoc indexTypeLoc;
  Type indexType;
  cuda_tile::TensorViewType tensorView;

  if (auto tileType = dyn_cast<cuda_tile::TileType>(firstType)) {
    Type elemType = tileType.getElementType();
    // Rank-0 tile of pointer:
    // - `base, index -> result` (comma before `->`): `base` is the pointer tile,
    //   `index` is the dynamic shape/stride element type.
    // - `base -> result` with no comma: if `result` has no dynamic shape or
    //   stride, `base` is the pointer tile (e.g. `tile<ptr<..., none>>` with a
    //   static tensor_view). Otherwise keep legacy behavior where the first
    //   type was treated as the dynamic-index tile type (invalid IR such as
    //   `tile<ptr<f64>> -> tensor_view<?x...>` is diagnosed in verify).
    if (isa<cuda_tile::PointerType>(elemType)) {
      if (succeeded(parser.parseOptionalComma())) {
        basePtrType = firstType;
        indexTypeLoc = parser.getCurrentLocation();
        if (parseCudaTileType(parser, indexType)) {
          return ParseResult::failure();
        }
        if (parser.parseArrow()) {
          return ParseResult::failure();
        }
        Type resultType;
        if (parseCudaTileType(parser, resultType)) {
          return ParseResult::failure();
        }
        tensorView = dyn_cast<TensorViewType>(resultType);
      } else {
        if (parser.parseArrow()) {
          return ParseResult::failure();
        }
        Type resultType;
        if (parseCudaTileType(parser, resultType)) {
          return ParseResult::failure();
        }
        tensorView = dyn_cast<TensorViewType>(resultType);
        if (tensorView && tensorView.dynamicShapeAmount() == 0 &&
            tensorView.dynamicStrideAmount() == 0)
          basePtrType = firstType;
        else {
          indexTypeLoc = firstTypeLoc;
          indexType = firstType;
        }
      }
    } else {
      // Index type (e.g. tile<i32>) for dynamic shapes/strides.
      indexTypeLoc = firstTypeLoc;
      indexType = firstType;
      if (parser.parseArrow()) {
        return ParseResult::failure();
      }
      Type resultType;
      if (parseCudaTileType(parser, resultType)) {
        return ParseResult::failure();
      }
      tensorView = dyn_cast<TensorViewType>(resultType);
    }
  } else if (auto tvType = dyn_cast<TensorViewType>(firstType)) {
    // No dynamic shapes/strides: the first type is the result tensor_view
    // directly (no index type, no arrow).
    tensorView = tvType;
  }

  if (!tensorView)
    return parser.emitError(parser.getCurrentLocation())
           << "expected 'tensor_view' type, but got " << firstType;

  bool hasDynamicOperands = !unresolvedDynShapeOperands.empty() ||
                            !unresolvedDynStrideOperands.empty();
  if (indexType && !hasDynamicOperands)
    return parser.emitError(indexTypeLoc)
           << "unexpected index type for static shape and strides";
  if (!indexType && hasDynamicOperands)
    return parser.emitError(firstTypeLoc)
           << "expected index type for dynamic shape or stride operands";

  if (!basePtrType) {
    // Base type not printed; infer `tile<ptr<elem>>` from the tensor_view
    // element type only. Do not copy `tensor_view`'s ptr_attr onto the
    // inferred pointer: the operand may be `tile<ptr<elem>>` without ptr_attr,
    // and MakeTensorViewOp::verify checks ptr_attr consistency between base
    // and result.
    Type elemTy = tensorView.getElementType();
    basePtrType = cuda_tile::TileType::get(
        ArrayRef<int64_t>{}, cuda_tile::PointerType::get(elemTy));
  }

  if (parser.resolveOperand(basePtrOperand, basePtrType, result.operands)) {
    return ParseResult::failure();
  }

  auto compareAndDiagnostic =
      [&](ArrayRef<std::tuple<int64_t, SMLoc>> fromOperands,
          ArrayRef<int64_t> fromTensorView, SMLoc fromOperandsDeclLocation,
          const char *name) -> ParseResult {
    if (fromTensorView.size() != fromOperands.size())
      return parser.emitError(fromOperandsDeclLocation)
             << "expected " << name << " declaration to contain "
             << fromTensorView.size()
             << " elements due to tensor_view type, but " << fromOperands.size()
             << " were provided";

    for (auto [i, fromOpAndLoc, fromTensorView] :
         llvm::enumerate(fromOperands, fromTensorView)) {
      auto [fromOp, operandLoc] = fromOpAndLoc;
      if (fromOp == fromTensorView)
        continue;

      auto diag = parser.emitError(operandLoc);
      diag << "input " << name << " dimension " << i
           << " does not match tensor_view type (expected ";
      if (fromTensorView == cuda_tile::TensorViewType::kDynamic)
        diag << "dynamic";
      else
        diag << fromTensorView;
      diag << ", got ";
      if (fromOp == cuda_tile::TensorViewType::kDynamic)
        diag << "dynamic";
      else
        diag << fromOp;
      diag << ")";
      return ParseResult::failure();
    }
    return ParseResult::success();
  };

  if (compareAndDiagnostic(opSideShape, tensorView.getShape(), shapeDeclLoc,
                           "shape") ||
      compareAndDiagnostic(opSideStrides, tensorView.getStrides(),
                           strideDeclLoc, "stride"))
    return ParseResult::failure();

  if (indexType) {
    if (parser.resolveOperands(unresolvedDynShapeOperands, indexType,
                               shapeDynamicValues))
      return ParseResult::failure();

    if (parser.resolveOperands(unresolvedDynStrideOperands, indexType,
                               strideDynamicValues))
      return ParseResult::failure();
  }

  result.addOperands(shapeDynamicValues);
  result.addOperands(strideDynamicValues);

  result.addAttributes(attributes);

  // Conversion is safe as it is checked above.
  std::array<int32_t, 3> operandSegmentSizes{
      1, static_cast<int32_t>(shapeDynamicValues.size()),
      static_cast<int32_t>(strideDynamicValues.size())};
  result.addAttribute(
      "operandSegmentSizes",
      DenseI32ArrayAttr::get(parser.getContext(), operandSegmentSizes));

  result.addTypes(tensorView);

  return ParseResult::success();
}

void cuda_tile::MakeTensorViewOp::print(OpAsmPrinter &p) {
  auto printMixedOperandList = [&](ArrayRef<int64_t> elements,
                                   operand_range dynamics) {
    auto dynIt = dynamics.begin();
    llvm::interleaveComma(elements, p, [&](int64_t elem) {
      if (elem != cuda_tile::TensorViewType::kDynamic) {
        p << elem;
        return;
      }

      assert(dynIt != dynamics.end());
      p.printOperand(*dynIt);
      dynIt++;
    });
  };

  p << " ";
  p.printOperand(getBase());
  p << ", shape = [";
  printMixedOperandList(getResult().getType().getShape(), getDynamicShape());
  p << "], strides = [";
  printMixedOperandList(getResult().getType().getStrides(),
                        getDynamicStrides());
  p << "]";
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"operandSegmentSizes"});
  p << " : ";

  Type baseElem = getBase().getType().getElementType();
  bool printExplicitBase = false;
  if (auto pe = dyn_cast<cuda_tile::PointerType>(baseElem)) {
    printExplicitBase = static_cast<bool>(pe.getPtrAttr());
  }
  if (printExplicitBase) {
    printCudaTileType(p, getBase().getType());
    if (!getDynamicShape().empty() || !getDynamicStrides().empty()) {
      p << ", ";
      Type dynamicType = !getDynamicShape().empty()
                             ? getDynamicShape().getTypes().front()
                             : getDynamicStrides().getTypes().front();
      printCudaTileType(p, dynamicType);
    }
    p << " -> ";
  } else {
    if (!getDynamicShape().empty() || !getDynamicStrides().empty()) {
      Type dynamicType = !getDynamicShape().empty()
                             ? getDynamicShape().getTypes().front()
                             : getDynamicStrides().getTypes().front();
      printCudaTileType(p, dynamicType);
      p << " -> ";
    }
  }

  printCudaTileType(p, getResult().getType());
}

LogicalResult cuda_tile::MakeTensorViewOp::verify() {
  Type baseElemType = getBase().getType().getElementType();
  Type basePointeeType;
  cuda_tile::PtrAttrAttr basePtrAttr{};

  if (auto ptrTy = dyn_cast<cuda_tile::PointerType>(baseElemType)) {
    basePointeeType = ptrTy.getPointeeType();
    basePtrAttr = ptrTy.getPtrAttr();
  } else {
    return emitOpError("expected base to be a pointer type, got ")
           << baseElemType;
  }

  if (getResult().getType().getElementType() != basePointeeType) {
    return emitOpError("expected pointer to ")
           << getResult().getType().getElementType()
           << " to build tensor_view of this type, got " << basePointeeType;
  }

  // Verify PtrAttr consistency between base pointer and result tensor_view.
  auto resultPtrAttr = getResult().getType().getPtrAttr();
  if (basePtrAttr && !resultPtrAttr) {
    return emitOpError("tensor_view must preserve ptr_attr from base pointer (")
           << basePtrAttr << ")";
  }
  if (!basePtrAttr && resultPtrAttr) {
    return emitOpError("pointer base without ptr_attr cannot produce "
                       "tensor_view with ptr_attr");
  }
  if (basePtrAttr && resultPtrAttr && basePtrAttr != resultPtrAttr) {
    return emitOpError("ptr_attr mismatch between base pointer (")
           << basePtrAttr << ") and result tensor_view (" << resultPtrAttr
           << ")";
  }

  if (getResult().getType().dynamicShapeAmount() != getDynamicShape().size())
    return emitOpError("expected ")
           << getResult().getType().dynamicShapeAmount()
           << " dynamic shape operands, got " << getDynamicShape().size();

  if (getResult().getType().dynamicStrideAmount() != getDynamicStrides().size())
    return emitOpError("expected ")
           << getResult().getType().dynamicStrideAmount()
           << " dynamic stride operands, got " << getDynamicStrides().size();

  Type dynamicValuesType;
  if (!getDynamicShape().empty()) {
    dynamicValuesType = getDynamicShape().getTypes().front();
    for (auto [i, dynamicValue] : llvm::enumerate(getDynamicShape()))
      if (dynamicValue.getType() != dynamicValuesType)
        return emitOpError("expected dynamic shape index ")
               << i << " to be of the same type as the other dynamic values ("
               << dynamicValuesType << "), got " << dynamicValue.getType();
  }

  if (!getDynamicStrides().empty()) {
    dynamicValuesType = dynamicValuesType
                            ? dynamicValuesType
                            : getDynamicStrides().getTypes().front();
    for (auto [i, dynamicValue] : llvm::enumerate(getDynamicStrides()))
      if (dynamicValue.getType() != dynamicValuesType)
        return emitOpError("expected dynamic stride index ")
               << i << " to be of the same type as the other dynamic values ("
               << dynamicValuesType << "), got " << dynamicValue.getType();
  }

  return success();
}

void MakeTensorViewOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "tview");
}

//===----------------------------------------------------------------------===//
// MakePartitionViewOp
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::MakePartitionViewOp::parse(OpAsmParser &parser,
                                                  OperationState &result) {
  OpAsmParser::UnresolvedOperand tensorView;
  NamedAttrList attributes;
  if (parser.parseOperand(tensorView) ||
      parser.parseOptionalAttrDict(attributes) || parser.parseColon())
    return ParseResult::failure();

  auto loc = parser.getCurrentLocation();
  Type maybePartitionView;
  if (parseCudaTileType(parser, maybePartitionView))
    return failure();
  PartitionViewType view = dyn_cast<PartitionViewType>(maybePartitionView);
  if (!view) {
    return parser.emitError(loc)
           << "expected 'partition_view' type, but got " << maybePartitionView;
  }

  if (parser.resolveOperand(tensorView, view.getTensorView(), result.operands))
    return ParseResult::failure();

  result.types.push_back(view);
  result.addAttributes(attributes);

  return ParseResult::success();
}

void cuda_tile::MakePartitionViewOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getTensorView());
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " : ";
  printCudaTileType(p, getResult().getType());
}

LogicalResult MakePartitionViewOp::verify() {
  PartitionViewType partition = getResult().getType();
  TensorViewType tensor_view = getTensorView().getType();

  if (tensor_view != partition.getTensorView())
    return emitOpError()
           << "expected the type of the provided tensor_view value ("
           << tensor_view << ") to be the same as the view's tensor_view type ("
           << partition.getTensorView() << ")";

  return success();
}

void MakePartitionViewOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "pview");
}

//===----------------------------------------------------------------------===//
// MakeStridedViewOp
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::MakeStridedViewOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  OpAsmParser::UnresolvedOperand tensorView;
  NamedAttrList attributes;
  if (parser.parseOperand(tensorView) ||
      parser.parseOptionalAttrDict(attributes) || parser.parseColon())
    return ParseResult::failure();

  auto loc = parser.getCurrentLocation();
  Type maybeStridedView;
  if (parseCudaTileType(parser, maybeStridedView))
    return failure();
  StridedViewType view = dyn_cast<StridedViewType>(maybeStridedView);
  if (!view) {
    return parser.emitError(loc)
           << "expected 'strided_view' type, but got " << maybeStridedView;
  }

  if (parser.resolveOperand(tensorView, view.getTensorView(), result.operands))
    return ParseResult::failure();

  result.types.push_back(view);
  result.addAttributes(attributes);

  return ParseResult::success();
}

void cuda_tile::MakeStridedViewOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getTensorView());
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " : ";
  printCudaTileType(p, getResult().getType());
}

LogicalResult MakeStridedViewOp::verify() {
  StridedViewType strided = getResult().getType();
  TensorViewType tensor_view = getTensorView().getType();

  if (tensor_view != strided.getTensorView())
    return emitOpError()
           << "expected the type of the provided tensor_view value ("
           << tensor_view << ") to be the same as the view's tensor_view type ("
           << strided.getTensorView() << ")";

  return success();
}

void MakeStridedViewOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "sview");
}

//===----------------------------------------------------------------------===//
// MakeGatherScatterViewOp
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::MakeGatherScatterViewOp::parse(OpAsmParser &parser,
                                                      OperationState &result) {
  OpAsmParser::UnresolvedOperand tensorView;
  NamedAttrList attributes;
  if (parser.parseOperand(tensorView) ||
      parser.parseOptionalAttrDict(attributes) || parser.parseColon())
    return ParseResult::failure();

  auto loc = parser.getCurrentLocation();
  Type maybeGatherScatterView;
  if (parseCudaTileType(parser, maybeGatherScatterView))
    return failure();
  GatherScatterViewType view =
      dyn_cast<GatherScatterViewType>(maybeGatherScatterView);
  if (!view) {
    return parser.emitError(loc)
           << "expected 'gather_scatter_view' type, but got "
           << maybeGatherScatterView;
  }

  if (parser.resolveOperand(tensorView, view.getTensorView(), result.operands))
    return ParseResult::failure();

  result.types.push_back(view);
  result.addAttributes(attributes);

  return ParseResult::success();
}

void cuda_tile::MakeGatherScatterViewOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getTensorView());
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " : ";
  printCudaTileType(p, getResult().getType());
}

LogicalResult MakeGatherScatterViewOp::verify() {
  GatherScatterViewType gatherScatter = getResult().getType();
  TensorViewType tensor_view = getTensorView().getType();

  if (tensor_view != gatherScatter.getTensorView())
    return emitOpError()
           << "expected the type of the provided tensor_view value ("
           << tensor_view << ") to be the same as the view's tensor_view type ("
           << gatherScatter.getTensorView() << ")";

  return success();
}

void MakeGatherScatterViewOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "gsview");
}

//===----------------------------------------------------------------------===//
// MaxFOp
//===----------------------------------------------------------------------===//

LogicalResult MaxFOp::verify() { return verifyFtz(*this, getFlushToZero()); }

//===----------------------------------------------------------------------===//
// MinFOp
//===----------------------------------------------------------------------===//

LogicalResult MinFOp::verify() { return verifyFtz(*this, getFlushToZero()); }

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void cuda_tile::ModuleOp::build(OpBuilder &builder, OperationState &result,
                                StringRef name, StringRef producer) {
  result.addRegion()->emplaceBlock();
  Properties &props = result.getOrAddProperties<Properties>();
  props.setSymName(builder.getStringAttr(name));
  if (!producer.empty())
    props.setProducer(builder.getStringAttr(producer));
}

LogicalResult cuda_tile::ModuleOp::verify() {
  auto *dialect = getContext()->getLoadedDialect<CudaTileDialect>();
  assert(dialect && "cuda_tile dialect must be loaded");
  if (failed(verifyPreviewFeatures(getOperation(),
                                   dialect->getEnabledPreviewFeatures()))) {
    return failure();
  }

  for (Operation &op : getBody().front()) {
    if (!isa<SymbolOpInterface>(&op)) {
      return op.emitOpError(
          "non-symbol operations are not allowed in a module body");
    }
  }
  if (failed(DebugInfoVerifier::verifyModule(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

LogicalResult MulFOp::verify() {
  if (failed(verifyIEEERoundingModes(*this)))
    return failure();
  return verifyFtz(*this, getFlushToZero());
}

//===----------------------------------------------------------------------===//
// NegIOp
//===----------------------------------------------------------------------===//

LogicalResult NegIOp::verify() {
  if (getOverflow() == cuda_tile::IntegerOverflow::NUW) {
    // The op has signed semantics.
    return emitOpError() << "'no_unsigned_wrap' overflow flag is not supported";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

/// Shared verifier for PackOp and UnpackOp.
template <typename opTy>
static LogicalResult verifyPackUnpackTypes(opTy *op) {
  /// Return the tile size in bits, accounting for sub-byte types.
  auto getTileSizeInBits = [](TileType ty) -> uint64_t {
    auto elementType = ty.getElementType();
    unsigned elementWidth;
    if (elementType.isInteger(1))
      elementWidth = 1;
    else if (isa<FloatTF32Type>(elementType))
      elementWidth = 32;
    else
      elementWidth = elementType.getIntOrFloatBitWidth();
    return elementWidth * ty.getNumElements();
  };

  auto srcTy = op->getSource().getType();
  auto resTy = op->getResult().getType();
  if (srcTy.getRank() != 1)
    return op->emitOpError() << "expects source and result to be rank-1 tiles";

  if (srcTy.getElementType().getIntOrFloatBitWidth() ==
      resTy.getElementType().getIntOrFloatBitWidth())
    return op->emitOpError()
           << "expects source and result to have different element type widths";

  auto srcSizeInBits = getTileSizeInBits(srcTy);
  auto resSizeInBits = getTileSizeInBits(resTy);
  // Alignment should be redundant with type verifiers, but kept for safety.
  if (srcSizeInBits % 8 != 0)
    return op->emitOpError() << "expects source tile to be byte-aligned";
  if (resSizeInBits % 8 != 0)
    return op->emitOpError() << "expects result tile to be byte-aligned";
  // Ensure tiles have same storage size
  if (srcSizeInBits != resSizeInBits)
    return op->emitOpError()
           << "expects source and result to have the same size in bytes, but "
              "got source tile size "
           << srcSizeInBits / 8 << " bytes and result tile size "
           << resSizeInBits / 8 << " bytes";

  return success();
}

LogicalResult PackOp::verify() { return verifyPackUnpackTypes<PackOp>(this); }

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

LogicalResult PermuteOp::verify() {
  auto srcTy = getSource().getType();
  size_t rank = srcTy.getRank();
  ArrayRef<int32_t> permutation = getPermutation();

  if (rank < 2)
    return emitOpError("expects at least rank 2, but got: ") << rank;

  // Check if the provided permutation is valid. A permutation is invalid if:
  // a) The number of elements in `permutation` is not equal to the `source`
  //    rank.
  // b) It contains duplicate.
  // c) At least one dimension is out of bound (`permutation[i]`
  //    is >= 0 and < rank).
  // d) result tile type matches the permuted source shape
  size_t permutationSize = permutation.size();
  if (permutationSize != rank) {
    return emitOpError() << "expect permutation size (" << permutationSize
                         << ") to equal the rank of the source (" << rank
                         << ")";
  }
  DenseSet<int32_t> uniqued(permutation.begin(), permutation.end());
  if (permutationSize != uniqued.size())
    return emitOpError() << "expect permutation elements to be unique";
  for (auto [idx, perm] : llvm::enumerate(permutation)) {
    if (perm < 0 || perm >= static_cast<int32_t>(rank)) {
      return emitOpError() << "permutation element at index " << idx << " ("
                           << perm << ") is out of bound [0, " << rank << ")";
    }
  }

  // Verify result shape is valid
  ArrayRef<int64_t> resultShape = getResult().getType().getShape();
  ArrayRef<int64_t> srcShape = srcTy.getShape();
  for (const auto &[idx, permutedIdx] : llvm::enumerate(permutation)) {
    if (resultShape[idx] != srcShape[permutedIdx]) {
      return emitOpError() << "result shape invalid at index " << idx
                           << ", expected: " << srcShape[permutedIdx]
                           << ", but got: " << resultShape[idx];
    }
  }
  return success();
}

LogicalResult PermuteOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/,
    ValueRange operands, DictionaryAttr /*attributes*/, PropertyRef properties,
    RegionRange /*regions*/, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto srcTy = dyn_cast<TileType>(operands[0].getType());
  if (!srcTy) {
    return failure();
  }
  auto *prop = properties.as<Properties *>();
  auto permutation = prop->permutation.asArrayRef();
  int64_t rank = srcTy.getRank();
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return failure();
  }
  SmallVector<int64_t> permutedShape;
  for (int32_t idx : permutation) {
    if (idx < 0 || idx >= rank) {
      return failure();
    }
    permutedShape.push_back(srcTy.getShape()[idx]);
  }
  inferredReturnTypes.push_back(
      TileType::get(permutedShape, srcTy.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// PrintOp / PrintTkoOp
//===----------------------------------------------------------------------===//

/// Extract a format expression from the given string, assuming that the
/// string begins directly with the expression.
static StringRef extractFormatExpression(StringRef str) {
  assert(str.front() == '%' && "expect format string to start with '%'");
  for (size_t i = 1, e = str.size(); i < e; ++i) {
    // Format string should end with one of these characters.
    // See https://cplusplus.com/reference/cstdio/printf/.
    // %S is CUDA-specific and indicates a wide string.
    if (std::strchr("diuoxXeEfFgGaAcsSpn%", str[i]))
      return str.substr(0, i + 1);
  }
  // Found a format string expression that does not end with a valid
  // character.
  return "";
}

LogicalResult PrintTkoOp::verify() {
  int expectedNumArgs = 0;
  for (int pos = 0, end = getStr().size(); pos < end; ++pos) {
    if (getStr()[pos] != '%')
      continue;
    StringRef formatExpr = extractFormatExpression(getStr().substr(pos));
    if (formatExpr.empty())
      return emitOpError("found unterminated format expression");
    if (formatExpr.compare("%%") == 0) {
      // This is an escaped '%' character.
      ++pos;
      continue;
    }
    ++expectedNumArgs;
  }
  if (expectedNumArgs != static_cast<int>(getArgs().size()))
    return emitOpError("incorrect number of operands: expected ")
           << expectedNumArgs << ", found " << getArgs().size();
  return success();
}

//===----------------------------------------------------------------------===//
// Reduce and Scan Ops helper functions
//===----------------------------------------------------------------------===//

// Common verification logic for operations with aggregation semantics
// (Reduce, Scan, etc.)
static LogicalResult verifyAggregateOpRegions(Operation *op, Region &region,
                                              size_t numOperands) {
  size_t expectedNumBlockOperands = numOperands * 2;
  Block &block = region.front();
  if (block.empty())
    return op->emitOpError("expect non-empty block");
  if (block.getNumArguments() != expectedNumBlockOperands)
    return op->emitOpError()
           << "expect " << expectedNumBlockOperands
           << " block arguments but got: " << block.getNumArguments();

  // All block operands must be cuda_tile.tile with 0 rank.
  auto blockArgs = block.getArgumentTypes();
  for (auto [idx, blockArg] : llvm::enumerate(blockArgs)) {
    if (TileType tileTy = dyn_cast<TileType>(blockArg)) {
      if (tileTy.getRank() == 0)
        continue;
    }
    return op->emitOpError() << "expect 0-rank tile type at index: " << idx
                             << " but got: " << blockArg;
  }

  // Block operand types must be equal "pair-wise":
  // [arg0_current_iter, %arg0_prev_iter, %arg1_current_iter,
  // %arg1_prev_iter...]
  // type(%arg0_current_iter) == type(%arg0_prev_iter)
  // type(%arg1_current_iter) == type(%arg1_prev_iter)
  // Note: The meaning of arg(i)_prev_iter is implementation defined, it can
  // either be: a) another element from the same operand b) the previous
  // reduction result c) the identity associated with the operand
  for (size_t idx = 0; idx < expectedNumBlockOperands - 1; idx += 2) {
    auto argTy = dyn_cast<TileType>(blockArgs[idx]);
    auto identityArgTy = dyn_cast<TileType>(blockArgs[idx + 1]);
    if (!argTy || !identityArgTy)
      return op->emitOpError()
             << "expected TileType for block arguments but got types: "
             << blockArgs[idx] << " and " << blockArgs[idx + 1];
    if (argTy.getElementType() != identityArgTy.getElementType())
      return op->emitOpError()
             << "expect same element type for block argument at index: " << idx
             << " and " << idx + 1 << " but got: " << argTy.getElementType()
             << " and " << identityArgTy.getElementType();
  }

  // Block operand types should match operand types.
  auto operandTypes = op->getOperandTypes();
  for (size_t idx = 0; idx < numOperands; idx++) {
    auto operandTy = dyn_cast<TileType>(operandTypes[idx]);
    auto argTy = dyn_cast<TileType>(blockArgs[idx * 2]);
    if (!operandTy || !argTy)
      return op->emitOpError()
             << "expected TileType for operand and block argument but got "
                "types: "
             << operandTypes[idx] << " and " << blockArgs[idx * 2];
    if (operandTy.getElementType() != argTy.getElementType())
      return op->emitOpError()
             << "expect same type for operand at index: " << idx
             << " and block argument at index: " << idx * 2
             << " but got: " << operandTy.getElementType() << " and "
             << argTy.getElementType();
  }

  auto term = cast<YieldOp>(block.getTerminator());
  auto termOperandTypes = term.getOperands().getTypes();
  size_t termOperands = term.getNumOperands();
  if (termOperands != numOperands)
    return op->emitOpError()
           << "expect number of terminators operands (" << termOperands
           << ") to match number of operands (" << numOperands << ")";

  // Terminator operand types must match operand types.
  for (size_t idx = 0; idx < numOperands; idx++) {
    auto operandTy = dyn_cast<TileType>(operandTypes[idx]);
    auto termTy = dyn_cast<TileType>(termOperandTypes[idx]);
    if (!operandTy || !termTy)
      return op->emitOpError()
             << "expected TileType for operand and terminator types but got: "
             << operandTypes[idx] << " and " << termOperandTypes[idx];
    if (operandTy.getElementType() != termTy.getElementType())
      return op->emitOpError()
             << "expect same type for operand at index: " << idx
             << " and terminator argument at index: " << idx
             << " but got: " << operandTy.getElementType() << " and "
             << termTy.getElementType();
  }

  auto isMemEffectFreeCudaTileOp = [](Operation *op) {
    // SIMT interop calls (preview$call) are permitted inside reduce/scan
    // regions. However, they must not have any memory effects apart from
    // reading from or writing to LOCAL fragments. The verifier will be
    // improved in the future to check for this.
    if (isa<CallOp>(op))
      return true;
    return isMemoryEffectFree(op);
  };

  for (Operation &innerOp : block.without_terminator()) {
    if (!isMemEffectFreeCudaTileOp(&innerOp)) {
      return innerOp.emitOpError(
                 "only memory-effect-free operations are allowed inside '")
             << op->getName() << "'";
    }
  }
  return success();
}

// Common verification logic for operations with aggregation semantics
// (Reduce, Scan, etc.)
static LogicalResult
verifyAggregateOp(Operation *op, ValueRange operands, TypeRange results,
                  int32_t dim, ArrayAttr identities,
                  bool requiresMatchingReturnShape = false) {
  size_t numOperands = operands.size();
  size_t numResults = results.size();

  if (numOperands == 0)
    return op->emitOpError() << "expect at least 1 operand";

  if (numOperands != numResults)
    return op->emitOpError() << "expect same number of operands and results";

  // Verify identities if provided:
  // a) #_identities == #_operands
  // b) type(identities[i]) == type(operands[i]) 0 <= i < operands.size
  if (identities) {
    size_t numIdentities = identities.size();
    if (numOperands != numIdentities)
      return op->emitOpError()
             << "expect identities to match the number of operands but got: "
             << numOperands << " operands and " << numIdentities
             << " identities";

    for (size_t idx = 0; idx < numOperands; idx++) {
      auto operandTy = dyn_cast<TileType>(operands[idx].getType());
      if (!operandTy)
        return op->emitOpError()
               << "expected TileType for operand at index " << idx
               << " but got: " << operands[idx].getType();
      auto identityTy = cast<TypedAttr>(identities[idx]).getType();
      if ((operandTy.getElementType().isBF16() && identityTy.isF16()) ||
          (operandTy.getElementType().isF16() && identityTy.isBF16()))
        continue;
      if (operandTy.getElementType() != identityTy) {
        return op->emitOpError()
               << "expect same type for operand at index: " << idx
               << " and identity at index: " << idx
               << " but got: " << operandTy.getElementType() << " and "
               << identityTy;
      }
    }
  }

  // All the operand have the same shape see: SameOperandsShape.
  auto firstOperandTy = dyn_cast<TileType>(operands[0].getType());
  if (!firstOperandTy)
    return op->emitOpError() << "expected TileType for first operand but got: "
                             << operands[0].getType();
  size_t rank = firstOperandTy.getRank();
  if (dim < 0 || dim >= static_cast<int32_t>(rank)) {
    return op->emitOpError()
           << "dimension (" << dim << ") is out of bound [0, " << rank << ")";
  }

  // If required, check that operand shapes match result shapes
  if (requiresMatchingReturnShape) {
    for (size_t idx = 0; idx < numOperands; idx++) {
      auto operandTy = dyn_cast<TileType>(operands[idx].getType());
      auto resultTy = dyn_cast<TileType>(results[idx]);
      if (!operandTy || !resultTy)
        return op->emitOpError()
               << "expected TileType for operand and result at index " << idx
               << " but got: " << operands[idx].getType() << " and "
               << results[idx];
      if (operandTy != resultTy)
        return op->emitOpError()
               << "expect same type for operand at index: " << idx
               << " and result at index: " << idx;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verifyRegions() {
  return verifyAggregateOpRegions(getOperation(), getRegion(),
                                  getNumOperands());
}

LogicalResult ReduceOp::verify() {
  return verifyAggregateOp(getOperation(), getOperands(), getResultTypes(),
                           getDim(), getIdentities());
}

LogicalResult
ReduceOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                           ReduceOp::Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  auto operands = adaptor.getOperands();
  if (operands.empty())
    return failure();

  int32_t dim = adaptor.getDim();
  for (Value operand : operands) {
    TileType operandTy = cast<TileType>(operand.getType());
    SmallVector<int64_t> targetShape;
    for (auto [dimIdx, dimSize] : llvm::enumerate(operandTy.getShape()))
      if (dim != static_cast<int32_t>(dimIdx))
        targetShape.push_back(dimSize);
    inferredReturnTypes.push_back(
        TileType::get(targetShape, operandTy.getElementType()));
  }
  return success();
}

void ReduceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getNumResults())
    setNameFn(getResult(0), "reduce");
}

void ReduceOp::getAsmBlockArgumentNames(Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
  for (auto [index, arg] : llvm::enumerate(region.getArguments())) {
    std::string name;
    if (index % 2 == 0)
      name = "reduce_lhs";
    else
      name = "reduce_rhs";

    if (region.getArguments().size() > 2)
      name += std::to_string(index / 2);

    setNameFn(arg, name);
  }
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto sourceTileType = cast<cuda_tile::TileType>(getSource().getType());
  auto resultTileType = cast<cuda_tile::TileType>(getResult().getType());
  // Note: Element type is verified by `SameOperandsAndResultElementType`.
  if (sourceTileType.getNumElements() != resultTileType.getNumElements())
    return emitOpError("expected source tile and result tile to have the "
                       "same number of elements");
  return success();
}

void ReshapeOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "reshape");
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  Operation *parentOp = (*this)->getParentOp();

  // Verify the invariants based on the parent operation.
  do {
    if (isa<IfOp>(parentOp)) {
      parentOp = parentOp->getParentOp();
      continue;
    }
    // Allow returns within loop operations (LoopOp).
    // The return should escape the loop and return from the enclosing function
    if (isa<LoopOp>(parentOp)) {
      parentOp = parentOp->getParentOp();
      continue;
    }
    if (auto funcOp = dyn_cast<FuncOp>(parentOp)) {
      // The operand number and types must match the function signature.
      const auto &results = funcOp.getFunctionType().getResults();
      if (getNumOperands() != results.size())
        return emitOpError("has ")
               << getNumOperands() << " operands, but enclosing function (@"
               << funcOp.getName() << ") returns " << results.size();

      for (size_t i = 0, e = results.size(); i != e; ++i)
        if (getOperand(i).getType() != results[i])
          return emitError() << "type of return operand " << i << " ("
                             << getOperand(i).getType()
                             << ") doesn't match function result type ("
                             << results[i] << ")"
                             << " in function @" << funcOp.getName();
      break;
    }
    if (auto entryOp = dyn_cast<EntryOp>(parentOp)) {
      // The operand number and types must match the function signature.
      const auto &results = entryOp.getFunctionType().getResults();
      if (getNumOperands() != results.size())
        return emitOpError("has ")
               << getNumOperands() << " operands, but enclosing function (@"
               << entryOp.getName() << ") returns " << results.size();
      // EntryOp must return zero results
      if (getNumOperands() != 0)
        return emitOpError("has ")
               << getNumOperands()
               << " operands, but entry function must return 0 operands";
      break;
    }

#ifdef TILE_IR_INCLUDE_TESTS
    if (auto funcOp = dyn_cast<Test_FuncOp>(parentOp)) {
      // The operand number and types must match the function signature.
      const auto &results = funcOp.getFunctionType().getResults();
      if (getNumOperands() != results.size())
        return emitOpError("has ")
               << getNumOperands() << " operands, but enclosing function (@"
               << funcOp.getName() << ") returns " << results.size();

      for (size_t i = 0, e = results.size(); i != e; ++i)
        if (getOperand(i).getType() != results[i])
          return emitError() << "type of return operand " << i << " ("
                             << getOperand(i).getType()
                             << ") doesn't match function result type ("
                             << results[i] << ")"
                             << " in function @" << funcOp.getName();
      break;
    }
#endif // TILE_IR_INCLUDE_TESTS

    return emitOpError(
        "must be used within a "
#ifdef TILE_IR_INCLUDE_TESTS
        "cuda_tile.testing$func, "
#endif
        "cuda_tile.entry, cuda_tile.if, or cuda_tile.loop operation");
  } while (true);

  return success();
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

LogicalResult RsqrtOp::verify() { return verifyFtz(*this, getFlushToZero()); }

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verifyRegions() {
  return verifyAggregateOpRegions(getOperation(), getRegion(),
                                  getNumOperands());
}

LogicalResult ScanOp::verify() {
  return verifyAggregateOp(getOperation(), getOperands(), getResultTypes(),
                           getDim(), getIdentities(),
                           /*requiresMatchingReturnShape=*/true);
}

LogicalResult
ScanOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                         ScanOp::Adaptor adaptor,
                         SmallVectorImpl<Type> &inferredReturnTypes) {

  auto operands = adaptor.getOperands();
  if (operands.empty())
    return failure();

  inferredReturnTypes.assign(operands.getTypes().begin(),
                             operands.getTypes().end());
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  TileType resTy = cast<TileType>(getResult().getType());
  Type elType = resTy.getElementType();
  if (isa<Float4E2M1FNType>(elType))
    return emitOpError("cannot operate on sub-byte type F4E2M1FN");
  return success();
}

struct SelectConsts : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value condition = op.getCond();
    Value trueVal = op.getValIfTrue();
    Value falseVal = op.getValIfFalse();
    TileType resTy = op.getType();

    // Constant-fold constant operands over non-splat constant condition.
    // select %cst_vec, %cst0, %cst1 => %cst2
    auto cond = condition.getDefiningOp<ConstantOp>();
    auto lhs = trueVal.getDefiningOp<ConstantOp>();
    auto rhs = falseVal.getDefiningOp<ConstantOp>();
    if (!cond || !lhs || !rhs)
      return failure();
    auto condVals = cond.getValue().getValues<bool>();
    auto lhsVals = lhs.getValue().getValues<Attribute>();
    auto rhsVals = rhs.getValue().getValues<Attribute>();
    llvm::SmallVector<Attribute, 8> out;
    out.reserve(lhs.getType().getNumElements());
    for (auto [c, l, r] : llvm::zip_equal(condVals, lhsVals, rhsVals)) {
      out.push_back(c ? l : r);
    }
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, resTy,
        cast<DenseTypedElementsAttr>(DenseElementsAttr::get(resTy, out)));
    return success();
  }
};

//  select %arg, %c1, %c0 => exti %arg unsigned
struct SelectToExtI : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Cannot exti i1 to i1, or i1 to f32
    TileType ty = op.getType();
    if (!llvm::isa<IntegerType>(ty.getElementType()) ||
        ty.getElementType().isInteger(1))
      return failure();

    // Apply the following folding pattern
    // select %x, c1, %c0 => extui %arg
    if (isConstantOnesValue(op.getValIfTrue()) &&
        isConstantZeroValue(op.getValIfFalse())) {
      rewriter.replaceOpWithNewOp<ExtIOp>(op, ty, op.getCond(),
                                          Signedness::Unsigned);
      return success();
    }

    // Apply the following folding pattern
    // select %x, c0, %c1 => extui (xor %arg, true)
    if (isConstantZeroValue(op.getValIfTrue()) &&
        isConstantOnesValue(op.getValIfFalse())) {
      rewriter.replaceOpWithNewOp<ExtIOp>(
          op, ty, createXOrForValue(rewriter, op.getLoc(), op.getCond()),
          Signedness::Unsigned);
      return success();
    }
    return failure();
  }
};

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SelectI1ToNot, SelectConsts, SelectToExtI>(context);
}

// 1) select c, x, x => x
static OpFoldResult tryFoldSelectSameOperands(SelectOp op,
                                              SelectOp::FoldAdaptor adaptor) {
  Value t = op.getValIfTrue();
  Value f = op.getValIfFalse();
  if (t == f)
    return t;
  return {};
}

// 2) select true, x, y => x
//    select false, x, y => y
static OpFoldResult tryFoldSelectConstCondition(SelectOp op,
                                                SelectOp::FoldAdaptor adaptor) {
  Value cond = op.getCond();
  if (isConstantTrueVal(cond))
    return op.getValIfTrue();
  if (isConstantFalseVal(cond))
    return op.getValIfFalse();
  return {};
}

// 3) Boolean identity: select c, true, false => c
//    (Safe because we return an existing value; the inverse case
//     `select c, false, true => !c` would require creating an op, so leave
//     that to canonicalization patterns.)
static OpFoldResult tryFoldSelectBoolIdentity(SelectOp op,
                                              SelectOp::FoldAdaptor adaptor) {
  // select %x, true, false => %x
  Value t = op.getValIfTrue();
  Value f = op.getValIfFalse();
  Value cond = op.getCond();
  auto tileTy = llvm::dyn_cast_or_null<TileType>(op.getType());
  if (tileTy && tileTy.getElementType().isSignlessInteger(1) &&
      isConstantTrueVal(t) && isConstantFalseVal(f))
    return cond;
  return {};
}

static OpFoldResult tryFoldSelectWithCmp(SelectOp op,
                                         SelectOp::FoldAdaptor adaptor) {
  Value t = op.getValIfTrue();
  Value f = op.getValIfFalse();
  Value cond = op.getCond();
  if (auto cmp = cond.getDefiningOp<CmpIOp>()) {
    auto pred = cmp.getComparisonPredicate();
    if (pred == ComparisonPredicate::EQUAL ||
        pred == ComparisonPredicate::NOT_EQUAL) {
      auto cmpLhs = cmp.getLhs();
      auto cmpRhs = cmp.getRhs();

      // Apply the following folding pattern
      // %0 = cmpi eq, %arg0, %arg1
      // %1 = select %0, %arg0, %arg1 => %arg1

      // or the following folding pattern
      // %0 = cmpi ne, %arg0, %arg1
      // %1 = select %0, %arg0, %arg1 => %arg0
      if ((cmpLhs == t && cmpRhs == f) || (cmpRhs == t && cmpLhs == f))
        return pred == ComparisonPredicate::NOT_EQUAL ? t : f;
    }
  }
  return {};
}

static OpFoldResult tryFoldSelectWithXor(SelectOp op,
                                         SelectOp::FoldAdaptor adaptor) {
  Value t = op.getValIfTrue();
  Value f = op.getValIfFalse();
  Value cond = op.getCond();

  // ---- Rule: select (xor pred, true), a, b  =>  select pred, b, a
  // Matches "Arith::SelectNotCond" pattern.
  if (auto xorOp = cond.getDefiningOp<XOrIOp>()) {
    Value lhs = xorOp.getLhs();
    Value rhs = xorOp.getRhs();

    // Recognize "not" encoded as xor with constant true.
    // Rhs only, XOrIOp is expected to be canonicalized itself
    if (isConstantTrueVal(rhs)) {
      // select(not(pred), a, b) -> select(pred, b, a)
      op.getCondMutable().assign(lhs);
      // swap true/false arms
      op.getValIfTrueMutable().assign(f);
      op.getValIfFalseMutable().assign(t);
      return op.getResult(); // in-place fold success
    }
  }
  return {};
}

static OpFoldResult tryFoldSelectWithSelect(SelectOp op,
                                            SelectOp::FoldAdaptor adaptor) {
  Value t = op.getValIfTrue();
  Value f = op.getValIfFalse();
  Value cond = op.getCond();
  // ---- Rule: select(pred, select(pred, a, b), c) => select(pred, a, c)
  // "RedundantSelectTrue"
  if (auto innerTrueSel = t.getDefiningOp<SelectOp>()) {
    if (innerTrueSel.getCond() == cond) {
      op.getValIfTrueMutable().assign(innerTrueSel.getValIfTrue());
      return op.getResult(); // in-place
    }
  }

  // ---- Rule: select(pred, a, select(pred, b, c)) => select(pred, a, c)
  // "RedundantSelectFalse"
  if (auto innerFalseSel = f.getDefiningOp<SelectOp>()) {
    if (innerFalseSel.getCond() == cond) {
      op.getValIfFalseMutable().assign(innerFalseSel.getValIfFalse());
      return op.getResult();
    }
  }

  return {};
}

namespace {
using SelectFoldRuleFn = OpFoldResult (*)(SelectOp, SelectOp::FoldAdaptor);

static constexpr SelectFoldRuleFn kSelectFoldRules[] = {
    tryFoldSelectSameOperands, tryFoldSelectConstCondition,
    tryFoldSelectBoolIdentity, tryFoldSelectWithCmp,
    tryFoldSelectWithXor,      tryFoldSelectWithSelect,
};
} // namespace

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  for (auto rule : kSelectFoldRules)
    if (OpFoldResult r = rule(*this, adaptor))
      return r;
  return {};
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

LogicalResult SqrtOp::verify() {
  auto rounding = getRoundingMode();
  if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                           RoundingMode::NEGATIVE_INF,
                           RoundingMode::POSITIVE_INF, RoundingMode::APPROX},
                          rounding)) {
    return emitOpError(
        "invalid rounding mode specified, expect "
        "one of [nearest_even, zero, negative_inf, positive_inf, approx]");
  }

  bool hasApprox = rounding == RoundingMode::APPROX;
  bool hasIEEERounding = !hasApprox;

  return verifySqrtFPModifiers(*this, hasIEEERounding, hasApprox,
                               /*full=*/false, getFlushToZero());
}

//===----------------------------------------------------------------------===//
// TanHOp
//===----------------------------------------------------------------------===//

LogicalResult TanHOp::verify() {
  auto rounding = getRoundingMode();
  if (!llvm::is_contained({RoundingMode::FULL, RoundingMode::APPROX},
                          rounding)) {
    return emitOpError(
        "invalid rounding mode specified, expect one of [approx, full]");
  }

  bool hasApprox = rounding == RoundingMode::APPROX;
  return verifyApprox(*this, hasApprox);
}

//===----------------------------------------------------------------------===//
// StoreOpBase
//===----------------------------------------------------------------------===//

LogicalResult
cuda_tile::impl::verifyMemoryModelStore(Operation *op,
                                        MemoryOrderingSemantics memoryOrdering,
                                        std::optional<MemoryScope> scope) {
  // First validate the memory ordering is supported
  switch (memoryOrdering) {
  case MemoryOrderingSemantics::WEAK:
  case MemoryOrderingSemantics::RELAXED:
  case MemoryOrderingSemantics::RELEASE:
    break; // Valid orderings
  default:
    return op->emitOpError(
               "expect one of: weak, relaxed, or release, but got: ")
           << stringifyMemoryOrderingSemantics(memoryOrdering);
  }

  // Then validate scope requirements based on ordering
  if (memoryOrdering == MemoryOrderingSemantics::WEAK) {
    if (scope.has_value())
      return op->emitOpError("weak store must not have memory scope");
  } else {
    // RELAXED or RELEASE require scope
    if (!scope.has_value())
      return op->emitOpError("memory scope is required for ")
             << stringifyMemoryOrderingSemantics(memoryOrdering) << " store";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StorePtrTkoOp
//===----------------------------------------------------------------------===//

LogicalResult StorePtrTkoOp::verify() {
  if (failed(verifyOptHintsCommon(this)))
    return failure();
  auto elemTy = getDestination().getType().getElementType();
  if (auto ptrTy = dyn_cast<PointerType>(elemTy)) {
    if (auto ptrAttr = ptrTy.getPtrAttr();
        ptrAttr && ptrAttr.getValue() == PtrAttr::MULTICAST) {
      return emitOpError("multicast pointer is not allowed");
    }
  }
  return impl::verifyMemoryModelStore(*this, getMemoryOrderingSemantics(),
                                      getMemoryScope());
}

//===----------------------------------------------------------------------===//
// StoreViewTkoOp
//===----------------------------------------------------------------------===//

ParseResult cuda_tile::StoreViewTkoOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::UnresolvedOperand tile, view;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  OpAsmParser::UnresolvedOperand token;
  NamedAttrList attributes;
  OptimizationHintsAttr optHints;
  DenseBoolArrayAttr inbounds;
  bool hasToken = false;
  SMLoc typeLoc;
  SMLoc boundAttributeLoc;
  SMLoc indicesLoc;

  // Parse memory attributes
  MemoryOrderingSemanticsAttr memSemantics;
  MemoryScopeAttr memScope;
  if (parseMemoryAttributes(parser, memSemantics, memScope))
    return failure();

  // Parse tile, view and indices
  if (parser.parseOperand(tile) || parser.parseComma() ||
      parser.parseOperand(view) || parser.parseLSquare()) {
    return failure();
  }
  indicesLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indices) || parser.parseRSquare()) {
    return failure();
  }

  // Parse common optional elements
  if (parseViewTkoCommon(parser, attributes, optHints, hasToken, token,
                         inbounds, boundAttributeLoc)) {
    return failure();
  }

  // Parse types: : tile_type, view_type, index_types... -> token_type
  if (parser.parseColon())
    return failure();

  Type tileType, viewType;
  SmallVector<Type> indexTypes;
  Type tokenType;

  if (parseCudaTileType(parser, tileType) || parser.parseComma() ||
      parseCudaTileType(parser, viewType))
    return failure();

  // Parse index types
  if (parseIndexTypes(parser, indices, indexTypes, typeLoc))
    return failure();

  // Parse arrow and result type
  if (parser.parseArrow() || parseCudaTileType(parser, tokenType))
    return failure();

  // Resolve tile and view operands
  if (parser.resolveOperand(tile, tileType, result.operands) ||
      parser.resolveOperand(view, viewType, result.operands))
    return failure();

  // Finalize parsing
  ViewTkoParseState parsed{
      /*indices=*/indices,
      /*indexTypes=*/indexTypes,
      /*hasToken=*/hasToken,
      /*token=*/token,
      /*memSemantics=*/memSemantics,
      /*memScope=*/memScope,
      /*optHints=*/optHints,
      /*inbounds=*/inbounds,
      /*attributes=*/std::move(attributes),
      /*segmentSizes=*/
      {1, 1, static_cast<int32_t>(indices.size()), hasToken ? 1 : 0},
      /*typeLoc=*/typeLoc,
      /*boundAttributeLoc=*/boundAttributeLoc,
      /*indicesLoc=*/indicesLoc,
  };
  if (finalizeViewTkoParse(parser, result, parsed)) {
    return failure();
  }

  // Set result type
  result.types.push_back(tokenType);

  return success();
}

void cuda_tile::StoreViewTkoOp::print(OpAsmPrinter &p) {
  printViewTkoCommon(p, *this);

  // Print tile, view and indices
  p << " ";
  p.printOperand(getTile());
  p << ", ";
  p.printOperand(getView());
  p << "[";
  llvm::interleaveComma(getIndex(), p, [&](Value idx) { p.printOperand(idx); });
  p << "]";

  printViewTkoOptionals(p, *this);

  // Print types
  p << " : ";
  printCudaTileType(p, getTile().getType());
  p << ", ";
  printCudaTileType(p, getView().getType());
  printIndexTypes(p, getIndex());

  // Print result type
  p << " -> ";
  printCudaTileType(p, getResultToken().getType());
}

LogicalResult StoreViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(this)))
    return failure();
  if (failed(verifyUnrestrictedViewPtrAttr(*this, getView()))) {
    return failure();
  }

  ArrayRef<bool> inbounds = getInbounds();
  if (inbounds.size() != getIndex().size()) {
    return emitOpError("inbounds size (")
           << inbounds.size() << ") must match the number of index dimensions ("
           << getIndex().size() << ")";
  }

  return impl::verifyMemoryModelStore(*this, getMemoryOrderingSemantics(),
                                      getMemoryScope());
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

LogicalResult SubFOp::verify() {
  if (failed(verifyIEEERoundingModes(*this)))
    return failure();
  return verifyFtz(*this, getFlushToZero());
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

LogicalResult TruncIOp::verify() {
  IntegerType from = cast<IntegerType>(getFrom().getType().getElementType());
  IntegerType to = cast<IntegerType>(getTo().getType().getElementType());

  if (to.getWidth() >= from.getWidth())
    return emitOpError("truncating to larger or identical integer");

  return success();
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

LogicalResult UnpackOp::verify() { return verifyPackUnpackTypes(this); }

//===----------------------------------------------------------------------===//
// Op Registration
//===----------------------------------------------------------------------===//

namespace {
struct CudaTileinlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation * /*call*/, Operation *callable,
                       bool /*wouldBeCloned*/) const final {
    return true;
  }

  bool isLegalToInline(Region * /*dest*/, Region * /*src*/,
                       bool /*wouldBeCloned*/,
                       IRMapping & /*valueMapping*/) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool /*wouldBeCloned*/,
                       IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = llvm::cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
    // This callback is invoked right before the blocks are inlined into the
    // position of the call operation. The main thing we're interested in
    // doing here is checking for the presence of early returns and handling
    // them appropriately. The rough transformation we do is to wrap the
    // inlined call into a loop, and transform the early returns into break
    // operations that exit the loop.
    Block &block = *inlinedBlocks.begin();

    // Walk the body of the inlined block looking for (and rewriting) early
    // returns.
    bool hadEarlyReturn = false;
    for (Operation &op : llvm::drop_end(block)) {
      op.walk([&](ReturnOp returnOp) {
        hadEarlyReturn = true;

        // Replace the return operation with a break operation.
        OpBuilder builder(returnOp);
        BreakOp::create(builder, returnOp.getLoc(), returnOp.getOperands());
        returnOp->erase();
      });
    }

    // If we didn't have an early return, nothing more to do here.
    if (!hadEarlyReturn)
      return;
    // Otherwise, we'll move the body of the inlined block into a new loop
    // operation, and replace the original return operation with a break
    // operation that will exit the loop.
    Operation *returnOp = block.getTerminator();
    OpBuilder builder(returnOp);

    // Build a break for the new loop wrapper.
    BreakOp::create(builder, returnOp->getLoc(), returnOp->getOperands());

    // Create a new loop operation that will contain the inlined block, and
    // update the original return to use the loops results.
    builder.setInsertionPointToStart(&block);
    auto loopOp = LoopOp::create(
        builder, block.front().getLoc(), returnOp->getOperandTypes(),
        /*operands=*/ValueRange(), /*attributes=*/ArrayRef<NamedAttribute>());
    returnOp->setOperands(loopOp.getResults());
    returnOp->moveAfter(loopOp);

    // Move the inlined block into the loop body.
    Block *loopBodyBlock = block.splitBlock(returnOp->getNextNode());
    loopBodyBlock->moveBefore(&loopOp.getBodyRegion(),
                              loopOp.getBodyRegion().begin());
  }
};

//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

struct CudaTileOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  // Provide custom aliasing for debug info attributes.
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    return TypeSwitch<Attribute, AliasResult>(attr)
        // Output mnemonic and return OverridableAlias.
        .Case<DICompileUnitAttr, DIFileAttr, DILexicalBlockAttr,
              DISubprogramAttr>([&](auto specificAttr) {
          os << std::decay_t<decltype(specificAttr)>::getMnemonic();
          return AliasResult::OverridableAlias;
        })
        .Default([](Attribute) { return AliasResult::NoAlias; });
  }
};

} // namespace

void CudaTileDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc"
      >();
  addInterfaces<CudaTileinlinerInterface, CudaTileOpAsmInterface>();
}
