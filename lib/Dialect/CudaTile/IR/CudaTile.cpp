//===- CudaTile.cpp - CUDA Tile Dialect Op Verifiers ------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/InliningUtils.h"

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

/// Parses a single function argument with cuda_tile type support.
static mlir::ParseResult parseSingleArgument(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments) {
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
  mlir::NamedAttrList attrs;
  if (parseCudaTileType(parser, arg.type) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseOptionalLocationSpecifier(arg.sourceLoc))
    return mlir::failure();

  arg.attrs = attrs.getDictionary(parser.getContext());
  arguments.push_back(arg);
  return mlir::success();
}

/// Parses function argument list with variadic support.
static mlir::ParseResult parseFunctionArgumentList(
    mlir::OpAsmParser &parser, bool allowVariadic,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments,
    bool &isVariadic) {
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

        return parseSingleArgument(parser, arguments);
      });
}

/// Parses type and attribute pairs for function results.
static mlir::ParseResult
parseTypeAndAttrList(mlir::OpAsmParser &parser,
                     llvm::SmallVectorImpl<mlir::Type> &types,
                     llvm::SmallVectorImpl<mlir::DictionaryAttr> &attrs) {
  return parser.parseCommaSeparatedList([&]() -> mlir::ParseResult {
    types.emplace_back();
    attrs.emplace_back();
    mlir::NamedAttrList attrList;
    if (parseCudaTileType(parser, types.back()) ||
        parser.parseOptionalAttrDict(attrList))
      return mlir::failure();
    attrs.back() = attrList.getDictionary(parser.getContext());
    return mlir::success();
  });
}

/// Parses function result list (single type or parenthesized type list).
static mlir::ParseResult parseFunctionResultList(
    mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Type> &resultTypes,
    llvm::SmallVectorImpl<mlir::DictionaryAttr> &resultAttrs) {
  if (mlir::failed(parser.parseOptionalLParen())) {
    // Single result type (no parentheses)
    mlir::Type resultType;
    if (parseCudaTileType(parser, resultType))
      return mlir::failure();
    resultTypes.push_back(resultType);
    resultAttrs.emplace_back();
    return mlir::success();
  }

  // Parenthesized result list
  if (mlir::succeeded(parser.parseOptionalRParen()))
    return mlir::success(); // Empty result list

  if (parseTypeAndAttrList(parser, resultTypes, resultAttrs))
    return mlir::failure();
  return parser.parseRParen();
}

} // namespace

/// Main function signature parser with cuda_tile dialect support.
mlir::ParseResult cuda_tile::parseFunctionSignatureWithArguments(
    mlir::OpAsmParser &parser, bool allowVariadic,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments,
    bool &isVariadic, llvm::SmallVectorImpl<mlir::Type> &resultTypes,
    llvm::SmallVectorImpl<mlir::DictionaryAttr> &resultAttrs) {
  if (parseFunctionArgumentList(parser, allowVariadic, arguments, isVariadic))
    return mlir::failure();
  if (mlir::succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs);
  return mlir::success();
}

/// Print function signature with cuda_tile dialect type support.
static void printFunctionSignatureWithCudaTileTypes(
    OpAsmPrinter &printer, TypeRange argTypes, ArrayAttr argAttrs,
    bool isVariadic, TypeRange resultTypes, Region *body) {
  printer << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      printer << ", ";
    auto arg = body->getArgument(i);
    ArrayRef<NamedAttribute> attrs;
    if (argAttrs)
      attrs = llvm::cast<DictionaryAttr>(argAttrs[i]).getValue();
    printer.printOperand(arg);
    printer << ": ";
    printCudaTileType(printer, arg.getType());
  }

  if (isVariadic) {
    if (!argTypes.empty())
      printer << ", ";
    printer << "...";
  }

  printer << ')';

  if (!resultTypes.empty()) {
    printer << " -> ";
    if (resultTypes.size() == 1) {
      printCudaTileType(printer, resultTypes[0]);
    } else {
      printer << '(';
      llvm::interleaveComma(resultTypes, printer, [&](Type resultType) {
        printCudaTileType(printer, resultType);
      });
      printer << ')';
    }
  }
}

/// Main function signature parser with cuda_tile dialect support, extracting
/// attributes and region from FunctionOpInterface
void cuda_tile::printFunctionSignatureWithCudaTileTypes(OpAsmPrinter &printer,
                                                        Operation *op,
                                                        TypeRange inputs,
                                                        TypeRange results) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  ::printFunctionSignatureWithCudaTileTypes(
      printer, inputs, funcOp.getArgAttrsAttr(),
      /*isVariadic=*/false, results, &funcOp.getFunctionBody());
}

//===----------------------------------------------------------------------===//
// Custom DenseIntOrFPElementsAttr Parsing
//===----------------------------------------------------------------------===//

// TODO: Leverage upstream changes to strip !cuda_tile. prefix.
static LogicalResult validateIntegerBounds(OpAsmParser &parser, int64_t intVal,
                                           Type elementType, SMLoc loc) {
  if (elementType.isInteger(1)) {
    // Union of signed [-1,1] and unsigned [0,1] = [-1,1]
    if (intVal < -1 || intVal > 1) {
      return parser.emitError(loc, "integer constant out of range for type");
    }
  } else if (elementType.isInteger(8)) {
    // Union of signed [-128,127] and unsigned [0,255] = [-128,255]
    if (intVal < -128 || intVal > 255) {
      return parser.emitError(loc, "integer constant out of range for type");
    }
  } else if (elementType.isInteger(16)) {
    // Union of signed [-32768,32767] and unsigned [0,65535] = [-32768,65535]
    if (intVal < -32768 || intVal > 65535) {
      return parser.emitError(loc, "integer constant out of range for type");
    }
  } else if (elementType.isInteger(32)) {
    // Union of signed [-2^31,2^31-1] and unsigned [0,2^32-1] = [-2^31,2^32-1]
    if (intVal < -2147483648LL || intVal > 4294967295LL) {
      return parser.emitError(loc, "integer constant out of range for type");
    }
  } else if (elementType.isInteger(64)) {
    // For i64, int64_t already covers the full signed range [-2^63,2^63-1]
    // The unsigned range [0,2^64-1] extends beyond int64_t, so we accept all
    // int64_t values negative values will be interpreted as large unsigned
    // values in two's complement
  }
  return success();
}

static bool isValidDenseElementType(Type elementType) {
  return elementType.isInteger(1) ||           // i1
         elementType.isInteger(8) ||           // i8
         elementType.isInteger(16) ||          // i16
         elementType.isInteger(32) ||          // i32
         elementType.isInteger(64) ||          // i64
         elementType.isF16() ||                // f16
         elementType.isBF16() ||               // bf16
         elementType.isF32() ||                // f32
         elementType.isF64() ||                // f64
         elementType.isTF32() ||               // tf32
         isa<Float8E4M3FNType>(elementType) || // f8E4M3FN
         isa<Float8E5M2Type>(elementType);     // f8E5M2
}

// Parse format: constant <f32: 0x7F800000> : tile<f32>
static ParseResult parseDenseIntOrFPElementsAttr(OpAsmParser &parser,
                                                 DenseIntOrFPElementsAttr &attr,
                                                 Type &resultType) {
  if (parser.parseLess())
    return failure();

  // We use the prefix element type to understand how to parse the dense values.
  Type prefixElementType;
  if (parseCudaTileType(parser, prefixElementType))
    return parser.emitError(parser.getCurrentLocation())
           << "expect element type to be one of i1 or i8 or i16 or i32 or i64 "
              "or f16 "
              "or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 values, but "
              "got "
           << prefixElementType;

  // Validate that prefixElementType is one of the allowed types
  if (!isValidDenseElementType(prefixElementType)) {
    return parser.emitError(parser.getCurrentLocation())
           << "expect element type to be one of i1 or i8 or i16 or i32 or i64 "
              "or f16 "
              "or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 values, but "
              "got "
           << prefixElementType;
  }

  bool isInteger = prefixElementType.isIntOrIndex();

  if (parser.parseColon())
    return failure();

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
      if (failed(parser.parseCommaSeparatedList(parseArrayElement))) {
        return failure();
      }

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
              "or bf16 or f32 or f64 or tf32 or f8E4M3FN or f8E5M2 values, but "
              "got "
           << resultType;
  }
  auto elementType = tileType.getElementType();
  if (prefixElementType != elementType)
    return parser.emitError(typeLoc)
           << "mismatch between the element type: " << prefixElementType
           << " and the tile element type " << elementType;

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
      attr = llvm::cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<bool>(boolValues)));
    } else if (elementType.isInteger(8)) {
      SmallVector<int8_t> i8Values;
      for (int64_t val : integerValues)
        i8Values.push_back(static_cast<int8_t>(val));
      attr = llvm::cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int8_t>(i8Values)));
    } else if (elementType.isInteger(16)) {
      SmallVector<int16_t> i16Values;
      for (int64_t val : integerValues)
        i16Values.push_back(static_cast<int16_t>(val));
      attr = llvm::cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int16_t>(i16Values)));
    } else if (elementType.isInteger(32)) {
      SmallVector<int32_t> i32Values;
      for (int64_t val : integerValues)
        i32Values.push_back(static_cast<int32_t>(val));
      attr = llvm::cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int32_t>(i32Values)));
    } else if (elementType.isInteger(64)) {
      attr = llvm::cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, ArrayRef<int64_t>(integerValues)));
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unsupported integer type");
    }
  } else { // Handle floating point numerical values.
    assert(isa_and_nonnull<FloatType>(elementType) && "expect a float type");
    attr = llvm::cast<DenseIntOrFPElementsAttr>(
        DenseElementsAttr::get(tileType, ArrayRef<APFloat>(floatValues)));
  }
  return success();
}

// constant <f32: 42.0> : tile<f32>
static void printDenseIntOrFPElementsAttr(OpAsmPrinter &p, Operation *op,
                                          DenseIntOrFPElementsAttr attr,
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
parseDenseIntOrFPElementsAttrNoResult(OpAsmParser &parser,
                                      DenseIntOrFPElementsAttr &attr) {
  Type resultType;
  return parseDenseIntOrFPElementsAttr(parser, attr, resultType);
}

static void
printDenseIntOrFPElementsAttrNoResult(OpAsmPrinter &p, Operation *op,
                                      DenseIntOrFPElementsAttr attr) {
  printDenseIntOrFPElementsAttr(p, op, attr, attr.getType());
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
    if (roundingMode != RoundingMode::NEAREST_INT_TO_ZERO) {
      return failure();
    }
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
  static const StringRef allowedModes[] = {"nearest_even", "zero",
                                           "negative_inf", "positive_inf"};

  auto ieeeValidator = [](OpAsmParser &parser, RoundingMode roundingMode,
                          StringRef roundingModeStr) -> ParseResult {
    // Only allow IEEE rounding modes
    if (roundingMode != RoundingMode::NEAREST_EVEN &&
        roundingMode != RoundingMode::ZERO &&
        roundingMode != RoundingMode::NEGATIVE_INF &&
        roundingMode != RoundingMode::POSITIVE_INF) {
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
                                isVariadic))
    return failure();
  return parser.parseRegion(region, arguments);
}

template <typename OpT>
void printArgumentRegion(OpAsmPrinter &p, OpT op, Region &region) {
  p.printNewline();
  printFunctionSignatureWithCudaTileTypes(p, region.getArgumentTypes(),
                                          /*argAttrs=*/{}, false,
                                          /*resultTypes=*/{}, &region);
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
  printer << memoryOrderingSemantics.getValue();
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
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.cpp.inc"

#define GET_OP_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/Enums.cpp.inc"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc"

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
    return op.emitOpError("invalid rounding error specified, expect "
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

//===----------------------------------------------------------------------===//
// AtomicRMWTkoOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicRMWTkoOp::verify() {
  auto ptrType =
      cast<cuda_tile::PointerType>(getPointers().getType().getElementType());
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

  // Check compatibility of RMW mode.
  switch (getMode()) {
  case AtomicRMWMode::AND:
  case AtomicRMWMode::OR:
  case AtomicRMWMode::XOR:
  case AtomicRMWMode::ADD:
  case AtomicRMWMode::MAX:
  case AtomicRMWMode::MIN:
  case AtomicRMWMode::UMAX:
  case AtomicRMWMode::UMIN: {
    auto integerTy = dyn_cast_or_null<IntegerType>(argElType);
    if (!integerTy || (!integerTy.isInteger(32) && !integerTy.isInteger(64)))
      return emitOpError("'") << stringifyAtomicRMWMode(getMode())
                              << "' works only with integers i32 and i64";
    break;
  }
  case AtomicRMWMode::ADDF: {
    auto floatTy = dyn_cast_or_null<FloatType>(argElType);
    if (!floatTy || (!floatTy.isF32() && !floatTy.isF64() && !floatTy.isF16()))
      return emitOpError("'") << stringifyAtomicRMWMode(getMode())
                              << "' works only with floats f16, f32, and f64";
    break;
  }
  case AtomicRMWMode::XCHG: {
    auto integerTy = dyn_cast_or_null<IntegerType>(argElType);
    auto floatTy = dyn_cast_or_null<FloatType>(argElType);
    if (!integerTy && !floatTy)
      return emitOpError("'")
             << stringifyAtomicRMWMode(getMode())
             << "' works only with integers or float of 32 or 64 bitwidth";
    int64_t bitwidth = argElType.getIntOrFloatBitWidth();
    if (bitwidth != 32 && bitwidth != 64)
      return emitOpError("'")
             << stringifyAtomicRMWMode(getMode())
             << "' works only with integers or float of 32 or 64 bitwidth";
  }
  }

  auto sem = getMemoryOrderingSemantics();
  // Check if memory ordering semantics is one of the allowed values
  if (sem != MemoryOrderingSemantics::RELAXED &&
      sem != MemoryOrderingSemantics::ACQUIRE &&
      sem != MemoryOrderingSemantics::RELEASE &&
      sem != MemoryOrderingSemantics::ACQ_REL) {
    return emitOpError("memory ordering semantics must be one of: "
                       "relaxed, acquire, release, acq_rel");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AtomicCASTkoOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicCASTkoOp::verify() {
  auto ptrType =
      cast<cuda_tile::PointerType>(getPointers().getType().getElementType());
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

  auto sem = getMemoryOrderingSemantics();
  // Check if memory ordering semantics is one of the allowed values
  if (sem != MemoryOrderingSemantics::RELAXED &&
      sem != MemoryOrderingSemantics::ACQUIRE &&
      sem != MemoryOrderingSemantics::RELEASE &&
      sem != MemoryOrderingSemantics::ACQ_REL) {
    return emitOpError("memory ordering semantics must be one of: "
                       "relaxed, acquire, release, acq_rel");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastOp::verify() {
  auto srcType = getSource().getType().getElementType();
  auto resType = getResult().getType().getElementType();

  // All numeric conversions are allowed if bitwidths match
  if (srcType.getIntOrFloatBitWidth() == resType.getIntOrFloatBitWidth()) {
    return success();
  }

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
      if (lhsShape[idx] != resultShape[idx] ||
          rhsShape[idx] != resultShape[idx]) {
        return emitOpError("expect {lhs, rhs, and result} shape to match at "
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
  DenseIntOrFPElementsAttr cstAttr = getValue();
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
    if (isa<cuda_tile::LoopOp, cuda_tile::ForOp>(parentLoop)) {
      break;
    }
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
    return emitOpError("invalid rounding error specified, expect "
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
    return emitOpError("invalid rounding error specified, expect "
                       "one of [zero, negative_inf, positive_inf]");
  }
  if (rounding == RoundingMode::NEGATIVE_INF &&
      getSignedness() == Signedness::Unsigned) {
    return emitOpError("rounding mode 'negative_inf' is not allowed with "
                       "'unsigned' flag");
  }
  return success();
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
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("source and result element type do not match");
  for (int i = 0, e = static_cast<int>(sourceType.getRank()); i < e; ++i) {
    if (sourceType.getDimSize(i) % resultType.getDimSize(i) != 0)
      return emitOpError("result dim size must divide source dim size evenly");
  }
  if (static_cast<int64_t>(getIndices().size()) != sourceType.getRank())
    return emitOpError("incorrect number of indices, expected ")
           << sourceType.getRank() << ", but found " << getIndices().size();
  return success();
}

//===----------------------------------------------------------------------===//
// IToFOp
//===----------------------------------------------------------------------===//

LogicalResult IToFOp::verify() {
  auto rounding = getRoundingMode();
  if (rounding != RoundingMode::NEAREST_EVEN)
    return emitOpError("invalid rounding error specified. Only "
                       "'nearest_even' is supported");
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
// MmaIOp
//===----------------------------------------------------------------------===//

LogicalResult MmaIOp::verify() {
  // Only need to verify shapes, as tablegen enforces element types
  return verifyMmaShapes(*this);
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
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands({lb, ub, step});
  result.addOperands(initArgs);
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

  return verifyLoopIterValues(*this, getResults(), getRegionIterValues());
}

void ForOp::print(OpAsmPrinter &p) {
  Value inductionVar = getInductionVar();
  p << " " << inductionVar << " in (" << getLowerBound() << " to "
    << getUpperBound() << ", step " << getStep() << ") : ";
  printCudaTileType(p, inductionVar.getType());
  p << " ";
  if (OperandRange initVals = getInitValues(); !initVals.empty())
    printLoopIteratorValues(p, initVals, getRegionIterValues());

  printControlFlowRegion<ContinueOp>(p, *this, getRegion());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
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
    return emitOpError("invalid rounding error specified. Only "
                       "'nearest_int_to_zero' is supported");
  return success();
}

//===----------------------------------------------------------------------===//
// FToFOp
//===----------------------------------------------------------------------===//

LogicalResult FToFOp::verify() {
  if (getFrom().getType() == getTo().getType())
    return emitOpError("converting tiles must not be a no-op");
  auto rounding = getRoundingMode();
  if (rounding != RoundingMode::NEAREST_EVEN)
    return emitOpError("invalid rounding mode specified for ftof. Only "
                       "'nearest_even' is supported");
  return success();
}

//===----------------------------------------------------------------------===//
// EntryOp
//===----------------------------------------------------------------------===//

constexpr char kOptimizationHintsAttr[] = "optimization_hints";

ParseResult EntryOp::parse(OpAsmParser &parser, OperationState &result) {
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
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  SmallVector<Attribute> argAttrs = llvm::to_vector(
      llvm::map_range(entryArgs, [](OpAsmParser::Argument arg) -> Attribute {
        return arg.attrs;
      }));
  result.addAttribute(getArgAttrsAttrName(result.name),
                      ArrayAttr::get(parser.getContext(), argAttrs));

  // Parse OptimizationHints attribute
  if (succeeded(parser.parseOptionalKeyword(kOptimizationHintsAttr))) {
    if (parser.parseEqual())
      return failure();
    Attribute opt_hint = OptimizationHintsAttr::parse(parser, Type{});
    if (opt_hint)
      result.addAttribute(getOptimizationHintsAttrName(result.name), opt_hint);
    else
      return failure();
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
  auto funcName =
      getOperation()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
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
      {getArgAttrsAttrName(), getFunctionTypeAttrName(),
       SymbolTable::getSymbolAttrName(), getResAttrsAttrName(),
       kOptimizationHintsAttr});
}

LogicalResult EntryOp::verify() {
  if (getNumResults() != 0)
    return emitOpError("entry op must not return values");

  for (Type operandTy : getArgumentTypes()) {
    auto tileTy = dyn_cast<cuda_tile::TileType>(operandTy);
    if (tileTy && tileTy.getRank() != 0)
      return emitOpError("entry op must have "
                         "scalar types (rank 0 !cuda_tile.tile)");
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
    if (hasResults) {
      return emitOpError("has non-empty return type, must define else branch");
    } else { // empty else block with no expected yield, nothing to check
      return success();
    }
  }
  return checkRegionYieldTypes(elseRegion, "else");
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

/// Get constant bool value or std::nullopt if not const
/// tile<i1> is expected for defining ConstantOp
static std::optional<bool> getConstantBoolValue(mlir::Value value) {
  auto cond = value.getDefiningOp<cuda_tile::ConstantOp>();
  if (!cond)
    return std::nullopt;
  auto type = cond.getType().getElementType();
  auto intType = llvm::dyn_cast<IntegerType>(type);
  if (!intType || intType.getWidth() != 1)
    return std::nullopt;
  DenseIntOrFPElementsAttr cstAttr = cond.getValue();
  if (cstAttr.size() != 1)
    return std::nullopt;
  return *cstAttr.getValues<bool>().begin();
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

  auto one = getConstantBoolValue(xorStmt.getRhs());
  if (!one || !one.value())
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
          return tileType && tileType.getShape().empty();
        }))
      return failure();

    SmallVector<Type> nonHoistable;
    for (auto [trueVal, falseVal] : llvm::zip(thenYieldArgs, elseYieldArgs)) {
      if (thenRegion == trueVal.getParentRegion() ||
          elseRegion == falseVal.getParentRegion())
        nonHoistable.push_back(trueVal.getType());
    }
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
        results[it.index()] =
            SelectOp::create(rewriter, op.getLoc(), cond, trueVal, falseVal);
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
          TileType constType = TileType::get({}, rewriter.getI1Type());
          llvm::APInt val(1, 1);
          auto constAttr = DenseIntElementsAttr::get(constType, val);
          auto constOp =
              ConstantOp::create(rewriter, op->getLoc(), constType, constAttr);
          Value notCond =
              XOrIOp::create(rewriter, op.getLoc(), op.getCondition(), constOp);
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
      auto one = getConstantBoolValue(notv.getRhs());
      if (one && one.value() && notv.getLhs() == prevIf.getCondition()) {
        nextElse = nextIf.getThenBlock();
        if (!nextIf.getElseRegion().empty())
          nextThen = nextIf.getElseBlock();
      }
    }
    if (XOrIOp notv = prevIf.getCondition().getDefiningOp<XOrIOp>()) {
      auto one = getConstantBoolValue(notv.getRhs());
      if (one && one.value() && notv.getLhs() == nextIf.getCondition()) {
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
    for (const auto &pair : llvm::enumerate(combinedIf.getResults())) {
      if (pair.index() < prevIf.getNumResults())
        prevValues.push_back(pair.value());
      else
        nextValues.push_back(pair.value());
    }
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
      if (!tileType || !tileType.getShape().empty())
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
      results[idx] = SelectOp::create(rewriter, op.getLoc(), op.getCondition(),
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
// LoadViewTkoOp
//===----------------------------------------------------------------------===//

LogicalResult LoadViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(this)))
    return failure();

  return impl::verifyMemoryModelLoad(*this, getMemoryOrderingSemantics(),
                                     getMemoryScope());
}

//===----------------------------------------------------------------------===//
// LoadPtrTkoOp
//===----------------------------------------------------------------------===//

LogicalResult LoadPtrTkoOp::verify() {
  if (failed(verifyOptHintsCommon(this)))
    return failure();
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
  if (hasIters || hasReturn) {
    p << ": ";
  }
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
    for (auto [regionArg, type] : llvm::zip_equal(regionArgs, iterTypes)) {
      regionArg.type = type;
    }
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

  Type indexType;
  if (!unresolvedDynShapeOperands.empty() ||
      !unresolvedDynStrideOperands.empty()) {
    if (parseCudaTileType(parser, indexType) || parser.parseArrow())
      return ParseResult::failure();
  }

  Type maybeTensorViewType;
  if (parseCudaTileType(parser, maybeTensorViewType))
    return ParseResult::failure();

  cuda_tile::TensorViewType tensorView =
      llvm::dyn_cast<TensorViewType>(maybeTensorViewType);
  if (!tensorView)
    return parser.emitError(parser.getCurrentLocation())
           << "expected 'tensor_view' type, but got " << maybeTensorViewType;

  if (parser.resolveOperand(
          basePtrOperand,
          cuda_tile::TileType::get(
              {}, cuda_tile::PointerType::get(tensorView.getElementType())),
          result.operands))
    return ParseResult::failure();

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

  if (!getDynamicShape().empty() || !getDynamicStrides().empty()) {
    Type dynamicType = !getDynamicShape().empty()
                           ? getDynamicShape().getTypes().front()
                           : getDynamicStrides().getTypes().front();
    printCudaTileType(p, dynamicType);
    p << " -> ";
  }

  printCudaTileType(p, getResult().getType());
}

LogicalResult cuda_tile::MakeTensorViewOp::verify() {
  Type baseElementType =
      llvm::cast<cuda_tile::PointerType>(getBase().getType().getElementType())
          .getPointeeType();
  if (getResult().getType().getElementType() != baseElementType)
    return emitOpError("expected pointer to ")
           << getResult().getType().getElementType()
           << " to build tensor_view of this type, got " << baseElementType;

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
    for (auto [i, dynamicValue] : llvm::enumerate(getDynamicShape())) {
      if (dynamicValue.getType() != dynamicValuesType)
        return emitOpError("expected dynamic shape index ")
               << i << " to be of the same type as the other dynamic values ("
               << dynamicValuesType << "), got " << dynamicValue.getType();
    }
  }

  if (!getDynamicStrides().empty()) {
    dynamicValuesType = dynamicValuesType
                            ? dynamicValuesType
                            : getDynamicStrides().getTypes().front();
    for (auto [i, dynamicValue] : llvm::enumerate(getDynamicStrides())) {
      if (dynamicValue.getType() != dynamicValuesType)
        return emitOpError("expected dynamic stride index ")
               << i << " to be of the same type as the other dynamic values ("
               << dynamicValuesType << "), got " << dynamicValue.getType();
    }
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
                                StringRef name) {
  result.addRegion()->emplaceBlock();
  Properties &props = result.getOrAddProperties<Properties>();
  props.setSymName(builder.getStringAttr(name));
}

LogicalResult cuda_tile::ModuleOp::verify() {
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
  if (permutationSize != uniqued.size()) {
    return emitOpError() << "expect permutation elements to be unique";
  }
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

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

/// Extract a format expression from the given string, assuming that the string
/// begins directly with the expression.
static StringRef extractFormatExpression(StringRef str) {
  assert(str.front() == '%' && "expect format string to start with '%'");
  for (size_t i = 1, e = str.size(); i < e; ++i) {
    // Format string should end with one of these characters.
    // See https://cplusplus.com/reference/cstdio/printf/.
    if (std::strchr("diuoxXeEfFgGaAcspn%", str[i]))
      return str.substr(0, i + 1);
  }
  // Found a format string expression that does not end with a valid
  // character.
  return "";
}

LogicalResult PrintOp::verify() {
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
#ifndef MIX_CUDA_TILE_TILE_AA
  // We allow only pure cuda_tile operations.
  Operation *invalidOp = nullptr;
  WalkResult status = block.walk([&](Operation *nestedOp) {
    if (!isPure(nestedOp)) {
      invalidOp = nestedOp;
      op->emitOpError("only pure operations allowed");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) {
    // Do not use invalidOp->emitRemark, as that may trigger recursive
    // verification of the reduce op again.
    mlir::emitRemark(invalidOp->getLoc(), "invalid op: ")
        << invalidOp->getName();
    return failure();
  }
#endif // MIX_CUDA_TILE_TILE_AA
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
    for (auto [dimIdx, dimSize] : llvm::enumerate(operandTy.getShape())) {
      if (dim != static_cast<int32_t>(dimIdx))
        targetShape.push_back(dimSize);
    }
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

    return emitOpError("must be used within a "
#ifdef TILE_IR_INCLUDE_TESTS
                       "cuda_tile.testing$func, "
#endif
                       "cuda_tile.entry, or cuda_tile.if operation");
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
// SqrtOp
//===----------------------------------------------------------------------===//

LogicalResult SqrtOp::verify() {
  auto rounding = getRoundingMode();
  if (!llvm::is_contained({RoundingMode::NEAREST_EVEN, RoundingMode::ZERO,
                           RoundingMode::NEGATIVE_INF,
                           RoundingMode::POSITIVE_INF, RoundingMode::APPROX},
                          rounding)) {
    return emitOpError(
        "invalid rounding error specified, expect "
        "one of [nearest_even, zero, negative_inf, positive_inf, approx]");
  }

  bool hasApprox = rounding == RoundingMode::APPROX;
  bool hasIEEERounding = !hasApprox;

  return verifySqrtFPModifiers(*this, hasIEEERounding, hasApprox,
                               /*full=*/false, getFlushToZero());
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
  return impl::verifyMemoryModelStore(*this, getMemoryOrderingSemantics(),
                                      getMemoryScope());
}

//===----------------------------------------------------------------------===//
// StoreViewTkoOp
//===----------------------------------------------------------------------===//

LogicalResult StoreViewTkoOp::verify() {
  if (failed(verifyViewLoadStoreCommon(this)))
    return failure();
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
    // position of the call operation. The main thing we're interested in doing
    // here is checking for the presence of early returns and handling them
    // appropriately. The rough transformation we do is to wrap the inlined call
    // into a loop, and transform the early returns into break operations that
    // exit the loop.
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
