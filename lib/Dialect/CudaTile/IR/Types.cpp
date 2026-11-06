//===- Types.cpp - CUDA Tile Type Verifiers and Parsers ---------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/CompactLayoutParser.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/LayoutAlgebra.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h"
#include <limits>

using namespace mlir;
using namespace mlir::cuda_tile;

#define GET_TYPEDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cuda_tile {
// Generate C++ functions for certain type constraints.
#include "cuda_tile/Dialect/CudaTile/IR/TypeConstraints.h.inc"
} // namespace cuda_tile
} // namespace mlir

bool cuda_tile::isPointerLike(Type t) {
  if (isa<PointerType>(t))
    return true;
  if (auto tileType = dyn_cast<cuda_tile::TileType>(t))
    return isPointerLike(tileType.getElementType());
  return false;
}

bool cuda_tile::isIntTile(Type type) {
  auto tileType = dyn_cast<cuda_tile::TileType>(type);
  if (!tileType) {
    return false;
  }
  return isa<IntegerType>(tileType.getElementType()) &&
         !tileType.getElementType().isInteger(1);
}

bool cuda_tile::isBoolTile(Type type) {
  auto tileType = dyn_cast<cuda_tile::TileType>(type);
  if (!tileType) {
    return false;
  }
  return tileType.getElementType().isInteger(1);
}

unsigned cuda_tile::getBitWidthFromType(Type type) {
  auto tileType = dyn_cast<cuda_tile::TileType>(type);
  if (!tileType) {
    return 0;
  }
  auto intTy = dyn_cast<IntegerType>(tileType.getElementType());
  return intTy ? intTy.getWidth() : 0;
}

bool CudaTileDialectType::classof(Type type) {
  return ::isa<CudaTileDialect>(type.getDialect());
}

/// Prints shape and element type in "8x16xf32" syntax.
static void printShapeAndElem(AsmPrinter &printer, ArrayRef<int64_t> shape,
                              Type elemType) {
  printer.printDimensionList(shape);
  if (!shape.empty())
    printer << "x";
  cuda_tile::printCudaTileType(printer, elemType);
  // printer << elemType;
}

static FailureOr<PaddingValueAttr>
parseOptionalPaddingValue(AsmParser &parser) {
  // Try to parse "padding_value = value"
  if (failed(parser.parseOptionalKeyword("padding_value")))
    return PaddingValueAttr();

  SMLoc loc = parser.getCurrentLocation();
  StringRef paddingValueStr;

  if (parser.parseEqual() || parser.parseKeyword(&paddingValueStr))
    return failure();

  auto attr = symbolizePaddingValue(paddingValueStr);
  if (!attr) {
    return parser.emitError(loc)
           << "invalid padding_value attribute specification. Got \""
           << paddingValueStr
           << "\" but expect one of: zero, neg_zero, nan, pos_inf, neg_inf";
  }

  return PaddingValueAttr::get(parser.getBuilder().getContext(), *attr);
}

static bool isTileShapeExpr(SymbolicExprAttr attr) {
  if (isa<SymbolicExprIntAttr, SymbolicExprVarAttr>(attr))
    return true;
  if (auto builtin = dyn_cast<SymbolicExprBuiltinAttr>(attr)) {
    Type type = builtin.getType().getValue();
    return isa<IndexType>(type);
  }
  if (auto mulAttr = dyn_cast<SymbolicExprMulAttr>(attr))
    return isTileShapeExpr(mulAttr.getLhs()) &&
           isTileShapeExpr(mulAttr.getRhs());
  if (auto addAttr = dyn_cast<SymbolicExprAddAttr>(attr))
    return isTileShapeExpr(addAttr.getLhs()) &&
           isTileShapeExpr(addAttr.getRhs());
  if (auto negAttr = dyn_cast<SymbolicExprNegAttr>(attr))
    return isTileShapeExpr(negAttr.getOperand());
  if (auto subAttr = dyn_cast<SymbolicExprSubAttr>(attr))
    return isTileShapeExpr(subAttr.getLhs()) &&
           isTileShapeExpr(subAttr.getRhs());
  return false;
}

static LogicalResult verifyConcreteTile(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape,
    Type elementType
) {
  if (auto pointerType = dyn_cast<cuda_tile::PointerType>(elementType);
      pointerType && isa<Float4E2M1FNType>(pointerType.getPointeeType()) &&
      !shape.empty()) {
    return emitError()
           << "tiles of ptr<f4E2M1FN> must have rank 0; use an f4 tile "
              "view for memory accesses";
  }
  if (isa<Float4E2M1FNType>(elementType)) {
    // f4 must have an even number of elements to pack 2 per byte.
    if (shape.empty() ||
        llvm::all_of(shape, [](int64_t dim) { return dim % 2 != 0; }))
      return emitError()
             << "F4E2M1FN tiles must have an even number of elements";
  }
  return verifyTileSize(
      emitError, shape
  );
}

//===----------------------------------------------------------------------===//
// Type Printing Utilities
//===----------------------------------------------------------------------===//

/// Parse a type, if type is unprefixed, assume it is from the cuda_tile dialect
ParseResult cuda_tile::parseCudaTileType(AsmParser &p, Type &type) {
  // The MLIR builtin dialect now provides an unprefixed `token` type whose
  // spelling collides with the cuda_tile `token` mnemonic. Without this guard,
  // the `parseOptionalType` call below would greedily parse the bare `token`
  // keyword as the builtin type, so route it to the cuda_tile token type first.
  if (succeeded(p.parseOptionalKeyword(cuda_tile::TokenType::getMnemonic()))) {
    type = cuda_tile::TokenType::get(p.getContext());
    return success();
  }

  auto result = p.parseOptionalType(type);
  if (result.has_value())
    return *result;

  StringRef mnemonic;
  result = generatedTypeParser(p, &mnemonic, type);
  if (result.has_value())
    return *result;

  return p.emitError(p.getCurrentLocation(), "unknown type: ") << mnemonic;
}

ParseResult cuda_tile::parseCudaTileType(AsmParser &p,
                                         SmallVectorImpl<Type> &types) {
  return p.parseCommaSeparatedList(
      AsmParser::Delimiter::None,
      [&]() -> ParseResult {
        Type type;
        if (failed(parseCudaTileType(p, type)))
          return failure();
        types.push_back(type);
        return success();
      },
      "Expected comma separated list of types");
}

ParseResult cuda_tile::parseCudaTileTypeSplat(
    AsmParser &p, SmallVectorImpl<Type> &types,
    ArrayRef<OpAsmParser::UnresolvedOperand> values) {

  if (failed(parseCudaTileType(p, types.emplace_back())))
    return failure();

  types.resize(values.size(), types.back());
  return success();
}

/// Print a type, stripping prefix if belonging to cuda_tile dialect
void cuda_tile::printCudaTileType(AsmPrinter &p, Type type) {
  if (isa<CudaTileDialect>(type.getDialect()) &&
      succeeded(generatedTypePrinter(type, p)))
    return;

  p.printType(type);
}

void cuda_tile::printCudaTileType(AsmPrinter &p, Operation *op, Type type) {
  printCudaTileType(p, type);
}

void cuda_tile::printCudaTileType(AsmPrinter &p, TypeRange types) {
  llvm::interleaveComma(types, p,
                        [&](Type type) { printCudaTileType(p, type); });
}

void cuda_tile::printCudaTileType(AsmPrinter &p, Operation *op,
                                  TypeRange types) {
  printCudaTileType(p, types);
}

void cuda_tile::printCudaTileTypeSplat(AsmPrinter &p, Operation *op,
                                       TypeRange types, ValueRange) {
  assert(llvm::all_equal(types) && "expected all types to be equal");
  assert(!types.empty() && "expected at least one type");

  printCudaTileType(p, types.front());
}

//===----------------------------------------------------------------------===//
// SymbolicType
//===----------------------------------------------------------------------===//

Type SymbolicType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  SymbolicExprAttr expr;

  if (parser.parseLess())
    return Type();

  // Keep the common dtype-parameter syntax compact (`<T>`), while still
  // allowing quoted names that cannot be printed as bare identifiers.
  std::string name;
  if (succeeded(parser.parseOptionalKeywordOrString(&name))) {
    expr = SymbolicExprVarAttr::get(parser.getContext(), name);
  } else {
    Attribute attr;
    if (parser.parseAttribute(attr))
      return Type();
    expr = llvm::dyn_cast<SymbolicExprAttr>(attr);
    if (!expr) {
      parser.emitError(loc, "symbolic type expression must be a "
                            "symbolic expression attribute");
      return Type();
    }
  }

  if (parser.parseGreater())
    return Type();

  return parser.getChecked<cuda_tile::SymbolicType>(loc, parser.getContext(),
                                                    expr);
}

void SymbolicType::print(AsmPrinter &printer) const {
  printer << "<";
  auto varAttr = llvm::cast<SymbolicExprVarAttr>(getExpr());
  printer.printKeywordOrString(varAttr.getName());
  printer << ">";
}

LogicalResult
cuda_tile::SymbolicType::verify(function_ref<InFlightDiagnostic()> emitError,
                                SymbolicExprAttr expr) {
  if (!expr)
    return emitError() << "symbolic type expression must be non-null";
  auto varAttr = llvm::dyn_cast<SymbolicExprVarAttr>(expr);
  if (!varAttr) {
    return emitError()
           << "symbolic type expression must be a SymbolicExprVarAttr, got "
           << expr;
  }
  if (varAttr.getName().empty())
    return emitError() << "symbolic type variable name must be non-empty";
  return success();
}

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

static ParseResult
parseTileDimensionList(AsmParser &parser,
                       SmallVectorImpl<int64_t> &concreteDims,
                       SmallVectorImpl<SymbolicExprAttr> &symbolicDims) {
  MLIRContext *context = parser.getContext();

  while (true) {
    if (succeeded(parser.parseOptionalLSquare())) {
      SMLoc exprLoc = parser.getCurrentLocation();
      bool emittedDiagnostic = false;
      CudaTileCompactLayoutParser exprParser(parser, context, exprLoc,
                                             emittedDiagnostic);
      FailureOr<SymbolicExprAttr> expr = exprParser.parseExpr();
      if (failed(expr) || parser.parseRSquare()) {
        return failure();
      }
      concreteDims.push_back(ShapedType::kDynamic);
      symbolicDims.push_back(*expr);
    } else {
      int64_t value = 0;
      OptionalParseResult intResult = parser.parseOptionalDecimalInteger(value);
      if (!intResult.has_value()) {
        break;
      }
      if (failed(*intResult)) {
        return failure();
      }
      concreteDims.push_back(value);
      symbolicDims.push_back(SymbolicExprIntAttr::get(context, value));
    }

    if (parser.parseXInDimensionList()) {
      return failure();
    }
  }

  return success();
}

Type cuda_tile::TileType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> concreteDims;
  SmallVector<SymbolicExprAttr> symbolicDims;
  Type elementType;
  if (parser.parseLess() ||
      parseTileDimensionList(parser, concreteDims, symbolicDims) ||
      parseCudaTileType(parser, elementType) || parser.parseGreater()) {
    return Type();
  }

  return cuda_tile::TileType::getChecked(
      [&]() { return parser.emitError(loc); },
      SymbolicTupleAttr::get(parser.getContext(), symbolicDims), elementType);
}

void cuda_tile::TileType::print(AsmPrinter &printer) const {
  printer << "<";
  auto symbolicShape = cast<SymbolicTupleAttr>(getSymbolicShapeAttr());
  llvm::interleave(
      symbolicShape.getValues(), printer,
      [&](SymbolicExprAttr dim) {
        if (auto intAttr = dyn_cast<SymbolicExprIntAttr>(dim)) {
          printer << intAttr.getValue();
          return;
        }
        printer << "[";
        CudaTileCompactLayoutParser::printCompactExpr(printer.getStream(), dim);
        printer << "]";
      },
      "x");
  if (!symbolicShape.getValues().empty()) {
    printer << "x";
  }
  cuda_tile::printCudaTileType(printer, getElementType());
  printer << ">";
}

LogicalResult
cuda_tile::TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                            SymbolicExprAttr symbolicShapeAttr,
                            ArrayRef<int64_t> concreteShapeCache,
                            Type elementType) {
  auto symbolicShape = dyn_cast<SymbolicTupleAttr>(symbolicShapeAttr);
  if (!symbolicShape)
    return emitError() << "symbolic shape must be a SymbolicTupleAttr";

  if (symbolicShape.getValues().size() != concreteShapeCache.size())
    return emitError() << "concrete shape cache rank "
                       << concreteShapeCache.size()
                       << " does not match symbolic shape rank "
                       << symbolicShape.getValues().size();

  for (auto [idx, dimAttr] : llvm::enumerate(symbolicShape.getValues())) {
    if (!isTileShapeExpr(dimAttr))
      return emitError() << "shape dimension " << idx
                         << " must be an integer-valued expression";

    if (auto intAttr = dyn_cast<SymbolicExprIntAttr>(dimAttr)) {
      if (concreteShapeCache[idx] != intAttr.getValue()) {
        return emitError() << "concrete shape cache dimension " << idx
                           << " does not match symbolic shape";
      }
      continue;
    }

    if (!ShapedType::isDynamic(concreteShapeCache[idx]))
      return emitError() << "concrete shape cache dimension " << idx
                         << " must be dynamic for symbolic shape dimension";
  }
  return verifyConcreteTile(
      emitError, concreteShapeCache, elementType
  );
}

ArrayRef<int64_t> cuda_tile::TileType::getShape() const {
  return getConcreteShapeCache();
}

cuda_tile::TileType
cuda_tile::TileType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                               Type elementType) const {
  if (shape)
    return cuda_tile::TileType::get(*shape, elementType);
  return cuda_tile::TileType::get(getContext(), getSymbolicShapeAttr(),
                                  getConcreteShapeCache(), elementType);
}

TileType cuda_tile::getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  auto tileType = dyn_cast<cuda_tile::TileType>(type);
  assert(tileType);
  return cuda_tile::TileType::get(tileType.getShape(), i1Type);
}

TileType cuda_tile::reshapeTileTypeToRank(TileType type, int targetRank) {
  int r = type.hasRank() ? type.getRank() : 0;
  assert(targetRank >= r);
  if (targetRank == r)
    return type;
  llvm::SmallVector<int64_t, 8> newShape;
  newShape.assign(targetRank - r, /*value=*/1);
  llvm::append_range(newShape, type.getShape());
  return cuda_tile::TileType::get(newShape, type.getElementType());
}

//===----------------------------------------------------------------------===//
// TensorViewType
//===----------------------------------------------------------------------===//

/// Parses the textural representation of a tensor_view stride.
static ParseResult parseStrideArray(AsmParser &parser,
                                    SmallVectorImpl<int64_t> &stride) {
  auto parseStrideElem = [&]() -> ParseResult {
    SMLoc loc = parser.getCurrentLocation();

    if (succeeded(parser.parseOptionalQuestion())) {
      stride.push_back(cuda_tile::TensorViewType::kDynamic);
      return success();
    }

    int64_t intStride = 0;
    OptionalParseResult intParseResult =
        parser.parseOptionalInteger<int64_t>(intStride);

    // If no hint of an integer was found.
    if (!intParseResult.has_value())
      return parser.emitError(
          loc, "expected either 64-bit integer or question mark");

    // If an invalid integer was found, an error has already been printed.
    if (failed(intParseResult.value()))
      return failure();

    // This is checked here to avoid accepting `kDynamic` as an explicit value.
    if (intStride <= 0)
      return parser.emitError(loc, "expected strictly positive integer, got ")
             << intStride;

    stride.push_back(intStride);
    return success();
  };

  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                        parseStrideElem, "strides array");
}

Type cuda_tile::TensorViewType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> shape;

  Type elementType;
  if (parser.parseLess() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/true) ||
      parseCudaTileType(parser, elementType))
    return Type();

  // Handle strides parsing based on tensor dimensionality
  SmallVector<int64_t> strides;
  cuda_tile::PtrAttrAttr ptrAttr{};

  if (shape.empty()) {
    // For 0-D tensors, check if strides are incorrectly provided
    if (succeeded(parser.parseOptionalComma())) {
      if (succeeded(parser.parseOptionalKeyword("strides"))) {
        parser.emitError(parser.getCurrentLocation())
            << "strides must not be provided for 0-d tiles";
        return Type();
      }
      // For 0-D tensors, comma after elementType may introduce ptrAttr.
      auto optResult = parser.parseOptionalAttribute(ptrAttr);
      if (optResult.has_value()) {
        if (failed(optResult.value())) {
          return Type();
        }
        if (parser.parseGreater()) {
          return Type();
        }
        return parser.getChecked<cuda_tile::TensorViewType>(
            loc, parser.getContext(), elementType, shape, strides, ptrAttr);
      }
      // If there's a comma but no valid token after it, that's an error
      parser.emitError(parser.getCurrentLocation())
          << "unexpected token after element type in 0-d tensor_view";
      return Type();
    }
  } else {
    // For non-0D tensors, strides are required
    if (failed(parser.parseOptionalComma())) {
      parser.emitError(parser.getCurrentLocation()) << "expected 'strides'";
      return Type();
    }
    if (parser.parseKeyword("strides") || parser.parseEqual() ||
        parseStrideArray(parser, strides))
      return Type();
  }

  // Optionally parse ptrAttr after strides (or after element type for 0-D).
  if (succeeded(parser.parseOptionalComma())) {
    auto optResult = parser.parseOptionalAttribute(ptrAttr);
    if (!optResult.has_value() || failed(optResult.value())) {
      parser.emitError(parser.getCurrentLocation())
          << "expected ptr_attr attribute";
      return Type();
    }
  }

  if (parser.parseGreater())
    return Type();

  return parser.getChecked<cuda_tile::TensorViewType>(
      loc, parser.getContext(), elementType, shape, strides, ptrAttr);
}

void cuda_tile::TensorViewType::print(AsmPrinter &printer) const {
  printer << "<";
  printShapeAndElem(printer, getShape(), getElementType());

  // Only print strides if tensor_view is not 0-D.
  if (!getShape().empty()) {
    printer << ", strides=[";
    llvm::interleave(
        getStrides(), printer,
        [&](int64_t strideElem) {
          if (strideElem == TensorViewType::kDynamic)
            printer << "?";
          else
            printer << strideElem;
        },
        ",");
    printer << "]";
  }

  if (auto ptrAttr = getPtrAttr()) {
    printer << ", " << ptrAttr;
  }

  printer << ">";
}

namespace {

/// Prints an array of dimensions in diagnostics, replacing
/// TensorViewType::kDynamic with a question mark.
struct PrintDynamic {
  PrintDynamic(ArrayRef<int64_t> values) : values(values) {}
  ArrayRef<int64_t> values;
};

Diagnostic &operator<<(Diagnostic &diag, PrintDynamic v) {
  diag << "[";
  llvm::interleaveComma(v.values, diag, [&](int64_t value) {
    if (value == cuda_tile::TensorViewType::kDynamic)
      diag << '?';
    else
      diag << value;
  });
  diag << "]";
  return diag;
}

} // namespace

LogicalResult
cuda_tile::TensorViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type elementType, ArrayRef<int64_t> shape,
                                  ArrayRef<int64_t> stride,
                                  cuda_tile::PtrAttrAttr ptrAttr) {
  if (ptrAttr) {
    switch (ptrAttr.getValue()) {
    case cuda_tile::PtrAttr::NONE:
    case cuda_tile::PtrAttr::UNICAST:
    case cuda_tile::PtrAttr::MULTICAST:
      break;
    default:
      return emitError()
             << "expected ptr_attr to be a valid pointer classification "
                "but got invalid enum value "
             << static_cast<int32_t>(ptrAttr.getValue());
    }
  }
  if (shape.size() != stride.size())
    return emitError() << "expected shape and stride to be of same rank but "
                          "got shape of rank "
                       << shape.size() << " and stride of rank "
                       << stride.size();

  if (any_of(shape, [](int64_t dim) {
        return dim <= 0 && dim != cuda_tile::TensorViewType::kDynamic;
      }))
    return emitError()
           << "dimensions must have strictly positive constant sizes but got "
           << PrintDynamic(shape);

  if (any_of(stride, [](int64_t dim) {
        return dim <= 0 && dim != cuda_tile::TensorViewType::kDynamic;
      }))
    return emitError()
           << "dimensions must have strictly positive constant strides but got "
           << PrintDynamic(stride);

  return success();
}

size_t cuda_tile::TensorViewType::dynamicShapeAmount() {
  return llvm::count_if(getShape(), [](int64_t x) {
    return x == cuda_tile::TensorViewType::kDynamic;
  });
}

size_t cuda_tile::TensorViewType::dynamicStrideAmount() {
  return llvm::count_if(getStrides(), [](int64_t x) {
    return x == cuda_tile::TensorViewType::kDynamic;
  });
}

std::optional<unsigned>
cuda_tile::TensorViewType::getF4PackingDimension() const {
  if (!isa<Float4E2M1FNType>(getElementType())) {
    return std::nullopt;
  }

  std::optional<unsigned> packingDimension;
  for (auto [dimension, stride] : llvm::enumerate(getStrides())) {
    if (stride != 1) {
      continue;
    }
    if (packingDimension) {
      return std::nullopt;
    }
    packingDimension = static_cast<unsigned>(dimension);
  }
  return packingDimension;
}

//===----------------------------------------------------------------------===//
// PartitionView Type
//===----------------------------------------------------------------------===//

static ParseResult parseViewTileShape(AsmParser &parser,
                                      SmallVector<int32_t> &tileShape) {
  if (parser.parseKeyword("tile") || parser.parseEqual())
    return ParseResult::failure();

  SMLoc dimListLoc = parser.getCurrentLocation();

  SmallVector<int64_t> tileShape64;
  if (parser.parseLParen() ||
      parser.parseDimensionList(tileShape64, /*allowDynamic=*/false,
                                /*withTrailingX=*/false) ||
      parser.parseRParen())
    return ParseResult::failure();

  for (auto [i, v] : llvm::enumerate(tileShape64)) {
    if (v <= 0 || v > std::numeric_limits<int32_t>::max()) {
      return parser.emitError(dimListLoc)
             << "tile dimension " << i << " exceeds i32 limitations (got " << v
             << ", expected strictly positive and less than or equal to "
             << std::numeric_limits<int32_t>::max() << ")";
    }
    tileShape.push_back(static_cast<int32_t>(v));
  }

  return ParseResult::success();
}

/// Parses an optional comma followed by a dimMap, returning the identity dimMap
/// if absent.
static ParseResult parseOptionalViewDimMap(AsmParser &parser,
                                           SmallVector<int32_t> &dimMap,
                                           size_t tileRank) {
  auto parseDimMapElem = [&]() -> ParseResult {
    int32_t dim;
    if (parser.parseInteger(dim))
      return ParseResult::failure();
    dimMap.push_back(dim);
    return ParseResult::success();
  };

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("dim_map") || parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parseDimMapElem, "dim map"))
      return ParseResult::failure();
  } else {
    // By default, dimMap is the identity mapping.
    for (int32_t i : llvm::seq(tileRank))
      dimMap.push_back(i);
  }

  return ParseResult::success();
}

Type cuda_tile::PartitionViewType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();

  if (parser.parseLess())
    return Type();

  SmallVector<int32_t> tileShape;
  if (parseViewTileShape(parser, tileShape) || parser.parseComma())
    return Type();

  FailureOr<PaddingValueAttr> paddingValue = parseOptionalPaddingValue(parser);
  if (failed(paddingValue))
    return Type();
  else if (*paddingValue && parser.parseComma())
    return Type();

  Type parsedType;
  if (parseCudaTileType(parser, parsedType))
    return Type();
  auto tensor_view = dyn_cast<TensorViewType>(parsedType);
  if (!tensor_view) {
    parser.emitError(parser.getCurrentLocation())
        << "expected 'tensor_view' type, but got " << parsedType;
    return Type();
  }

  SmallVector<int32_t> dimMap;
  if (parseOptionalViewDimMap(parser, dimMap, tileShape.size()))
    return Type();

  if (parser.parseGreater())
    return Type();

  return parser.getChecked<cuda_tile::PartitionViewType>(
      loc, parser.getContext(),
      DenseI32ArrayAttr::get(parser.getContext(), tileShape), tensor_view,
      dimMap, *paddingValue);
}

static void printViewTileShape(AsmPrinter &printer,
                               ArrayRef<int32_t> tileShape) {
  printer << "tile=(";
  llvm::interleave(tileShape, printer, "x");
  printer << ")";
}

static void printOptionalViewPaddingValue(AsmPrinter &printer,
                                          PaddingValueAttr paddingValue) {
  if (paddingValue)
    printer << "padding_value = "
            << stringifyPaddingValue(paddingValue.getValue()) << ", ";
}

static void printOptionalViewDimMap(AsmPrinter &printer,
                                    ArrayRef<int32_t> dimMap) {
  if (!llvm::equal(dimMap, llvm::seq(dimMap.size())))
    printer << ", dim_map=[" << dimMap << "]";
}

void cuda_tile::PartitionViewType::print(AsmPrinter &printer) const {
  printer << "<";
  printViewTileShape(printer, getTileShape().asArrayRef());
  printer << ", ";
  printOptionalViewPaddingValue(printer, getPaddingValue());
  printCudaTileType(printer, getTensorView());
  printOptionalViewDimMap(printer, getDimMap());
  printer << ">";
}

/// Verifies requirements that are shared between all grid-like views.
static LogicalResult verifyPartitionViewLike(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI32ArrayAttr tileShapeAttr, cuda_tile::TensorViewType tensorView,
    ArrayRef<int32_t> dimMap, PaddingValueAttr paddingValue) {
  ArrayRef<int32_t> tileShape = tileShapeAttr.asArrayRef();

  if (tileShape.empty())
    return emitError() << "0-dimension tile shape is not supported";

  if (tileShape.size() != tensorView.getShape().size())
    return emitError() << "expected tensor_view rank and tile rank "
                          "to match, got tensor_view of rank "
                       << tensorView.getShape().size() << " and tiles of rank "
                       << tileShape.size();

  if (tileShape.size() > std::numeric_limits<int32_t>::max())
    return emitError() << "tile rank cannot be more than "
                       << std::numeric_limits<int32_t>::max() << ", got "
                       << tileShape.size();

  if (dimMap.size() != tileShape.size())
    return emitError() << "expected dim_map to map exactly all "
                       << tileShape.size() << " dimensions of the tile, got "
                       << dimMap.size() << " mappings";

  if (any_of(tileShape, [](int32_t dim) { return dim <= 0; }))
    return emitError()
           << "tile shape dimensions must have positive length but got ["
           << tileShape << "]";

  if (any_of(tileShape, [](int32_t dim) { return (dim & (dim - 1)) != 0; }))
    return emitError()
           << "tile shape dimensions must have power of two length but got ["
           << tileShape << "]";

  SmallVector<std::optional<int32_t>> usedTensorViewDim(tileShape.size(),
                                                        std::nullopt);
  for (auto [tileDim, tensorViewDim] : llvm::enumerate(dimMap)) {
    if (tensorViewDim < 0)
      return emitError() << "target dimension must not be negative, got "
                         << tensorViewDim;

    if (static_cast<uint32_t>(tensorViewDim) >= tensorView.getShape().size())
      return emitError()
             << "target dimension is outside of tensor view dimensions, "
                "expected strictly less than "
             << tensorView.getShape().size() << ", got " << tensorViewDim;

    if (usedTensorViewDim[tensorViewDim].has_value())
      return emitError() << "target dimension " << tensorViewDim
                         << " mapped at least twice (for tile dimensions "
                         << usedTensorViewDim[tensorViewDim].value() << " and "
                         << tileDim << ")";

    usedTensorViewDim[tensorViewDim] = tileDim;
  }

  // Run the Tile type verifier to catch invalid tiles in the partition type
  SmallVector<int64_t> shape64(tileShape.begin(), tileShape.end());
  if (failed(verifyConcreteTile([&]() { return emitError(); }, shape64,
                                tensorView.getElementType())))
    return failure();

  // Verify that special padding values are only used with floating point types
  if (paddingValue) {
    switch (paddingValue.getValue()) {
    default:
      break;
    case PaddingValue::neg_zero:
    case PaddingValue::nan:
    case PaddingValue::pos_inf:
    case PaddingValue::neg_inf:
      if (!llvm::isa<FloatType>(tensorView.getElementType()))
        return emitError()
               << "padding_value "
               << stringifyPaddingValue(paddingValue.getValue())
               << " can only be used with floating point element types, got "
               << tensorView.getElementType();
      break;
    }
  }

  return success();
}

LogicalResult cuda_tile::PartitionViewType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI32ArrayAttr tileShapeAttr, cuda_tile::TensorViewType tensorView,
    ArrayRef<int32_t> dimMap, PaddingValueAttr paddingValue) {
  return verifyPartitionViewLike(emitError, tileShapeAttr, tensorView, dimMap,
                                 paddingValue);
}

size_t cuda_tile::PartitionViewType::getViewIndexRank() const {
  return getTileShape().size();
}

Type cuda_tile::PartitionViewType::getViewTileType() const {
  SmallVector<int64_t> shape64(getTileShape().size());
  llvm::transform(getTileShape().asArrayRef(), shape64.begin(),
                  [](int32_t x) -> int64_t { return x; });
  return cuda_tile::TileType::get(shape64, getTensorView().getElementType());
}

LogicalResult cuda_tile::PartitionViewType::verifyIndices(
    function_ref<InFlightDiagnostic()> emitError, TypeRange indexTypes) const {
  if (indexTypes.empty())
    return success();

  // All indices must be scalar tiles.
  for (Type t : indexTypes) {
    auto tileType = dyn_cast<TileType>(t);
    if (!tileType) {
      return emitError() << "expected index type to be a tile type, got " << t;
    }
    if (tileType.getRank() != 0) {
      return emitError() << "expected index type to be a scalar tile, got "
                         << tileType;
    }
  }

  // All indices must have the same type.
  Type baseIndexType = indexTypes.front();
  for (const auto &[i, indexType] : llvm::enumerate(indexTypes)) {
    if (indexType != baseIndexType)
      return emitError() << "expected index type " << i
                         << " to be the same as other index types ("
                         << baseIndexType << "), got " << indexType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StridedView Type
//===----------------------------------------------------------------------===//

static ParseResult
parseViewTraversalStrides(AsmParser &parser,
                          SmallVector<int32_t> &traversalStrides) {
  if (parser.parseKeyword("traversal_strides") || parser.parseEqual())
    return ParseResult::failure();

  auto parseStrideElement = [&]() {
    int32_t element;
    ParseResult result = parser.parseInteger(element);
    if (succeeded(result))
      traversalStrides.push_back(element);
    return result;
  };

  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                        parseStrideElement);
}

Type cuda_tile::StridedViewType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();

  if (parser.parseLess())
    return Type();

  SmallVector<int32_t> tileShape;
  SmallVector<int32_t> traversalStrides;
  if (parseViewTileShape(parser, tileShape) || parser.parseComma() ||
      parseViewTraversalStrides(parser, traversalStrides) ||
      parser.parseComma())
    return Type();

  FailureOr<PaddingValueAttr> paddingValue = parseOptionalPaddingValue(parser);
  if (failed(paddingValue))
    return Type();
  else if (*paddingValue && parser.parseComma())
    return Type();

  Type parsedType;
  if (parseCudaTileType(parser, parsedType))
    return Type();
  auto tensor_view = dyn_cast<TensorViewType>(parsedType);
  if (!tensor_view) {
    parser.emitError(parser.getCurrentLocation())
        << "expected 'tensor_view' type, but got " << parsedType;
    return Type();
  }

  SmallVector<int32_t> dimMap;
  if (parseOptionalViewDimMap(parser, dimMap, tileShape.size()))
    return Type();

  if (parser.parseGreater())
    return Type();

  return parser.getChecked<cuda_tile::StridedViewType>(
      loc, parser.getContext(),
      DenseI32ArrayAttr::get(parser.getContext(), tileShape),
      DenseI32ArrayAttr::get(parser.getContext(), traversalStrides),
      tensor_view, dimMap, *paddingValue);
}

static void printViewTraversalStrides(AsmPrinter &printer,
                                      ArrayRef<int32_t> traversalStrides) {
  printer << "traversal_strides=[";
  llvm::interleave(traversalStrides, printer, ",");
  printer << "]";
}

void cuda_tile::StridedViewType::print(AsmPrinter &printer) const {
  printer << "<";
  printViewTileShape(printer, getTileShape().asArrayRef());
  printer << ", ";
  printViewTraversalStrides(printer, getTraversalStrides().asArrayRef());
  printer << ", ";
  printOptionalViewPaddingValue(printer, getPaddingValue());
  printCudaTileType(printer, getTensorView());
  printOptionalViewDimMap(printer, getDimMap());
  printer << ">";
}

LogicalResult cuda_tile::StridedViewType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI32ArrayAttr tileShapeAttr, DenseI32ArrayAttr traversalStridesAttr,
    cuda_tile::TensorViewType tensorView, ArrayRef<int32_t> dimMap,
    PaddingValueAttr paddingValue) {
  if (failed(verifyPartitionViewLike(emitError, tileShapeAttr, tensorView,
                                     dimMap, paddingValue)))
    return failure();

  if (tileShapeAttr.asArrayRef().size() !=
      traversalStridesAttr.asArrayRef().size())
    return emitError() << "expected " << tileShapeAttr.asArrayRef().size()
                       << " traversal strides, got "
                       << traversalStridesAttr.asArrayRef().size();

  if (any_of(traversalStridesAttr.asArrayRef(),
             [](int32_t dim) { return dim <= 0; }))
    return emitError()
           << "traversal strides must be strictly positive but got ["
           << traversalStridesAttr.asArrayRef() << "]";

  return success();
}

size_t cuda_tile::StridedViewType::getViewIndexRank() const {
  return getTileShape().size();
}

Type cuda_tile::StridedViewType::getViewTileType() const {
  SmallVector<int64_t> shape64(getTileShape().size());
  llvm::transform(getTileShape().asArrayRef(), shape64.begin(),
                  [](int32_t x) -> int64_t { return x; });
  return cuda_tile::TileType::get(shape64, getTensorView().getElementType());
}

LogicalResult cuda_tile::StridedViewType::verifyIndices(
    function_ref<InFlightDiagnostic()> emitError, TypeRange indexTypes) const {
  if (indexTypes.empty())
    return success();

  // All indices must be scalar tiles.
  for (Type t : indexTypes) {
    auto tileType = dyn_cast<TileType>(t);
    if (!tileType) {
      return emitError() << "expected index type to be a tile type, got " << t;
    }
    if (tileType.getRank() != 0) {
      return emitError() << "expected index type to be a scalar tile, got "
                         << tileType;
    }
  }

  // All indices must have the same type.
  Type baseIndexType = indexTypes.front();
  for (const auto &[i, indexType] : llvm::enumerate(indexTypes)) {
    if (indexType != baseIndexType)
      return emitError() << "expected index type " << i
                         << " to be the same as other index types ("
                         << baseIndexType << "), got " << indexType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GatherScatterView Type
//===----------------------------------------------------------------------===//

Type cuda_tile::GatherScatterViewType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tileShape;

  if (parser.parseLess())
    return Type();

  if (parser.parseKeyword("tile") || parser.parseEqual())
    return Type();

  SMLoc dimListLoc = parser.getCurrentLocation();
  if (parser.parseLParen() ||
      parser.parseDimensionList(tileShape, /*allowDynamic=*/false,
                                /*withTrailingX=*/false) ||
      parser.parseRParen() || parser.parseComma())
    return Type();

  FailureOr<PaddingValueAttr> paddingValue = parseOptionalPaddingValue(parser);
  if (failed(paddingValue))
    return Type();
  else if (*paddingValue && parser.parseComma())
    return Type();

  Type parsedType;
  if (parseCudaTileType(parser, parsedType))
    return Type();
  auto tensor_view = dyn_cast<TensorViewType>(parsedType);
  if (!tensor_view) {
    parser.emitError(parser.getCurrentLocation())
        << "expected 'tensor_view' type, but got " << parsedType;
    return Type();
  }

  uint32_t sparse_dim;
  if (parser.parseComma() || parser.parseKeyword("sparse_dim") ||
      parser.parseEqual())
    return Type();

  if (parser.parseInteger(sparse_dim))
    return Type();

  if (parser.parseGreater())
    return Type();

  SmallVector<int32_t> tileShape32;
  for (auto [i, v] : llvm::enumerate(tileShape)) {
    if (v <= 0 || v > std::numeric_limits<int32_t>::max()) {
      parser.emitError(dimListLoc)
          << "tile dimension " << i << " exceeds i32 limitations (got " << v
          << ", expected strictly positive and less than or equal to "
          << std::numeric_limits<int32_t>::max() << ")";
      return Type();
    }
    tileShape32.push_back(static_cast<int32_t>(v));
  }

  return parser.getChecked<cuda_tile::GatherScatterViewType>(
      loc, parser.getContext(),
      DenseI32ArrayAttr::get(parser.getContext(), tileShape32), tensor_view,
      sparse_dim, *paddingValue);
}

void cuda_tile::GatherScatterViewType::print(AsmPrinter &printer) const {
  printer << "<";
  printer << "tile=(";
  llvm::interleave(getTileShape().asArrayRef(), printer, "x");
  printer << "), ";
  if (getPaddingValue())
    printer << "padding_value = "
            << stringifyPaddingValue(getPaddingValue().getValue()) << ", ";

  printCudaTileType(printer, getTensorView());

  printer << ", sparse_dim=" << getSparseDim();

  printer << ">";
}

LogicalResult cuda_tile::GatherScatterViewType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI32ArrayAttr tileShapeAttr, cuda_tile::TensorViewType tensorView,
    uint32_t sparse_dim, PaddingValueAttr paddingValue) {
  ArrayRef<int32_t> tileShape = tileShapeAttr.asArrayRef();

  if (tileShape.empty())
    return emitError() << "0-dimension tile shape is not supported";

  if (tileShape.size() != tensorView.getShape().size())
    return emitError() << "expected tensor_view rank and tile rank "
                          "to match, got tensor_view of rank "
                       << tensorView.getShape().size() << " and tiles of rank "
                       << tileShape.size();

  if (tileShape.size() > std::numeric_limits<int32_t>::max())
    return emitError() << "tile rank cannot be more than "
                       << std::numeric_limits<int32_t>::max() << ", got "
                       << tileShape.size();

  if (any_of(tileShape, [](int32_t d) { return d <= 0; }))
    return emitError()
           << "tile shape dimensions must have positive length but got ["
           << tileShape << "]";

  if (any_of(tileShape, [](int32_t d) { return (d & (d - 1)) != 0; }))
    return emitError()
           << "tile shape dimensions must have power of two length but got ["
           << tileShape << "]";

  // Verify sparse_dim parameter
  if (sparse_dim >= tensorView.getShape().size())
    return emitError()
           << "gather dimension is outside of tensor view dimensions, "
              "expected strictly less than "
           << tensorView.getShape().size() << ", got " << sparse_dim;

  // Run the Tile type verifier to catch invalid tiles in the gather type
  SmallVector<int64_t> shape64(tileShape.begin(), tileShape.end());
  if (failed(verifyConcreteTile([&]() { return emitError(); }, shape64,
                                tensorView.getElementType())))
    return failure();

  // Verify that special padding values are only used with floating point types
  if (paddingValue) {
    switch (paddingValue.getValue()) {
    default:
      break;
    case PaddingValue::neg_zero:
    case PaddingValue::nan:
    case PaddingValue::pos_inf:
    case PaddingValue::neg_inf:
      if (!llvm::isa<FloatType>(tensorView.getElementType()))
        return emitError()
               << "padding_value "
               << stringifyPaddingValue(paddingValue.getValue())
               << " can only be used with floating point element types, got "
               << tensorView.getElementType();
      break;
    }
  }

  return success();
}

size_t cuda_tile::GatherScatterViewType::getViewIndexRank() const {
  return getTileShape().size();
}

Type cuda_tile::GatherScatterViewType::getViewTileType() const {
  SmallVector<int64_t> shape64(getTileShape().size());
  llvm::transform(getTileShape().asArrayRef(), shape64.begin(),
                  [](int32_t x) -> int64_t { return x; });
  return cuda_tile::TileType::get(shape64, getTensorView().getElementType());
}

LogicalResult cuda_tile::GatherScatterViewType::verifyIndices(
    function_ref<InFlightDiagnostic()> emitError, TypeRange indexTypes) const {
  // For GatherScatterView, the index at position `sparse_dim` MUST be a 1D
  // tensor (gather/scatter indices) while the rest MUST be scalar tiles.
  if (indexTypes.empty())
    return success();

  uint32_t sparse_dim = getSparseDim();

  // Verify sparse_dim is within index range
  if (sparse_dim >= indexTypes.size())
    return emitError() << "gather/scatter dimension " << sparse_dim
                       << " is out of range for index count "
                       << indexTypes.size();

  // Verify index at sparse_dim position is a 1D tensor
  auto gatherIndexTileType = dyn_cast<TileType>(indexTypes[sparse_dim]);
  if (!gatherIndexTileType)
    return emitError() << "expected index at sparse_dim position " << sparse_dim
                       << " to be a tile type for GatherScatterView";

  if (gatherIndexTileType.getShape().size() != 1)
    return emitError() << "expected index at sparse_dim position " << sparse_dim
                       << " to be a 1D tensor for GatherScatterView, got rank "
                       << gatherIndexTileType.getShape().size();

  // Verify that the gather/scatter index size matches the tile shape at the
  // gather dimension
  int64_t gatherIndexSize = gatherIndexTileType.getShape()[0];
  int32_t tileSizeAtDim = getTileShape()[sparse_dim];
  if (gatherIndexSize != tileSizeAtDim)
    return emitError() << "expected gather/scatter index size ("
                       << gatherIndexSize
                       << ") to match tile shape at gather/scatter dimension "
                       << sparse_dim << " (" << tileSizeAtDim << ")";

  // Verify all other indices are scalar tiles and have the same type
  Type baseScalarType;
  for (size_t i = 0; i < indexTypes.size(); ++i) {
    if (i == sparse_dim)
      continue; // Skip the gather index

    auto idxTileType = dyn_cast<TileType>(indexTypes[i]);
    if (!idxTileType || !idxTileType.getShape().empty())
      return emitError() << "expected index " << i
                         << " to be a scalar tile for GatherScatterView";

    if (!baseScalarType) {
      baseScalarType = indexTypes[i];
    } else if (indexTypes[i] != baseScalarType) {
      return emitError() << "expected scalar index " << i
                         << " to have the same type as other scalar indices ("
                         << baseScalarType << "), got " << indexTypes[i];
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FragmentType
//===----------------------------------------------------------------------===//

Type cuda_tile::FragmentType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  SMLoc loc = parser.getCurrentLocation();

  // Shape and element type: `NxMx...xElem`. Concrete dims are bare integers,
  // symbolic dims are written in brackets, e.g. `2x[F]xf32`. Shared with
  // TileType via parseTileDimensionList.
  SmallVector<int64_t> concreteDims;
  SmallVector<SymbolicExprAttr> symbolicDims;
  Type elementType;
  if (parser.parseLess() ||
      parseTileDimensionList(parser, concreteDims, symbolicDims) ||
      parseCudaTileType(parser, elementType)) {
    return {};
  }

  // Memory space.
  if (parser.parseComma()) {
    return {};
  }
  SMLoc memLoc = parser.getCurrentLocation();
  StringRef memStr;
  if (parser.parseKeyword(&memStr)) {
    return {};
  }
  std::optional<MemorySpace> memorySpace = symbolizeMemorySpace(memStr);
  if (!memorySpace) {
    parser.emitError(memLoc) << "unknown memory space '" << memStr << "'";
    return {};
  }

  // mem_layout = <layout>. Parsed via FieldParser so the layout may be written
  // in the stripped `<(...):(...)>` form without the `#cuda_tile.preview$layout`
  // prefix.
  if (parser.parseComma() || parser.parseKeyword("mem_layout") ||
      parser.parseEqual()) {
    return {};
  }
  FailureOr<LayoutAttr> memLayout = FieldParser<LayoutAttr>::parse(parser);
  if (failed(memLayout)) {
    return {};
  }

  // fragmap = <layout>. Same as above, we use FieldParser so we can parse it in
  // the stripped form.
  if (parser.parseComma() || parser.parseKeyword("fragmap") ||
      parser.parseEqual()) {
    return {};
  }
  FailureOr<LayoutAttr> fragmap = FieldParser<LayoutAttr>::parse(parser);
  if (failed(fragmap)) {
    return {};
  }

  // Optional: `, alignment = <expr>` (elided in the IR when equal to 1).
  SymbolicExprAttr alignment;
  if (succeeded(parser.parseOptionalComma())) {
    SMLoc alignLoc = parser.getCurrentLocation();
    if (parser.parseKeyword("alignment") || parser.parseEqual()) {
      return {};
    }
    bool emittedDiagnostic = false;
    CudaTileCompactLayoutParser exprParser(parser, ctx, alignLoc,
                                           emittedDiagnostic);
    FailureOr<SymbolicExprAttr> expr = exprParser.parseExpr();
    if (failed(expr)) {
      if (!emittedDiagnostic) {
        parser.emitError(alignLoc, "expected alignment expression");
      }
      return {};
    }
    alignment = *expr;
  }

  // Absent alignment defaults to a concrete 1 element.
  if (!alignment) {
    alignment = SymbolicExprIntAttr::get(ctx, 1);
  }

  if (parser.parseGreater()) {
    return {};
  }

  return FragmentType::getChecked([&] { return parser.emitError(loc); },
                                  SymbolicTupleAttr::get(ctx, symbolicDims),
                                  elementType, *memorySpace, *memLayout,
                                  *fragmap, alignment);
}

void cuda_tile::FragmentType::print(AsmPrinter &printer) const {
  printer << "<";
  auto symbolicShape = cast<SymbolicTupleAttr>(getSymbolicShapeAttr());
  llvm::interleave(
      symbolicShape.getValues(), printer,
      [&](SymbolicExprAttr dim) {
        if (auto intAttr = dyn_cast<SymbolicExprIntAttr>(dim)) {
          printer << intAttr.getValue();
          return;
        }
        printer << "[";
        CudaTileCompactLayoutParser::printCompactExpr(printer.getStream(), dim);
        printer << "]";
      },
      "x");
  if (!symbolicShape.getValues().empty()) {
    printer << "x";
  }
  cuda_tile::printCudaTileType(printer, getElementType());
  printer << ", " << stringifyMemorySpace(getMemorySpace());
  printer << ", mem_layout = ";
  printer.printStrippedAttrOrType(getMemLayout());
  printer << ", fragmap = ";
  printer.printStrippedAttrOrType(getFragmap());
  // Elide alignment when it is the default (concrete 1 element).
  auto alignmentInt = dyn_cast<SymbolicExprIntAttr>(getAlignment());
  if (!alignmentInt || alignmentInt.getValue() != 1) {
    printer << ", alignment = ";
    CudaTileCompactLayoutParser::printCompactExpr(printer.getStream(),
                                                  getAlignment());
  }
  printer << ">";
}

/// Element count of a fragment shape, or nullopt when a dimension is still
/// symbolic or the product overflows.
static std::optional<int64_t>
staticFragmentElementCount(ArrayRef<int64_t> concreteShapeCache) {
  // tryGetNumElements() multiplies kDynamic in as a plain factor, so the
  // symbolic dimensions have to be ruled out first.
  if (ShapedType::isDynamicShape(concreteShapeCache)) {
    return std::nullopt;
  }
  return ShapedType::tryGetNumElements(concreteShapeCache);
}

/// Verify the storage-safety contract on a fragment's `mem_layout`.
///
/// A fragment buffer holds exactly the footprint elements of one invocation and
/// the SIMT ABI hands the implementation a bare pointer to it, so the layout
/// must be a permutation of `[0, elementCount)`.
///
/// A parametric layout is re-checked once concretization substitutes its
/// variables.
static LogicalResult
verifyFragmentMemLayout(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<int64_t> concreteShapeCache,
                        cuda_tile::MemorySpace memorySpace,
                        cuda_tile::LayoutAttr memLayout) {
  FailureOr<SmallVector<LayoutLeaf>> leaves = flattenStaticLayout(memLayout);
  if (failed(leaves)) {
    return success();
  }

  // A symbolic fragment shape leaves the element count unknown, so the two
  // checks that relate the layout to it are deferred with the layout itself.
  std::optional<int64_t> elementCount =
      staticFragmentElementCount(concreteShapeCache);

  FailureOr<int64_t> memLayoutSize = staticLayoutSize(*leaves);
  if (failed(memLayoutSize)) {
    return emitError() << "mem_layout domain size must be representable as a "
                          "signed 64-bit integer";
  }
  if (elementCount && *memLayoutSize != *elementCount) {
    return emitError() << "mem_layout domain size (" << *memLayoutSize
                       << ") must equal the fragment element count ("
                       << *elementCount << ")";
  }

  FailureOr<int64_t> storageCosize = staticLayoutCosize(*leaves);
  if (failed(storageCosize)) {
    return emitError() << "mem_layout storage span must be representable as a "
                          "signed 64-bit integer";
  }
  if (memorySpace == cuda_tile::MemorySpace::SMEM &&
      *storageCosize > std::numeric_limits<int32_t>::max()) {
    return emitError() << "smem mem_layout storage span (" << *storageCosize
                       << ") exceeds the current signed 32-bit offset ABI "
                          "limit ("
                       << std::numeric_limits<int32_t>::max() << ")";
  }
  if (elementCount && *storageCosize != *elementCount) {
    return emitError()
           << "mem_layout must be a bijection on the footprint elements (a "
              "permutation of [0, "
           << *elementCount << ")), but its codomain spans " << *storageCosize
           << " element(s)";
  }

  if (!hasNoBroadcastModes(*leaves) || !hasCompactStrideChain(*leaves)) {
    return emitError() << "mem_layout must be a bijection on the footprint "
                          "elements, but its modes overlap or leave gaps";
  }

  return success();
}

LogicalResult cuda_tile::FragmentType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    SymbolicExprAttr symbolicShapeAttr, ArrayRef<int64_t> concreteShapeCache,
    Type elementType, cuda_tile::MemorySpace memorySpace,
    cuda_tile::LayoutAttr memLayout, cuda_tile::LayoutAttr fragmap,
    SymbolicExprAttr alignment) {
  auto symbolicShape = dyn_cast<SymbolicTupleAttr>(symbolicShapeAttr);
  if (!symbolicShape) {
    return emitError() << "symbolic shape must be a SymbolicTupleAttr";
  }

  if (!alignment) {
    return emitError() << "alignment must not be null (use 1 for the default)";
  }

  // An alignment is a single number of elements, so it must be integer-valued:
  // a tuple, a datatype, or a toward-zero division has no meaning here. This
  // mirrors the scratch alignment check in ScratchRequestAttr::verify.
  if (failed(verifySymbolicExprKind(alignment, SymbolicExprKind::Int, [&] {
        return emitError() << "fragment alignment: ";
      }))) {
    return failure();
  }

  // A concrete alignment must additionally be a positive power-of-two number
  // of elements. The check runs on the simplified expression so that an
  // alignment stays rejected however it was spelled: `-(4)` and `4 - 8` mean
  // the same thing as `-4`, and the compact printer spells all three the same
  // way once they are folded.
  if (auto alignInt = dyn_cast<SymbolicExprIntAttr>(alignment.simplify())) {
    int64_t raw = alignInt.getValue();
    if (raw <= 0) {
      return emitError() << "fragment alignment must be a positive number of "
                            "elements";
    }
    if (!llvm::isPowerOf2_64(raw)) {
      return emitError() << "fragment alignment must be a power of two number "
                            "of elements, got "
                         << raw;
    }
  }

  if (symbolicShape.getValues().size() != concreteShapeCache.size()) {
    return emitError() << "concrete shape cache rank ("
                       << concreteShapeCache.size()
                       << ") does not match symbolic shape rank ("
                       << symbolicShape.getValues().size() << ")";
  }

  for (auto [idx, dim] : llvm::enumerate(symbolicShape.getValues())) {
    if (!isTileShapeExpr(dim)) {
      return emitError() << "shape dimension " << idx
                         << " must be an integer-valued expression";
    }

    // If the dimension is an int expression (i.e. concrete), then verify it
    // matches the concrete shape cache.
    if (auto intDim = dyn_cast<SymbolicExprIntAttr>(dim)) {
      int64_t value = intDim.getValue();
      if (value <= 0) {
        return emitError() << "fragment shape dimension " << idx
                           << " must be positive, got " << value;
      }
      if (concreteShapeCache[idx] != value) {
        return emitError() << "concrete shape cache dimension " << idx
                           << " does not match symbolic shape";
      }
      continue;
    }

    // Otherwise, the concrete shape cache *must* hold a "dynamic" entry at this
    // index.
    if (!ShapedType::isDynamic(concreteShapeCache[idx])) {
      return emitError() << "concrete shape cache dimension " << idx
                         << " must be dynamic for a symbolic shape dimension";
    }
  }

  if (ShapedType::isStaticShape(concreteShapeCache) &&
      !staticFragmentElementCount(concreteShapeCache)) {
    return emitError()
           << "fragment element count must fit in a signed 64-bit integer";
  }

  // The memory space is constrained to `local`/`smem` by the MemorySpace enum
  // itself; there is no `tmem` value.

  if (!memLayout) {
    return emitError() << "mem_layout must not be null";
  }
  if (!fragmap) {
    return emitError() << "fragmap must not be null";
  }

  return verifyFragmentMemLayout(emitError, concreteShapeCache, memorySpace,
                                 memLayout);
}

/// Forward the getShape for ShapedTypeInterface to the concrete shape cache on
/// the type.
ArrayRef<int64_t> cuda_tile::FragmentType::getShape() const {
  return getConcreteShapeCache();
}

static bool fragmentHasUnresolvedSymbolicExpr(Attribute attr) {
  return attr
      .walk([](Attribute nested) {
        return isa<SymbolicExprVarAttr, SymbolicExprBuiltinAttr>(nested)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

bool cuda_tile::FragmentType::isConcrete() const {
  if (llvm::any_of(getShape(), ShapedType::isDynamic)) {
    return false;
  }
  return !fragmentHasUnresolvedSymbolicExpr(getMemLayout()) &&
         !fragmentHasUnresolvedSymbolicExpr(getFragmap()) &&
         !fragmentHasUnresolvedSymbolicExpr(getAlignment());
}

//===----------------------------------------------------------------------===//
// Type Registration
//===----------------------------------------------------------------------===//

void CudaTileDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cuda_tile/Dialect/CudaTile/IR/Types.cpp.inc"
      >();
}
