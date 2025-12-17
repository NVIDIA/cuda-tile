//===- Types.cpp - CUDA Tile Type Verifiers and Parsers ---------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"

#define GET_TYPEDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.cpp.inc"

using namespace mlir;
using namespace mlir::cuda_tile;

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

bool CudaTileType::classof(Type type) {
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

//===----------------------------------------------------------------------===//
// Type Printing Utilities
//===----------------------------------------------------------------------===//

/// Parse a type, if type is unprefixed, assume it is from the cuda_tile dialect
ParseResult cuda_tile::parseCudaTileType(AsmParser &p, Type &type) {
  auto result = p.parseOptionalType(type);
  if (result.has_value() && succeeded(*result))
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
// TileType
//===----------------------------------------------------------------------===//

cuda_tile::TileType cuda_tile::TileType::get(ArrayRef<int64_t> shape,
                                             Type elementType) {
  return cuda_tile::TileType::get(elementType.getContext(), shape, elementType);
}

Type cuda_tile::TileType::parse(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> dims;
  Type elementType;
  if (parser.parseLess() ||
      parser.parseDimensionList(dims, /*allowDynamic=*/false) ||
      parseCudaTileType(parser, elementType) || parser.parseGreater())
    return Type();
  return parser.getChecked<cuda_tile::TileType>(loc, elementType.getContext(),
                                                dims, elementType);
}

void cuda_tile::TileType::print(AsmPrinter &printer) const {
  printer << "<";
  printShapeAndElem(printer, getShape(), getElementType());
  printer << ">";
}

LogicalResult
cuda_tile::TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                            ArrayRef<int64_t> shape, Type elementType) {
  return verifyTileSize(emitError, shape);
}

cuda_tile::TileType
cuda_tile::TileType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                               Type elementType) const {
  return cuda_tile::TileType::get(shape.has_value() ? *shape : getShape(),
                                  elementType);
}

Type cuda_tile::getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tileType = dyn_cast<cuda_tile::TileType>(type))
    return cuda_tile::TileType::get(tileType.getShape(), i1Type);

  return i1Type;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

cuda_tile::PointerType cuda_tile::PointerType::get(Type pointeeType) {
  return cuda_tile::PointerType::get(pointeeType.getContext(), pointeeType);
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

  if (shape.empty()) {
    // For 0-D tensors, check if strides are incorrectly provided
    if (succeeded(parser.parseOptionalComma())) {
      if (succeeded(parser.parseOptionalKeyword("strides"))) {
        parser.emitError(parser.getCurrentLocation())
            << "strides must not be provided for 0-d tiles";
        return Type();
      } else {
        // If there's a comma but no 'strides' keyword, that's also an error
        parser.emitError(parser.getCurrentLocation())
            << "unexpected token after element type in 0-d tensor_view";
        return Type();
      }
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

  if (parser.parseGreater())
    return Type();

  return parser.getChecked<cuda_tile::TensorViewType>(
      loc, parser.getContext(), elementType, shape, strides);
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
                                  ArrayRef<int64_t> stride) {
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

//===----------------------------------------------------------------------===//
// PartitionView Type
//===----------------------------------------------------------------------===//

Type cuda_tile::PartitionViewType::parse(AsmParser &parser) {
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

  TensorViewType tensor_view;
  if (parseCudaTileType(parser, tensor_view))
    return Type();

  SmallVector<int32_t> dimMap;
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
      return Type();
  } else {
    // By default, dimMap is the identity mapping.
    for (int32_t i : llvm::seq(tileShape.size()))
      dimMap.push_back(i);
  }

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

  return parser.getChecked<cuda_tile::PartitionViewType>(
      loc, parser.getContext(),
      DenseI32ArrayAttr::get(parser.getContext(), tileShape32), tensor_view,
      dimMap, *paddingValue);
}

void cuda_tile::PartitionViewType::print(AsmPrinter &printer) const {
  printer << "<";
  printer << "tile=(";
  llvm::interleave(getTileShape().asArrayRef(), printer, "x");
  printer << "), ";
  if (getPaddingValue())
    printer << "padding_value = "
            << stringifyPaddingValue(getPaddingValue().getValue()) << ", ";

  printCudaTileType(printer, getTensorView());

  // Only print mapping if non-trivial.
  bool isIdentityMapping =
      llvm::equal(getDimMap(), llvm::seq(getTileShape().size()));
  if (!isIdentityMapping)
    printer << ", dim_map=[" << getDimMap() << "]";

  printer << ">";
}

LogicalResult cuda_tile::PartitionViewType::verify(
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
             << tileShape.size() << ", got " << tensorViewDim;

    if (usedTensorViewDim[tensorViewDim].has_value())
      return emitError() << "target dimension " << tensorViewDim
                         << " mapped at least twice (for tile dimensions "
                         << usedTensorViewDim[tensorViewDim].value() << " and "
                         << tileDim << ")";

    usedTensorViewDim[tensorViewDim] = tileDim;
  }

  // Run the Tile type verifier to catch invalid tiles in the partition type
  SmallVector<int64_t> shape64(tileShape.begin(), tileShape.end());
  if (failed(cuda_tile::TileType::verify([&]() { return emitError(); }, shape64,
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

size_t cuda_tile::PartitionViewType::getViewIndexRank() const {
  return getTileShape().size();
}

Type cuda_tile::PartitionViewType::getViewTileType() const {
  SmallVector<int64_t> shape64(getTileShape().size());
  llvm::transform(getTileShape().asArrayRef(), shape64.begin(),
                  [](int32_t x) -> int64_t { return x; });
  return cuda_tile::TileType::get(shape64, getTensorView().getElementType());
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
