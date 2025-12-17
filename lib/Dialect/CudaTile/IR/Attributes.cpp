//===- Attributes.cpp - CUDA Tile Attribute Verifiers -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <optional>

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc"

LogicalResult OptimizationHintsAttr::verifyParamWithContext(
    llvm::function_ref<InFlightDiagnostic()> emitError, llvm::StringRef context,
    ArrayRef<StringRef> allowedKeys, DictionaryAttr &attr) {
  for (auto param : attr) {
    llvm::StringRef key = param.getName().strref();
    if (!allowedKeys.empty() && !llvm::is_contained(allowedKeys, key))
      return emitError() << key << " is not allowed for current Operation";
    if (key == kNumCTAInCGA) {
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(param.getValue())) {
        uint64_t numCTA = intAttr.getInt();
        if ((numCTA == 0) || (numCTA > 16) || ((numCTA & (numCTA - 1)) != 0))
          return emitError()
                 << "expected power-of-two ≤ 16 for " << context << "." << key;
      } else {
        return emitError() << "integer value expected for " << context << "."
                           << key;
      }
    } else if (key == kAllowTMA) {
      if (!llvm::dyn_cast_or_null<BoolAttr>(param.getValue()))
        return emitError() << "boolean value expected for " << context << "."
                           << key;
    } else if (key == kLatency) {
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(param.getValue())) {
        int64_t val = intAttr.getInt();
        if ((val < 1) || (val > 10))
          return emitError()
                 << "integer value in the range [1, 10] is expected for "
                 << context << "." << key;
      } else {
        return emitError() << "integer value expected for " << context << "."
                           << key;
      }
    } else if (key == kOccupancy) {
      if (auto intAttr =
              llvm::dyn_cast_or_null<IntegerAttr>(param.getValue())) {
        int64_t val = intAttr.getInt();
        if ((val < 1) || (val > 32))
          return emitError()
                 << "integer value in the range [1, 32] is expected for "
                 << context << "." << key;
      } else {
        return emitError() << "integer value expected for " << context << "."
                           << key;
      }
    } else {
      return emitError() << "unknown param " << key << " for " << context;
    }
  }
  return success();
}

LogicalResult OptimizationHintsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, DictionaryAttr value) {
  return verifyWithOp(nullptr, emitError, value);
}

LogicalResult OptimizationHintsAttr::verifyWithOp(
    Operation *op, llvm::function_ref<InFlightDiagnostic()> emitError,
    DictionaryAttr value) {
  llvm::SmallVector<llvm::StringRef, 4> allowedKeys;
  if (op != nullptr) {
    // Initialize list of supported hints for EntryOp
    if (llvm::isa<EntryOp>(op)) {
      allowedKeys.push_back(kNumCTAInCGA);
      allowedKeys.push_back(kOccupancy);
    }
    // Initialize list of supported hints for Load/Store Ops
    if (llvm::isa<LoadViewTkoOp, StoreViewTkoOp, LoadPtrTkoOp, StorePtrTkoOp>(
            op)) {
      allowedKeys.push_back(kLatency);
      if (llvm::isa<LoadViewTkoOp, StoreViewTkoOp>(op))
        allowedKeys.push_back(kAllowTMA);
    }
  }
  for (NamedAttribute entry : value.getValue()) {
    llvm::StringRef key = entry.getName().strref();
    if (!isAllowedKey(key))
      return emitError() << "unallowed key " << key;
    auto innerDict = llvm::dyn_cast_or_null<DictionaryAttr>(entry.getValue());
    if (!innerDict)
      return emitError()
             << "expected dictionary attribute for optimization_hints entry `"
             << key << "` got value=" << entry.getValue();
    if (innerDict)
      if (failed(
              verifyParamWithContext(emitError, key, allowedKeys, innerDict)))
        return failure();
  }
  return success();
}

std::optional<int> OptimizationHintsAttr::getNumCTAInCGA(StringRef sm) {
  std::optional<int> res = std::nullopt;
  if (!getValue().empty()) {
    auto smEntry = llvm::dyn_cast_or_null<DictionaryAttr>(getValue().get(sm));
    if (smEntry)
      if (auto numCTAInCGA =
              llvm::dyn_cast_or_null<IntegerAttr>(smEntry.get(kNumCTAInCGA)))
        res = numCTAInCGA.getInt();
  }
  return res;
}

std::optional<bool> OptimizationHintsAttr::getAllowTMA(StringRef sm) {
  std::optional<bool> res = std::nullopt;
  if (!getValue().empty()) {
    auto smEntry = llvm::dyn_cast_or_null<DictionaryAttr>(getValue().get(sm));
    if (smEntry)
      if (auto allowTMA =
              llvm::dyn_cast_or_null<BoolAttr>(smEntry.get(kAllowTMA)))
        res = allowTMA.getValue();
  }
  return res;
}

std::optional<int> OptimizationHintsAttr::getLatency(StringRef sm) {
  std::optional<int> res = std::nullopt;
  if (!getValue().empty()) {
    auto smEntry = llvm::dyn_cast_or_null<DictionaryAttr>(getValue().get(sm));
    if (smEntry)
      if (auto latency =
              llvm::dyn_cast_or_null<IntegerAttr>(smEntry.get(kLatency)))
        res = latency.getInt();
  }
  return res;
}

std::optional<int> OptimizationHintsAttr::getOccupancy(StringRef sm) {
  std::optional<int> res = std::nullopt;
  if (!getValue().empty()) {
    auto smEntry = llvm::dyn_cast_or_null<DictionaryAttr>(getValue().get(sm));
    if (smEntry)
      if (auto cost =
              llvm::dyn_cast_or_null<IntegerAttr>(smEntry.get(kOccupancy)))
        res = cost.getInt();
  }
  return res;
}

Attribute OptimizationHintsAttr::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLess())
    return {};
  if (llvm::succeeded(parser.parseOptionalGreater()))
    return OptimizationHintsAttr::get(parser.getContext(),
                                      DictionaryAttr::get(parser.getContext()));

  NamedAttrList entries;
  auto parseOneEntry = [&]() -> ParseResult {
    std::string key;
    Attribute rawAttr;
    DictionaryAttr dataDict;
    if (parser.parseKeywordOrString(&key) || parser.parseEqual() ||
        parser.parseAttribute(rawAttr))
      return failure();

    if (entries.get(key))
      return parser.emitError(parser.getCurrentLocation())
             << "duplicate optimization_hints key `" << key << "`";

    if (!isAllowedKey(key))
      return parser.emitError(parser.getCurrentLocation())
             << "unallowed key " << key;

    dataDict = llvm::dyn_cast_or_null<DictionaryAttr>(rawAttr);
    if (!dataDict)
      return parser.emitError(parser.getCurrentLocation())
             << "expected dictionary attribute for optimization_hints entry `"
             << key << "` got value=" << rawAttr;

    if (failed(verifyParamWithContext(
            [&]() -> InFlightDiagnostic {
              return parser.emitError(parser.getCurrentLocation());
            },
            key, {}, dataDict))) {
      return failure();
    }

    entries.append(key, dataDict);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::None, parseOneEntry))
    return {};
  if (parser.parseGreater())
    return {};

  return OptimizationHintsAttr::get(
      parser.getContext(), parser.getBuilder().getDictionaryAttr(entries));
}

void OptimizationHintsAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getValue(), printer, [&](NamedAttribute attr) {
    printer << attr.getName().strref() << " = {";
    llvm::interleaveComma(mlir::cast<DictionaryAttr>(attr.getValue()), printer,
                          [&](NamedAttribute na) {
                            printer << na.getName().strref() << " = ";
                            printer.printAttributeWithoutType(na.getValue());
                          });
    printer << "}";
  });
  printer << ">";
}

LogicalResult DivByAttr::verifyWithAssumeOp(Operation *op) const {
  auto assumeOp = llvm::cast<AssumeOp>(op);

  // Make sure divisor is a positive power of 2.
  uint64_t divisor = getDivisor();
  bool isPowerOfTwo = divisor > 0 && ((divisor & (divisor - 1)) == 0);
  if (!isPowerOfTwo)
    return op->emitOpError() << "'" << name << "' divisor must be a power of 2";

  if (!llvm::all_equal({getEvery().has_value(), getAlong().has_value()}))
    return op->emitOpError()
           << "'" << name << "' 'every'/'along' must be used in combination";

  // Verify that the divisor is not larger than 4611686018427387904. This is a
  // technical limitation of the current implementation that could be lifted.
  if (divisor > 4611686018427387904)
    return op->emitOpError() << "'" << name << "' divisor is too large";

  // TensorViewType
  if (auto tensorViewType =
          llvm::dyn_cast<cuda_tile::TensorViewType>(assumeOp.getType())) {
    if (getEvery().has_value())
      return op->emitOpError() << "'" << name
                               << "' 'every'/'along' cannot be used if the "
                                  "constrained value is a tensor_view";
    return success();
  }

  // TileType
  auto tileType = llvm::dyn_cast<cuda_tile::TileType>(assumeOp.getType());
  if (!tileType)
    return op->emitOpError() << "'" << name
                             << "' is valid only for tile of integer/pointer "
                                "or tensor_view values";
  if (tileType.getRank() == 0 && getEvery().has_value())
    return op->emitOpError() << "'" << name
                             << "' 'every'/'along' cannot be used if the "
                                "constrained value is a 0D tile";
  Type elType = tileType.getElementType();
  if (!llvm::isa<cuda_tile::PointerType, IntegerType>(elType))
    return op->emitOpError() << "'" << name
                             << "' is valid only for tile of integer/pointer "
                                "or tensor_view values";

  // Verify every/along.
  if (!getEvery().has_value())
    return success();
  if (*getAlong() < 0 || *getAlong() >= tileType.getRank())
    return op->emitOpError()
           << "'" << name << "' every_dim (" << *getAlong()
           << ") must be >= 0 and < tile rank (" << tileType.getRank() << ")";
  if (*getEvery() < 0 || *getEvery() > tileType.getDimSize(*getAlong()))
    return op->emitOpError() << "expected '" << name
                             << "' every_dim to be within 0 and the size of "
                                "the respective dimension ("
                             << tileType.getDimSize(*getAlong()) << ")";
  return success();
}

Attribute DivByAttr::parse(AsmParser &parser, Type odsType) {
  // Parse literal '<'.
  if (parser.parseLess())
    return {};

  // Parse variable 'divisor'.
  uint64_t divisor = 0;
  if (parser.parseInteger(divisor)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse parameter 'divisor' which is expected to "
                     "be an integer");
    return {};
  }

  // Parse 'every' and 'along'.
  std::optional<int64_t> every = std::nullopt;
  std::optional<int64_t> along = std::nullopt;
  if (succeeded(parser.parseOptionalComma())) {
    // Parse optional every/along.
    int64_t everyVal = -1, alongVal = -1;
    if (parser.parseKeyword("every") || parser.parseInteger(everyVal) ||
        parser.parseKeyword("along") || parser.parseInteger(alongVal))
      return {};
    every = everyVal;
    along = alongVal;
  }

  // Parse literal '>'.
  if (parser.parseGreater())
    return {};

  return DivByAttr::get(parser.getContext(), divisor, every, along);
}

void DivByAttr::print(AsmPrinter &printer) const {
  printer << "<" << getDivisor();
  if (getEvery().has_value())
    printer << ", every " << *getEvery() << " along " << *getAlong();
  printer << ">";
}

LogicalResult SameElementsAttr::verifyWithAssumeOp(Operation *op) const {
  auto assumeOp = llvm::cast<AssumeOp>(op);
  auto tileType = llvm::dyn_cast<cuda_tile::TileType>(assumeOp.getType());
  if (!tileType)
    return op->emitOpError()
           << "'" << name
           << "' is valid only for tile of integer/pointer values";
  if (!llvm::isa<cuda_tile::PointerType, IntegerType>(
          tileType.getElementType()))
    return op->emitOpError()
           << "'" << name
           << "' is valid only for tile of integer/pointer values";
  if (getValues().size() != tileType.getRank())
    return op->emitOpError()
           << "expected number of values in '" << name << "' ("
           << getValues().size() << ") to match rank of constrained tile ("
           << tileType.getRank() << ")";
  for (int64_t i = 0, e = tileType.getRank(); i < e; ++i) {
    if (getValues()[i] < 0 || getValues()[i] > tileType.getDimSize(i))
      return op->emitOpError()
             << "expected '" << name << "' value " << i
             << " to be within 0 and the size of the respective dimension ("
             << tileType.getDimSize(i) << ")";
  }
  return success();
}

LogicalResult BoundedAttr::verifyWithAssumeOp(Operation *op) const {
  auto tileType =
      llvm::dyn_cast<cuda_tile::TileType>(llvm::cast<AssumeOp>(op).getType());
  if (!tileType)
    return op->emitOpError()
           << "'" << name << "' is valid only for tile of integer values";
  auto intType = llvm::dyn_cast<IntegerType>(tileType.getElementType());
  if (!intType)
    return op->emitOpError()
           << "'" << name << "' is valid only for tile of integer values";
  int64_t minVal = getMinSignedValueForBitwidth(intType.getWidth());
  int64_t maxVal = getMaxSignedValueForBitwidth(intType.getWidth());
  if (getLb().has_value() && (*getLb() > maxVal || *getLb() < minVal))
    return op->emitOpError()
           << "'" << name << "' expects lower bound to be within [" << minVal
           << ", " << maxVal << "]";
  if (getUb().has_value() && (*getUb() > maxVal || *getUb() < minVal))
    return op->emitOpError()
           << "'" << name << "' expects upper bound to be within [" << minVal
           << ", " << maxVal << "]";
  if (getLb().has_value() && getUb().has_value() && *getLb() > *getUb())
    return op->emitOpError()
           << "'" << name
           << "' expects lower bound to be less than or equal to upper bound";
  return success();
}

//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILexicalBlockAttr,
                   DISubprogramAttr>(attr);
}

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILocalScopeAttr>(attr);
}

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DISubprogramAttr>(attr);
}

void CudaTileDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc"
      >();
}
