//===- Attributes.cpp - CUDA Tile Attribute Verifiers -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
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
#include "cuda_tile/Dialect/CudaTile/IR/Remark.h"
#include <optional>
#include <string>
#include <type_traits>

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Optimization Hints Validation Helpers
//===----------------------------------------------------------------------===//

namespace {
/// Result of validation - contains optional value and error message
template <typename T>
struct ValidationResult {
  std::optional<T> value;
  std::string errorMessage;

  bool isValid() const { return value.has_value(); }
  explicit operator bool() const { return isValid(); }
};
} // namespace

static InFlightDiagnostic
emitDiagnostic(Location loc, StringRef msg = StringRef(),
               DiagnosticSeverity severity = DiagnosticSeverity::Warning) {
  return severity == DiagnosticSeverity::Error ? emitError(loc, msg)
                                               : emitWarning(loc, msg);
}

// Validators are auto-generated from HintKey enum metadata in AttrDefs.td.
#define CUDA_TILE_HINT_VALIDATORS
#include "cuda_tile/Dialect/CudaTile/IR/HintKeyImpl.inc"
#undef CUDA_TILE_HINT_VALIDATORS

/// Retrieve and validate a hint value from the optimization_hints dictionary.
///
/// Looks up the hint in the SM-specific entry first (e.g. "sm_100"), falling
/// back to "default". Validates the attribute using the provided validator and
/// casts the result to the requested return type. When \p op is non-null,
/// emits remarks for "used default" (value came from default entry) or
/// "clamped" (validator returned a value but with a non-empty error message).
///
/// \tparam RetT        The desired return type (e.g., int, bool, std::string).
/// \tparam ValidateRetT The type returned by the validator (typically int64_t).
/// \param value    The optimization_hints DictionaryAttr.
/// \param sm       Target SM string (e.g. "sm_100").
/// \param key      The HintKey enum value to look up.
/// \param validate Validator function for this hint type.
/// \param op       When non-null, used to emit optimization hint remarks.
/// \return The validated hint value, or std::nullopt if absent or invalid.
template <typename RetT, typename ValidateRetT>
static std::optional<RetT>
getHintValue(DictionaryAttr value, StringRef sm, HintKey key,
             ValidationResult<ValidateRetT> (*validate)(Attribute, StringRef),
             Operation *op = nullptr) {
  if (value.empty())
    return std::nullopt;
  Attribute attr;
  bool usedDefault = false;
  if (auto smEntry = value.getAs<DictionaryAttr>(sm))
    attr = smEntry.get(stringifyHintKey(key));
  if (!attr) {
    if (auto defEntry = value.getAs<DictionaryAttr>("default")) {
      attr = defEntry.get(stringifyHintKey(key));
      usedDefault = (attr != nullptr);
    }
  }
  if (!attr)
    return std::nullopt;
  auto result = validate(attr, sm);
  if (op != nullptr) {
    if (result.value) {
      if (!result.errorMessage.empty()) {
        cuda_tile::remark::reportRemark(
            op, cuda_tile::remark::RemarkID::RemarkHintOutOfRange_Missed,
            result.errorMessage);
      } else if (usedDefault) {
        cuda_tile::remark::reportRemark(
            op, cuda_tile::remark::RemarkID::RemarkHintDefault_Succeeded,
            stringifyHintKey(key));
      } else {
        cuda_tile::remark::reportRemark(
            op, cuda_tile::remark::RemarkID::RemarkHint_Succeeded,
            stringifyHintKey(key));
      }
      // Validators return int64_t, bool, or std::string; callers request RetT.
      // Guard that the return type can represent all valid hint values.
      static_assert(std::is_same_v<RetT, bool> ||
                        std::is_same_v<RetT, std::string> ||
                        sizeof(RetT) >= sizeof(int32_t),
                    "RetT must be large enough to hold hint values");
      return static_cast<RetT>(*result.value);
    }
    cuda_tile::remark::reportRemark(
        op, cuda_tile::remark::RemarkID::RemarkHint_Failed,
        stringifyHintKey(key));
  }
  return std::nullopt;
}

/// Validate a single hint attribute using the provided validator callback.
///
/// Calls \p validate on \p value with \p context.  If the validator returns
/// a non-empty errorMessage, emits a diagnostic at \p loc and sets \p res
/// to the diagnostic result (warning or error, depending on dialect config).
///
/// \tparam ValidateRetT The value type produced by the validator (e.g.
/// int64_t).
/// \param value    The hint Attribute to validate.
/// \param context  Validation context string (typically the architecture key).
/// \param loc      Location used for diagnostic emission.
/// \param res      LogicalResult reference; set to failure on validation error.
/// \param validate Validator function returning ValidationResult<ValidateRetT>.
template <typename ValidateRetT>
static void verifyOneHint(
    Attribute value, StringRef context, Location loc, LogicalResult &res,
    ValidationResult<ValidateRetT> (*validate)(Attribute, StringRef)) {
  auto result = validate(value, context);
  if (!result.errorMessage.empty()) {
    res = emitDiagnostic(loc) << result.errorMessage;
  }
}

// Return failure() if hints are not supported for current operations/target
LogicalResult OptimizationHintsAttr::verifyParamWithContext(
    Location loc, StringRef context, ArrayRef<StringRef> keysValidForOperation,
    DictionaryAttr &attr) {
  // Fast return if not warning about hints
  if (!cast<CudaTileDialect>(getDialect()).getWarnUnsupportedHints())
    return success();

  // Handle unallowed architecture key
  if (!isKnownKey(context))
    return emitDiagnostic(loc) << "unknown hint key " << context;

  LogicalResult res = success();
  for (auto param : attr) {
    StringRef key = param.getName().strref();

    if (!keysValidForOperation.empty() &&
        !is_contained(keysValidForOperation, key)) {
      res = emitDiagnostic(loc)
            << key << " is not known hint for current Operation";
      continue;
    }

#define CUDA_TILE_HINT_KEY(Name, Str, RetType, Version)                        \
  if (key == stringifyHintKey(HintKey::Name)) {                                \
    verifyOneHint(param.getValue(), context, loc, res, validate##Name);        \
  } else
#include "cuda_tile/Dialect/CudaTile/IR/HintKeyAccessors.inc"
    {
      res = emitDiagnostic(loc)
            << "unknown param " << key << " for " << context;
    }
  }

  // Return failure if warnings were printed
  // Error (if errorOnHints not set) would be suppressed outside
  return res;
}

LogicalResult
OptimizationHintsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DictionaryAttr value) {
  for (NamedAttribute entry : value.getValue()) {
    if (!isa<DictionaryAttr>(entry.getValue()))
      return emitError()
             << "expected dictionary attribute for optimization_hints entry `"
             << entry.getName().strref() << "` got value=" << entry.getValue();
  }
  return success();
}

LogicalResult OptimizationHintsAttr::verifyWithOp(Operation *op,
                                                  DictionaryAttr value) {
  Location loc = op->getLoc();
  bool errorOnHints = cast<CudaTileDialect>(getDialect()).getErrorOnHints();
  SmallVector<StringRef, 4> keysValidForOperation;
  if (op != nullptr) {
#define CUDA_TILE_HINT_OP_KEYS
#include "cuda_tile/Dialect/CudaTile/IR/HintKeyImpl.inc"
#undef CUDA_TILE_HINT_OP_KEYS
  }

  for (NamedAttribute entry : value.getValue()) {
    StringRef key = entry.getName().strref();

    auto innerDict = dyn_cast<DictionaryAttr>(entry.getValue());
    if (!innerDict)
      return op->emitOpError()
             << "expected dictionary attribute for optimization_hints entry `"
             << key << "` got value=" << entry.getValue();

    if (failed(verifyParamWithContext(loc, key, keysValidForOperation,
                                      innerDict))) {
      if (errorOnHints)
        return failure();
    }
  }

  return success();
}

// Getter definitions — template + X-macro expansion from HintKeyAccessors.inc.
#define CUDA_TILE_HINT_KEY(Name, Str, RetType, Version)                        \
  std::optional<RetType> OptimizationHintsAttr::get##Name(StringRef sm,        \
                                                          Operation *op) {     \
    return getHintValue<RetType>(getValue(), sm, HintKey::Name,                \
                                 validate##Name, op);                          \
  }
#include "cuda_tile/Dialect/CudaTile/IR/HintKeyAccessors.inc"

Attribute OptimizationHintsAttr::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLess())
    return {};
  if (succeeded(parser.parseOptionalGreater()))
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

    dataDict = dyn_cast<DictionaryAttr>(rawAttr);
    if (!dataDict)
      return parser.emitError(parser.getCurrentLocation())
             << "expected dictionary attribute for optimization_hints entry `"
             << key << "` got value=" << rawAttr;

    entries.append(key, dataDict);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::None,
                                     parseOneEntry))
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
