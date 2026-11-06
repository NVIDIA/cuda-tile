//===- BytecodeGenUtilities.cpp - Bytecode Gen Utilities --------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common utilities used across multiple bytecode
// generation TableGen backends for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#include "BytecodeGenUtilities.h"

#include "llvm/TableGen/Error.h"

#include <map>
#include <string>
#include <tuple>

using namespace llvm;
using namespace mlir;

std::pair<StringRef, StringRef> mlir::tblgen::parseVersion(StringRef version) {
  const size_t dotPos = version.find('.');
  const StringRef majorStr = version.substr(0, dotPos);
  const StringRef minorStr =
      (dotPos != StringRef::npos) ? version.substr(dotPos + 1) : "0";
  return {majorStr, minorStr};
}

std::pair<StringRef, StringRef>
mlir::tblgen::extractVersionFromAttribute(const NamedAttribute &namedAttr,
                                          const Operator &op) {
  const StringRef attrName = namedAttr.name;

  // Search through operation arguments for matching attribute.
  for (unsigned i = 0, e = op.getNumArgs(); i != e; ++i) {
    const auto arg = op.getArg(i);
    const auto *argAttr = arg.dyn_cast<NamedAttribute *>();
    if (!argAttr || argAttr->name != attrName)
      continue;

    // Found matching attribute - look for version metadata
    for (const auto &decorator : op.getArgDecorators(i)) {
      if (!decorator.getDef().isSubClassOf("CudaTileArgMetadata"))
        continue;

      StringRef version = decorator.getDef().getValueAsString("sinceVersion");
      return parseVersion(version);
    }

    // Found attribute but missing required metadata.
    PrintFatalError(op.getLoc(),
                    "attribute '" + attrName.str() + "' in operation '" +
                        op.getOperationName() +
                        "' is missing version metadata (CudaTileArgMetadata)");
  }

  // Attribute not found in operation arguments.
  PrintFatalError(op.getLoc(), "attribute '" + attrName.str() +
                                   "' not found in operation '" +
                                   op.getOperationName() + "'");
}

std::optional<std::string>
mlir::tblgen::extractDefaultValue(const NamedAttribute &namedAttr) {
  if (namedAttr.attr.hasDefaultValue())
    return namedAttr.attr.getDefaultValue().str();
  return std::nullopt;
}

std::optional<std::string>
mlir::tblgen::extractSameOperandRankName(const NamedAttribute &namedAttr,
                                         const Operator &op) {
  const StringRef attrName = namedAttr.name;

  // Search through operation arguments for matching attribute.
  for (unsigned i = 0, e = op.getNumArgs(); i != e; ++i) {
    const auto arg = op.getArg(i);
    const auto *argAttr = arg.dyn_cast<NamedAttribute *>();
    if (!argAttr || argAttr->name != attrName) {
      continue;
    }

    for (const auto &decorator : op.getArgDecorators(i)) {
      if (!decorator.getDef().isSubClassOf("RequireSameOperandRank")) {
        continue;
      }
      return decorator.getDef().getValueAsString("rankOperand").str();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::string> mlir::tblgen::getAttributeEncodingRequiredPredicate(
    const NamedAttribute &namedAttr, const Operator &op, StringRef opExpr) {
  const std::string getter = op.getGetterName(namedAttr.name);

  if (std::optional<std::string> defaultValue =
          extractDefaultValue(namedAttr)) {
    return formatv("{0}.{1}() != {2}", opExpr, getter, *defaultValue).str();
  }

  // A `RequireSameOperandRank` attribute has no ODS default because its shape
  // depends on the operand: a reader predating it rebuilds an all-`false`
  // vector sized from that operand, so only a set bit needs the field.
  if (extractSameOperandRankName(namedAttr, op)) {
    return formatv("::llvm::is_contained({0}.{1}(), true)", opExpr, getter)
        .str();
  }

  return std::nullopt;
}

std::string mlir::tblgen::extractVersionFromOperation(const Operator &op) {
  const auto &def = op.getDef();
  if (const auto *metadata = def.getValueAsOptionalDef("metadata"))
    return metadata->getValueAsString("sinceVersion").str();

  PrintFatalError(op.getLoc(), "operation '" + op.getOperationName() +
                                   "' is missing version metadata");
}

bool mlir::tblgen::isPreviewOperation(const Operator &op) {
  return op.getDef().isSubClassOf("CudaTilePreviewOpDef");
}

StringRef mlir::tblgen::getOwningPreviewFeatureName(const Record *record) {
  const Record *feature = record->getValueAsDef("previewFeature");
  return feature->getValueAsString("cppName");
}

bool mlir::tblgen::isVersionlessOperation(const Operator &op) {
  if (isPreviewOperation(op))
    return true;
  return false;
}

bool mlir::tblgen::isExternallyEncodedOpAttribute(StringRef attrName) {
  // Function symbol visibility is encoded in the function-table entry flags,
  // not in the generated operation payload.
  return attrName == "sym_visibility";
}

std::pair<StringRef, StringRef>
mlir::tblgen::extractVersionFromOperand(unsigned operandIndex,
                                        const Operator &op) {
  assert(operandIndex < static_cast<unsigned>(op.getNumOperands()) &&
         "TableGen backend bug: operand index out of bounds");

  // Find argument index by scanning for operands only.
  uint32_t currentOperandIndex = 0;
  for (int argIndex = 0; argIndex < op.getNumArgs(); ++argIndex) {
    auto argToOpOrAttr = op.getArgToOperandAttrOrProp(argIndex);

    // Check if this argument is an operand.
    if (argToOpOrAttr.kind() == Operator::OperandAttrOrProp::Kind::Operand) {
      if (currentOperandIndex == operandIndex) {
        // Found the argument index for this operand - get its decorators.
        for (const auto &decorator : op.getArgDecorators(argIndex)) {
          if (!decorator.getDef().isSubClassOf("CudaTileArgMetadata"))
            continue;

          StringRef version =
              decorator.getDef().getValueAsString("sinceVersion");
          return parseVersion(version);
        }

        // Found operand but no metadata.
        const auto &operand = op.getOperand(operandIndex);
        PrintFatalError(
            op.getLoc(),
            "operand '" + operand.name.str() + "' in operation '" +
                op.getOperationName() +
                "' is missing version metadata (CudaTileArgMetadata)");
      }
      ++currentOperandIndex;
    }
  }

  // Operand not found in operation arguments.
  const auto &operand = op.getOperand(operandIndex);
  PrintFatalError(op.getLoc(), "operand '" + operand.name.str() +
                                   "' not found in operation '" +
                                   op.getOperationName() + "'");
}

std::pair<StringMap<size_t>, std::optional<std::pair<std::string, std::string>>>
mlir::tblgen::getVersionOrderedBitAssignments(const Operator &op) {
  StringMap<size_t> bitAssignments;
  std::optional<std::pair<std::string, std::string>> minVersion;
  size_t bitIndex = 0;

  // Fast path for versionless operations: simple declaration-order assignment
  // with no version tracking.
  if (isVersionlessOperation(op)) {
    // Assign bits to optional attributes first (declaration order).
    for (const auto &namedAttr : op.getAttributes())
      if (namedAttr.attr.isOptional() &&
          !isExternallyEncodedOpAttribute(namedAttr.name)) {
        bitAssignments[namedAttr.name.str()] = bitIndex++;
      }

    // Then assign bits to optional operands (declaration order).
    if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments"))
      for (const auto &odsOperand : op.getOperands())
        if (odsOperand.isOptional())
          bitAssignments[odsOperand.name.str()] = bitIndex++;

    return {bitAssignments, std::nullopt};
  }

  // Path for public operations: version-ordered assignment.
  struct VersionKey {
    int major;
    int minor;
    bool operator<(const VersionKey &other) const {
      return std::tie(major, minor) < std::tie(other.major, other.minor);
    }
  };
  std::map<VersionKey, std::vector<StringRef>> versionGroups;

  // Helper to parse version string to VersionKey.
  auto parseToVersionKey = [&op](StringRef majorStr, StringRef minorStr) {
    int major, minor;
    if (majorStr.getAsInteger(10, major) || minorStr.getAsInteger(10, minor)) {
      PrintFatalError(op.getLoc(), "invalid version string in operation '" +
                                       op.getOperationName() + "': " +
                                       majorStr.str() + "." + minorStr.str());
    }
    return VersionKey{major, minor};
  };

  // Group optional attributes by version (attributes processed first within
  // each version).
  for (const auto &namedAttr : op.getAttributes()) {
    if (namedAttr.attr.isOptional() &&
        !isExternallyEncodedOpAttribute(namedAttr.name)) {
      auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
      versionGroups[parseToVersionKey(majorStr, minorStr)].push_back(
          namedAttr.name);
    }
  }

  // Group optional operands by version (operands processed second within each
  // version).
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    for (const auto &[operandIndex, odsOperand] :
         llvm::enumerate(op.getOperands())) {
      if (odsOperand.isOptional()) {
        auto [majorStr, minorStr] = extractVersionFromOperand(operandIndex, op);
        versionGroups[parseToVersionKey(majorStr, minorStr)].push_back(
            odsOperand.name);
      }
    }
  }

  // Capture the minimum version (first key in versionGroups).
  if (!versionGroups.empty()) {
    const auto &firstVersion = versionGroups.begin()->first;
    minVersion = {std::to_string(firstVersion.major),
                  std::to_string(firstVersion.minor)};
  }

  // Assign bit indices in version order.
  for (const auto &[version, names] : versionGroups)
    for (StringRef name : names)
      bitAssignments[name.str()] = bitIndex++;

  return {bitAssignments, minVersion};
}

std::pair<StringRef, StringRef>
mlir::tblgen::extractVersionFromResult(unsigned resultIndex,
                                       const Operator &op) {
  assert(resultIndex < static_cast<unsigned>(op.getNumResults()) &&
         "TableGen backend bug: result index out of bounds");

  // Look for version metadata in result decorators.
  for (const auto &decorator : op.getResultDecorators(resultIndex)) {
    if (!decorator.getDef().isSubClassOf("CudaTileArgMetadata"))
      continue;

    StringRef version = decorator.getDef().getValueAsString("sinceVersion");
    return parseVersion(version);
  }

  // Result missing required metadata.
  const auto &result = op.getResult(resultIndex);
  PrintFatalError(op.getLoc(),
                  "result '" + result.name.str() + "' in operation '" +
                      op.getOperationName() +
                      "' is missing version metadata (CudaTileArgMetadata)");
}

std::string
mlir::tblgen::replaceTableGenPlaceholders(StringRef code, StringRef ctxtVar,
                                          ArrayRef<ParamSubst> paramSubsts) {
  FmtContext ctx;
  ctx.addSubst("_ctxt", ctxtVar);
  for (const auto &subst : paramSubsts) {
    ctx.addSubst(("_param_" + subst.paramName).str(), subst.expr);
  }
  return tgfmt(code, &ctx);
}
