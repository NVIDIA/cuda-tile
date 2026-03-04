//===- BytecodeGenUtilities.cpp ---------------------------------*- C++ -*-===//
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

static std::pair<std::string, std::string>
parseVersionString(StringRef version) {
  const size_t dotPos = version.find('.');
  const StringRef majorStr = version.substr(0, dotPos);
  const StringRef minorStr =
      (dotPos != StringRef::npos) ? version.substr(dotPos + 1) : "0";
  return {majorStr.str(), minorStr.str()};
}

std::pair<std::string, std::string>
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

      const std::string version =
          decorator.getDef().getValueAsString("sinceVersion").str();
      return parseVersionString(version);
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

std::string mlir::tblgen::extractVersionFromOperation(const Operator &op) {
  const auto &def = op.getDef();
  if (const auto *metadata = def.getValueAsOptionalDef("metadata"))
    return metadata->getValueAsString("sinceVersion").str();

  PrintFatalError(op.getLoc(), "operation '" + op.getOperationName() +
                                   "' is missing version metadata");
}

std::pair<std::string, std::string>
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

          const std::string version =
              decorator.getDef().getValueAsString("sinceVersion").str();
          return parseVersionString(version);
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

  // Path for public operations: version-ordered assignment.
  struct VersionKey {
    int major;
    int minor;
    bool operator<(const VersionKey &other) const {
      return std::tie(major, minor) < std::tie(other.major, other.minor);
    }
  };
  std::map<VersionKey, std::vector<StringRef>> versionGroups;

  // Group optional attributes by version (attributes processed first within
  // each version).
  for (const auto &namedAttr : op.getAttributes()) {
    if (namedAttr.attr.isOptional()) {
      auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
      VersionKey version{std::stoi(majorStr), std::stoi(minorStr)};
      versionGroups[version].push_back(namedAttr.name);
    }
  }

  // Group optional operands by version (operands processed second within each
  // version).
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    for (const auto &[operandIndex, odsOperand] :
         llvm::enumerate(op.getOperands())) {
      if (odsOperand.isOptional()) {
        auto [majorStr, minorStr] = extractVersionFromOperand(operandIndex, op);
        VersionKey version{std::stoi(majorStr), std::stoi(minorStr)};
        versionGroups[version].push_back(odsOperand.name);
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

std::pair<std::string, std::string>
mlir::tblgen::extractVersionFromResult(unsigned resultIndex,
                                       const Operator &op) {
  assert(resultIndex < static_cast<unsigned>(op.getNumResults()) &&
         "TableGen backend bug: result index out of bounds");

  // Look for version metadata in result decorators.
  for (const auto &decorator : op.getResultDecorators(resultIndex)) {
    if (!decorator.getDef().isSubClassOf("CudaTileArgMetadata"))
      continue;

    const std::string version =
        decorator.getDef().getValueAsString("sinceVersion").str();
    return parseVersionString(version);
  }

  // Result missing required metadata.
  const auto &result = op.getResult(resultIndex);
  PrintFatalError(op.getLoc(),
                  "result '" + result.name.str() + "' in operation '" +
                      op.getOperationName() +
                      "' is missing version metadata (CudaTileArgMetadata)");
}
