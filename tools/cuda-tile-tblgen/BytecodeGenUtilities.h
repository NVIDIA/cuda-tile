//===- BytecodeGenUtilities.h -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities used across multiple bytecode generation
// TableGen backends for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODEGEN_UTILITIES_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODEGEN_UTILITIES_H_

#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/TableGen/Error.h"

#include <optional>
#include <string>

namespace mlir {
namespace tblgen {

/// Extract version information from an attribute's TableGen metadata.
std::pair<std::string, std::string>
extractVersionFromAttribute(const NamedAttribute &namedAttr,
                            const Operator &op);

/// Extract the default value from an attribute if it has one.
std::optional<std::string> extractDefaultValue(const NamedAttribute &namedAttr);

/// Extract the version string from an operation's metadata.
std::string extractVersionFromOperation(const Operator &op);

/// Get version-ordered bit assignments for optional fields.
/// Returns map from field name to bit position, and optionally the earliest
/// version among all optional fields (if any exist).
std::pair<llvm::StringMap<size_t>,
          std::optional<std::pair<std::string, std::string>>>
getVersionOrderedBitAssignments(const Operator &op);

/// Extract version information from an operand's TableGen metadata.
std::pair<std::string, std::string>
extractVersionFromOperand(unsigned operandIndex, const Operator &op);

/// Extract version information from a result's TableGen metadata.
std::pair<std::string, std::string>
extractVersionFromResult(unsigned resultIndex, const Operator &op);

/// Shared structure to capture version info for result
/// serialization/deserialization.
struct ResultVersionInfo {
  std::string majorStr, minorStr;
  std::string name;
  bool requiresVersionCheck;

  ResultVersionInfo(int idx, const NamedTypeConstraint &result,
                    const Operator &op, const std::string &opVersion)
      : name(result.name.str()) {
    std::tie(majorStr, minorStr) = extractVersionFromResult(idx, op);
    requiresVersionCheck = (majorStr + "." + minorStr != opVersion);

    // Validate that required results added after operation version are
    // buildable.
    if (requiresVersionCheck) {
      std::optional<StringRef> builderCall = result.constraint.getBuilderCall();
      if (!builderCall.has_value())
        llvm::PrintFatalError("Required result '" + result.name.str() +
                              "' in operation '" + op.getOperationName() +
                              "' was introduced after the operation (version " +
                              majorStr + "." + minorStr + " vs " + opVersion +
                              ") and has non-buildable type constraint '" +
                              result.constraint.getDefName().str() +
                              "'. Results added after operation version must "
                              "have buildable types.");
    }
  }
};

} // namespace tblgen
} // namespace mlir

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODEGEN_UTILITIES_H_
