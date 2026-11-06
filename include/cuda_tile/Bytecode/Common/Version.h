//===- Version.h - CUDA Tile Bytecode Version Utilities ---------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_COMMON_VERSION_H
#define CUDA_TILE_BYTECODE_COMMON_VERSION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdint>
#include <optional>

namespace mlir::cuda_tile {
/// This class represents the version of the bytecode format.
/// The version is used to determine the compatibility of the bytecode with
/// different versions of the Cuda Toolkit and Driver.
class BytecodeVersion {
public:
  /// Construct a bytecode version, which by default will target the current
  /// compatibility version of the bytecode format.
  BytecodeVersion() : BytecodeVersion(kCurrentCompatibilityVersion) {}

  /// Construct a bytecode version from the given major, minor, etc.
  /// version numbers. Returns nullopt if the version is not supported.
  static std::optional<BytecodeVersion>
  fromVersion(uint8_t verMajor, uint8_t verMinor, uint16_t verTag = 0);

  /// Returns the major version number.
  uint8_t getMajor() const { return verMajor; }

  /// Returns the minor version number.
  uint8_t getMinor() const { return verMinor; }

  /// Returns the version tag.
  uint16_t getTag() const { return verTag; }

  /// Various comparison operators for comparing versions.
  bool operator==(const BytecodeVersion &other) const {
    return verMajor == other.verMajor && verMinor == other.verMinor &&
           verTag == other.verTag;
  }
  bool operator!=(const BytecodeVersion &other) const {
    return !(*this == other);
  }
  bool operator<(const BytecodeVersion &other) const {
    if (verMajor != other.verMajor) {
      return verMajor < other.verMajor;
    }
    if (verMinor != other.verMinor) {
      return verMinor < other.verMinor;
    }
    return verTag < other.verTag;
  }
  bool operator<=(const BytecodeVersion &other) const {
    return *this < other || *this == other;
  }
  bool operator>(const BytecodeVersion &other) const { return other < *this; }
  bool operator>=(const BytecodeVersion &other) const {
    return !(*this < other);
  }

  /// Convert the version to a human-readable string format.
  std::string toString() const {
    if (verTag) {
      return llvm::formatv("{0}.{1}.{2}", verMajor, verMinor, verTag).str();
    }
    return llvm::formatv("{0}.{1}", verMajor, verMinor).str();
  }

  //===--------------------------------------------------------------------===//
  // Version Definitions
  //===--------------------------------------------------------------------===//

  /// The current "compatibility" version of the bytecode format. This version
  /// is the one with the widest compatibility range within a major version of
  /// the Cuda Toolkit and Driver (generally corresponding to the last major
  /// version).
  static const BytecodeVersion kCurrentCompatibilityVersion;

  /// The current version of the bytecode format. This version corresponds to
  /// the most recent version of CUDA Tile IR.
  static const BytecodeVersion kCurrentVersion;

  /// The version when unified bitfield for optional parameters was introduced.
  /// For versions >= 13.3, all optional parameters (Type, Enum, etc.) use a
  /// single bitfield. For versions < 13.3, OptionalEnum uses inline flags.
  static const BytecodeVersion kUnifiedBitfieldVersion;

  /// The version starting from which unknown function-table flag bits above
  /// Visibility/KindKernel/HasOptimizationHints are hard errors.
  static const BytecodeVersion kFunctionFlagsExtendedVersion;

  /// The version that introduces the Form field in bits [3:2] and stops using
  /// HasOptimizationHints. KindKernel entries always carry optimization_hints;
  /// extern FuncOp is FormNormal with body length 0 and always-present
  /// const_params.
  static const BytecodeVersion kFunctionFlagsFormVersion;

  /// The minimum supported version of the bytecode format.
  static const BytecodeVersion kMinSupportedVersion;

  /// The first version that uses the canonical bit-packed wire format for
  /// `i1` `DenseElementsAttr` payloads.
  static const BytecodeVersion kCanonicalI1Version;

  /// The first version that encodes function arg attrs.
  static const BytecodeVersion kMinArgAttrsVersion;

  /// The first version that encodes the `flags` field of a `DISubprogramAttr`.
  ///
  ///   * Bytecode below this version carries no flags field; a subprogram read
  ///     from it defaults to `DISubprogramFlags::None`.
  ///   * At or above it, every set bit must be one the build recognizes;
  ///     unknown bits are a hard error.
  static const BytecodeVersion kMinDISubprogramFlagsVersion;

private:
  /// Constructs a BytecodeVersion object with the given version components.
  constexpr BytecodeVersion(uint8_t verMajor, uint8_t verMinor, uint16_t verTag)
      : verMajor(verMajor), verMinor(verMinor), verTag(verTag) {}

  /// The major version number.
  uint8_t verMajor;

  /// The minor version number.
  uint8_t verMinor;

  /// The tag version number.
  uint16_t verTag;
};

/// Streams the bytecode version to the given output stream, formatted as
/// "major.minor.tag".
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const BytecodeVersion &version) {
  return os << version.toString();
}

/// Returns a list of all supported bytecode versions.
/// This list is generated from the SupportedVersion definitions in TableGen.
llvm::SmallVector<BytecodeVersion> getSupportedVersions();

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_BYTECODE_COMMON_VERSION_H
