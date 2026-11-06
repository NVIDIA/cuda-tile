//===- Version.cpp - CUDA Tile Bytecode Versioning --------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Common/Version.h"

#include "VersionUtils.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
// Include auto-generated version constants from TableGen
#define GEN_VERSION_CONSTANTS
#include "StaticOpcodes.inc"

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// BytecodeVersion
//===----------------------------------------------------------------------===//

std::optional<BytecodeVersion> BytecodeVersion::fromVersion(uint8_t verMajor,
                                                            uint8_t verMinor,
                                                            uint16_t verTag) {
  // Include auto-generated version validation from TableGen.
#define GEN_VERSION_VALIDATION
#include "StaticOpcodes.inc"
}

//===----------------------------------------------------------------------===//
// Version Definitions

/// The current "compatibility" version of the bytecode format. This should
/// generally correspond to the last major version of the Cuda Toolkit and
/// Driver.
const BytecodeVersion BytecodeVersion::kCurrentCompatibilityVersion = {
    /*verMajor=*/13,
    /*verMinor=*/1,
    /*verTag=*/0,
};

/// The current version of the bytecode format.
const BytecodeVersion BytecodeVersion::kCurrentVersion = {
    /*verMajor=*/13,
    /*verMinor=*/5,
    /*verTag=*/0,
};

/// The version when unified bitfield for optional parameters was introduced.
const BytecodeVersion BytecodeVersion::kUnifiedBitfieldVersion = {
    /*verMajor=*/13,
    /*verMinor=*/3,
    /*verTag=*/0,
};

/// Unknown function-table flag bits above the pre-Form base mask become hard
/// errors at this version.
const BytecodeVersion BytecodeVersion::kFunctionFlagsExtendedVersion = {
    /*verMajor=*/13,
    /*verMinor=*/4,
    /*verTag=*/0,
};

/// Introduces the Form field in bits [3:2] and drops HasOptimizationHints.
const BytecodeVersion BytecodeVersion::kFunctionFlagsFormVersion = {
    /*verMajor=*/13,
    /*verMinor=*/5,
    /*verTag=*/0,
};

/// The lowest supported version of the bytecode format.
const BytecodeVersion BytecodeVersion::kMinSupportedVersion = {
    /*verMajor=*/13,
    /*verMinor=*/1,
    /*verTag=*/0,
};

/// The version when the canonical i1 bit-packed encoding was introduced.
const BytecodeVersion BytecodeVersion::kCanonicalI1Version = {
    /*verMajor=*/13,
    /*verMinor=*/4,
    /*verTag=*/0,
};

/// The version where arg attrs were introduced for function ops.
const BytecodeVersion BytecodeVersion::kMinArgAttrsVersion = {
    /*verMajor=*/13,
    /*verMinor=*/5,
    /*verTag=*/0,
};

/// The version where the `flags` field of DISubprogramAttr was introduced.
const BytecodeVersion BytecodeVersion::kMinDISubprogramFlagsVersion = {
    /*verMajor=*/13,
    /*verMinor=*/5,
    /*verTag=*/0,
};

//===----------------------------------------------------------------------===//
// Opcode Version Checking
//===----------------------------------------------------------------------===//

bool mlir::cuda_tile::detail::isPreviewOpcode(uint32_t opcode) {
  return opcode >= kPreviewOpcodesMin && opcode <= kPreviewOpcodesMax;
}

bool mlir::cuda_tile::detail::isOpcodeAvailableInVersion(
    uint32_t opcode, const BytecodeVersion &version) {
  // Preview opcodes are non-portable: accept only when major.minor matches the
  // exact current version. Named preview-feature opt-in is enforced elsewhere.
  if (isPreviewOpcode(opcode)) {
    return version.getMajor() == BytecodeVersion::kCurrentVersion.getMajor() &&
           version.getMinor() == BytecodeVersion::kCurrentVersion.getMinor();
  }

  auto it = getVersionToMaxOpcodeMap().find(
      std::make_pair(version.getMajor(), version.getMinor()));
  if (it == getVersionToMaxOpcodeMap().end()) {
    return false;
  }
  return opcode <= it->second;
}

//===----------------------------------------------------------------------===//
// Supported Versions List
//===----------------------------------------------------------------------===//

// Include auto-generated getSupportedVersions() function from TableGen.
#define GEN_SUPPORTED_VERSIONS_LIST
#include "StaticOpcodes.inc"
