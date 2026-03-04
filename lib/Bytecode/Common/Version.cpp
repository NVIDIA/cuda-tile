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
    /*verMinor=*/2,
    /*verTag=*/0,
};

/// The lowest supported version of the bytecode format.
const BytecodeVersion BytecodeVersion::kMinSupportedVersion = {
    /*verMajor=*/13,
    /*verMinor=*/1,
    /*verTag=*/0,
};

//===----------------------------------------------------------------------===//
// Opcode Version Checking
//===----------------------------------------------------------------------===//

bool mlir::cuda_tile::detail::isOpcodeAvailableInVersion(
    uint32_t opcode, const BytecodeVersion &version) {
  auto it = getVersionToMaxOpcodeMap().find(
      std::make_pair(version.getMajor(), version.getMinor()));
  if (it == getVersionToMaxOpcodeMap().end())
    return false;
  return opcode <= it->second;
}
