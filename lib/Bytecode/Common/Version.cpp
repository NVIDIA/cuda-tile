//===- Version.cpp - CUDA Tile Bytecode Versioning --------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile/Bytecode/Common/Version.h"

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// BytecodeVersion
//===----------------------------------------------------------------------===//

std::optional<BytecodeVersion> BytecodeVersion::fromVersion(uint8_t verMajor,
                                                            uint8_t verMinor,
                                                            uint16_t verTag) {
  // Check support within the known major versions.
  if (verMajor == 13) {
    // [13.1, 13.1]
    if (verMinor < 1 || verMinor > 1)
      return std::nullopt;
    return BytecodeVersion(verMajor, verMinor, verTag);
  }
  // This is an unknown major version.
  return std::nullopt;
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
    /*verMinor=*/1,
    /*verTag=*/0,
};

/// The lowest supported version of the bytecode format.
const BytecodeVersion BytecodeVersion::kMinSupportedVersion = {
    /*verMajor=*/13,
    /*verMinor=*/1,
    /*verTag=*/0,
};
