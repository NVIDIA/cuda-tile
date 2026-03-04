//===- VersionUtils.h -------------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for version checking during bytecode operations.
// This is not part of the public API - only for bytecode
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_COMMON_VERSION_UTILS_H
#define CUDA_TILE_BYTECODE_COMMON_VERSION_UTILS_H

#include "cuda_tile/Bytecode/Common/Version.h"

namespace mlir::cuda_tile::detail {

/// Utility for bytecode encoding/decoding.
/// Check if an opcode is available in the given bytecode version.
bool isOpcodeAvailableInVersion(uint32_t opcode,
                                const BytecodeVersion &version);

} // namespace mlir::cuda_tile::detail

#endif // CUDA_TILE_BYTECODE_COMMON_VERSION_UTILS_H
