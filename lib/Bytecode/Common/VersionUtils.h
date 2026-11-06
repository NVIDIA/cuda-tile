//===- VersionUtils.h - CUDA Tile Bytecode Version Utils --------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for version checking bytecode operations.
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

/// Returns true if the opcode is in the public-preview range (0x4000 - 0x4FFF).
bool isPreviewOpcode(uint32_t opcode);

} // namespace mlir::cuda_tile::detail

#endif // CUDA_TILE_BYTECODE_COMMON_VERSION_UTILS_H
