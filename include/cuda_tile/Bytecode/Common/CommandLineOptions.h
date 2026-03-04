//===- CommandLineOptions.h -------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_COMMON_COMMANDLINEOPTIONS_H
#define CUDA_TILE_BYTECODE_COMMON_COMMANDLINEOPTIONS_H

#include "cuda_tile/Bytecode/Common/Version.h"

namespace mlir {
namespace cuda_tile {

/// Register command line options for Cuda Tile IR bytecode version.
void registerTileIRBytecodeVersionOption();

/// Get the current bytecode version from command line options.
/// Returns the default version if no command line option was set.
BytecodeVersion getCurrentBytecodeVersion();

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_BYTECODE_COMMON_COMMANDLINEOPTIONS_H
