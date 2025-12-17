//===- BytecodeWriter.h - CUDA Tile Bytecode Writer -------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_BYTECODE_WRITER_H
#define CUDA_TILE_BYTECODE_WRITER_H

#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::cuda_tile {

/// Writes a cuda_tile module to the provided output stream in bytecode format.
LogicalResult writeBytecode(raw_ostream &os, cuda_tile::ModuleOp module,
                            BytecodeVersion targetVersion =
                                BytecodeVersion::kCurrentCompatibilityVersion);

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_BYTECODE_WRITER_H
