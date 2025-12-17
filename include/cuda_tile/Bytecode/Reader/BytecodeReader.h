//===- BytecodeReader.h - CUDA Tile Bytecode Reader -------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_BYTECODE_READER_H
#define CUDA_TILE_BYTECODE_READER_H

#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/MemoryBuffer.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::cuda_tile {

/// Returns if the given bytecode buffer is a valid cuda_tile bytecode.
bool isTileIRBytecode(llvm::MemoryBufferRef bytecodeBuffer);
bool isTileIRBytecode(const char *bytecodeBuffer);

/// Returns the size of the bytecode defined in the given buffer.
std::optional<size_t> getBytecodeSize(const char *bytecodeBuffer);

/// Reads a cuda_tile module from the provided bytecode data.
OwningOpRef<cuda_tile::ModuleOp>
readBytecode(llvm::MemoryBufferRef bytecodeBuffer, MLIRContext &context);

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_BYTECODE_READER_H
