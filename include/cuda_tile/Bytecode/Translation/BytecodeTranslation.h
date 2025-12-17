//===- BytecodeTranslation.h - CUDA Tile Bytecode Translation ---*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BYTECODE_TRANSLATION_H
#define BYTECODE_TRANSLATION_H

namespace mlir::cuda_tile {

void registerTileIRTranslations();

} // namespace mlir::cuda_tile

#endif // BYTECODE_TRANSLATION_H
