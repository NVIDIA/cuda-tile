//===- Traits.h - CUDA Tile Traits ------------------------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_TRAITS_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_TRAITS_H

#include "mlir/IR/Types.h"

namespace mlir {

namespace OpTrait {

namespace cuda_tile {

namespace impl {

/// Verify destination and source shape for load and store OPs
bool verifyLoadStoreType(Type dstType, Type srcType);

/// Verify destination and mask shape for load and store OPs
bool verifyLoadStoreMask(Type dstType, Type maskType);

/// Verify destination and padding shape for load OP
bool verifyLoadPadding(Type dstType, Type paddingType);

} // namespace impl
} // namespace cuda_tile
} // namespace OpTrait
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_TRAITS_H
