//===- Traits.cpp - CUDA Tile Traits Utilities ------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile/Dialect/CudaTile/IR/Traits.h"

#include "llvm/Support/Casting.h"

#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

using namespace mlir;
using namespace mlir::cuda_tile;

bool OpTrait::cuda_tile::impl::verifyLoadStoreType(Type dstType, Type srcType) {
  if (!isPointerLike(srcType) || !isa<TileType>(dstType))
    return false;

  auto dstTensorType = cast<TileType>(dstType);
  auto srcTensorType = cast<TileType>(srcType);
  auto srcPointerType = cast<PointerType>(srcTensorType.getElementType());

  return srcTensorType.getShape() == dstTensorType.getShape() &&
         srcPointerType.getPointeeType() == dstTensorType.getElementType();
}

bool OpTrait::cuda_tile::impl::verifyLoadStoreMask(Type resultType,
                                                   Type maskType) {
  if (!isa<TileType>(resultType) || !isa<TileType>(maskType))
    return false;

  return cast<TileType>(resultType).getShape() ==
         cast<TileType>(maskType).getShape();
}

bool OpTrait::cuda_tile::impl::verifyLoadPadding(Type resultType,
                                                 Type paddingType) {
  if (!isa<TileType>(resultType) || !isa<TileType>(paddingType))
    return false;
  auto resultTensorType = cast<TileType>(resultType);
  auto paddingTensorType = cast<TileType>(paddingType);

  return resultTensorType.getShape() == paddingTensorType.getShape() &&
         resultTensorType.getElementType() ==
             paddingTensorType.getElementType();
}
