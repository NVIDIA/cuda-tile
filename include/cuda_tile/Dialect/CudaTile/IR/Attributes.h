//===- Attributes.h - CUDA Tile Debug Info Attributes -----------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_ATTRIBUTES_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.h.inc"
#include "cuda_tile/Dialect/CudaTile/IR/Enums.h.inc"

namespace mlir::cuda_tile {

//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

/// Base class for all debug info attributes.
class DINodeAttr : public Attribute {
public:
  using Attribute::Attribute;
  static bool classof(Attribute attr);
};

/// Represents a debug info scope.
class DIScopeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;
  static bool classof(Attribute attr);
};

/// Represents a local debug info scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;
  static bool classof(Attribute attr);
};

} // namespace mlir::cuda_tile

#define GET_ATTRDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.h.inc"

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_ATTRIBUTES_H
