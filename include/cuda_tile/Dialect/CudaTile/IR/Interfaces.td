//===- Interfaces.td - CUDA Tile Interface Definitions -----*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDATILE_DIALECT_CUDATILE_IR_INTERFACES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def CudaTile_AssumePredicateAttrInterface
    : AttrInterface<"AssumePredicateAttrInterface"> {
  let description = [{
    This interface must be implemented by all attributes that can be used as a
    `cuda_tile.assume` predicate.
  }];
  let cppNamespace = "::mlir::cuda_tile";
  let methods = [
    InterfaceMethod<[{
        Verifies this attribute in the context of the given `cuda_tile.assume`
        op. Returns "success" if the attribute is semantically valid on the op
        and "failure" otherwise.
      }],
      "LogicalResult", "verifyWithAssumeOp", (ins "::mlir::Operation *":$op)>
  ];
}

def CudaTile_TileView : TypeInterface<"TileView"> {
  let cppNamespace = "::mlir::cuda_tile";
  let description = [{
    Represents a view within a memref from which tiles can be loaded/stored. It
    acts as a converter from a coordinate in an abstract tile space and tiles,
    communicating a loading/storing strategy.

    Views must always access tiles of the same type no matter the index.

    For an example, see `!cuda_tile.partition_view`.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the rank of tile indices (tile-space coordinates).
      }],
      /*retTy=*/"size_t",
      /*methodName=*/"getViewIndexRank",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the type of tiles loaded from/stored to the view.
      }],
      // FIXME: The return type should be constrainted to
      // cuda_tile::TileType, but due to circular dependencies this is
      // tricky to achieve with ODS.
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getViewTileType",
      /*args=*/(ins)
    >,
  ];
}

class AllElementTypeMatch<string summary, list<string> names>
  : PredOpTrait<summary,
                AllMatchSameOperatorPred<names, 
                  "::llvm::cast<::mlir::cuda_tile::TileType>($_self.getType()).getElementType()">> {
  list<string> values = names;
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_INTERFACES_TD
