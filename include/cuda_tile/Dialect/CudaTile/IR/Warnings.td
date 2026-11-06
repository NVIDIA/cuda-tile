//===- Warnings.td - CUDA Tile Warning Definitions ---------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Always-on warning diagnostics. See Remark.h for the developer guide
// (how to add new warnings, ID conventions, call sites).
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_WARNINGS_TD
#define CUDATILE_DIALECT_WARNINGS_TD

include "cuda_tile/Dialect/CudaTile/IR/RemarkUtils.td"

def GenericWarning : TileIRWarning<"GenericWarning", 0, "Compiler warning: {0}">;

def RegisterPressure
    : TileIRWarning<"RegisterPressure", 1, "High register usage">;

def WarningID
    : I32EnumAttr<"WarningID", "All TileIR warning IDs",
                  [GenericWarning, RegisterPressure]> {
  let cppNamespace = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}

#endif // CUDATILE_DIALECT_WARNINGS_TD
