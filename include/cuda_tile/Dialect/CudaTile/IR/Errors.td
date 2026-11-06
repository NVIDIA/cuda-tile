//===- Errors.td - CUDA Tile Error Definitions -------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Always-on error diagnostics. See Remark.h for the developer guide
// (how to add new errors, ID conventions, call sites).
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_ERRORS_TD
#define CUDATILE_DIALECT_ERRORS_TD

include "cuda_tile/Dialect/CudaTile/IR/RemarkUtils.td"

def GenericError : TileIRError<"GenericError", 0, "Internal compiler error: {0}">;

def ErrorID
    : I32EnumAttr<"ErrorID", "All TileIR error IDs",
                  [GenericError]> {
  let cppNamespace = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}

#endif // CUDATILE_DIALECT_ERRORS_TD
