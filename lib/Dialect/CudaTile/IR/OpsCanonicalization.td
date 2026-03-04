//===- OpsCanonicalization.td ------------------------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_OPS_PATTERNS
#define CUDA_TILE_OPS_PATTERNS

include "mlir/IR/PatternBase.td"
include "cuda_tile/Dialect/CudaTile/IR/Ops.td"

//===----------------------------------------------------------------------===//
// Common helpers
//===----------------------------------------------------------------------===//

// A native constraint that is true iff the given Value is a constant `true`.
def IsConstTrueVal :
  Constraint<CPred<"isConstantTrueVal($0)">,
             "is const true">;

// A native constraint that is true iff the given Value is a constant `true`.
def IsConstFalseVal :
  Constraint<CPred<"isConstantFalseVal($0)">,
             "is const false">;

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// select(pred, false, true) => not(pred)
def SelectI1ToNot :
    Pat<(CudaTile_SelectOp $pred, $falseVal, $trueVal),
        (CudaTile_XOrIOp $pred, $trueVal),
        [
          (IsConstFalseVal $falseVal),
          (IsConstTrueVal $trueVal)
        ]>;

#endif // CUDA_TILE_OPS_PATTERNS
