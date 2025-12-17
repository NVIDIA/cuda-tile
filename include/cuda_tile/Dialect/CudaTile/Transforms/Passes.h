//===- Passes.h - CUDA Tile Dialect Passes ----------------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_TRANSFORMS_PASSES_H
#define CUDA_TILE_DIALECT_CUDATILE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir::cuda_tile {

static constexpr llvm::StringLiteral kLoopSplitThresholdAttrName =
    "cuda_tile.loop_split";

struct TileIROptimizationsOpts {
  // Sets default threshold for Loop Split optimization
  // Set to -1 to disable pass completely
  int loop_split_threshold = 1;
  // Run CSE
  bool enable_cse = true;
  // Run canonicalization pass before optimizations
  bool canonicalize_before = true;
  // Run canonicalization pass after optimizations
  bool canonicalize_after = true;
};

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc"

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_DIALECT_CUDATILE_TRANSFORMS_PASSES_H
