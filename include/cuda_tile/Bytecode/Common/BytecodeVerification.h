//===- BytecodeVerification.h - Bytecode format verification ----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verification that a cuda_tile module conforms to bytecode format invariants
// (self-contained, only allowed op types). Used by both Reader and Writer so
// that bytecode produced by any writer (including custom clients) is validated
// on read.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_COMMON_BYTECODE_VERIFICATION_H
#define CUDA_TILE_BYTECODE_COMMON_BYTECODE_VERIFICATION_H

#include "mlir/Support/LogicalResult.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace mlir {
namespace cuda_tile {

/// Verifies that the module conforms to bytecode format invariants:
/// 1. Top-level only function and global operations.
/// 2. In public builds, preview FuncOps are extern declarations.
/// 3. All operations (at any nesting level) are from the CudaTile dialect.
/// 4. Calls with a resolved callee reference an extern function.
/// 5. Entry arguments are rank-0 tile types.
///
/// Use this when reading bytecode to reject invalid bytecode produced by
/// custom or buggy writers; use when writing to ensure we only serialize
/// valid structure.
LogicalResult verifySelfContainedModuleAndOperationInvariants(ModuleOp module);

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_BYTECODE_COMMON_BYTECODE_VERIFICATION_H
