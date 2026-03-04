//===- CudaTileOptimizer.h --------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_C_DIALECT_CUDATILEOPTIMIZER_H
#define CUDA_TILE_C_DIALECT_CUDATILEOPTIMIZER_H

#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// CudaTile optimizations flags
enum {
  CUDATILE_OPT_FLAG_NONE = 0u,
  CUDATILE_OPT_FLAG_ENABLE_MULTITHREAD = 1u << 0,
  CUDATILE_OPT_FLAG_FUSE_FMA = 1u << 1,
};

/// Callback function type for handling diagnostics.
/// userData: User-provided context pointer
/// diagnostic: The diagnostic being emitted
/// Returns: MlirLogicalResult indicating whether the diagnostic was handled
typedef MlirLogicalResult (*MlirDiagnosticCallback)(MlirDiagnostic diagnostic,
                                                    void *userData);

/// Structure that holds configuration for CUDA Tile IR passes
typedef struct {
  uint64_t flags;
  int optLevel;
  int loopSplitThreshold;
  // Optional diagnostic handler callback and user data
  MlirDiagnosticCallback diagnosticCallback;
  void *diagnosticUserData;
} mlirCudaTileOptConfig;

/// Initialize CUDA Tile IR Optimization config with default values
MLIR_CAPI_EXPORTED void mlirCudaTileOptFlagsInit(mlirCudaTileOptConfig *config);

/// Applies TileIR optimizations to a cuda_tile module operation.
/// Returns true on success, false on failure.
/// Note: This function extracts the cuda_tile module and applies the
/// configured optimization pipeline.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirCudaTileApplyOptimizations(
    MlirOperation moduleOp, mlirCudaTileOptConfig *config);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TILE_C_DIALECT_CUDATILEOPTIMIZER_H
