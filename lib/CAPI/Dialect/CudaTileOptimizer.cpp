//===- CudaTileOptimizer.cpp ------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile-c/Dialect/CudaTileOptimizer.h"

#include "mlir/CAPI/Diagnostics.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Diagnostics.h"

#include "cuda_tile/Dialect/CudaTile/Optimizer/CudaTileOptimizer.h"

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// CUDA Tile IR -> CUDA Tile IR optimization pipeline
//===----------------------------------------------------------------------===//
void mlirCudaTileOptFlagsInit(mlirCudaTileOptConfig *config) {
  if (!config)
    return;
  // Clear config
  memset(config, 0, sizeof(*config));

  // Set default values
  config->flags = 0;              // Default
  config->loopSplitThreshold = 1; // Default - run for all loops
  config->optLevel = 3;           // Default - run all opts
  config->diagnosticCallback = nullptr;
  config->diagnosticUserData = nullptr;
}

// Initialize CPP struct cuda_tile::TileIROptimizerOptions
// based on values from C API mlirCudaTileOptConfig struct
static TileIROptimizerOptions toCpp(const mlirCudaTileOptConfig &c) {
  TileIROptimizerOptions o;
  o.enableMultithread = (c.flags & CUDATILE_OPT_FLAG_ENABLE_MULTITHREAD) != 0;
  o.enableFuseFMA = (c.flags & CUDATILE_OPT_FLAG_FUSE_FMA) != 0;
  o.optLevel = c.optLevel;
  o.loopSplitThreshold = c.loopSplitThreshold;
  return o;
}

MlirLogicalResult
mlirCudaTileApplyOptimizations(MlirOperation moduleOp,
                               mlirCudaTileOptConfig *config) {
  auto *cppOp = unwrap(moduleOp);
  if (!cppOp)
    return mlirLogicalResultFailure();
  if (!config)
    return mlirLogicalResultFailure();
  auto cudaTileModuleOp = extractCudaTileModuleOp(cppOp);
  if (!cudaTileModuleOp)
    return mlirLogicalResultFailure();

  // Register all CUDA Tile IR optimization passes
  registerTileIROptPasses();

  // Set up diagnostic handler if callback is provided
  MLIRContext *context = cppOp->getContext();
  std::optional<DiagnosticEngine::HandlerID> handlerID = std::nullopt;

  if (config->diagnosticCallback) {
    auto handler = [callback = config->diagnosticCallback,
                    userData = config->diagnosticUserData](Diagnostic &diag) {
      MlirDiagnostic cDiag = wrap(diag);
      return mlirLogicalResultIsSuccess(callback(cDiag, userData)) ? success()
                                                                   : failure();
    };
    handlerID = context->getDiagEngine().registerHandler(handler);
  }

  // Run optimizations
  auto result = optimizeTileIRModule(cudaTileModuleOp, toCpp(*config),
                                     handlerID.has_value());

  // Unregister handler if we registered one
  if (handlerID.has_value())
    context->getDiagEngine().eraseHandler(*handlerID);

  return succeeded(result) ? mlirLogicalResultSuccess()
                           : mlirLogicalResultFailure();
}
