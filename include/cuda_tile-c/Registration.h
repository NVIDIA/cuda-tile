//===- Registration.h - CUDA Tile C API Registration ------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_C_REGISTRATION_H
#define CUDA_TILE_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Add all the dialects provided by cuda_tile to the registry.
MLIR_CAPI_EXPORTED void
mlirCudaTileRegisterAllDialects(MlirDialectRegistry registry);

/// Add all the passes provided by cuda_tile.
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // CUDA_TILE_C_REGISTRATION_H
