//===- Registration.cpp - CUDA Tile CAPI Registration -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile-c/Registration.h"

#include "mlir/CAPI/IR.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

void mlirCudaTileRegisterAllDialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::cuda_tile::CudaTileDialect>();
}

void mlirCudaTileRegisterAllPasses() {
  mlir::cuda_tile::registerCudaTilePasses();
}
