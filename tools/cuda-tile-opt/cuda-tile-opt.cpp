//===- cuda-tile-opt.cpp - CUDA Tile Dialect Test Driver --------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::cuda_tile::CudaTileDialect>();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::cuda_tile::registerCudaTilePasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CudaTile test driver\n", registry));
}
