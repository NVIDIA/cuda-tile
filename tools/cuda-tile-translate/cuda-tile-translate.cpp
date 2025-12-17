//===- cuda-tile-translate.cpp - CUDA Tile Translation Tool -------*- C++
//-*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "cuda_tile/Bytecode/Translation/BytecodeTranslation.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "test/RoundTripTestRegistration.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::cuda_tile::CudaTileDialect>();

  mlir::cuda_tile::registerTileIRTranslations();
  mlir::cuda_tile::registerTileIRTestTranslations();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "CUDA Tile Translation Tool"));
}
