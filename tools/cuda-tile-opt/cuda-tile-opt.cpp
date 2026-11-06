//===- cuda-tile-opt.cpp - CUDA Tile Dialect Test Driver --------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"

#include "cuda_tile/Bytecode/Common/CommandLineOptions.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

#ifdef CUDA_TILE_ENABLE_TESTING
namespace mlir::cuda_tile::test {
void registerTransformsUtilsTestPasses();
} // namespace mlir::cuda_tile::test
#endif // CUDA_TILE_ENABLE_TESTING

int main(int argc, char **argv) {
  // Command line options for hint diagnostics
  static llvm::cl::opt<bool> warnUnsupportedHints(
      "Wunsupported-hints",
      llvm::cl::desc(
          "Enable warnings for unsupported/invalid optimization hints"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> errorOnHints(
      "Werr-hints",
      llvm::cl::desc("Treat unsupported/invalid optimization hints as errors"),
      llvm::cl::init(false));

  mlir::cuda_tile::registerTileIRPreviewOption();

  mlir::DialectRegistry registry;
  registry.insert<mlir::cuda_tile::CudaTileDialect>();
  // Add an extension to configure the dialect when it's loaded
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::cuda_tile::CudaTileDialect *dialect) {
        // Configure hint diagnostics based on command-line options
        dialect->setWarnUnsupportedHints(warnUnsupportedHints);
        dialect->setErrorOnHints(errorOnHints);
        dialect->setEnabledPreviewFeatures(
            mlir::cuda_tile::getPreviewFeaturesFromCommandLine());
      });

  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::registerStripDebugInfoPass();
  mlir::cuda_tile::registerCudaTilePasses();

#ifdef CUDA_TILE_ENABLE_TESTING
  mlir::cuda_tile::test::registerTransformsUtilsTestPasses();
#endif // CUDA_TILE_ENABLE_TESTING

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CudaTile test driver\n", registry));
}
