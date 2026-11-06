//===- BytecodeTranslation.cpp - CUDA Tile Bytecode Xlation -----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Translation/BytecodeTranslation.h"

#include "mlir/Tools/mlir-translate/Translation.h"

#include "cuda_tile/Bytecode/Analysis/MinVersionAnalyzer.h"
#include "cuda_tile/Bytecode/Common/CommandLineOptions.h"
#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"

using namespace mlir;
using namespace mlir::cuda_tile;

static void registerConfiguredCudaTileDialect(DialectRegistry &registry) {
  registry.insert<CudaTileDialect>();
  registry.addExtension(+[](MLIRContext *, CudaTileDialect *dialect) {
    dialect->setEnabledPreviewFeatures(getPreviewFeaturesFromCommandLine());
  });
}

//===----------------------------------------------------------------------===//
// Deserialization registration
//===----------------------------------------------------------------------===//

static OwningOpRef<Operation *> deserializeModule(llvm::StringRef bytecodeStr,
                                                  MLIRContext *context) {
  llvm::MemoryBufferRef bytecodeBufferRef(bytecodeStr,
                                          "deserializeModuleBuffer");
  auto dialect = context->getOrLoadDialect<CudaTileDialect>();
  dialect->setWarnUnsupportedHints(getWarnUnsupportedHints());
  dialect->setErrorOnHints(getErrorUnsupportedHints());
  return readBytecode(bytecodeBufferRef, *context);
}

static void registerFromTileIRBytecodeTranslation() {
  TranslateToMLIRRegistration fromBytecode(
      "cudatilebc-to-mlir", "Translate CUDA Tile IR bytecode to MLIR",
      [](llvm::StringRef bytecode, MLIRContext *context) {
        return deserializeModule(bytecode, context);
      },
      [](DialectRegistry &registry) {
        registerConfiguredCudaTileDialect(registry);
      });
}

//===----------------------------------------------------------------------===//
// Serialization registration
//===----------------------------------------------------------------------===//
static FailureOr<cuda_tile::ModuleOp> getCudaTileModuleOp(Operation *op) {
  cuda_tile::ModuleOp moduleOp = dyn_cast<cuda_tile::ModuleOp>(op);
  if (moduleOp) {
    return moduleOp;
  }
  // Also support a CUDA Tile IR Module nested in a MLIR Module for convenience
  // since the MLIR parse is adding one implicitly by default.
  if (auto moduleOp = dyn_cast<mlir::ModuleOp>(op)) {
    if (!llvm::hasSingleElement(*moduleOp.getBody()) ||
        !llvm::isa<cuda_tile::ModuleOp>(moduleOp.getBody()->front())) {
      return op->emitError(
          "expected a single CUDA Tile IR module in the MLIR module");
    }
    return cast<cuda_tile::ModuleOp>(moduleOp.getBody()->front());
  }
  return op->emitError("expected a CUDA Tile IR module, but got a " +
                       op->getName().getStringRef());
}

static void registerToTileIRBytecodeTranslation() {
  TranslateFromMLIRRegistration toBytecode(
      "mlir-to-cudatilebc", "Translate MLIR to CUDA Tile IR bytecode",
      [](Operation *op, raw_ostream &output) {
        BytecodeVersion targetVersion = getCurrentBytecodeVersion();
        auto moduleOp = getCudaTileModuleOp(op);
        if (failed(moduleOp)) {
          return failure();
        }
        auto dialect =
            cast<CudaTileDialect>(moduleOp->getOperation()->getDialect());
        dialect->setWarnUnsupportedHints(getWarnUnsupportedHints());
        dialect->setErrorOnHints(getErrorUnsupportedHints());
        BytecodeWriterConfig writerConfig(targetVersion,
                                          isBytecodeVersionExplicitlySet());
        return writeBytecode(output, *moduleOp, writerConfig);
      },
      [](DialectRegistry &registry) {
        registerConfiguredCudaTileDialect(registry);
      });
}

//===----------------------------------------------------------------------===//
// Min version analysis registration
//===----------------------------------------------------------------------===//

static void registerMinVersionAnalysis() {
  TranslateFromMLIRRegistration minVersion(
      "analyze-min-version",
      "Analyze IR and report minimum CUDA Tile bytecode version required "
      "(outputs one version per cuda_tile.module)",
      [](Operation *op, raw_ostream &output) {
        auto moduleOp = getCudaTileModuleOp(op);
        if (failed(moduleOp)) {
          return failure();
        }

        auto minVersion = getMinBytecodeVersion(*moduleOp);
        if (failed(minVersion)) {
          return failure();
        }

        output << minVersion->toString() << "\n";
        return success();
      },
      [](DialectRegistry &registry) {
        registerConfiguredCudaTileDialect(registry);
      });
}

void mlir::cuda_tile::registerTileIRTranslations() {
  registerFromTileIRBytecodeTranslation();
  registerToTileIRBytecodeTranslation();
  registerMinVersionAnalysis();
}
