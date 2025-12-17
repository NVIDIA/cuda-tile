//===- BytecodeTranslation.cpp - CUDA Tile Bytecode Xlation -----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile/Bytecode/Translation/BytecodeTranslation.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// Deserialization registration
//===----------------------------------------------------------------------===//

static OwningOpRef<Operation *> deserializeModule(llvm::StringRef bytecodeStr,
                                                  MLIRContext *context) {
  llvm::MemoryBufferRef bytecodeBufferRef(bytecodeStr,
                                          "deserializeModuleBuffer");
  context->getOrLoadDialect<CudaTileDialect>();
  return readBytecode(bytecodeBufferRef, *context);
}

static void registerFromTileIRBytecodeTranslation() {
  TranslateToMLIRRegistration fromBytecode(
      "cudatilebc-to-mlir", "Translate CUDA Tile IR bytecode to MLIR",
      [](llvm::StringRef bytecode, MLIRContext *context) {
        return deserializeModule(bytecode, context);
      },
      [](DialectRegistry &registry) { registry.insert<CudaTileDialect>(); });
}

//===----------------------------------------------------------------------===//
// Serialization registration
//===----------------------------------------------------------------------===//

static void registerToTileIRBytecodeTranslation() {
  TranslateFromMLIRRegistration toBytecode(
      "mlir-to-cudatilebc", "Translate MLIR to CUDA Tile IR bytecode",
      [](Operation *op, raw_ostream &output) {
        cuda_tile::ModuleOp moduleOp = dyn_cast<cuda_tile::ModuleOp>(op);
        if (moduleOp)
          return writeBytecode(output, moduleOp,
                               BytecodeVersion::kCurrentVersion);

        // Also support a CUDA Tile IR Module nested in a MLIR Module for
        // convenience since the MLIR parse is adding one implicitly by default.
        if (auto moduleOp = dyn_cast<mlir::ModuleOp>(op)) {
          if (!llvm::hasSingleElement(*moduleOp.getBody()) ||
              !llvm::isa<cuda_tile::ModuleOp>(moduleOp.getBody()->front())) {
            op->emitError(
                "expected a single CUDA Tile IR module in the MLIR module");
            return failure();
          }
          return writeBytecode(
              output, cast<cuda_tile::ModuleOp>(moduleOp.getBody()->front()),
              BytecodeVersion::kCurrentVersion);
        }

        op->emitError("expected a CUDA Tile IR module, but got a " +
                      op->getName().getStringRef());
        return failure();
      },
      [](DialectRegistry &registry) { registry.insert<CudaTileDialect>(); });
}

void mlir::cuda_tile::registerTileIRTranslations() {
  registerFromTileIRBytecodeTranslation();
  registerToTileIRBytecodeTranslation();
}
