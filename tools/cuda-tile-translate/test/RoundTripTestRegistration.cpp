//===- RoundTripTestRegistration.cpp - Round-trip Testing -------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundTripTestRegistration.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"

#include "cuda_tile/Bytecode/Common/CommandLineOptions.h"
#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// Round-trip registration
//===----------------------------------------------------------------------===//

static LogicalResult roundTripModule(cuda_tile::ModuleOp op,
                                     raw_ostream &output,
                                     BytecodeVersion version,
                                     bool useGenericForm) {
  // First, serialize the module to bytecode
  SmallVector<char, 4096> bytecodeBuffer;
  llvm::raw_svector_ostream rvo(bytecodeBuffer);
  if (failed(writeBytecode(rvo, op, version)))
    return failure();
  MLIRContext *context = op->getContext();
  llvm::MemoryBufferRef bytecodeBufferRef(
      llvm::StringRef(bytecodeBuffer.data(), bytecodeBuffer.size()),
      "roundTripModuleBuffer");
  OwningOpRef<cuda_tile::ModuleOp> deserializedModule =
      readBytecode(bytecodeBufferRef, *context);
  if (!deserializedModule) {
    op->emitError("Failed to deserialize bytecode");
    return failure();
  }
  // Print the deserialized module for visual comparison
  OpPrintingFlags flags;
  if (useGenericForm)
    flags.printGenericOpForm();
  deserializedModule->print(output, flags);
  output << "\n";
  return success();
}

void mlir::cuda_tile::registerTileIRTestTranslations() {
  static llvm::cl::opt<bool> useGenericForm(
      "generic-form", llvm::cl::desc("Print operations in generic form"),
      llvm::cl::init(false));

  TranslateFromMLIRRegistration roundtrip(
      "test-cudatile-roundtrip",
      "Test bytecode serialization and deserialization round-trip",
      [](cuda_tile::ModuleOp op, llvm::raw_ostream &output) {
        return roundTripModule(op, output, getCurrentBytecodeVersion(),
                               useGenericForm);
      },
      [](DialectRegistry &registry) { registry.insert<CudaTileDialect>(); });
}
