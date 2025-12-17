//===- RoundTripTestRegistration.cpp - Round-trip Testing -------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "RoundTripTestRegistration.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {
class BytecodeVersionParser : public llvm::cl::parser<BytecodeVersion> {
public:
  BytecodeVersionParser(llvm::cl::Option &o)
      : llvm::cl::parser<BytecodeVersion>(o) {}

  bool parse(llvm::cl::Option &o, StringRef /*argName*/, StringRef arg,
             BytecodeVersion &v) {
    StringRef versionStr = arg;

    // Parse the `major.minor`.
    uint8_t verMajor, verMinor;
    if (versionStr.consumeInteger(10, verMajor) ||
        !versionStr.consume_front(".") ||
        versionStr.consumeInteger(10, verMinor))
      return o.error("Invalid argument '" + arg + "'");

    // Parse the `.tag`.
    uint16_t tag = 0;
    if (versionStr.consume_front(".") && versionStr.consumeInteger(10, tag))
      return o.error("Invalid argument '" + arg + "'");
    if (!versionStr.empty())
      return o.error("Invalid argument '" + arg + "'");

    std::optional<BytecodeVersion> version =
        BytecodeVersion::fromVersion(verMajor, verMinor, tag);
    if (!version) {
      return o.error(
          llvm::formatv(
              "Invalid argument '{0}': the supported versions are [{1}, {2}]",
              arg, BytecodeVersion::kMinSupportedVersion,
              BytecodeVersion::kCurrentVersion)
              .str());
    }

    // Set the version and return false to indicate success.
    v = *version;
    return false;
  }
  static void print(raw_ostream &os, const BytecodeVersion &v) { os << v; };
};
} // namespace

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
  static llvm::cl::opt<BytecodeVersion, /*ExternalStorage=*/false,
                       BytecodeVersionParser>
      bytecodeVersion(
          "bytecode-version",
          llvm::cl::desc("Bytecode version to use for roundtrip testing"),
          llvm::cl::init(BytecodeVersion::kCurrentVersion));

  static llvm::cl::opt<bool> useGenericForm(
      "generic-form", llvm::cl::desc("Print operations in generic form"),
      llvm::cl::init(false));

  TranslateFromMLIRRegistration roundtrip(
      "test-cudatile-roundtrip",
      "Test bytecode serialization and deserialization round-trip",
      [](cuda_tile::ModuleOp op, llvm::raw_ostream &output) {
        return roundTripModule(op, output, bytecodeVersion, useGenericForm);
      },
      [](DialectRegistry &registry) { registry.insert<CudaTileDialect>(); });
}
