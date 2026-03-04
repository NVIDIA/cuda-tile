//===- CommandLineOptions.cpp -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Common/CommandLineOptions.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::cuda_tile;
using llvm::raw_ostream;
using llvm::StringRef;

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

  static void print(raw_ostream &os, const BytecodeVersion &v) { os << v; }
};

// Static storage for command line option value.
static BytecodeVersion *bytecodeVersionPtr = nullptr;
} // namespace

void mlir::cuda_tile::registerTileIRBytecodeVersionOption() {
  // Register command line option.
  static llvm::cl::opt<BytecodeVersion, /*ExternalStorage=*/false,
                       BytecodeVersionParser>
      bytecodeVersion("bytecode-version",
                      llvm::cl::desc("Bytecode version to use for translation"),
                      llvm::cl::init(BytecodeVersion::kCurrentVersion));

  bytecodeVersionPtr = &bytecodeVersion;
}

BytecodeVersion mlir::cuda_tile::getCurrentBytecodeVersion() {
  return bytecodeVersionPtr ? *bytecodeVersionPtr
                            : BytecodeVersion::kCurrentVersion;
}
