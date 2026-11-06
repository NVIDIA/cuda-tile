//===- CommandLineOptions.cpp - CUDA Tile Bytecode CLI ----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Common/CommandLineOptions.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

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
        versionStr.consumeInteger(10, verMinor)) {
      return o.error("Invalid argument '" + arg + "'");
    }

    // Parse the `.tag`.
    uint16_t tag = 0;
    if (versionStr.consume_front(".") && versionStr.consumeInteger(10, tag)) {
      return o.error("Invalid argument '" + arg + "'");
    }
    if (!versionStr.empty()) {
      return o.error("Invalid argument '" + arg + "'");
    }

    std::optional<BytecodeVersion> version =
        BytecodeVersion::fromVersion(verMajor, verMinor, tag);
    if (!version) {
      return o.error(
          llvm::formatv(
              "Invalid argument '{0}': the supported versions are [{1} - {2}]",
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

// Pointer to the registered bytecode-version option.
static llvm::cl::opt<BytecodeVersion, /*ExternalStorage=*/false,
                     BytecodeVersionParser> *bytecodeVersionOpt = nullptr;

// Static storage for hints command line option value.
static bool warnUnsupportedHintsVar = false;
static bool errorUnsupportedHintsVar = false;

static llvm::cl::list<PreviewFeature, /*StorageClass=*/bool,
                      PreviewFeatureCLParser> *previewFeaturesOpt = nullptr;
} // namespace

void mlir::cuda_tile::registerTileIRBytecodeVersionOption() {
  // Register command line option.
  static llvm::cl::opt<BytecodeVersion, /*ExternalStorage=*/false,
                       BytecodeVersionParser>
      bytecodeVersion("bytecode-version",
                      llvm::cl::desc("Bytecode version to use for translation"),
                      llvm::cl::init(BytecodeVersion::kCurrentVersion));

  bytecodeVersionOpt = &bytecodeVersion;
}

BytecodeVersion mlir::cuda_tile::getCurrentBytecodeVersion() {
  return bytecodeVersionOpt ? bytecodeVersionOpt->getValue()
                            : BytecodeVersion::kCurrentVersion;
}

bool mlir::cuda_tile::isBytecodeVersionExplicitlySet() {
  return bytecodeVersionOpt && bytecodeVersionOpt->getNumOccurrences() > 0;
}

void mlir::cuda_tile::registerTileIROptimizationHintsOptions() {
  // Register command line option for hint diagnostics
  static llvm::cl::opt<bool, true> warnUnsupportedHints(
      "Wunsupported-hints",
      llvm::cl::desc(
          "Enable warnings for unsupported/invalid optimization hints."),
      llvm::cl::location(warnUnsupportedHintsVar), llvm::cl::init(false));

  static llvm::cl::opt<bool, true> errorOnHints(
      "Werr-hints",
      llvm::cl::desc("Treat unsupported/invalid optimization hints as errors."),
      llvm::cl::location(errorUnsupportedHintsVar), llvm::cl::init(false));
}

bool PreviewFeatureCLParser::parse(llvm::cl::Option &option,
                                   llvm::StringRef /*argName*/,
                                   llvm::StringRef arg, PreviewFeature &value) {
  std::optional<PreviewFeature> feature = symbolizePreviewFeature(arg);
  if (!feature) {
    std::string supported;
    llvm::raw_string_ostream os(supported);
    llvm::interleaveComma(getSupportedPreviewFeatures(), os, [&](auto item) {
      os << stringifyPreviewFeature(item);
    });
    return option.error("unknown preview feature '" + arg +
                        "'; supported features: " + os.str());
  }
  if (isRetiredPreviewFeature(*feature)) {
    return option.error("preview feature '" + arg + "' has been retired");
  }
  value = *feature;
  return false;
}

void mlir::cuda_tile::registerTileIRPreviewOption() {
  static llvm::cl::list<PreviewFeature, /*StorageClass=*/bool,
                        PreviewFeatureCLParser>
      previewFeatures(
          "preview",
          llvm::cl::desc("Enable one or more named preview features"),
          llvm::cl::value_desc("feature[,feature...]"),
          llvm::cl::CommaSeparated, llvm::cl::ValueRequired);
  previewFeaturesOpt = std::addressof(previewFeatures);
}

PreviewFeatures mlir::cuda_tile::getPreviewFeaturesFromCommandLine() {
  PreviewFeatures features;
  if (!previewFeaturesOpt) {
    return features;
  }
  for (PreviewFeature feature : *previewFeaturesOpt) {
    features.enable(feature);
  }
  return features;
}

bool mlir::cuda_tile::getWarnUnsupportedHints() {
  return warnUnsupportedHintsVar;
}

bool mlir::cuda_tile::getErrorUnsupportedHints() {
  return errorUnsupportedHintsVar;
}

void mlir::cuda_tile::registerListVersionsOption() {
  static llvm::cl::opt<bool> listVersions(
      "list-versions",
      llvm::cl::desc("List all supported bytecode versions and exit"),
      llvm::cl::init(false), llvm::cl::callback([](const bool &val) {
        if (val) {
          auto versions = getSupportedVersions();
          for (const auto &version : versions) {
            llvm::outs() << version.toString() << "\n";
          }
          exit(0);
        }
      }));
}
