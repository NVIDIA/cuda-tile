//===- CommandLineOptions.h - CUDA Tile Bytecode CLI Options ----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_COMMON_COMMANDLINEOPTIONS_H
#define CUDA_TILE_BYTECODE_COMMON_COMMANDLINEOPTIONS_H

#include "llvm/Support/CommandLine.h"

#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.h"
namespace mlir {
namespace cuda_tile {

/// Register command line options for Cuda Tile IR bytecode version.
void registerTileIRBytecodeVersionOption();

/// Register command line options for OptimizationHints diagnostics
void registerTileIROptimizationHintsOptions();

/// Command-line parser for a named preview feature.
class PreviewFeatureCLParser : public llvm::cl::basic_parser<PreviewFeature> {
public:
  PreviewFeatureCLParser(llvm::cl::Option &option)
      : llvm::cl::basic_parser<PreviewFeature>(option) {}

  bool parse(llvm::cl::Option &option, llvm::StringRef argName,
             llvm::StringRef arg, PreviewFeature &value);
  llvm::StringRef getValueName() const override {
    return "feature[,feature...]";
  }
  void printOptionDiff(const llvm::cl::Option &option, PreviewFeature,
                       llvm::cl::OptionValue<PreviewFeature>,
                       size_t globalWidth) const {
    printOptionNoValue(option, globalWidth);
  }
};

/// Register `--preview=<feature>[,<feature>...]`.
void registerTileIRPreviewOption();

/// Get the preview features enabled by command-line options. Returns an empty
/// set if the option was not provided.
PreviewFeatures getPreviewFeaturesFromCommandLine();

/// Get the current bytecode version from command line options.
/// Returns the default version if no command line option was set.
BytecodeVersion getCurrentBytecodeVersion();

/// Returns whether the bytecode version was explicitly set on the command line
/// (as opposed to taking its default value).
bool isBytecodeVersionExplicitlySet();

/// Get the unsupported optimization hint warning flag from command line
/// options. Returns the default (false) option if no command line option was
/// set.
bool getWarnUnsupportedHints();

/// Get the unsupported optimization hint error flag from command line options.
/// Returns the default (false) option if no command line option was set.
bool getErrorUnsupportedHints();

/// Register command line option to list supported bytecode versions.
void registerListVersionsOption();

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_BYTECODE_COMMON_COMMANDLINEOPTIONS_H
