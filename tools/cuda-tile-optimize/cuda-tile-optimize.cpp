//===- cuda-tile-optimize.cpp -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA Tile Optimizer,
// which is a standalone tool that performs CUDA Tile IR Bytecode -> CUDA Tile
// IR Bytecode optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include "cuda_tile/Dialect/CudaTile/Optimizer/CudaTileOptimizer.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {
//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//
struct Options {
  llvm::cl::OptionCategory optCategory{"TileIR Optimizer Options"};

  llvm::cl::opt<std::string> inputFile{
      llvm::cl::Positional,
      llvm::cl::desc("<tile bytecode file>"),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<std::string> outputFile{
      "output-file",
      llvm::cl::desc("Specify name of output file."),
      llvm::cl::value_desc("file"),
      llvm::cl::init(""),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::alias outputFileAlias{
      "o",
      llvm::cl::desc("Alias for --output-file"),
      llvm::cl::aliasopt(outputFile),
      llvm::cl::NotHidden,
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<bool> quiet{
      "quiet",
      llvm::cl::desc("Don't produce output to file/screen"),
      llvm::cl::init(false),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::alias quietAlias{
      "q",
      llvm::cl::desc("Alias for --quiet"),
      llvm::cl::aliasopt(quiet),
      llvm::cl::NotHidden,
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<bool> emitBytecode{
      "emit-bytecode",
      llvm::cl::desc("Emit Bytecode to output file"),
      llvm::cl::init(false),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<std::string> pipelinePre{
      "run-before-default-pipeline",
      llvm::cl::desc("Add passes before default pipeline"),
      llvm::cl::value_desc("MLIR Pipeline"),
      llvm::cl::init(""),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::alias preAlias{
      "before",
      llvm::cl::desc("Alias for --run-before-default-pipeline"),
      llvm::cl::aliasopt(pipelinePre),
      llvm::cl::NotHidden,
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<std::string> pipelinePost{
      "run-after-default-pipeline",
      llvm::cl::desc("Add passes after default pipeline"),
      llvm::cl::value_desc("MLIR Pipeline"),
      llvm::cl::init(""),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::alias postAlias{
      "after",
      llvm::cl::desc("Alias for --run-after-default-pipeline"),
      llvm::cl::aliasopt(pipelinePost),
      llvm::cl::NotHidden,
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<bool> fuseFMA{
      "fuse-fma",
      llvm::cl::desc("Enable FMA Fusion pass"),
      llvm::cl::init(false),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<int> optLevel{
      "opt-level",
      llvm::cl::desc("Specify optimization level. Default Value: 3."),
      llvm::cl::value_desc("N"),
      llvm::cl::init(3),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::alias optLevelAlias{
      "O",
      llvm::cl::desc("Alias for --opt-level"),
      llvm::cl::aliasopt(optLevel),
      llvm::cl::NotHidden,
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<bool> verbose{
      "verbose",
      llvm::cl::desc("Enable verbose output from CudaTileOptimizer"),
      llvm::cl::init(false),
      llvm::cl::cat(optCategory),
  };
  llvm::cl::alias verboseAlias{
      "v",
      llvm::cl::desc("Alias for --verbose"),
      llvm::cl::aliasopt(verbose),
      llvm::cl::NotHidden,
      llvm::cl::cat(optCategory),
  };
  llvm::cl::opt<bool> enableMultithread{
      "enable-multithread",
      llvm::cl::desc("Enable MLIR Multithreading"),
      llvm::cl::init(false),
      llvm::cl::cat(optCategory),
  };
};

} // namespace

int main(int argc, char **argv) {
  Options options;

  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    StringRef date(STD_DATE);
    StringRef dateYear = date.take_back(4);

    StringRef toolVersion;
#ifdef TOOLS_VERSION
#ifdef TOOLS_VERSION_EXTENDED
    toolVersion = TOOLS_VERSION "\n" TOOLS_VERSION_EXTENDED "\n";
#else
    toolVersion = TOOLS_VERSION "\n";
#endif // TOOLS_VERSION_EXTENDED
#endif // TOOLS_VERSION

    // Format for the version string:
    //   {0}: The current year.
    //   {1}: The build date.
    //   {2}: Optional tool version.
    constexpr StringLiteral versionFormat = R"(
cuda-tile-optimize: NVIDIA (R) Cuda TileIR->TileIR optimizer
Copyright (c) 2005-{0} NVIDIA Corporation
Built on {1}
{2})";

    os << llvm::formatv(versionFormat.ltrim().data(), dateYear, date,
                        toolVersion);
  });

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "cuda-tile-optimize: NVIDIA (R) CUDA "
                                         "Tile IR -> CUDA Tile IR optimizer\n"))
    return 1;

  if (options.inputFile.empty()) {
    llvm::errs() << "error: no input file provided\n\n";
    llvm::cl::PrintHelpMessage();
    return 1;
  }

  TileIROptimizerConfig cfg;
  cfg.verbose = options.verbose;
  cfg.input = TileIROptInput::fromFile(options.inputFile);

  // Pipeline toggles (positive logic now)
  cfg.opt.enableFuseFMA = options.fuseFMA;
  cfg.opt.optLevel = options.optLevel;
  cfg.opt.enableMultithread = options.enableMultithread;
  // User specified pipeline
  cfg.opt.pipelinePreText = options.pipelinePre;
  cfg.opt.pipelinePostText = options.pipelinePost;

  // Output selection
  // Output mode priority:
  // 1. File output (bytecode/MLIR) when outputFile specified
  // 2. Add stdout in verbose mode
  // 3. Default to stdout when not quiet and no output file
  cfg.output.mode = TileIROptOutputMode::None;
  if (!options.outputFile.empty()) {
    // File output specified
    if (options.emitBytecode) {
      cfg.output.mode = TileIROptOutputMode::BytecodeFile;
      cfg.output.bytecodeFile = options.outputFile;
    } else {
      cfg.output.mode = TileIROptOutputMode::MlirFile;
      cfg.output.mlirFile = options.outputFile;
    }
    // Verbose mode adds stdout output alongside file output
    if (options.verbose)
      cfg.output.mode |= TileIROptOutputMode::MlirStdout;
  } else if (!options.quiet) {
    // Default to stdout when no file output and not quiet
    cfg.output.mode = TileIROptOutputMode::MlirStdout;
  }

  // Set up diagnostic handler to print errors/remarks to stderr
  // Note: The context is created inside optimizeTileIR, so we can't set up
  // the handler before calling it. The diagnostics will be handled by the
  // default MLIR diagnostic handler which prints to stderr.

  if (failed(optimizeTileIR(cfg))) {
    // Error diagnostics have already been emitted to stderr
    return 1;
  }

  return 0;
}
