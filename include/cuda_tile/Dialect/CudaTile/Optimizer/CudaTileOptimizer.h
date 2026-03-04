//===- CudaTileOptimizer.h --------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_OPTIMIZER_H
#define CUDA_TILE_DIALECT_CUDATILE_OPTIMIZER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/MemoryBuffer.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <cstdint>
#include <string>

namespace mlir::cuda_tile {

/// Where to emit results.
/// Can be combined as a bitmask, e.g. MlirFile | Screen
enum class TileIROptOutputMode : uint32_t {
  None = 0,
  // write CUDA Tile IR bytecode to file
  BytecodeFile = 1u << 0,
  // return CUDA Tile IR bytecode in memory (std::string*)
  BytecodeMemory = 1u << 1,
  // write MLIR textual IR to file
  MlirFile = 1u << 2,
  // print MLIR textual IR to screen (llvm::outs by default)
  MlirStdout = 1u << 3
};
} // namespace mlir::cuda_tile

namespace llvm {
LLVM_DECLARE_ENUM_AS_BITMASK(
    mlir::cuda_tile::TileIROptOutputMode,
    (uint32_t)mlir::cuda_tile::TileIROptOutputMode::MlirStdout);
}

namespace mlir::cuda_tile {

/// Pipeline optimization options.
struct TileIROptimizerOptions {
  bool enableMultithread = false;
  bool enableFuseFMA = false;
  int optLevel = 3;
  int loopSplitThreshold = 1;

  // User can specify additional passes to be added
  // before and/or after default pipeline.
  // Note: Textual pipeline (MLIR pass pipeline grammar)
  // is parsed into the nested OpPassManager on cuda_tile::EntryOp
  std::string pipelinePreText = "";
  std::string pipelinePostText = "";
};

void registerTileIROptPasses();

LogicalResult optimizeTileIRModule(ModuleOp module,
                                   const TileIROptimizerOptions &opts,
                                   bool verbose = false);

struct TileIROptInput {
  using BufferT = llvm::StringRef;
  using FileT = std::string;

  // The actual payload
  std::variant<BufferT, FileT> value;

  static TileIROptInput fromBuffer(BufferT buf) {
    TileIROptInput in;
    in.value = buf;
    return in;
  }

  static TileIROptInput fromFile(FileT filename) {
    TileIROptInput in;
    in.value = std::move(filename);
    return in;
  }
};

struct TileIROptOutput {
  // Output selection.
  TileIROptOutputMode mode = TileIROptOutputMode::None;
  // Bytecode outputs:
  // used if outputMode has BytecodeFile
  std::string bytecodeFile;
  // used if outputMode has BytecodeMemory
  std::string *bytecodeBuffer = nullptr;

  // MLIR outputs:
  // used if outputMode has MlirFile
  std::string mlirFile;

  // Screen output (MLIR text). If null, defaults to llvm::outs().
  // used if outputMode has MlirStdout
  llvm::raw_ostream *screenOS = nullptr;
};

/// Options for bytecode -> optimize -> bytecode.
struct TileIROptimizerConfig {
  // Input configuration
  TileIROptInput input;

  // Output configuration
  TileIROptOutput output;

  // Optimization pipeline configuration.
  TileIROptimizerOptions opt;

  // Enable verbose output
  bool verbose = false;
};

/// Optimize a CUDA Tile IR bytecode buffer and re-emit bytecode according to
/// options. On success(), writes to file and/or memory per `opts.outputMode`.
mlir::LogicalResult optimizeTileIR(TileIROptimizerConfig &cfg);

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_DIALECT_CUDATILE_OPTIMIZER_H
