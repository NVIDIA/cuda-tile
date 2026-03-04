//===- CudaTileOptimizer.cpp ------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CUDA Tile IR -> CUDA Tile IR Bytecode optimization flow
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/Optimizer/CudaTileOptimizer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {

/// Parse optimization pipeline from text
static LogicalResult parseTextInto(llvm::StringRef text, OpPassManager &PM,
                                   MLIRContext *context) {
  if (text.empty())
    return success();
  // Parsing textual pipeline into an existing (nested) OpPassManager.
  // NOTE: because opPM is already nested for cuda_tile::EntryOp, the text
  // should NOT include an op anchor.
  if (failed(parsePassPipeline(text, PM))) {
    emitError(UnknownLoc::get(context)) << "Failed to parse pipeline: " << text;
    return failure();
  }
  return success();
}

/// Build default optimization pipeline
static LogicalResult
buildDefaultCudaTilePipeline(OpPassManager &nested,
                             const TileIROptimizerOptions &opts) {
  // 1) Optional FMA fusion
  if (opts.enableFuseFMA)
    nested.addPass(createFuseFMAPass());

  if (opts.optLevel >= 1) {
    // 2) Canonicalize + CSE before further opts
    nested.addPass(createCanonicalizerPass());
    nested.addPass(createCSEPass());

    if (opts.optLevel >= 2) {
      nested.addPass(createLoopInvariantCodeMotionPass());

      if (opts.optLevel >= 3) {
        // 3) loop split, followed by another canonicalization sweep.
        nested.addPass(createLoopSplitPass({opts.loopSplitThreshold}));
        nested.addPass(createCanonicalizerPass());
      }
    }
  }
  return success();
}

/// Build optimization pipeline (default or with builder/text overrides)
static LogicalResult
buildCudaTileOptimizationPipeline(PassManager &pm,
                                  const TileIROptimizerOptions &opts) {
  // Pipeline is nested under cuda_tile::EntryOp.
  auto &nested = pm.nestAny();

  // Add additional passes before default pipeline
  if (failed(parseTextInto(opts.pipelinePreText, nested, pm.getContext())))
    return failure();

  // Add default pipeline
  if (failed(buildDefaultCudaTilePipeline(nested, opts)))
    return failure();

  // Add additional passes after default pipeline
  return parseTextInto(opts.pipelinePostText, nested, pm.getContext());
}

//===----------------------------------------------------------------------===//
// CUDA Tile IR parsing
//===----------------------------------------------------------------------===//

/// Parses the given bytecode buffer into a CUDA Tile IR module. Returns null if
/// the buffer is not valid bytecode.
OwningOpRef<mlir::ModuleOp> parseTileIRBytecode(llvm::MemoryBufferRef bytecode,
                                                MLIRContext &context) {
  // Check if this is CUDA Tile IR bytecode.
  if (!isTileIRBytecode(bytecode))
    return {};
  context.loadDialect<cuda_tile::CudaTileDialect>();

  OwningOpRef<cuda_tile::ModuleOp> tileIRModule =
      cuda_tile::readBytecode(bytecode, context);
  if (!tileIRModule)
    return {};

  // Wrap the bytecode module into a builtin module.
  OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(UnknownLoc::get(&context));
  module->getBody()->push_back(tileIRModule.release());
  return module;
}

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

// write Bytecode to buffer
static LogicalResult writeBytecodeToBuffer(cuda_tile::ModuleOp module,
                                           std::string &out) {
  out.clear();
  llvm::raw_string_ostream os(out);
  if (failed(cuda_tile::writeBytecode(os, module,
                                      BytecodeVersion::kCurrentVersion)))
    return failure();
  os.flush();
  return success();
}

// Utility: emit error and return failure().
static LogicalResult emitConfigError(MLIRContext *context, const char *msg) {
  return emitError(UnknownLoc::get(context)) << msg;
}
static LogicalResult emitConfigError(MLIRContext *context, std::string msg) {
  return emitError(UnknownLoc::get(context)) << msg;
}

// Validate provided configuration
static LogicalResult validateConfig(TileIROptimizerConfig &cfg,
                                    MLIRContext *context) {
  using O = TileIROptOutputMode;
  if ((cfg.output.mode & O::BytecodeFile) != O::None &&
      cfg.output.bytecodeFile.empty())
    return emitConfigError(
        context, "BytecodeFile requested but bytecodeOutputFile is empty");
  if ((cfg.output.mode & O::BytecodeMemory) != O::None &&
      !cfg.output.bytecodeBuffer)
    return emitConfigError(
        context, "BytecodeMemory requested but bytecodeOutputBuffer==nullptr");
  if ((cfg.output.mode & O::MlirFile) != O::None && cfg.output.mlirFile.empty())
    return emitConfigError(context,
                           "MlirFile requested but mlirOutputFile is empty");
  // Input Buffer case
  if (auto buf = std::get_if<TileIROptInput::BufferT>(&cfg.input.value))
    if (buf->empty())
      return emitConfigError(context, "loadInputModule: input buffer is empty");
  // Input File case
  if (auto fname = std::get_if<TileIROptInput::FileT>(&cfg.input.value))
    if (fname->empty())
      return emitConfigError(context, "loadInputModule: filename is empty");

  return success();
}

// Loads/produces a ModuleOp into `outMod` based on cfg.input.kind.
// Returns success() on success, failure() on any error.
static LogicalResult loadInputModule(TileIROptimizerConfig &cfg,
                                     MLIRContext &context,
                                     OwningOpRef<mlir::ModuleOp> &outMod) {
  // The values of cfg.input.buffer & cfg.input.filename are already checked
  // during the call of validateConfig()
  // 1) Materialize a MemoryBuffer + MemoryBufferRef regardless of source
  std::unique_ptr<llvm::MemoryBuffer> owned;
  llvm::MemoryBufferRef ref;
  StringRef *buf = nullptr;

  if (auto fname = std::get_if<TileIROptInput::FileT>(&cfg.input.value)) {
    // Read raw bytes (no text-mode CRLF translation), so detection is reliable.
    auto bufOrErr = llvm::MemoryBuffer::getFile(*fname, /*IsText=*/false);
    if (!bufOrErr)
      return emitConfigError(
          &context, (llvm::Twine("Failed to read file: ") + *fname).str());

    owned = std::move(*bufOrErr);
    ref = owned->getMemBufferRef();
  } else {
    buf = std::get_if<TileIROptInput::BufferT>(&cfg.input.value);
    // No copy here. Build a non-owning view onto caller's memory.
    ref = llvm::MemoryBufferRef(*buf, "");
  }

  // Parse depending on detected type
  if (cuda_tile::isTileIRBytecode(ref)) {
    // CUDA Tile IR bytecode
    outMod = parseTileIRBytecode(ref, context);
  } else {
    // MLIR textual IR
    llvm::SourceMgr sm;
    if (buf) {
      // Create an owned, null-terminated copy ONLY for the Buffer path.
      // This guarantees ownership + '\0' for SourceMgr.
      owned = llvm::MemoryBuffer::getMemBufferCopy(*buf, "");
      if (!owned)
        return emitConfigError(&context,
                               "Failed to allocate buffer copy for MLIR text.");
    }
    // If cfg.input.kind == K::File, 'owned' was already set from getFile()
    // above.
    sm.AddNewSourceBuffer(std::move(owned), llvm::SMLoc());
    outMod = parseSourceFile<mlir::ModuleOp>(sm, &context);
  }

  if (!outMod)
    return emitConfigError(&context, "Failed to parse input");
  return success();
}

static LogicalResult emitOutputs(TileIROptimizerConfig &cfg,
                                 OwningOpRef<mlir::ModuleOp> &parentModule) {
  using O = TileIROptOutputMode;
  MLIRContext *context = parentModule->getContext();

  // 1) Bytecode: file / memory
  if ((cfg.output.mode & (O::BytecodeFile | O::BytecodeMemory)) != O::None) {
    // → Generate bytecode once to memory, then branch.
    std::string bc;
    if (failed(
            writeBytecodeToBuffer(extractCudaTileModuleOp(*parentModule), bc)))
      return emitConfigError(context, "Bytecode generation failed");

    if ((cfg.output.mode & O::BytecodeFile) != O::None) {
      std::string err;
      if (auto of = openOutputFile(cfg.output.bytecodeFile, &err)) {
        of->os().write(bc.data(), bc.size());
        of->keep();
      } else {
        return emitConfigError(context, err);
      }
    }
    if ((cfg.output.mode & O::BytecodeMemory) != O::None)
      *cfg.output.bytecodeBuffer = std::move(bc);
  }

  // 2) MLIR textual: file / screen
  if ((cfg.output.mode & (O::MlirFile | O::MlirStdout)) != O::None) {
    // Print once to a string and reuse for file / screen.
    std::string mlirText;
    {
      llvm::raw_string_ostream os(mlirText);
      // Optional: pass OpPrintingFlags if you want elideAttrs(), etc.
      parentModule->print(os);
    }

    if ((cfg.output.mode & O::MlirFile) != O::None) {
      std::string err;
      if (auto of = openOutputFile(cfg.output.mlirFile, &err)) {
        of->os() << mlirText;
        of->keep();
      } else {
        return emitConfigError(context, err);
      }
    }

    if ((cfg.output.mode & O::MlirStdout) != O::None)
      (cfg.output.screenOS ? *cfg.output.screenOS : llvm::outs())
          << mlirText << '\n';
  }

  return success();
}

} // namespace

namespace mlir::cuda_tile {

void registerTileIROptPasses() {
  registerCudaTilePasses();
  registerTransformsPasses();
}

// -----------------------------------------------------------------------------
// 2) optimize CUDA Tile IR module - shared optimization pass with CAPI
// -----------------------------------------------------------------------------
LogicalResult optimizeTileIRModule(ModuleOp module,
                                   const TileIROptimizerOptions &opts,
                                   bool verbose) {
  // Build a PassManager specialized for cuda_tile::ModuleOp.
  PassManager pm = PassManager::on<ModuleOp>(module->getContext());

  if (failed(buildCudaTileOptimizationPipeline(pm, opts)))
    return failure();

  if (verbose) {
    pm.enableVerifier(true);
    std::string pipe;
    llvm::raw_string_ostream os(pipe);
    pm.printAsTextualPipeline(os, true);
    emitRemark(UnknownLoc::get(module->getContext())) << "Pipeline: " << pipe;
  }
  return pm.run(module);
}

// -----------------------------------------------------------------------------
// optimizeTileIR - calls:
// 1) loadInputModule - from file or buffer: Bytecode or MLIR Text format
// 2) optimizeTileIR - run optimization pipeline
// 3) emitOutputs - writes output to file, buffer or screen: Bytecode or MLIR
// -----------------------------------------------------------------------------
LogicalResult optimizeTileIR(TileIROptimizerConfig &cfg) {
  // Create a context and register the CudaTile dialect.
  DialectRegistry registry;
  registry.insert<CudaTileDialect>();

  MLIRContext context(registry, cfg.opt.enableMultithread
                                    ? MLIRContext::Threading::ENABLED
                                    : MLIRContext::Threading::DISABLED);

  // Enable printing of remarks if verbose mode is on
  if (cfg.verbose) {
    context.getDiagEngine().registerHandler([](Diagnostic &diag) {
      // Print all diagnostics (including remarks) to stderr
      if (diag.getSeverity() == DiagnosticSeverity::Remark ||
          diag.getSeverity() == DiagnosticSeverity::Warning ||
          diag.getSeverity() == DiagnosticSeverity::Error) {
        llvm::errs() << diag << "\n";
      }
      return success();
    });
  }

  // Validate user-provided configuration.
  if (failed(validateConfig(cfg, &context)))
    return failure();

  // Parse the input
  OwningOpRef<mlir::ModuleOp> parentModule;
  if (failed(loadInputModule(cfg, context, parentModule)))
    return failure();

  auto module = extractCudaTileModuleOp(*parentModule);
  if (!module)
    return emitError(UnknownLoc::get(&context))
           << "cuda_tile.ModuleOp not found in input";

  registerTileIROptPasses();

  // Build & run the optimization pipeline
  if (failed(optimizeTileIRModule(module, cfg.opt, cfg.verbose)))
    return emitError(UnknownLoc::get(&context))
           << "Failed to optimize CUDA Tile IR program";

  // No output is requested by caller
  if (cfg.output.mode == TileIROptOutputMode::None)
    return success();

  return emitOutputs(cfg, parentModule);
}

} // namespace mlir::cuda_tile
