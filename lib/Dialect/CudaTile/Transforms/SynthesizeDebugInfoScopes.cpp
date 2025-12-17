//===- SynthesizeDebugInfoScopes.cpp - Debug Info Scopes --------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

namespace mlir::cuda_tile {
#define GEN_PASS_DEF_SYNTHESIZEDEBUGINFOSCOPESPASS
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc"
} // namespace mlir::cuda_tile

using namespace mlir;
using namespace mlir::cuda_tile;

/// Attempt to extract a filename for the given loc.
static FileLineColLoc extractFileLoc(Location loc) {
  return TypeSwitch<Location, FileLineColLoc>(loc)
      .Case([](FileLineColLoc loc) { return loc; })
      .Case([](NameLoc loc) { return extractFileLoc(loc.getChildLoc()); })
      .Case([](OpaqueLoc loc) {
        return extractFileLoc(loc.getFallbackLocation());
      })
      .Default(FileLineColLoc());
}

/// Returns a new file attribute based on the given file location.
static DIFileAttr createFileForLoc(FileLineColLoc loc) {
  Builder builder(loc.getContext());
  StringRef inputFilePath = loc.getFilename().getValue();
  return builder.getType<DIFileAttr>(
      builder.getStringAttr(llvm::sys::path::filename(inputFilePath)),
      builder.getStringAttr(llvm::sys::path::parent_path(inputFilePath)));
}

/// Returns a new compile unit based on the file location contained within
/// `loc`.
static DICompileUnitAttr createCompileUnitForLoc(Location loc) {
  Builder builder(loc->getContext());

  // Create a fileAttr
  DIFileAttr fileAttr;
  if (FileLineColLoc fileLoc = extractFileLoc(loc)) {
    fileAttr = createFileForLoc(fileLoc);
  } else {
    fileAttr = builder.getType<DIFileAttr>(builder.getStringAttr("<unknown>"),
                                           builder.getStringAttr(""));
  }

  return DICompileUnitAttr::get(builder.getContext(), fileAttr);
}

/// Synthesize a scope for the given function operation. This essentially just
/// attaches a new `DISubprogram` to the operation.
static void synthesizeScopeForFunction(FunctionOpInterface funcOp,
                                       DICompileUnitAttr compileUnitAttr) {
  MLIRContext *context = funcOp.getContext();
  Location loc = funcOp.getLoc();

  // Skip functions that already have a scope.
  if (loc->findInstanceOf<DILocAttr>())
    return;

  // Filename, line and colmun to associate to the function. If we don't have a
  // proper line, just use 1 (the start of the file) as a reasonable default.
  DIFileAttr fileAttr = compileUnitAttr.getFile();
  unsigned line = 1;
  FileLineColLoc fileLoc = extractFileLoc(loc);
  if (fileLoc) {
    fileAttr = createFileForLoc(fileLoc);
    line = fileLoc.getLine();
  } else {
    fileLoc = FileLineColLoc::get(context, fileAttr.getName(),
                                  /*line=*/1, /*column=*/1);
  }

  // Create a new subprogram for the function.
  auto funcName = funcOp.getNameAttr();
  auto subprogramAttr =
      DISubprogramAttr::get(context, fileAttr, line, funcName, funcName,
                            compileUnitAttr, /*scopeLine=*/line);
  funcOp->walk([&](Operation *op) {
    FileLineColLoc opFileLoc = extractFileLoc(op->getLoc());
    if (!opFileLoc)
      opFileLoc = fileLoc;
    op->setLoc(DILocAttr::get(context, opFileLoc, subprogramAttr));
  });
}

namespace {
struct SynthesizeDebugInfoScopesPass
    : public cuda_tile::impl::SynthesizeDebugInfoScopesPassBase<
          SynthesizeDebugInfoScopesPass> {
  using Base::Base;

  void runOnOperation() override {
    cuda_tile::ModuleOp module = getOperation();

    // Create a compile unit for the module.
    DICompileUnitAttr compileUnit = createCompileUnitForLoc(module.getLoc());

    // Create subprograms for each function within the module.
    for (auto funcOp : module.getOps<FunctionOpInterface>())
      synthesizeScopeForFunction(funcOp, compileUnit);
  }
};
} // end anonymous namespace
