//===- BytecodeVerification.cpp - Bytecode format verification --*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Common/BytecodeVerification.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
using namespace mlir;
using namespace mlir::cuda_tile;

LogicalResult
cuda_tile::verifySelfContainedModuleAndOperationInvariants(ModuleOp module) {
  if (module.getBody().empty()) {
    return module.emitOpError("module body is empty");
  }

  // Validate that we have a self-contained module that matches what we can
  // encode within the bytecode (e.g. no non-functions/globals nested in the
  // module).
  for (Operation &op : module.getBody().front()) {
    if (!isa<FunctionOpInterface, GlobalOp>(&op)) {
      auto diag = module.emitOpError(
          "only function and global ops are allowed in the body");
      diag.attachNote(op.getLoc()) << "invalid op: " << op.getName();
      return diag;
    }

    if (auto func = dyn_cast<FuncOp>(&op); func && !func.getIsExtern())
      return func.emitOpError("must be extern when serialized to bytecode");
    auto entry = dyn_cast<EntryOp>(&op);
    if (!entry)
      continue;
    for (auto [index, type] : llvm::enumerate(entry.getArgumentTypes())) {
      // EntryOp::verify() rejects non-scalar TileType arguments.
      if (!isa<TileType>(type)) {
        return entry.emitError()
               << "unsupported type in bytecode for entry argument #" << index
               << ": " << type << "; expected a rank-0 !cuda_tile.tile type";
      }
    }
  }

  // Allow only ops from the CudaTile dialect inside of the module (at any
  // nesting level). A call to a body-bearing function is valid in memory so
  // that it can be inlined, but it cannot cross the bytecode boundary.
  Dialect *dialect = module->getDialect();
  WalkResult status = module->walk([&](Operation *op) {
    if (op->getDialect() != dialect) {
      auto diag = module.emitOpError("only ops from the '")
                  << dialect->getNamespace() << "' dialect are allowed";
      diag.attachNote(op->getLoc()) << "invalid op: " << op->getName();
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (status.wasInterrupted()) {
    return failure();
  }

  SymbolTableCollection symbolTables;
  status = module->walk([&](CallOp call) {
    // An unresolved callee may be supplied by a later linking step. If the
    // callee is present in this module, bytecode may only reference an extern
    // function declaration.
    Operation *callee =
        symbolTables.lookupNearestSymbolFrom(call, call.getCalleeAttr());
    if (!callee) {
      return WalkResult::advance();
    }
    auto func = dyn_cast<FuncOp>(callee);
    if (!func || !func.getIsExtern()) {
      call.emitOpError() << "'" << call.getCallee()
                         << "' must reference an extern preview$func when "
                            "serialized to bytecode";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return status.wasInterrupted() ? failure() : success();
}
