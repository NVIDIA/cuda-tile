//===- LICM.cpp - CUDA Tile Loop Invariant Code Motion Pass -----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs loop-invariant code motion (LICM) on cuda_tile.for and
// cuda_tile.loop operations. The standard MLIR LICM pass requires
// LoopLikeOpInterface, which these ops do not implement due to their
// non-standard control flow (ContinueOp/BreakOp inside IfOp regions).
//
// An operation is hoisted before the enclosing loop if ALL of the following
// hold:
//   1. It is side-effect-free (isMemoryEffectFree).
//   2. It is safe to speculate (isSpeculatable) -- this prevents hoisting
//      ops like divi with a dynamic divisor that could introduce division-by-
//      zero on zero-trip loops.
//   3. It is not a terminator (ContinueOp, BreakOp, YieldOp, ReturnOp).
//   4. All of its operands are either defined outside the loop region or are
//      produced by operations that will themselves be hoisted.
//
// The pass uses a single pre-order walk over the loop body (including nested
// non-isolated regions such as IfOp regions) to collect hoistable ops.
// Pre-order ensures that definitions are visited before uses, so a
// "willHoist" set can track transitive invariance in one pass without a
// fixpoint loop. Operations with the IsolatedFromAbove trait are considered
// for hoisting as a whole, but their regions are not traversed: moving an
// operation out of an isolated region without moving the isolated parent
// would create an illegal cross-region SSA use. Nested ForOp/LoopOp regions
// are also skipped (WalkResult::skip) since they are processed by their own
// LICM invocation.
//
// The outer driver uses a post-order walk over the function, processing inner
// loops before outer loops. This allows invariants to cascade outward through
// nested loop nests (e.g., a constant inside a loop-inside-a-loop gets hoisted
// all the way out in two steps).
//
// Differences from the standard MLIR LICM (moveLoopInvariantCode):
//
//   - Targets ForOp/LoopOp directly instead of requiring LoopLikeOpInterface,
//     which CudaTile loops cannot implement due to ContinueOp/BreakOp early
//     exits through nested IfOp regions.
//
//   - Hoists from inside non-isolated nested regions such as IfOp, ReduceOp
//     and ScanOp regions, not only from the top-level loop body block.
//     IsolatedFromAbove operations remain indivisible hoisting boundaries. The
//     standard MLIR LICM only inspects ops returned by getLoopRegions(), which
//     are the immediate loop body blocks.
//
//   - Uses a single pre-order walk with a "willHoist" set for transitive
//     invariance, instead of the standard approach of iterating over top-level
//     ops and relying on the caller to re-run for transitive cases.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace mlir::cuda_tile {

/// Return true if |op| can be hoisted out of |loopRegion|. The |willHoist|
/// set contains ops that have already been marked for hoisting during the
/// current pre-order walk; their results are treated as defined outside the
/// loop for the purpose of checking operand invariance.
static bool isHoistableOp(Operation *op, Region &loopRegion,
                          const DenseSet<Operation *> &willHoist) {
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  if (!isMemoryEffectFree(op) || !isSpeculatable(op)) {
    return false;
  }

  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      if (loopRegion.isAncestor(operand.getParentRegion())) {
        return false;
      }
      continue;
    }
    if (willHoist.contains(defOp)) {
      continue;
    }
    if (loopRegion.isAncestor(defOp->getParentRegion())) {
      return false;
    }
  }

  return true;
}

/// Collect and hoist all loop-invariant ops from |loopOp|'s body region.
/// Uses a pre-order walk to discover invariant ops in a single pass,
/// descending into non-isolated regions but skipping the regions of
/// IsolatedFromAbove operations. The isolated operation itself is still
/// considered for whole-operation hoisting. Nested ForOp/LoopOp operations
/// are skipped entirely and handled by the outer post-order driver. Returns
/// true if any ops were hoisted.
static bool hoistFromLoop(Operation *loopOp) {
  Region &loopRegion = loopOp->getRegion(0);
  DenseSet<Operation *> willHoist;
  SmallVector<Operation *> toHoist;

  loopRegion.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ForOp, LoopOp>(op)) {
      return WalkResult::skip();
    }

    if (isHoistableOp(op, loopRegion, willHoist)) {
      willHoist.insert(op);
      toHoist.push_back(op);
    }

    // An IsolatedFromAbove operation may be hoisted as a whole, but none of
    // its nested operations may be moved across the isolation boundary on
    // their own.
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  for (Operation *op : toHoist) {
    op->moveBefore(loopOp);
  }

  return !toHoist.empty();
}

#define GEN_PASS_DEF_CUDATILELICMPASS
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc"

struct CudaTileLICMPass : public impl::CudaTileLICMPassBase<CudaTileLICMPass> {
public:
  using impl::CudaTileLICMPassBase<CudaTileLICMPass>::CudaTileLICMPassBase;

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PostOrder>([](Operation *op) {
      if (isa<ForOp, LoopOp>(op)) {
        hoistFromLoop(op);
      }
      return WalkResult::advance();
    });
  }
};

} // namespace mlir::cuda_tile
