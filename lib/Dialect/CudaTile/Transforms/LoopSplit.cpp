//===- LoopSplit.cpp - CUDA Tile Loop Split Optimization Pass ---*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"
#include <regex>
#include <string>

using namespace mlir;
using namespace mlir::cuda_tile;

namespace mlir::cuda_tile {

/// Normalize a comparison to always be "iv <op> value"
//  Return false if comparison is not with induction variable
//  Return false if comparison signedness doesn't match ForOp signedness
static bool normalizeForOpCmp(ForOp forOp, CmpIOp cmp,
                              ComparisonPredicate &normalizedPred, Value &rhs) {
  // Currently ForOp always use signed comparison for bound checks,
  // so don't perform split if signedness of cmp doesn't match
  if (cmp.getSignedness() == Signedness::Unsigned)
    return false;
  auto iv = forOp.getInductionVar();
  auto pred = cmp.getComparisonPredicate();
  if (cmp.getLhs() == iv) {
    normalizedPred = pred;
    rhs = cmp.getRhs();
    return true;
  } else if (cmp.getRhs() == iv) {
    rhs = cmp.getLhs();
    switch (pred) {
    case ComparisonPredicate::LESS_THAN:
      normalizedPred = ComparisonPredicate::GREATER_THAN;
      break;
    case ComparisonPredicate::LESS_THAN_OR_EQUAL:
      normalizedPred = ComparisonPredicate::GREATER_THAN_OR_EQUAL;
      break;
    case ComparisonPredicate::GREATER_THAN:
      normalizedPred = ComparisonPredicate::LESS_THAN;
      break;
    case ComparisonPredicate::GREATER_THAN_OR_EQUAL:
      normalizedPred = ComparisonPredicate::LESS_THAN_OR_EQUAL;
      break;
    default:
      return false;
    }
    return true;
  }
  return false;
}

/// Return True if splitting loop for current branch seems profitable for
///  performance
static bool isSplitProfitable(ForOp forOp, IfOp ifOp, int threshold) {
  // If threshold is 1, splitting will occur regardless of the content of the
  // IfOp. In that case, we can short-circuit.
  if (threshold == 1)
    return true;
  auto countOps = [&](auto &opRange) {
    bool hasHeavyOps = false;
    int opCount = 0;
    for (Operation &op : opRange.getOps()) {
      hasHeavyOps |=
          isa<LoadPtrTkoOp, LoadViewTkoOp, StorePtrTkoOp, StoreViewTkoOp,
              MmaFOp, MmaIOp, ReduceOp, IfOp, ForOp>(op);
      opCount++;
    }
    return std::tuple(opCount - 1, hasHeavyOps);
  };
  auto [thenSize, hasHeavyOps] = countOps(ifOp.getThenRegion());
  int elseSize = 0;
  if (ifOp->getNumRegions() > 1) {
    bool elseHasHeavyOps = false;
    std::tie(elseSize, elseHasHeavyOps) = countOps(ifOp.getElseRegion());
    hasHeavyOps |= elseHasHeavyOps;
  }

  // Only split loop if there are either many operations
  // inside either the then or else block, or if any op is "expensive"
  return thenSize >= threshold || elseSize >= threshold || hasHeavyOps;
}

/// Check if an cuda_tile.if condition is a cmpi with induction variable.
//  Collect all branches with the same predicate into `ifOps` vector
static bool isSplittableCondition(ForOp forOp, IfOp ifOp,
                                  SmallPtrSet<Operation *, 4> &ifOps,
                                  ComparisonPredicate &predOut, Value &rhsOut,
                                  CmpIOp &cmpOpOut, bool &secondThen,
                                  bool &copyCmp, int threshold) {
  // Optimization hint says not to split loop at this branch
  if (!threshold)
    return false;

  // Condition is not Cmp operation
  auto cmp = ifOp.getCondition().getDefiningOp<CmpIOp>();
  if (!cmp)
    return false;

  ComparisonPredicate normalizedPred;
  Value rhs;
  // Normalizes the comparison so that induction variables are on the left.
  // If the comparison does not involve the induction variable (or not in a
  // tractable way), abort.
  if (!normalizeForOpCmp(forOp, cmp, normalizedPred, rhs))
    return false;

  // Check that we compare induction variable with loop invariant
  auto rhsOp = rhs.getDefiningOp();
  if (rhsOp && forOp.getBody()->findAncestorOpInBlock(*rhsOp))
    return false;

  // Check that predicate is supported and determine what block goes to the
  // first loop
  switch (normalizedPred) {
  case ComparisonPredicate::GREATER_THAN:
  case ComparisonPredicate::GREATER_THAN_OR_EQUAL:
    secondThen = true;
    break;
  case ComparisonPredicate::LESS_THAN:
  case ComparisonPredicate::LESS_THAN_OR_EQUAL:
    secondThen = false;
    break;
  default:
    return false;
  }

  copyCmp = false;
  // Collect all IfOps with the same predicate and check for profitability
  bool isProfitable = false;
  Value pred = cmp->getResult(0);
  for (Operation *user : pred.getUsers()) {
    // In order to delete CmpOp and copy only one side of IfOp during cloning
    // IfOp should be in the same loop as CmpOp
    if (isa<IfOp>(user)) {
      // Check whether there is at least one IfOp using the same predicate that
      // would benefit from splitting.
      isProfitable |= isSplitProfitable(forOp, cast<IfOp>(user), threshold);
      // Collect IfOps for partial copy only directly nested into ForOp
      if (user->getParentOp() == forOp.getOperation()) {
        ifOps.insert(user);
        continue;
      }
      // If the IfOp is nested, it will not be split, so we fall through to
      // ensure the comparison is kept.
    }
    // CmpOp has other uses, except directly nested IfOps - need to keep it
    copyCmp = true;
  }

  // No profitable IfOps found for splitting
  if (!isProfitable)
    return false;

  cmpOpOut = cmp;
  predOut = normalizedPred;
  rhsOut = rhs;
  return true;
}

/// Create a copy of the loop with new bounds & partial copy of if-blocks
static ForOp copyLoop(RewriterBase &rewriter, ForOp forOp, CmpIOp cmpOp,
                      SmallPtrSet<Operation *, 4> &ifOps,
                      SmallVector<Operation *> &opsToClone, Value lowerBound,
                      Value upperBound, ValueRange iterArgs, bool cloneThen) {
  Location loc = forOp.getLoc();
  Value step = forOp.getStep();
  IRMapping mapper;
  auto newLoop =
      ForOp::create(rewriter, loc, lowerBound, upperBound, step, iterArgs);
  rewriter.setInsertionPointToStart(newLoop.getBody());

  for (auto [orig, repl] : llvm::zip(forOp.getBody()->getArguments(),
                                     newLoop.getBody()->getArguments()))
    mapper.map(orig, repl);

  // Process all operations selected for copy
  for (Operation *op : opsToClone) {
    if (cmpOp.getOperation() == op) {
      // Replace CmpOp with constant value
      TileType constType = TileType::get({}, rewriter.getI1Type());
      llvm::APInt val(1, cloneThen ? 1 : 0);
      auto constAttr = DenseIntElementsAttr::get(constType, val);
      auto cmpConst =
          ConstantOp::create(rewriter, op->getLoc(), constType, constAttr);
      mapper.map(cmpOp.getResult(), cmpConst);
      continue;
    } else if (!ifOps.contains(op)) {
      rewriter.clone(*op, mapper);
      continue;
    }
    // Current operation is IfOp that we split
    IfOp ifOp = cast<IfOp>(op);
    bool is_continue = false;
    Region &region = cloneThen ? ifOp.getThenRegion() : ifOp.getElseRegion();
    for (Operation &subOp : region.front()) {
      // Copy all operations from one of the regions
      if (isa<ContinueOp>(subOp)) {
        rewriter.clone(subOp, mapper);
        // Stop cloning operations at ContinueOp
        is_continue = true;
        break;
      }
      if (isa<YieldOp>(subOp)) {
        // Map ifResult to the YieldOp
        auto yieldOp = cast<YieldOp>(subOp);
        for (auto [ifResult, yieldArg] :
             llvm::zip_equal(ifOp.getResults(), yieldOp.getOperands()))
          mapper.map(ifResult, mapper.lookupOrDefault(yieldArg));
      } else {
        // General operation
        rewriter.clone(subOp, mapper);
      }
    }
    // Continue was met inside if-block - don't need to copy operations below
    if (is_continue)
      break;
  }
  return newLoop;
}

// Helper function to return if step is equal to one
static inline bool isConstOne(ConstantOp op) {
  auto type = op.getType().getElementType();
  auto intType = llvm::dyn_cast<IntegerType>(type);
  if (!intType)
    return false;
  DenseIntOrFPElementsAttr cstAttr = op.getValue();
  if (cstAttr.size() != 1)
    return false;
  auto intData = cstAttr.tryGetValues<APInt>();
  if (succeeded(intData))
    return (*intData->begin() == 1);
  return false;
}

/// Split the loop at the correct threshold based on predicate.
static void performLoopSplit(RewriterBase &rewriter, ForOp forOp,
                             ComparisonPredicate pred, Value splitValue,
                             SmallPtrSet<Operation *, 4> &ifOps, CmpIOp cmp,
                             bool secondThen, bool copyCmp) {
  Location loc = forOp.getLoc();
  Value step = forOp.getStep();
  auto constStep = llvm::dyn_cast<ConstantOp>(step.getDefiningOp());
  ValueRange iterArgs = forOp.getInitValues();
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();

  // Compute split point depending on predicate.
  // Increase splitPoint by 1 in the case of GT or LTE
  Value splitPoint = splitValue;
  if (pred == ComparisonPredicate::GREATER_THAN ||
      pred == ComparisonPredicate::LESS_THAN_OR_EQUAL) {
    TileType constType = llvm::dyn_cast<TileType>(step.getType());
    auto intType = llvm::dyn_cast<IntegerType>(constType.getElementType());
    llvm::APInt val(intType.getWidth(), 1);
    auto constAttr = DenseIntElementsAttr::get(constType, val);
    auto constOp = ConstantOp::create(rewriter, loc, constType, constAttr);
    splitPoint = AddIOp::create(rewriter, loc, splitValue, constOp);
  }
  if (!constStep || !isConstOne(constStep)) {
    // Step is not equal to one (or dynamic)
    // Need special handling, so that loop split point is aligned (i.e. == lb +
    // k * step) So, splitPoint = start + Ceil(splitPoint - lb, step) * step
    Value diff = SubIOp::create(rewriter, loc, splitPoint, lb);
    Value k = DivIOp::create(rewriter, loc, diff, step, Signedness::Signed,
                             RoundingMode::POSITIVE_INF);
    Value kstep = MulIOp::create(rewriter, loc, k, step);
    splitPoint = AddIOp::create(rewriter, loc, lb, kstep);
  }

  Value minSplitPoint =
      MinIOp::create(rewriter, loc, splitPoint, ub, Signedness::Signed);
  Value maxSplitPoint =
      MaxIOp::create(rewriter, loc, splitPoint, lb, Signedness::Signed);

  ForOp firstLoop, secondLoop;
  Block *originalBody = forOp.getBody();
  SmallVector<Operation *> opsToClone;
  // Collect operations for cloning
  for (Operation &op : *originalBody) {
    if (copyCmp || (&op != cmp.getOperation()))
      opsToClone.push_back(&op);
  }

  // First loop: before the condition flips true
  rewriter.setInsertionPoint(forOp);
  firstLoop = copyLoop(rewriter, forOp, cmp, ifOps, opsToClone, lb,
                       minSplitPoint, iterArgs, !secondThen);
  // Second loop: after the condition is true
  rewriter.setInsertionPointAfter(firstLoop);
  secondLoop = copyLoop(rewriter, forOp, cmp, ifOps, opsToClone, maxSplitPoint,
                        ub, firstLoop.getResults(), secondThen);

  rewriter.replaceOp(forOp, secondLoop);
};

/// Merge optimization hints - more precise hint (if any) gets priority
//  Default value is splitThreshold == 1 defined in pass options
//  Return threshold (minimum number of operations inside if-block)
//  that will be used for determine if splitting should be performed
//  1 - effectively enables splitting for any branch
static int getSplitThreshold(std::optional<int> entryHint,
                             std::optional<int> forHint,
                             std::optional<int> ifHint, int splitThreshold) {
  if (ifHint)
    return ifHint.value();
  if (forHint)
    return forHint.value();
  if (entryHint)
    return entryHint.value();
  return splitThreshold;
}

static std::optional<int> getLoopSplitThresholdAttr(Operation *op) {
  std::optional<int> res = std::nullopt;
  Attribute loopSplitThreshold =
      op->getDiscardableAttr(kLoopSplitThresholdAttrName);
  if (loopSplitThreshold)
    res = cast<IntegerAttr>(loopSplitThreshold).getInt();
  return res;
}

#define GEN_PASS_DEF_LOOPSPLITPASS
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc"

struct LoopSplitPass : public impl::LoopSplitPassBase<LoopSplitPass> {
public:
  using impl::LoopSplitPassBase<LoopSplitPass>::LoopSplitPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    std::optional<int> entryHint = getLoopSplitThresholdAttr(op);
    op->walk([&](ForOp forOp) {
      std::optional<int> forHint = getLoopSplitThresholdAttr(forOp);
      forOp->walk([&](IfOp ifOp) {
        CmpIOp cmp;
        ComparisonPredicate pred;
        Value rhs;
        bool secondThen;
        bool copyCmp;
        std::optional<int> ifHint = getLoopSplitThresholdAttr(ifOp);
        SmallPtrSet<Operation *, 4> ifOps;
        if (isSplittableCondition(forOp, ifOp, ifOps, pred, rhs, cmp,
                                  secondThen, copyCmp,
                                  getSplitThreshold(entryHint, forHint, ifHint,
                                                    splitThreshold))) {
          rewriter.setInsertionPoint(forOp);
          performLoopSplit(rewriter, forOp, pred, rhs, ifOps, cmp, secondThen,
                           copyCmp);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      return WalkResult::advance();
    });
  }
};

} // namespace mlir::cuda_tile
