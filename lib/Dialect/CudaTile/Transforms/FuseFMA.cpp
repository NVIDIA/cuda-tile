//===- FuseFMA.cpp - CUDA Tile FMA Fusion Optimization Pass -----*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {

class MulAddPattern final : public OpRewritePattern<cuda_tile::AddFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cuda_tile::AddFOp op,
                                PatternRewriter &rewriter) const override {
    Value c;
    cuda_tile::MulFOp ab;
    if ((ab = op.getLhs().getDefiningOp<cuda_tile::MulFOp>()) &&
        ab.getResult().hasOneUse()) {
      c = op.getRhs();
    } else {
      return rewriter.notifyMatchFailure(op, "no mulf op with one use");
    }

    Value a = ab.getLhs();
    Value b = ab.getRhs();

    // Only fuse if rounding modes and modifiers are the same.
    auto ftz = op.getFlushToZero();
    auto rm = op.getRoundingMode();

    if (ftz != ab.getFlushToZero() || rm != ab.getRoundingMode())
      return rewriter.notifyMatchFailure(
          op, "rounding modes and modifiers are not the same");

    rewriter.replaceOpWithNewOp<cuda_tile::FmaOp>(
        op, a, b, c,
        cuda_tile::RoundingModeAttr::get(rewriter.getContext(), rm),
        ftz ? rewriter.getUnitAttr() : nullptr);
    rewriter.eraseOp(ab); // drop the now-dead multiplication
    return success();
  }
};

class MulSubPattern : public OpRewritePattern<cuda_tile::SubFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cuda_tile::SubFOp op,
                                PatternRewriter &rewriter) const override {
    Value c;
    cuda_tile::MulFOp ab;
    Location loc = op.getLoc();

    if ((ab = op.getLhs().getDefiningOp<cuda_tile::MulFOp>()) &&
        ab.getResult().hasOneUse()) {
      c = rewriter.createOrFold<cuda_tile::NegFOp>(loc, op.getRhs());
    } else {
      return rewriter.notifyMatchFailure(op, "no mulf op on LHS with one use");
    }

    Value a = ab.getLhs();
    Value b = ab.getRhs();

    // Only fuse if rounding modes and modifiers are the same.
    auto ftz = op.getFlushToZero();
    auto rm = op.getRoundingMode();

    if (ftz != ab.getFlushToZero() || rm != ab.getRoundingMode())
      return rewriter.notifyMatchFailure(
          op, "rounding modes and modifiers are not the same");

    rewriter.replaceOpWithNewOp<cuda_tile::FmaOp>(
        op, a, b, c,
        cuda_tile::RoundingModeAttr::get(rewriter.getContext(), rm),
        ftz ? rewriter.getUnitAttr() : nullptr);
    rewriter.eraseOp(ab); // drop the now-dead multiplication
    return success();
  }
};

} // namespace

namespace mlir::cuda_tile {
#define GEN_PASS_DEF_FUSEFMAPASS
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc"

struct FuseFMAPass : public cuda_tile::impl::FuseFMAPassBase<FuseFMAPass> {
public:
  FuseFMAPass() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Add canonicalization patterns to reorder operands
    cuda_tile::AddFOp::getCanonicalizationPatterns(patterns, &getContext());
    // Add FMA fusion patterns
    patterns.add<MulAddPattern, MulSubPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace mlir::cuda_tile
