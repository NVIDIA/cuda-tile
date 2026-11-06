//===- PreviewFeatures.cpp - CUDA Tile Preview Features ---------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

using namespace mlir;
using namespace mlir::cuda_tile;

#define GEN_PREVIEW_FEATURE_DEFS
#define GEN_PREVIEW_IR_CLASSIFICATION
#include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.inc"

PreviewFeatures
mlir::cuda_tile::makePreviewFeatures(llvm::ArrayRef<PreviewFeature> features) {
  PreviewFeatures result;
  for (PreviewFeature feature : features) {
    result.enable(feature);
  }
  return result;
}

namespace {

template <typename AttrOrType>
static void collectPreviewFeatures(AttrOrType attrOrType,
                                   PreviewFeatures &features) {
  AttrTypeWalker walker;
  walker.addWalk([&](Attribute nested) {
    if (std::optional<PreviewFeature> feature =
            mlir::cuda_tile::detail::getPreviewFeature(nested)) {
      features.enable(*feature);
    }
    return WalkResult::advance();
  });
  walker.addWalk([&](Type nested) {
    if (std::optional<PreviewFeature> feature =
            mlir::cuda_tile::detail::getPreviewFeature(nested)) {
      features.enable(*feature);
    }
    // TileType's symbolic-shape storage always contains preview attribute
    // classes, even when it is merely the synthesized representation of a
    // stable concrete shape. The generated TileType classifier above decides
    // whether that extension is actually present. Avoid recursively treating
    // the fallback storage as independently authored preview IR, but still
    // inspect the element type.
    if (auto tileType = dyn_cast<cuda_tile::TileType>(nested)) {
      collectPreviewFeatures(tileType.getElementType(), features);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  // Pre-order is required so TileType can skip its synthesized symbolic-shape
  // storage before the walker visits those nested attributes.
  (void)walker.walk<WalkOrder::PreOrder>(attrOrType);
}

static PreviewFeatures getOperationPreviewFeatures(Operation *op) {
  PreviewFeatures features;
  if (std::optional<PreviewFeature> feature =
          mlir::cuda_tile::detail::getPreviewFeature(op)) {
    features.enable(*feature);
  }

  // Discardable attributes are not serialized and carry no CUDA Tile
  // semantics, so preview content in them does not require the opt-in.
  for (NamedAttribute namedAttr : op->getAttrs()) {
    if (op->getDiscardableAttr(namedAttr.getName())) {
      continue;
    }
    collectPreviewFeatures(namedAttr.getValue(), features);
  }
  for (Type type : op->getOperandTypes()) {
    collectPreviewFeatures(type, features);
  }
  for (Type type : op->getResultTypes()) {
    collectPreviewFeatures(type, features);
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments()) {
        collectPreviewFeatures(argument.getType(), features);
      }
    }
  }
  return features;
}

} // namespace

PreviewFeatures mlir::cuda_tile::getRequiredPreviewFeatures(Operation *root) {
  PreviewFeatures required = getOperationPreviewFeatures(root);
  root->walk([&](Operation *op) {
    if (op != root) {
      required |= getOperationPreviewFeatures(op);
    }
  });
  return required;
}

LogicalResult
mlir::cuda_tile::verifyPreviewFeatures(Operation *root,
                                       PreviewFeatures enabledFeatures) {
  auto verifyOp = [&](Operation *op) -> LogicalResult {
    PreviewFeatures required = getOperationPreviewFeatures(op);
    for (PreviewFeature feature : getAllPreviewFeatures()) {
      if (required.isEnabled(feature) && !enabledFeatures.isEnabled(feature)) {
        op->emitError() << "uses preview feature '"
                        << stringifyPreviewFeature(feature)
                        << "', which is not enabled";
        return failure();
      }
    }
    return success();
  };

  if (failed(verifyOp(root))) {
    return failure();
  }
  WalkResult result = root->walk([&](Operation *op) {
    if (op == root) {
      return WalkResult::advance();
    }
    return failed(verifyOp(op)) ? WalkResult::interrupt()
                                : WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}
