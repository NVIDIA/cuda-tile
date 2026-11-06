//===- PreviewFeatures.h - CUDA Tile Preview Features -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_PREVIEWFEATURES_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_PREVIEWFEATURES_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
class Operation;

namespace cuda_tile {

#define GEN_PREVIEW_FEATURE_DECLS
#include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.inc"

/// A set of independently enabled preview features.
class PreviewFeatures {
public:
  PreviewFeatures() = default;
  explicit PreviewFeatures(PreviewFeature feature) { enable(feature); }

  void enable(PreviewFeature feature) { features.insert(feature); }
  bool isEnabled(PreviewFeature feature) const {
    return features.contains(feature);
  }
  bool empty() const { return features.empty(); }

  PreviewFeatures &operator|=(const PreviewFeatures &other) {
    features.insert(other.features.begin(), other.features.end());
    return *this;
  }

private:
  // Keep common cases allocation-free while allowing the set to grow without
  // imposing a fixed feature-count limit.
  llvm::SmallDenseSet<PreviewFeature, 4> features;
};

/// Build a set from a sequence of feature values.
PreviewFeatures makePreviewFeatures(llvm::ArrayRef<PreviewFeature> features);

/// Return all preview features used by `root`, including features appearing in
/// nested attributes and types.
PreviewFeatures getRequiredPreviewFeatures(Operation *root);

/// Reject preview IR whose owning feature is not enabled.
LogicalResult verifyPreviewFeatures(Operation *root,
                                    PreviewFeatures enabledFeatures);

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_PREVIEWFEATURES_H
