//===- LayoutAlgebra.cpp - Static CuTe layout algebra -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/LayoutAlgebra.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::cuda_tile;

/// Walk a (shape, stride) pair in lockstep, appending integer leaves.
static LogicalResult flattenInto(SymbolicExprAttr shape,
                                 SymbolicExprAttr stride,
                                 SmallVectorImpl<LayoutLeaf> &leaves) {
  shape = shape.simplify();
  stride = stride.simplify();

  auto shapeTuple = llvm::dyn_cast<SymbolicTupleAttr>(shape);
  auto strideTuple = llvm::dyn_cast<SymbolicTupleAttr>(stride);
  if (static_cast<bool>(shapeTuple) != static_cast<bool>(strideTuple)) {
    // The verifier only lets a tuple face a scalar mode while that scalar is
    // still parametric, because substituting a shape parameter can expand it
    // into a tuple of modes. Defer until then.
    assert(!llvm::isa<SymbolicExprIntAttr>(shapeTuple ? stride : shape) &&
           "expected the verifier to reject an integer facing a tuple mode");
    return failure();
  }
  if (shapeTuple) {
    // Nesting is never introduced below the top level, so the verifier can and
    // does require the two trees to agree from here down.
    assert(shapeTuple.getValues().size() == strideTuple.getValues().size() &&
           "expected shape and stride modes to have the same rank");
    for (auto [shapeElem, strideElem] :
         llvm::zip_equal(shapeTuple.getValues(), strideTuple.getValues())) {
      if (failed(flattenInto(shapeElem, strideElem, leaves))) {
        return failure();
      }
    }
    return success();
  }

  auto shapeInt = llvm::dyn_cast<SymbolicExprIntAttr>(shape);
  auto strideInt = llvm::dyn_cast<SymbolicExprIntAttr>(stride);
  if (!shapeInt || !strideInt) {
    return failure();
  }
  leaves.push_back({shapeInt.getValue(), strideInt.getValue()});
  return success();
}

FailureOr<SmallVector<LayoutLeaf>>
mlir::cuda_tile::flattenStaticLayout(LayoutAttr layout) {
  assert(layout && "expected a non-null layout");
  SmallVector<LayoutLeaf> leaves;
  if (failed(flattenInto(layout.getShapes(), layout.getStrides(), leaves))) {
    return failure();
  }
  return leaves;
}

FailureOr<int64_t>
mlir::cuda_tile::staticLayoutSize(ArrayRef<LayoutLeaf> leaves) {
  int64_t size = 1;
  for (const LayoutLeaf &leaf : leaves) {
    assert(leaf.shape > 0 && "expected the verifier to reject a non-positive "
                             "shape extent");
    std::optional<int64_t> next = llvm::checkedMul(size, leaf.shape);
    if (!next) {
      return failure();
    }
    size = *next;
  }
  return size;
}

FailureOr<int64_t>
mlir::cuda_tile::staticLayoutCosize(ArrayRef<LayoutLeaf> leaves) {
  int64_t maxOffset = 0;
  for (const LayoutLeaf &leaf : leaves) {
    assert(leaf.shape > 0 && "expected the verifier to reject a non-positive "
                             "shape extent");
    assert(leaf.stride >= 0 &&
           "expected the verifier to reject a negative stride");
    // A unit extent is never stepped, so it contributes a zero term.
    std::optional<int64_t> next =
        llvm::checkedMulAdd(leaf.shape - 1, leaf.stride, maxOffset);
    if (!next) {
      return failure();
    }
    maxOffset = *next;
  }
  std::optional<int64_t> cosize = llvm::checkedAdd(maxOffset, int64_t{1});
  if (!cosize) {
    return failure();
  }
  return *cosize;
}

bool mlir::cuda_tile::hasNoBroadcastModes(ArrayRef<LayoutLeaf> leaves) {
  return llvm::all_of(leaves, [](const LayoutLeaf &leaf) {
    return leaf.shape == 1 || leaf.stride != 0;
  });
}

bool mlir::cuda_tile::hasCompactStrideChain(ArrayRef<LayoutLeaf> leaves) {
  // Unit extents and broadcasts occupy no codomain of their own.
  SmallVector<LayoutLeaf> sorted;
  for (const LayoutLeaf &leaf : leaves) {
    assert(leaf.shape > 0 && "expected the verifier to reject a non-positive "
                             "shape extent");
    assert(leaf.stride >= 0 &&
           "expected the verifier to reject a negative stride");
    if (leaf.shape == 1 || leaf.stride == 0) {
      continue;
    }
    sorted.push_back(leaf);
  }
  llvm::sort(sorted, [](const LayoutLeaf &lhs, const LayoutLeaf &rhs) {
    return lhs.stride < rhs.stride;
  });

  // Every mode has to start exactly where its predecessor stopped, so
  // `covered` is at once the stride the next mode must carry and the running
  // product of the extents below it.
  int64_t covered = 1;
  for (const LayoutLeaf &leaf : sorted) {
    if (leaf.stride != covered) {
      return false;
    }
    std::optional<int64_t> next = llvm::checkedMul(covered, leaf.shape);
    if (!next) {
      return false;
    }
    covered = *next;
  }
  return true;
}
