//===- LayoutAlgebra.h - Static CuTe layout algebra -------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A small, self-contained subset of the CuTe layout algebra over fully static
// layouts.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_LAYOUTALGEBRA_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_LAYOUTALGEBRA_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace cuda_tile {

class LayoutAttr;

/// One leaf of a flattened layout: a static extent paired with its stride.
struct LayoutLeaf {
  int64_t shape;
  int64_t stride;

  bool operator==(const LayoutLeaf &other) const {
    return shape == other.shape && stride == other.stride;
  }
};

/// Flatten `layout` into its integer leaves in left-to-right order.
///
/// Fails while the layout is still parametric: either a leaf does not simplify
/// to an integer, or a parameter has not yet expanded into the modes its stride
/// describes.
///
/// Example: `(2, (4, 8)):(1, (2, 8))` -> `{2:1, 4:2, 8:8}`.
FailureOr<SmallVector<LayoutLeaf>> flattenStaticLayout(LayoutAttr layout);

/// Size of the layout's domain: the product of the shape leaves.
///
/// Fails on a product that overflows int64.
///
/// Example: `(4, 8):(1, 4)` has size 32.
FailureOr<int64_t> staticLayoutSize(ArrayRef<LayoutLeaf> leaves);

/// Cosize of the layout: one past the largest reachable offset, i.e. the
/// storage it occupies. A unit extent is never stepped, so its stride adds
/// nothing.
///
/// Fails on an offset that overflows int64.
///
/// Example: `(4):(2)` reaches offsets 0, 2, 4, 6, so its cosize is 7.
FailureOr<int64_t> staticLayoutCosize(ArrayRef<LayoutLeaf> leaves);

/// Whether the layout is free of broadcast modes, i.e. whether every
/// stride-zero leaf has unit extent.
///
/// Example: true for `(1, 8):(0, 1)`, false for `(4, 8):(0, 1)`.
bool hasNoBroadcastModes(ArrayRef<LayoutLeaf> leaves);

/// Whether the modes tile the offsets they reach without gap or overlap. Order
/// the modes with extent `n_j > 1` and stride `s_j > 0` by non-decreasing
/// stride; the chain is compact when every `s_j` equals `n_1 * ... * n_{j-1}`.
///
/// A unit extent or a zero stride reaches offset 0 only, so such modes are
/// excluded from the chain. A layout free of broadcast modes -- see
/// hasNoBroadcastModes -- is therefore a permutation of `[0, size)` exactly
/// when its chain is compact. A chain whose running product overflows int64 is
/// rejected, since such a layout is not addressable anyway.
///
/// Examples: `(4, 8):(1, 4)` is compact, as is its transpose `(4, 8):(8, 1)`.
/// `(2, 2):(1, 1)` is not: the two modes share a stride, and indices (0, 1)
/// and (1, 0) do share offset 1. `(2, 2):(1, 8)` is not either, because it
/// leaves offsets 2-7 unreachable, even though it stays injective.
bool hasCompactStrideChain(ArrayRef<LayoutLeaf> leaves);

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_LAYOUTALGEBRA_H
