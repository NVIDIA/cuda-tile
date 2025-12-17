//===- SharedVerifiers.h - CUDA Tile Shared Verifiers -----------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H

#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

namespace mlir {
namespace cuda_tile {

//===----------------------------------------------------------------------===//
// View Load and Store Utilities
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult verifyOptHintsCommon(Op op) {
  auto hints = op->getOptimizationHints();
  if (hints && !hints->getValue().empty() &&
      failed(op->getOptimizationHintsAttr().verifyWithOp(
          op->getOperation(), [&]() { return op->emitOpError(); },
          hints->getValue())))
    return op->emitOpError("Optimization hints verification failed");

  return success();
}

template <typename LoadStoreOp>
static LogicalResult verifyViewLoadStoreCommon(LoadStoreOp op) {
  TileView viewType = op->getView().getType();
  Operation::operand_range::type_range indexTypes = op->getIndex().getTypes();
  Type tileType = op->getTile().getType();

  if (indexTypes.size() != viewType.getViewIndexRank())
    return op->emitOpError()
           << "expected " << viewType.getViewIndexRank()
           << " index operands (based on view type), got " << indexTypes.size();

  if (tileType != viewType.getViewTileType())
    return op->emitOpError()
           << "expected tile type to be " << viewType.getViewTileType()
           << " (based on view type), got " << tileType;

  if (!indexTypes.empty()) {
    Type baseIndexType = indexTypes.front();
    for (const auto &[i, indexType] : llvm::enumerate(indexTypes)) {
      if (indexType != baseIndexType)
        return op->emitOpError() << "expected index type " << i
                                 << " to be the same as other index types ("
                                 << baseIndexType << "), got " << indexType;
    }
  }

  return verifyOptHintsCommon(op);
}

/// Verifies that every dimension in `shape`
///   • is a positive compile‑time constant,
///   • is a power of two, and
///   • the total element count does not exceed `maxTileNumElements`.
static inline LogicalResult
verifyTileSize(function_ref<InFlightDiagnostic()> emitError,
               ArrayRef<int64_t> shape) {
  constexpr int64_t kMaxElems = maxTileNumElements;

  int64_t numElems = 1;
  for (int64_t dim : shape) {
    // Dimension must be positive.
    if (dim <= 0)
      return emitError() << "all dimensions must be positive constants, got "
                         << shape;

    // Dimension must be a power of two.
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(dim)))
      return emitError() << "all dimensions must be powers of two, got "
                         << shape;

    // Guard against overflow before multiplying.
    if (numElems > kMaxElems / dim)
      return emitError() << "tile would exceed the maximum of " << kMaxElems
                         << " elements";

    numElems *= dim;
  }

  return success();
}

template <typename OpTy>
static inline LogicalResult verifyFtz(OpTy op, bool ftz) {
  Type ty = getElementTypeOrSelf(op.getResult().getType());

  // Check flush-to-zero modifier compatibility
  // FTZ: When set, subnormal inputs and results are flushed to sign-preserving
  // zero.
  if (ftz && !ty.isF32())
    return op.emitOpError("flush_to_zero modifier only supported for f32 data "
                          "type, but got: ")
           << ty;
  return success();
}

namespace detail {

template <typename OpTy>
static inline LogicalResult
verifyDivSqrtCommonFPModifiers(OpTy op, bool hasRoundingMode, bool approx,
                               bool full, bool ftz) {
  if (approx && full)
    return op.emitOpError(
        "approx modifier and full modifier are mutually exclusive");

  Type ty = getElementTypeOrSelf(op.getResult().getType());
  if (ftz && !ty.isF32())
    return op.emitOpError("flush_to_zero modifier only supported for f32 data "
                          "type, but got: ")
           << ty;

  if (approx && !ty.isF32())
    return op.emitOpError(
               "approx modifier only supported for f32 data type, but got: ")
           << ty;

  if (full && !ty.isF32())
    return op.emitOpError(
               "full modifier only supported for f32 data type, but got: ")
           << ty;

  if (hasRoundingMode) {
    if (approx)
      return op.emitOpError("rounding mode does not allow approx modifier");
    if (full)
      return op.emitOpError("rounding mode does not allow full modifier");
  }
  return success();
}

} // namespace detail

template <typename OpTy>
static inline LogicalResult verifyDivFPModifiers(OpTy op, bool hasRoundingMode,
                                                 bool approx, bool full,
                                                 bool ftz) {

  return detail::verifyDivSqrtCommonFPModifiers(op, hasRoundingMode, approx,
                                                full, ftz);
}

template <typename OpTy>
static inline LogicalResult verifySqrtFPModifiers(OpTy op, bool hasRoundingMode,
                                                  bool approx, bool full,
                                                  bool ftz) {

  return detail::verifyDivSqrtCommonFPModifiers(op, hasRoundingMode, approx,
                                                full, ftz);
}

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H
