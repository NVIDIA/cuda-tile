//===- SharedVerifiers.h - CUDA Tile Shared Verifiers -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
namespace mlir {
namespace cuda_tile {

enum class SymbolicExprKind { Int, Tuple, Datatype, String };

LogicalResult
verifySymbolicExprKind(SymbolicExprAttr expr, SymbolicExprKind expectedKind,
                       function_ref<InFlightDiagnostic()> emitError);

/// Verify that every symbolic variable referenced by a function signature is
/// bound by `constParams`, that function's constant-parameter scope.
LogicalResult verifySignatureSymbolicScope(Operation *op, TypeRange inputs,
                                           TypeRange results,
                                           ArrayRef<Attribute> constParams);

LogicalResult verifyConstantArgumentKind(Operation *op, unsigned index,
                                         ConstParamAttr param, Attribute arg);

//===----------------------------------------------------------------------===//
// View Load and Store Utilities
//===----------------------------------------------------------------------===//

static inline TensorViewType getUnderlyingTensorViewType(Type viewType) {
  if (auto partitionView = dyn_cast<PartitionViewType>(viewType)) {
    return partitionView.getTensorView();
  }
  if (auto stridedView = dyn_cast<StridedViewType>(viewType)) {
    return stridedView.getTensorView();
  }
  if (auto gatherScatterView = dyn_cast<GatherScatterViewType>(viewType)) {
    return gatherScatterView.getTensorView();
  }
  return {};
}

/// Verify the compiler-checked portion of the f4 view memory contract. The
/// remaining evenness requirements are execution preconditions whose violation
/// is undefined behavior; they intentionally are not verifier requirements.
static inline LogicalResult verifyF4ViewAccessContract(Operation *op,
                                                       Type viewType) {
  TensorViewType tensorView = getUnderlyingTensorViewType(viewType);
  if (!tensorView || !isa<Float4E2M1FNType>(tensorView.getElementType())) {
    return success();
  }

  std::optional<unsigned> packingDimension = tensorView.getF4PackingDimension();
  if (!packingDimension) {
    size_t unitStrideCount = llvm::count(tensorView.getStrides(), int64_t{1});
    return op->emitOpError()
           << "an f4 view access requires exactly one statically encoded "
              "unit stride to identify the packing dimension; got "
           << unitStrideCount << " in " << tensorView;
  }

  if (auto gatherScatterView = dyn_cast<GatherScatterViewType>(viewType);
      gatherScatterView &&
      gatherScatterView.getSparseDim() == *packingDimension) {
    return op->emitOpError()
           << "an f4 gather/scatter view cannot use packing dimension "
           << *packingDimension << " as its sparse dimension";
  }

  return success();
}

template <typename Op>
static LogicalResult verifyOptHintsCommon(Op op) {
  Operation *opInst = op->getOperation();
  auto hintsAttr = opInst->getAttrOfType<OptimizationHintsAttr>("optimization_hints");
  if (!hintsAttr || hintsAttr.getValue().empty())
    return success();
  if (failed(hintsAttr.verifyWithOp(op->getOperation(), hintsAttr.getValue())))
    return op->emitOpError("Optimization hints verification failed");
  return success();
}

static inline LogicalResult verifyViewIndexTileMatch(Operation *op,
                                                     Value viewVal,
                                                     ValueRange indexVals,
                                                     Value tileVal) {
  auto viewIface = dyn_cast<TileView>(viewVal.getType());
  if (!viewIface)
    return op->emitOpError("expected a tile view type for the view operand");
  TypeRange indexTypes = indexVals.getTypes();
  Type tileType = tileVal.getType();

  if (indexTypes.size() != viewIface.getViewIndexRank())
    return op->emitOpError()
           << "expected " << viewIface.getViewIndexRank()
           << " index operands (based on view type), got " << indexTypes.size();

  if (tileType != viewIface.getViewTileType())
    return op->emitOpError()
           << "expected tile type to be " << viewIface.getViewTileType()
           << " (based on view type), got " << tileType;

  if (failed(verifyF4ViewAccessContract(op, viewVal.getType()))) {
    return failure();
  }

  if (failed(viewIface.verifyIndices([&]() { return op->emitOpError(); },
                                     indexTypes)))
    return failure();

  return success();
}

template <typename LoadStoreOp>
static LogicalResult verifyViewLoadStoreCommon(LoadStoreOp op) {
  if (failed(verifyViewIndexTileMatch(op->getOperation(), op->getView(),
                                     op->getIndex(), op->getTile())))
    return failure();
  return verifyOptHintsCommon(op);
}

static inline LogicalResult verifyViewLoadStoreCommon(Operation *op,
                                                      Value viewVal,
                                                      ValueRange indexVals,
                                                      Value tileVal) {
  return verifyViewIndexTileMatch(op, viewVal, indexVals, tileVal);
}

/// Verifies that every *static* dimension in `shape`
///   • is a positive compile‑time constant,
///   • is a power of two, and
///   • the product of the static dimensions does not exceed
///     `maxTileNumElements`.
///
/// Symbolic dimensions (`ShapedType::isDynamic(dim)`) are skipped: they come
/// from symbolic shapes whose extent is not known at verification time, so
/// none of the above can be checked for them. As a consequence, once a shape
/// contains any symbolic dimension we can no longer guarantee the
/// `maxTileNumElements` bound on the *full* tile — the count enforced here is
/// only over the known (static) dimensions, and the symbolic ones may push the
/// real element count past the limit at runtime. This bound is therefore a
/// best-effort, static-only check for symbolic tiles.
static inline LogicalResult
verifyTileSize(function_ref<InFlightDiagnostic()> emitError,
               ArrayRef<int64_t> shape
) {
  constexpr int64_t kMaxElems = maxTileNumElements;

  int64_t numElems = 1;
  for (int64_t dim : shape) {
    // If a dimension is symbolic skip it.
    if (ShapedType::isDynamic(dim))
      continue;
    // Dimension must be positive.
    if (dim <= 0)
      return emitError() << "all dimensions must be positive constants, got "
                         << shape;

    // Dimension must be a power of two.
    if (
        !llvm::isPowerOf2_64(static_cast<uint64_t>(dim))) {
      return emitError() << "all dimensions must be powers of two, got "
                         << shape;
    }

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

template <typename OpTy>
static inline LogicalResult verifyApprox(OpTy op, bool approx) {
  Type ty = getElementTypeOrSelf(op.getResult().getType());
  if (approx && !ty.isF32())
    return op.emitOpError(
               "approx modifier only supported for f32 data type, but got: ")
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

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

template <typename OpTy, typename PointeeType>
static inline LogicalResult verifyAlloca(OpTy op) {
  // Alignment must be power of 2.
  int64_t alignment = op.getAlignment();
  bool isPowerOfTwo = alignment > 0 && ((alignment & (alignment - 1)) == 0);
  if (!isPowerOfTwo)
    return op.emitOpError() << "'alignment' must be power of two";

  auto ptrType =
      cast<PointeeType>(getElementTypeOrSelf(op.getResult().getType()));
  Type pointeeTy = ptrType.getPointeeType();
  int64_t sizeInBytes = 0;
  if (auto intTy = dyn_cast<IntegerType>(pointeeTy)) {
    sizeInBytes = (intTy.getWidth() + 7) / 8;
  }
  if (auto floatTy = dyn_cast<FloatType>(pointeeTy)) {
    int64_t sizeInBits = APFloat::getSizeInBits(floatTy.getFloatSemantics());
    sizeInBytes = (sizeInBits + 7) / 8;
  }
  if (alignment < sizeInBytes)
    return op.emitOpError()
           << "'alignment' (" << alignment
           << ") must be at least the natural size (" << sizeInBytes
           << " bytes) for element type " << pointeeTy;

  return success();
}

//===----------------------------------------------------------------------===//
// Atomic Operations Utilities
//===----------------------------------------------------------------------===//

/// Verifies RMW mode compatibility with element type for atomic operations.
template <typename OpTy>
static inline LogicalResult verifyAtomicRMWMode(OpTy op, AtomicRMWMode mode,
                                                Type elementType) {
  switch (mode) {
  case AtomicRMWMode::AND:
  case AtomicRMWMode::OR:
  case AtomicRMWMode::XOR:
  case AtomicRMWMode::ADD:
  case AtomicRMWMode::MAX:
  case AtomicRMWMode::MIN:
  case AtomicRMWMode::UMAX:
  case AtomicRMWMode::UMIN: {
    auto integerTy = dyn_cast_or_null<IntegerType>(elementType);
    if (!integerTy || (!integerTy.isInteger(32) && !integerTy.isInteger(64)))
      return op.emitOpError("'") << stringifyAtomicRMWMode(mode)
                                 << "' works only with integers i32 and i64";
    break;
  }
  case AtomicRMWMode::ADDF: {
    auto floatTy = dyn_cast_or_null<FloatType>(elementType);
    if (!floatTy || (!floatTy.isF32() && !floatTy.isF64() && !floatTy.isF16() &&
                     !floatTy.isBF16())) {
      return op.emitOpError("'")
             << stringifyAtomicRMWMode(mode)
             << "' works only with floats f16, bf16, f32, and f64";
    }
    break;
  }
  case AtomicRMWMode::XCHG: {
    auto integerTy = dyn_cast_or_null<IntegerType>(elementType);
    auto floatTy = dyn_cast_or_null<FloatType>(elementType);
    if (!integerTy && !floatTy)
      return op.emitOpError("'")
             << stringifyAtomicRMWMode(mode)
             << "' works only with integers or float of 32 or 64 bitwidth";
    int64_t bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth != 32 && bitwidth != 64)
      return op.emitOpError("'")
             << stringifyAtomicRMWMode(mode)
             << "' works only with integers or float of 32 or 64 bitwidth";
  }
  }
  return success();
}

/// Verifies memory ordering semantics for atomic operations.
template <typename OpTy>
static inline LogicalResult
verifyAtomicMemoryOrdering(OpTy op, MemoryOrderingSemantics semantics) {
  if (semantics != MemoryOrderingSemantics::RELAXED &&
      semantics != MemoryOrderingSemantics::ACQUIRE &&
      semantics != MemoryOrderingSemantics::RELEASE &&
      semantics != MemoryOrderingSemantics::ACQ_REL) {
    return op.emitOpError("memory ordering semantics must be one of: "
                          "relaxed, acquire, release, acq_rel");
  }
  return success();
}

/// Extract PtrAttrAttr from a TileView type's embedded TensorViewType.
/// Returns failure if the view type is not recognized.
static inline FailureOr<PtrAttrAttr> getViewPtrAttr(Operation *op,
                                                    Type viewType) {
  if (auto pv = dyn_cast<PartitionViewType>(viewType)) {
    return pv.getTensorView().getPtrAttr();
  }
  if (auto sv = dyn_cast<StridedViewType>(viewType)) {
    return sv.getTensorView().getPtrAttr();
  }
  if (auto gv = dyn_cast<GatherScatterViewType>(viewType)) {
    return gv.getTensorView().getPtrAttr();
  }
  if (auto tv = dyn_cast<TensorViewType>(viewType)) {
    return tv.getPtrAttr();
  }
  return op->emitOpError("unknown TileView type: ") << viewType;
}

/// Verify that a TileView operand does not have a restricted ptr_attr.
template <typename OpT>
static LogicalResult verifyUnrestrictedViewPtrAttr(OpT op, Value viewOperand) {
  auto ptrAttrOrErr = getViewPtrAttr(op.getOperation(), viewOperand.getType());
  if (failed(ptrAttrOrErr)) {
    return failure();
  }
  if (auto ptrAttr = *ptrAttrOrErr) {
    if (ptrAttr.getValue() == PtrAttr::MULTICAST) {
      return op.emitOpError("multicast ptr_attr is not allowed");
    }
  }
  return success();
}

/// Verify that a TileView operand has multicast ptr_attr.
/// Used by multimem ops.
template <typename OpT>
static LogicalResult verifyMultimemPtrAttr(OpT op, Value viewOperand) {
  auto ptrAttrOrErr = getViewPtrAttr(op.getOperation(), viewOperand.getType());
  if (failed(ptrAttrOrErr)) {
    return failure();
  }
  if (!*ptrAttrOrErr) {
    return op.emitOpError("requires multicast ptr_attr, got none");
  }
  if (auto ptrAttr = *ptrAttrOrErr; ptrAttr.getValue() != PtrAttr::MULTICAST) {
    return op.emitOpError("requires multicast ptr_attr, got ") << ptrAttr;
  }
  return success();
}

/// Verify the target-independent relationship between a synchronization
/// contract and a concrete physical participant count. CTA synchronization is
/// checked at the downstream call site, where the target's thread count is
/// known.
inline LogicalResult verifySynchronizationParticipation(
    Synchronization synchronization, int64_t participants,
    function_ref<InFlightDiagnostic()> emitError) {
  int64_t granularity;
  llvm::StringRef scope;
  switch (synchronization) {
  case Synchronization::None:
  case Synchronization::CTA:
    return success();
  case Synchronization::Warp:
    granularity = 32;
    scope = "warp";
    break;
  case Synchronization::Warpgroup:
    granularity = 128;
    scope = "warpgroup";
    break;
  }

  if (participants <= 0 || participants % granularity != 0) {
    return emitError() << "synchronization '" << scope
                       << "' requires the fragmap participant mode size ("
                       << participants << ") to be a positive multiple of "
                       << granularity;
  }
  return success();
}

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H
