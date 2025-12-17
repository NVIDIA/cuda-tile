//===- CudaTileDialect.h - CUDA Tile C API Dialect Utilities ----*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_C_DIALECT_CUDATILEDIALECT_H
#define CUDA_TILE_C_DIALECT_CUDATILEDIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CUDATILE, cuda_tile);

// PointerType

/// Returns true if the given type is a cuda_tile PointerType.
MLIR_CAPI_EXPORTED bool mlirCudaTileTypeIsAPointerType(MlirType type);

/// Returns the TypeID for cuda_tile PointerType.
MLIR_CAPI_EXPORTED MlirTypeID mlirCudaTilePointerTypeGetTypeID(void);

/// Returns a cuda_tile PointerType with the given pointee type in the given
/// context.
MLIR_CAPI_EXPORTED MlirType mlirCudaTilePointerTypeGet(MlirContext ctx,
                                                       MlirType pointeeType);

/// Returns the pointee type of the given cuda_tile PointerType.
MLIR_CAPI_EXPORTED MlirType
mlirCudaTilePointerTypeGetPointeeType(MlirType type);

// TileType

/// Returns true if the given type is a cuda_tile TileType.
MLIR_CAPI_EXPORTED bool mlirCudaTileTypeIsATileType(MlirType type);

/// Returns the TypeID for cuda_tile TileType.
MLIR_CAPI_EXPORTED MlirTypeID mlirCudaTileTileTypeGetTypeID(void);

/// Returns a cuda_tile TileType with the given shape and element type.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileTileTypeGet(MlirContext ctx,
                                                    intptr_t rank,
                                                    const int64_t *shape,
                                                    MlirType elementType);

/// Returns the element type of the given cuda_tile TileType.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileTileTypeGetElementType(MlirType type);

/// Returns the rank of the given cuda_tile TileType.
MLIR_CAPI_EXPORTED intptr_t mlirCudaTileTileTypeGetRank(MlirType type);

/// Returns the shape of the given cuda_tile TileType at the given index.
MLIR_CAPI_EXPORTED int64_t mlirCudaTileTileTypeGetDimSize(MlirType type,
                                                          intptr_t pos);

/// Returns a cuda_tile TileType with the given shape and element type,
/// performing verification. Returns a null type if verification fails.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileTileTypeGetChecked(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType);

// TokenType

/// Returns true if the given type is a cuda_tile TokenType.
MLIR_CAPI_EXPORTED bool mlirCudaTileTypeIsATokenType(MlirType type);

/// Returns the TypeID for cuda_tile TokenType.
MLIR_CAPI_EXPORTED MlirTypeID mlirCudaTileTokenTypeGetTypeID(void);

/// Returns a cuda_tile TokenType.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileTokenTypeGet(MlirContext ctx);

// TensorViewType

/// Returns true if the given type is a cuda_tile TensorViewType.
MLIR_CAPI_EXPORTED bool mlirCudaTileTypeIsATensorViewType(MlirType type);

/// Returns the TypeID for cuda_tile TensorViewType.
MLIR_CAPI_EXPORTED MlirTypeID mlirCudaTileTensorViewTypeGetTypeID(void);

/// Returns a cuda_tile TensorViewType with the given element type, shape, and
/// strides.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileTensorViewTypeGet(
    MlirContext ctx, MlirType elementType, intptr_t shapeRank,
    const int64_t *shape, intptr_t strideRank, const int64_t *strides);

/// Returns the element type of the given cuda_tile TensorViewType.
MLIR_CAPI_EXPORTED MlirType
mlirCudaTileTensorViewTypeGetElementType(MlirType type);

/// Returns the rank of the given cuda_tile TensorViewType.
MLIR_CAPI_EXPORTED intptr_t mlirCudaTileTensorViewTypeGetRank(MlirType type);

/// Returns the shape of the given cuda_tile TensorViewType at the given index.
MLIR_CAPI_EXPORTED int64_t mlirCudaTileTensorViewTypeGetDimSize(MlirType type,
                                                                intptr_t pos);

/// Returns the stride of the given cuda_tile TensorViewType at the given index.
MLIR_CAPI_EXPORTED int64_t mlirCudaTileTensorViewTypeGetStride(MlirType type,
                                                               intptr_t pos);

/// Returns the dynamic dimension constant for TensorViewType.
MLIR_CAPI_EXPORTED int64_t mlirCudaTileTensorViewTypeGetDynamicSize(void);

/// Returns a cuda_tile TensorViewType with the given element type, shape, and
/// strides, performing verification. Returns a null type if verification fails.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileTensorViewTypeGetChecked(
    MlirContext ctx, MlirType elementType, intptr_t shapeRank,
    const int64_t *shape, intptr_t strideRank, const int64_t *strides);

// PartitionViewType

/// Returns true if the given type is a cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED bool mlirCudaTileTypeIsAPartitionViewType(MlirType type);

/// Returns the TypeID for cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED MlirTypeID mlirCudaTilePartitionViewTypeGetTypeID(void);

/// Returns a cuda_tile PartitionViewType with the given tile shape, tensor
/// view, dim map, and optional padding value.
MLIR_CAPI_EXPORTED MlirType mlirCudaTilePartitionViewTypeGet(
    MlirContext ctx, MlirAttribute tileShapeAttr, MlirType tensorViewType,
    intptr_t dimMapRank, const int32_t *dimMap, MlirAttribute paddingValue);

/// Returns the tile shape attribute of the given cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTilePartitionViewTypeGetTileShape(MlirType type);

/// Returns the tensor view type of the given cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED MlirType
mlirCudaTilePartitionViewTypeGetTensorView(MlirType type);

/// Returns the rank of the dim map of the given cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED intptr_t
mlirCudaTilePartitionViewTypeGetDimMapRank(MlirType type);

/// Returns the dim map element at the given index of the given cuda_tile
/// PartitionViewType.
MLIR_CAPI_EXPORTED int32_t
mlirCudaTilePartitionViewTypeGetDimMapElement(MlirType type, intptr_t pos);

/// Returns the padding value attribute of the given cuda_tile PartitionViewType
/// (may be null).
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTilePartitionViewTypeGetPaddingValue(MlirType type);

/// Returns the view tile type of the given cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED MlirType
mlirCudaTilePartitionViewTypeGetViewTileType(MlirType type);

/// Returns the view index rank of the given cuda_tile PartitionViewType.
MLIR_CAPI_EXPORTED intptr_t
mlirCudaTilePartitionViewTypeGetViewIndexRank(MlirType type);

/// Returns a cuda_tile PartitionViewType with the given tile shape, tensor
/// view, dim map, and padding value, performing verification. Returns a null
/// type if verification fails.
MLIR_CAPI_EXPORTED MlirType mlirCudaTilePartitionViewTypeGetChecked(
    MlirContext ctx, MlirAttribute tileShapeAttr, MlirType tensorViewType,
    intptr_t dimMapRank, const int32_t *dimMap, MlirAttribute paddingValue);

// RoundingModeAttr

/// Returns true if the given attribute is a cuda_tile RoundingModeAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsARoundingModeAttr(MlirAttribute attr);

/// Returns a cuda_tile RoundingModeAttr with the given rounding mode string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileRoundingModeAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the rounding mode string of the given cuda_tile RoundingModeAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileRoundingModeAttrGetValue(MlirAttribute attr);

// ComparisonOrderingAttr

/// Returns true if the given attribute is a cuda_tile ComparisonOrderingAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAComparisonOrderingAttr(MlirAttribute attr);

/// Returns a cuda_tile ComparisonOrderingAttr with the given ordering string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileComparisonOrderingAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the comparison ordering string of the given cuda_tile
/// ComparisonOrderingAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileComparisonOrderingAttrGetValue(MlirAttribute attr);

// ComparisonPredicateAttr

/// Returns true if the given attribute is a cuda_tile ComparisonPredicateAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAComparisonPredicateAttr(MlirAttribute attr);

/// Returns a cuda_tile ComparisonPredicateAttr with the given predicate string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileComparisonPredicateAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the comparison predicate string of the given cuda_tile
/// ComparisonPredicateAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileComparisonPredicateAttrGetValue(MlirAttribute attr);

// DenseI32ArrayAttr helpers

/// Creates a DenseI32ArrayAttr with the given values.
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaTileDenseI32ArrayAttrGet(
    MlirContext ctx, intptr_t numElements, const int32_t *values);

/// Returns the number of elements in a DenseI32ArrayAttr.
MLIR_CAPI_EXPORTED intptr_t
mlirCudaTileDenseI32ArrayAttrGetNumElements(MlirAttribute attr);

/// Returns the element at the given index in a DenseI32ArrayAttr.
MLIR_CAPI_EXPORTED int32_t
mlirCudaTileDenseI32ArrayAttrGetElement(MlirAttribute attr, intptr_t pos);

// MemoryOrderingSemanticsAttr

/// Returns true if the given attribute is a cuda_tile
/// MemoryOrderingSemanticsAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAMemoryOrderingSemanticsAttr(MlirAttribute attr);

/// Returns a cuda_tile MemoryOrderingSemanticsAttr with the given semantics
/// string.
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaTileMemoryOrderingSemanticsAttrGet(
    MlirContext ctx, MlirStringRef value);

/// Returns the memory ordering semantics string of the given cuda_tile
/// MemoryOrderingSemanticsAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileMemoryOrderingSemanticsAttrGetValue(MlirAttribute attr);

// MemoryScopeAttr

/// Returns true if the given attribute is a cuda_tile MemoryScopeAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAMemoryScopeAttr(MlirAttribute attr);

/// Returns a cuda_tile MemoryScopeAttr with the given scope string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileMemoryScopeAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the memory scope string of the given cuda_tile MemoryScopeAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileMemoryScopeAttrGetValue(MlirAttribute attr);

// PaddingValueAttr

/// Returns true if the given attribute is a cuda_tile PaddingValueAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAPaddingValueAttr(MlirAttribute attr);

/// Returns a cuda_tile PaddingValueAttr with the given padding value string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTilePaddingValueAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the padding value string of the given cuda_tile PaddingValueAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTilePaddingValueAttrGetValue(MlirAttribute attr);

// AtomicRMWModeAttr

/// Returns true if the given attribute is a cuda_tile AtomicRMWModeAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAAtomicRMWModeAttr(MlirAttribute attr);

/// Returns a cuda_tile AtomicRMWModeAttr with the given mode string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileAtomicRMWModeAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the atomic RMW mode string of the given cuda_tile AtomicRMWModeAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileAtomicRMWModeAttrGetValue(MlirAttribute attr);

// IntegerOverflowAttr

/// Returns true if the given attribute is a cuda_tile IntegerOverflowAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAIntegerOverflowAttr(MlirAttribute attr);

/// Returns a cuda_tile IntegerOverflowAttr with the given overflow string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileIntegerOverflowAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the integer overflow string of the given cuda_tile
/// IntegerOverflowAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileIntegerOverflowAttrGetValue(MlirAttribute attr);

// SignednessAttr

/// Returns true if the given attribute is a cuda_tile SignednessAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsASignednessAttr(MlirAttribute attr);

/// Returns a cuda_tile SignednessAttr with the given signedness string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileSignednessAttrGet(MlirContext ctx, MlirStringRef value);

/// Returns the signedness string of the given cuda_tile SignednessAttr.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileSignednessAttrGetValue(MlirAttribute attr);

// OptimizationHintsAttr

/// Returns true if the given attribute is a cuda_tile OptimizationHintsAttr.
MLIR_CAPI_EXPORTED bool
mlirCudaTileAttributeIsAOptimizationHintsAttr(MlirAttribute attr);

/// Returns an empty cuda_tile OptimizationHintsAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileOptimizationHintsAttrGetEmpty(MlirContext ctx);

/// Returns a cuda_tile OptimizationHintsAttr with EntryOp hints for the given
/// architecture. Pass 0 for unused parameters.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileOptimizationHintsAttrGetEntryOpHint(MlirContext ctx,
                                                MlirStringRef arch,
                                                int32_t numCta,
                                                int32_t occupancy);

/// Returns a cuda_tile OptimizationHintsAttr with LoadStore hints for the given
/// architecture. Pass 0 for latency and false for allowTma if unused.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileOptimizationHintsAttrGetLoadStoreOpHint(MlirContext ctx,
                                                    MlirStringRef arch,
                                                    int8_t allowTma,
                                                    int32_t latency);

// Pass Management and Optimization Functions (Future CAPI Extensions)

/// Returns true if the operation is a cuda_tile ModuleOp.
MLIR_CAPI_EXPORTED bool mlirCudaTileOperationIsAModuleOp(MlirOperation op);

/// Returns true if the operation is a standard MLIR ModuleOp.
MLIR_CAPI_EXPORTED bool mlirOperationIsAModuleOp(MlirOperation op);

/// Applies TileIR optimizations to a cuda_tile module operation.
/// Returns true on success, false on failure.
/// Note: This is a high-level function that would need extensive CAPI
/// infrastructure.
MLIR_CAPI_EXPORTED bool
mlirCudaTileApplyOptimizations(MlirOperation moduleOp,
                               int32_t loopSplitThreshold, bool enableCSE,
                               bool canonicalizeBefore, bool canonicalizeAfter);

/// Writes a cuda_tile module to bytecode format using a file descriptor.
/// Returns true on success, false on failure.
/// Note: This function would need CAPI for bytecode writing and operation
/// casting.
MLIR_CAPI_EXPORTED bool mlirCudaTileWriteBytecode(MlirOperation moduleOp,
                                                  int fileDescriptor);

/// Writes a cuda_tile module to bytecode format to a memory buffer.
/// Returns an MlirStringRef containing the bytecode data (with length).
/// Returns empty string ref on failure.
/// Caller must free the buffer using mlirCudaTileFreeBuffer.
MLIR_CAPI_EXPORTED MlirStringRef
mlirCudaTileWriteBytecodeToBuffer(MlirOperation moduleOp);

/// Frees a buffer returned by mlirCudaTileWriteBytecodeToBuffer.
MLIR_CAPI_EXPORTED void mlirCudaTileFreeBuffer(MlirStringRef buffer);

// Helper functions for operation attribute manipulation

/// Creates an integer type with the given width.
MLIR_CAPI_EXPORTED MlirType mlirCudaTileIntegerTypeGet(MlirContext ctx,
                                                       unsigned width);

/// Creates an integer attribute with the given type and value.
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaTileIntegerAttrGet(MlirType type,
                                                            int64_t value);

/// Sets a discardable attribute on an operation by name.
MLIR_CAPI_EXPORTED void mlirCudaTileOperationSetDiscardableAttributeByName(
    MlirOperation op, MlirStringRef name, MlirAttribute attr);

// Pass Registration Functions

/// Registers all CudaTile passes with the global pass registry.
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterPasses(void);

/// Registers individual CudaTile passes with the global pass registry.
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterSynthesizeDebugInfoScopesPass(void);
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterFuseFMAPass(void);
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterLoopSplitPass(void);

/// Registers standard MLIR passes with the global pass registry.
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterCanonicalizerPass(void);
MLIR_CAPI_EXPORTED void mlirCudaTileRegisterCSEPass(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TILE_C_DIALECT_CUDATILEDIALECT_H
