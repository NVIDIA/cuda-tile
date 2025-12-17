//===- CudaTileDialect.cpp - CUDA Tile CAPI ---------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile-c/Dialect/CudaTileDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cuda_tile;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CUDATILE, cuda_tile,
                                      cuda_tile::CudaTileDialect)

/// Construct the specified type with the given parameters and verify the type.
/// If the type fails to verify, an error is printed and the function returns
/// a "null" type.
template <typename T, typename... ParamsT>
static T getCheckedType(MLIRContext *ctx, ParamsT &&...params) {
  return T::getChecked([&] { return emitError(UnknownLoc::get(ctx)); }, ctx,
                       std::forward<ParamsT>(params)...);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

bool mlirCudaTileTypeIsAPointerType(MlirType type) {
  return isa<PointerType>(unwrap(type));
}

MlirTypeID mlirCudaTilePointerTypeGetTypeID(void) {
  return wrap(PointerType::getTypeID());
}

MlirType mlirCudaTilePointerTypeGet(MlirContext ctx, MlirType pointeeType) {
  return wrap(PointerType::get(unwrap(ctx), unwrap(pointeeType)));
}

MlirType mlirCudaTilePointerTypeGetPointeeType(MlirType type) {
  return wrap(cast<PointerType>(unwrap(type)).getPointeeType());
}

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

bool mlirCudaTileTypeIsATileType(MlirType type) {
  return isa<TileType>(unwrap(type));
}

MlirTypeID mlirCudaTileTileTypeGetTypeID(void) {
  return wrap(TileType::getTypeID());
}

MlirType mlirCudaTileTileTypeGet(MlirContext ctx, intptr_t rank,
                                 const int64_t *shape, MlirType elementType) {
  ArrayRef<int64_t> shapeRef(shape, rank);
  return wrap(TileType::get(unwrap(ctx), shapeRef, unwrap(elementType)));
}

MlirType mlirCudaTileTileTypeGetElementType(MlirType type) {
  return wrap(cast<TileType>(unwrap(type)).getElementType());
}

intptr_t mlirCudaTileTileTypeGetRank(MlirType type) {
  return cast<TileType>(unwrap(type)).getRank();
}

int64_t mlirCudaTileTileTypeGetDimSize(MlirType type, intptr_t pos) {
  return cast<TileType>(unwrap(type)).getDimSize(pos);
}

MlirType mlirCudaTileTileTypeGetChecked(MlirContext ctx, intptr_t rank,
                                        const int64_t *shape,
                                        MlirType elementType) {
  ArrayRef<int64_t> shapeRef(shape, rank);
  return wrap(
      getCheckedType<TileType>(unwrap(ctx), shapeRef, unwrap(elementType)));
}

//===----------------------------------------------------------------------===//
// TokenType
//===----------------------------------------------------------------------===//

bool mlirCudaTileTypeIsATokenType(MlirType type) {
  return isa<TokenType>(unwrap(type));
}

MlirTypeID mlirCudaTileTokenTypeGetTypeID(void) {
  return wrap(TokenType::getTypeID());
}

MlirType mlirCudaTileTokenTypeGet(MlirContext ctx) {
  return wrap(TokenType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// TensorViewType
//===----------------------------------------------------------------------===//

bool mlirCudaTileTypeIsATensorViewType(MlirType type) {
  return isa<TensorViewType>(unwrap(type));
}

MlirTypeID mlirCudaTileTensorViewTypeGetTypeID(void) {
  return wrap(TensorViewType::getTypeID());
}

MlirType mlirCudaTileTensorViewTypeGet(MlirContext ctx, MlirType elementType,
                                       intptr_t shapeRank, const int64_t *shape,
                                       intptr_t strideRank,
                                       const int64_t *strides) {
  ArrayRef<int64_t> shapeRef(shape, shapeRank);
  ArrayRef<int64_t> strideRef(strides, strideRank);
  return wrap(TensorViewType::get(unwrap(ctx), unwrap(elementType), shapeRef,
                                  strideRef));
}

MlirType mlirCudaTileTensorViewTypeGetElementType(MlirType type) {
  return wrap(cast<TensorViewType>(unwrap(type)).getElementType());
}

intptr_t mlirCudaTileTensorViewTypeGetRank(MlirType type) {
  return cast<TensorViewType>(unwrap(type)).getShape().size();
}

int64_t mlirCudaTileTensorViewTypeGetDimSize(MlirType type, intptr_t pos) {
  return cast<TensorViewType>(unwrap(type)).getShape()[pos];
}

int64_t mlirCudaTileTensorViewTypeGetStride(MlirType type, intptr_t pos) {
  return cast<TensorViewType>(unwrap(type)).getStrides()[pos];
}

int64_t mlirCudaTileTensorViewTypeGetDynamicSize(void) {
  return TensorViewType::kDynamic;
}

MlirType mlirCudaTileTensorViewTypeGetChecked(
    MlirContext ctx, MlirType elementType, intptr_t shapeRank,
    const int64_t *shape, intptr_t strideRank, const int64_t *strides) {
  ArrayRef<int64_t> shapeRef(shape, shapeRank);
  ArrayRef<int64_t> strideRef(strides, strideRank);
  return wrap(getCheckedType<TensorViewType>(unwrap(ctx), unwrap(elementType),
                                             shapeRef, strideRef));
}

//===----------------------------------------------------------------------===//
// PartitionViewType
//===----------------------------------------------------------------------===//

bool mlirCudaTileTypeIsAPartitionViewType(MlirType type) {
  return isa<PartitionViewType>(unwrap(type));
}

MlirTypeID mlirCudaTilePartitionViewTypeGetTypeID(void) {
  return wrap(PartitionViewType::getTypeID());
}

MlirType mlirCudaTilePartitionViewTypeGet(
    MlirContext ctx, MlirAttribute tileShapeAttr, MlirType tensorViewType,
    intptr_t dimMapRank, const int32_t *dimMap, MlirAttribute paddingValue) {
  ArrayRef<int32_t> dimMapRef(dimMap, dimMapRank);
  auto denseI32Attr = cast<DenseI32ArrayAttr>(unwrap(tileShapeAttr));
  auto tensorView = cast<TensorViewType>(unwrap(tensorViewType));
  PaddingValueAttr paddingValueAttr = nullptr;
  if (!mlirAttributeIsNull(paddingValue))
    paddingValueAttr = cast<PaddingValueAttr>(unwrap(paddingValue));
  return wrap(PartitionViewType::get(unwrap(ctx), denseI32Attr, tensorView,
                                     dimMapRef, paddingValueAttr));
}

MlirAttribute mlirCudaTilePartitionViewTypeGetTileShape(MlirType type) {
  return wrap(cast<PartitionViewType>(unwrap(type)).getTileShape());
}

MlirType mlirCudaTilePartitionViewTypeGetTensorView(MlirType type) {
  return wrap(cast<PartitionViewType>(unwrap(type)).getTensorView());
}

intptr_t mlirCudaTilePartitionViewTypeGetDimMapRank(MlirType type) {
  return cast<PartitionViewType>(unwrap(type)).getDimMap().size();
}

int32_t mlirCudaTilePartitionViewTypeGetDimMapElement(MlirType type,
                                                      intptr_t pos) {
  return cast<PartitionViewType>(unwrap(type)).getDimMap()[pos];
}

MlirAttribute mlirCudaTilePartitionViewTypeGetPaddingValue(MlirType type) {
  auto paddingValue = cast<PartitionViewType>(unwrap(type)).getPaddingValue();
  if (paddingValue)
    return wrap(paddingValue);
  return {nullptr};
}

MlirType mlirCudaTilePartitionViewTypeGetViewTileType(MlirType type) {
  return wrap(cast<PartitionViewType>(unwrap(type)).getViewTileType());
}

intptr_t mlirCudaTilePartitionViewTypeGetViewIndexRank(MlirType type) {
  return cast<PartitionViewType>(unwrap(type)).getViewIndexRank();
}

MlirType mlirCudaTilePartitionViewTypeGetChecked(
    MlirContext ctx, MlirAttribute tileShapeAttr, MlirType tensorViewType,
    intptr_t dimMapRank, const int32_t *dimMap, MlirAttribute paddingValue) {
  ArrayRef<int32_t> dimMapRef(dimMap, dimMapRank);
  auto denseI32Attr = cast<DenseI32ArrayAttr>(unwrap(tileShapeAttr));
  auto tensorView = cast<TensorViewType>(unwrap(tensorViewType));
  PaddingValueAttr paddingValueAttr = nullptr;
  if (!mlirAttributeIsNull(paddingValue))
    paddingValueAttr = cast<PaddingValueAttr>(unwrap(paddingValue));
  return wrap(getCheckedType<PartitionViewType>(
      unwrap(ctx), denseI32Attr, tensorView, dimMapRef, paddingValueAttr));
}

//===----------------------------------------------------------------------===//
// RoundingModeAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsARoundingModeAttr(MlirAttribute attr) {
  return isa<RoundingModeAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileRoundingModeAttrGet(MlirContext ctx,
                                              MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto roundingMode = symbolizeRoundingMode(valueStr);
  if (!roundingMode.has_value())
    return {nullptr};
  return wrap(RoundingModeAttr::get(unwrap(ctx), roundingMode.value()));
}

MlirStringRef mlirCudaTileRoundingModeAttrGetValue(MlirAttribute attr) {
  auto roundingModeAttr = cast<RoundingModeAttr>(unwrap(attr));
  StringRef result = stringifyRoundingMode(roundingModeAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// ComparisonOrderingAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAComparisonOrderingAttr(MlirAttribute attr) {
  return isa<ComparisonOrderingAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileComparisonOrderingAttrGet(MlirContext ctx,
                                                    MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto ordering = symbolizeComparisonOrdering(valueStr);
  if (!ordering.has_value())
    return {nullptr};
  return wrap(ComparisonOrderingAttr::get(unwrap(ctx), ordering.value()));
}

MlirStringRef mlirCudaTileComparisonOrderingAttrGetValue(MlirAttribute attr) {
  auto orderingAttr = cast<ComparisonOrderingAttr>(unwrap(attr));
  StringRef result = stringifyComparisonOrdering(orderingAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// ComparisonPredicateAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAComparisonPredicateAttr(MlirAttribute attr) {
  return isa<ComparisonPredicateAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileComparisonPredicateAttrGet(MlirContext ctx,
                                                     MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto predicate = symbolizeComparisonPredicate(valueStr);
  if (!predicate.has_value())
    return {nullptr};
  return wrap(ComparisonPredicateAttr::get(unwrap(ctx), predicate.value()));
}

MlirStringRef mlirCudaTileComparisonPredicateAttrGetValue(MlirAttribute attr) {
  auto predicateAttr = cast<ComparisonPredicateAttr>(unwrap(attr));
  StringRef result = stringifyComparisonPredicate(predicateAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// DenseI32ArrayAttr helpers
//===----------------------------------------------------------------------===//

MlirAttribute mlirCudaTileDenseI32ArrayAttrGet(MlirContext ctx,
                                               intptr_t numElements,
                                               const int32_t *values) {
  ArrayRef<int32_t> valuesRef(values, numElements);
  return wrap(DenseI32ArrayAttr::get(unwrap(ctx), valuesRef));
}

intptr_t mlirCudaTileDenseI32ArrayAttrGetNumElements(MlirAttribute attr) {
  auto denseAttr = cast<DenseI32ArrayAttr>(unwrap(attr));
  return denseAttr.size();
}

int32_t mlirCudaTileDenseI32ArrayAttrGetElement(MlirAttribute attr,
                                                intptr_t pos) {
  auto denseAttr = cast<DenseI32ArrayAttr>(unwrap(attr));
  return denseAttr[pos];
}

//===----------------------------------------------------------------------===//
// MemoryOrderingSemanticsAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAMemoryOrderingSemanticsAttr(MlirAttribute attr) {
  return isa<MemoryOrderingSemanticsAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileMemoryOrderingSemanticsAttrGet(MlirContext ctx,
                                                         MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto semantics = symbolizeMemoryOrderingSemantics(valueStr);
  if (!semantics.has_value())
    return {nullptr};
  return wrap(MemoryOrderingSemanticsAttr::get(unwrap(ctx), semantics.value()));
}

MlirStringRef
mlirCudaTileMemoryOrderingSemanticsAttrGetValue(MlirAttribute attr) {
  auto semanticsAttr = cast<MemoryOrderingSemanticsAttr>(unwrap(attr));
  StringRef result = stringifyMemoryOrderingSemantics(semanticsAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// MemoryScopeAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAMemoryScopeAttr(MlirAttribute attr) {
  return isa<MemoryScopeAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileMemoryScopeAttrGet(MlirContext ctx,
                                             MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto scope = symbolizeMemoryScope(valueStr);
  if (!scope.has_value())
    return {nullptr};
  return wrap(MemoryScopeAttr::get(unwrap(ctx), scope.value()));
}

MlirStringRef mlirCudaTileMemoryScopeAttrGetValue(MlirAttribute attr) {
  auto scopeAttr = cast<MemoryScopeAttr>(unwrap(attr));
  StringRef result = stringifyMemoryScope(scopeAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// PaddingValueAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAPaddingValueAttr(MlirAttribute attr) {
  return isa<PaddingValueAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTilePaddingValueAttrGet(MlirContext ctx,
                                              MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto paddingValue = symbolizePaddingValue(valueStr);
  if (!paddingValue.has_value())
    return {nullptr};
  return wrap(PaddingValueAttr::get(unwrap(ctx), paddingValue.value()));
}

MlirStringRef mlirCudaTilePaddingValueAttrGetValue(MlirAttribute attr) {
  auto paddingAttr = cast<PaddingValueAttr>(unwrap(attr));
  StringRef result = stringifyPaddingValue(paddingAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// AtomicRMWModeAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAAtomicRMWModeAttr(MlirAttribute attr) {
  return isa<AtomicRMWModeAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileAtomicRMWModeAttrGet(MlirContext ctx,
                                               MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto mode = symbolizeAtomicRMWMode(valueStr);
  if (!mode.has_value())
    return {nullptr};
  return wrap(AtomicRMWModeAttr::get(unwrap(ctx), mode.value()));
}

MlirStringRef mlirCudaTileAtomicRMWModeAttrGetValue(MlirAttribute attr) {
  auto modeAttr = cast<AtomicRMWModeAttr>(unwrap(attr));
  StringRef result = stringifyAtomicRMWMode(modeAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// IntegerOverflowAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAIntegerOverflowAttr(MlirAttribute attr) {
  return isa<IntegerOverflowAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileIntegerOverflowAttrGet(MlirContext ctx,
                                                 MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto overflow = symbolizeIntegerOverflow(valueStr);
  if (!overflow.has_value())
    return {nullptr};
  return wrap(IntegerOverflowAttr::get(unwrap(ctx), overflow.value()));
}

MlirStringRef mlirCudaTileIntegerOverflowAttrGetValue(MlirAttribute attr) {
  auto overflowAttr = cast<IntegerOverflowAttr>(unwrap(attr));
  StringRef result = stringifyIntegerOverflow(overflowAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// SignednessAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsASignednessAttr(MlirAttribute attr) {
  return isa<SignednessAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileSignednessAttrGet(MlirContext ctx,
                                            MlirStringRef value) {
  StringRef valueStr = unwrap(value);
  auto signedness = symbolizeSignedness(valueStr);
  if (!signedness.has_value())
    return {nullptr};
  return wrap(SignednessAttr::get(unwrap(ctx), signedness.value()));
}

MlirStringRef mlirCudaTileSignednessAttrGetValue(MlirAttribute attr) {
  auto signednessAttr = cast<SignednessAttr>(unwrap(attr));
  StringRef result = stringifySignedness(signednessAttr.getValue());
  return wrap(result);
}

//===----------------------------------------------------------------------===//
// OptimizationHintsAttr
//===----------------------------------------------------------------------===//

bool mlirCudaTileAttributeIsAOptimizationHintsAttr(MlirAttribute attr) {
  return isa<OptimizationHintsAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileOptimizationHintsAttrGetEmpty(MlirContext ctx) {
  auto context = unwrap(ctx);
  auto emptyDict = DictionaryAttr::get(context);
  return wrap(OptimizationHintsAttr::get(context, emptyDict));
}

MlirAttribute mlirCudaTileOptimizationHintsAttrGetEntryOpHint(
    MlirContext ctx, MlirStringRef arch, int32_t numCta, int32_t occupancy) {
  auto context = unwrap(ctx);
  StringRef archStr = unwrap(arch);

  // Build the inner dictionary with EntryOp hints
  SmallVector<NamedAttribute, 4> innerAttrs;
  IntegerType i32 = IntegerType::get(context, 32);

  if (numCta != 0) {
    innerAttrs.emplace_back(StringAttr::get(context, "num_cta_in_cga"),
                            IntegerAttr::get(i32, numCta));
  }

  if (occupancy != 0) {
    innerAttrs.emplace_back(StringAttr::get(context, "occupancy"),
                            IntegerAttr::get(i32, occupancy));
  }

  auto innerDict = DictionaryAttr::get(context, innerAttrs);

  // Create the outer dictionary with architecture as key
  NamedAttribute outerEntry(StringAttr::get(context, archStr), innerDict);
  auto outerDict = DictionaryAttr::get(context, {outerEntry});

  return wrap(OptimizationHintsAttr::get(context, outerDict));
}

MlirAttribute mlirCudaTileOptimizationHintsAttrGetLoadStoreOpHint(
    MlirContext ctx, MlirStringRef arch, int8_t allowTma, int32_t latency) {
  auto context = unwrap(ctx);
  StringRef archStr = unwrap(arch);

  // Build the inner dictionary with LoadStore hints
  SmallVector<NamedAttribute, 4> innerAttrs;
  IntegerType i32 = IntegerType::get(context, 32);

  // Only emit allow_tma if explicitly specified (not -1)
  if (allowTma != -1) {
    innerAttrs.emplace_back(StringAttr::get(context, "allow_tma"),
                            BoolAttr::get(context, allowTma != 0));
  }

  if (latency != 0) {
    innerAttrs.emplace_back(StringAttr::get(context, "latency"),
                            IntegerAttr::get(i32, latency));
  }

  auto innerDict = DictionaryAttr::get(context, innerAttrs);

  // Create the outer dictionary with architecture as key
  NamedAttribute outerEntry(StringAttr::get(context, archStr), innerDict);
  auto outerDict = DictionaryAttr::get(context, {outerEntry});

  return wrap(OptimizationHintsAttr::get(context, outerDict));
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static cuda_tile::ModuleOp extractCudaTileModuleOp(Operation *op) {
  // Try direct cast first
  if (auto directCudaTileModule = dyn_cast<cuda_tile::ModuleOp>(op))
    return directCudaTileModule;

  // Try nested case: look inside a regular ModuleOp
  if (auto moduleOp = dyn_cast<mlir::ModuleOp>(op)) {
    if (!moduleOp.getBody()->empty()) {
      if (auto nestedCudaTileModule =
              dyn_cast<cuda_tile::ModuleOp>(&moduleOp.getBody()->front()))
        return nestedCudaTileModule;
    }
  }

  // Not found
  return {};
}

//===----------------------------------------------------------------------===//
// Pass Management and Optimization Functions
//===----------------------------------------------------------------------===//

bool mlirCudaTileOperationIsAModuleOp(MlirOperation op) {
  return isa<cuda_tile::ModuleOp>(unwrap(op));
}

bool mlirOperationIsAModuleOp(MlirOperation op) {
  return isa<mlir::ModuleOp>(unwrap(op));
}

bool mlirCudaTileApplyOptimizations(MlirOperation moduleOp,
                                    int32_t loopSplitThreshold, bool enableCSE,
                                    bool canonicalizeBefore,
                                    bool canonicalizeAfter) {
  auto *op = unwrap(moduleOp);

  // Extract cuda_tile::ModuleOp (handles both direct and nested cases)
  auto cudaTileModule = extractCudaTileModuleOp(op);
  if (!cudaTileModule)
    return false;

  // Create PassManager for cuda_tile::ModuleOp
  PassManager pm = PassManager::on<cuda_tile::ModuleOp>(op->getContext());
  auto &nested = pm.nest<cuda_tile::EntryOp>();

  // Add canonicalization pass before optimizations if requested
  if (canonicalizeBefore)
    nested.addPass(createCanonicalizerPass());

  // Add loop split pass if threshold is >= 0
  if (loopSplitThreshold >= 0)
    nested.addPass(cuda_tile::createLoopSplitPass({loopSplitThreshold}));

  // Add CSE pass if requested
  if (enableCSE)
    nested.addPass(createCSEPass());

  // Add canonicalization pass after optimizations if requested
  if (canonicalizeAfter)
    nested.addPass(createCanonicalizerPass());

  // Run the passes
  if (failed(pm.run(cudaTileModule)))
    return false;

  return true;
}

MlirStringRef mlirCudaTileWriteBytecodeToBuffer(MlirOperation moduleOp) {
  auto *op = unwrap(moduleOp);

  // Extract cuda_tile::ModuleOp (handles both direct and nested cases)
  auto cudaTileModule = extractCudaTileModuleOp(op);
  if (!cudaTileModule)
    return mlirStringRefCreateFromCString("");

  // Allocate buffer that caller must free
  std::string temp;
  llvm::raw_string_ostream stream(temp);
  if (failed(cuda_tile::writeBytecode(stream, cudaTileModule,
                                      BytecodeVersion::kCurrentVersion)))
    return mlirStringRefCreateFromCString("");

  stream.flush();

  // Allocate persistent buffer
  char *buffer = static_cast<char *>(malloc(temp.size()));
  if (!buffer)
    return mlirStringRefCreateFromCString("");

  memcpy(buffer, temp.data(), temp.size());
  return mlirStringRefCreate(buffer, temp.size());
}

void mlirCudaTileFreeBuffer(MlirStringRef buffer) {
  if (buffer.data && buffer.length > 0)
    free(const_cast<char *>(buffer.data));
}

//===----------------------------------------------------------------------===//
// Helper functions for operation attribute manipulation
//===----------------------------------------------------------------------===//

MlirType mlirCudaTileIntegerTypeGet(MlirContext ctx, unsigned width) {
  return wrap(IntegerType::get(unwrap(ctx), width));
}

MlirAttribute mlirCudaTileIntegerAttrGet(MlirType type, int64_t value) {
  return wrap(IntegerAttr::get(unwrap(type), value));
}

void mlirCudaTileOperationSetDiscardableAttributeByName(MlirOperation op,
                                                        MlirStringRef name,
                                                        MlirAttribute attr) {
  StringRef nameStr = unwrap(name);
  unwrap(op)->setDiscardableAttr(nameStr, unwrap(attr));
}

//===----------------------------------------------------------------------===//
// Pass Registration Functions
//===----------------------------------------------------------------------===//

void mlirCudaTileRegisterPasses(void) {
  // Register all CudaTile passes
  registerSynthesizeDebugInfoScopesPass();
  registerFuseFMAPass();
  registerLoopSplitPass();

  // Register standard MLIR passes
  registerCanonicalizerPass();
  registerCSEPass();
}

void mlirCudaTileRegisterSynthesizeDebugInfoScopesPass(void) {
  registerSynthesizeDebugInfoScopesPass();
}

void mlirCudaTileRegisterFuseFMAPass(void) { registerFuseFMAPass(); }

void mlirCudaTileRegisterLoopSplitPass(void) { registerLoopSplitPass(); }

void mlirCudaTileRegisterCanonicalizerPass(void) {
  registerCanonicalizerPass();
}

void mlirCudaTileRegisterCSEPass(void) { registerCSEPass(); }
