//===- DialectCudaTile.cpp - CUDA Tile dialect python bindings --*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile-c/Dialect/CudaTileDialect.h"
#include "cuda_tile-c/Dialect/CudaTileOptimizer.h"
namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_cuda_tile, m) {
  //===--------------------------------------------------------------------===//
  // CudaTile dialect/pass registration
  //===--------------------------------------------------------------------===//
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__cuda_tile__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context") = nb::none(), nb::arg("load") = true);

  m.def("register_passes", []() { mlirCudaTileRegisterPasses(); });

  // Create a simple struct to avoid C++ symbol binding issues with
  // EMBED_CAPI_LINK_LIBS
  struct TileIROptimizationsOptsWrapper {
    int opt_level = 3;
    bool fuse_fma = false;
  };

  nb::class_<TileIROptimizationsOptsWrapper>(m, "TileIROptimizationsOpts")
      .def(nb::init<>())
      .def_rw("opt_level", &TileIROptimizationsOptsWrapper::opt_level)
      .def_rw("fuse_fma", &TileIROptimizationsOptsWrapper::fuse_fma);

  // TODO: Add CudaTile python bindings tests for ir passes
  m.def(
      "applyTileIROptimizations",
      [](nb::object &moduleOp, const TileIROptimizationsOptsWrapper &opts) {
        mlirCudaTileOptConfig config;
        mlirCudaTileOptFlagsInit(&config);
        config.optLevel = opts.opt_level;
        if (opts.fuse_fma)
          config.flags |= CUDATILE_OPT_FLAG_FUSE_FMA;
        MlirOperation mlirOp = nb::cast<MlirOperation>(moduleOp);
        return mlirLogicalResultIsSuccess(
            mlirCudaTileApplyOptimizations(mlirOp, &config));
      },
      nb::arg("module"), nb::arg("opts") = TileIROptimizationsOptsWrapper{},
      "Perform CUDA Tile IR optimizations using CAPI wrapper");

  m.def(
      "addLoopSplitThresholdAttr",
      [](nb::object &Op, const int threshold) {
        MlirOperation mlirOp = nb::cast<MlirOperation>(Op);
        MlirContext ctx = mlirOperationGetContext(mlirOp);
        MlirType i32Type = mlirCudaTileIntegerTypeGet(ctx, 32);
        MlirAttribute thresholdAttr =
            mlirCudaTileIntegerAttrGet(i32Type, threshold);
        MlirStringRef attrName =
            mlirStringRefCreateFromCString("loop_split_threshold");
        mlirCudaTileOperationSetDiscardableAttributeByName(mlirOp, attrName,
                                                           thresholdAttr);
      },
      nb::arg("op"), nb::arg("threshold"),
      "Set Loop Split optimization hint for operation using CAPI wrapper");

  m.def(
      "writeBytecode",
      [](const nb::object &file_obj, const nb::object &moduleOp) -> bool {
        // Convert the Python object to MLIR module
        MlirOperation mlirOp = nb::cast<MlirOperation>(moduleOp);

        // Platform-independent approach: write to memory buffer via CAPI,
        // then let Python handle file I/O
        MlirStringRef bytecode_buffer =
            mlirCudaTileWriteBytecodeToBuffer(mlirOp);

        // Check for failure (empty buffer)
        if (bytecode_buffer.length == 0)
          return false;

        // Write buffer to Python file object
        nb::bytes data(bytecode_buffer.data, bytecode_buffer.length);
        file_obj.attr("write")(data);
        if (nb::hasattr(file_obj, "flush"))
          file_obj.attr("flush")();

        // Free the C-allocated buffer
        mlirCudaTileFreeBuffer(bytecode_buffer);

        return true;
      },
      nb::arg("file"), nb::arg("module"),
      "Write cuda_tile module to bytecode file object using CAPI wrapper");

  // TODO: Implement CudaTile C API wrappers using tablegen.
  // For now we implemented C-API wrappers manually.

  mlir_type_subclass(m, "PointerType",
                     [](MlirType type) -> bool {
                       return mlirCudaTileTypeIsAPointerType(type);
                     })
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirType pointeeType,
             MlirContext context) -> nb::object {
            // Note: PointerType does not have a verifier, so `getCheckedType`
            // cannot be used.
            return cls(mlirCudaTilePointerTypeGet(context, pointeeType));
          },
          nb::arg("cls"), nb::arg("pointee_type"),
          nb::arg("context") = nb::none())
      .def_classmethod(
          "upcast_type",
          [](const nb::object &cls, MlirType type) -> nb::object {
            if (mlirCudaTileTypeIsAPointerType(type))
              return cls(type);
            return nb::none();
          },
          nb::arg("cls"), nb::arg("type"))
      .def_property_readonly("pointee_type", [](MlirType self) -> MlirType {
        return mlirCudaTilePointerTypeGetPointeeType(self);
      });

  mlir_type_subclass(
      m, "TileType",
      [](MlirType type) -> bool { return mlirCudaTileTypeIsATileType(type); })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::vector<int64_t> &shape,
             MlirType elementType, MlirContext context) -> nb::object {
            MlirType type = mlirCudaTileTileTypeGetChecked(
                context, shape.size(), shape.data(), elementType);
            if (mlirTypeIsNull(type))
              return nb::none();
            return cls(type);
          },
          nb::arg("cls"), nb::arg("shape"), nb::arg("element_type"),
          nb::arg("context") = nb::none())
      .def_classmethod(
          "upcast_type",
          [](const nb::object &cls, MlirType type) -> nb::object {
            if (mlirCudaTileTypeIsATileType(type))
              return cls(type);
            return nb::none();
          },
          nb::arg("cls"), nb::arg("type"))
      .def_property_readonly("shape",
                             [](MlirType type) -> std::vector<int64_t> {
                               intptr_t rank =
                                   mlirCudaTileTileTypeGetRank(type);
                               std::vector<int64_t> shape(rank);
                               for (intptr_t i = 0; i < rank; ++i) {
                                 shape[i] =
                                     mlirCudaTileTileTypeGetDimSize(type, i);
                               }
                               return shape;
                             })
      .def_property_readonly("element_type", [](MlirType type) -> MlirType {
        return mlirCudaTileTileTypeGetElementType(type);
      });

  mlir_type_subclass(
      m, "TokenType",
      [](MlirType type) -> bool { return mlirCudaTileTypeIsATokenType(type); })
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context) -> nb::object {
            return cls(mlirCudaTileTokenTypeGet(context));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(m, "TensorViewType",
                     [](MlirType type) -> bool {
                       return mlirCudaTileTypeIsATensorViewType(type);
                     })
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirType elementType,
             const std::vector<std::optional<int64_t>> &shape,
             const std::vector<std::optional<int64_t>> &stride,
             MlirContext context) -> nb::object {
            auto transformDynamic = [](std::optional<int64_t> val) {
              if (!val.has_value())
                return mlirCudaTileTensorViewTypeGetDynamicSize();

              if (val.value() > 0)
                return val.value();

              // Reject negative values early so kDynamic is not passed as is.
              std::string errorMsg;
              llvm::raw_string_ostream oss(errorMsg);
              oss << "expected strictly positive value for tensor_view "
                     "dimension, got "
                  << val.value();
              throw std::invalid_argument(errorMsg);
            };

            std::vector<int64_t> shapeEncoded(shape.size());
            llvm::transform(shape, shapeEncoded.begin(), transformDynamic);
            std::vector<int64_t> strideEncoded(stride.size());
            llvm::transform(stride, strideEncoded.begin(), transformDynamic);

            MlirType type = mlirCudaTileTensorViewTypeGetChecked(
                context, elementType, shape.size(), shapeEncoded.data(),
                stride.size(), strideEncoded.data());
            if (mlirTypeIsNull(type))
              return nb::none();
            return cls(type);
          },
          nb::arg("cls"), nb::arg("element_type"), nb::arg("shape"),
          nb::arg("stride"), nb::arg("context") = nb::none())
      .def_property_readonly("element_type",
                             [](MlirType type) -> MlirType {
                               return mlirCudaTileTensorViewTypeGetElementType(
                                   type);
                             })
      .def_property_readonly(
          "shape",
          [](MlirType type) -> std::vector<std::optional<int64_t>> {
            intptr_t rank = mlirCudaTileTensorViewTypeGetRank(type);
            std::vector<std::optional<int64_t>> shapeOptional(rank);
            int64_t dynamicSize = mlirCudaTileTensorViewTypeGetDynamicSize();
            for (intptr_t i = 0; i < rank; ++i) {
              int64_t val = mlirCudaTileTensorViewTypeGetDimSize(type, i);
              shapeOptional[i] =
                  val == dynamicSize ? std::nullopt : std::optional{val};
            }
            return shapeOptional;
          })
      .def_property_readonly(
          "strides", [](MlirType type) -> std::vector<std::optional<int64_t>> {
            intptr_t rank = mlirCudaTileTensorViewTypeGetRank(type);
            std::vector<std::optional<int64_t>> strideOptional(rank);
            int64_t dynamicSize = mlirCudaTileTensorViewTypeGetDynamicSize();
            for (intptr_t i = 0; i < rank; ++i) {
              int64_t val = mlirCudaTileTensorViewTypeGetStride(type, i);
              strideOptional[i] =
                  val == dynamicSize ? std::nullopt : std::optional{val};
            }
            return strideOptional;
          });

  mlir_type_subclass(m, "PartitionViewType",
                     [](MlirType type) -> bool {
                       return mlirCudaTileTypeIsAPartitionViewType(type);
                     })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::vector<int32_t> &tileShape,
             MlirType wrappedTensorViewType,
             const std::optional<std::vector<int32_t>> &dimMap,
             const nb::object &paddingValue,
             MlirContext context) -> nb::object {
            if (!mlirCudaTileTypeIsATensorViewType(wrappedTensorViewType)) {
              throw nb::type_error("expected tensor_view type");
            }

            std::vector<int32_t> dimMapInPlace;
            const std::vector<int32_t> *dimMapParam;
            if (dimMap.has_value()) {
              dimMapParam = &dimMap.value();
            } else {
              dimMapInPlace.resize(tileShape.size());
              for (size_t i = 0; i < tileShape.size(); ++i) {
                dimMapInPlace[i] = static_cast<int32_t>(i);
              }
              dimMapParam = &dimMapInPlace;
            }

            MlirAttribute paddingValueAttr = {nullptr};
            if (!paddingValue.is_none()) {
              paddingValueAttr = nb::cast<MlirAttribute>(paddingValue);
            }

            // Create DenseI32ArrayAttr for tile shape
            MlirAttribute tileShapeAttr = mlirCudaTileDenseI32ArrayAttrGet(
                context, tileShape.size(), tileShape.data());

            MlirType type = mlirCudaTilePartitionViewTypeGetChecked(
                context, tileShapeAttr, wrappedTensorViewType,
                dimMapParam->size(), dimMapParam->data(), paddingValueAttr);
            if (mlirTypeIsNull(type))
              return nb::none();
            return cls(type);
          },
          nb::arg("cls"), nb::arg("tile_shape"), nb::arg("tensor_view_type"),
          nb::arg("dim_map") = nb::none(),
          nb::arg("padding_value") = nb::none(),
          nb::arg("context") = nb::none())
      .def_property_readonly(
          "tile_shape",
          [](MlirType type) -> std::vector<int32_t> {
            MlirAttribute shapeAttr =
                mlirCudaTilePartitionViewTypeGetTileShape(type);
            intptr_t numElements =
                mlirCudaTileDenseI32ArrayAttrGetNumElements(shapeAttr);
            std::vector<int32_t> result(numElements);
            for (intptr_t i = 0; i < numElements; ++i) {
              result[i] = mlirCudaTileDenseI32ArrayAttrGetElement(shapeAttr, i);
            }
            return result;
          })
      .def_property_readonly(
          "tensor_view",
          [](MlirType type) -> MlirType {
            return mlirCudaTilePartitionViewTypeGetTensorView(type);
          })
      .def_property_readonly(
          "dim_map",
          [](MlirType type) -> std::vector<int32_t> {
            intptr_t rank = mlirCudaTilePartitionViewTypeGetDimMapRank(type);
            std::vector<int32_t> result(rank);
            for (intptr_t i = 0; i < rank; ++i) {
              result[i] =
                  mlirCudaTilePartitionViewTypeGetDimMapElement(type, i);
            }
            return result;
          })
      .def_property_readonly(
          "padding_value",
          [](MlirType type) -> MlirAttribute {
            return mlirCudaTilePartitionViewTypeGetPaddingValue(type);
          })
      .def_property_readonly(
          "view_tile_type",
          [](MlirType type) -> MlirType {
            return mlirCudaTilePartitionViewTypeGetViewTileType(type);
          })
      .def_property_readonly("view_index_rank", [](MlirType type) -> size_t {
        return mlirCudaTilePartitionViewTypeGetViewIndexRank(type);
      });

  mlir_attribute_subclass(m, "RoundingModeAttr",
                          [](MlirAttribute attr) -> bool {
                            return mlirCudaTileAttributeIsARoundingModeAttr(
                                attr);
                          })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileRoundingModeAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr)) {
              // Fallback to default if invalid value
              MlirStringRef defaultStr =
                  mlirStringRefCreateFromCString("nearest_even");
              attr = mlirCudaTileRoundingModeAttrGet(context, defaultStr);
            }
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef = mlirCudaTileRoundingModeAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });

  mlir_attribute_subclass(
      m, "OptimizationHintsAttr",
      [](MlirAttribute attr) -> bool {
        return mlirCudaTileAttributeIsAOptimizationHintsAttr(attr);
      })
      .def_classmethod(
          "getEntryOpHint",
          [](const nb::object &cls, const std::string &arch, const int &num_cta,
             const int &occupancy, MlirContext context) -> nb::object {
            MlirStringRef archStr =
                mlirStringRefCreateFromCString(arch.c_str());
            MlirAttribute attr =
                mlirCudaTileOptimizationHintsAttrGetEntryOpHint(
                    context, archStr, num_cta, occupancy);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("arch"), nb::arg("num_cta"),
          nb::arg("occupancy"), nb::arg("context") = nb::none())
      .def_classmethod(
          "getLoadStoreOpHint",
          [](const nb::object &cls, const std::string &arch,
             const nb::object &allow_tma, const int &latency,
             MlirContext context) -> nb::object {
            MlirStringRef archStr =
                mlirStringRefCreateFromCString(arch.c_str());
            // Convert Python None/True/False to -1/1/0
            int8_t allowTmaValue = -1; // default: not specified
            if (!allow_tma.is_none())
              allowTmaValue = nb::cast<bool>(allow_tma) ? 1 : 0;
            MlirAttribute attr =
                mlirCudaTileOptimizationHintsAttrGetLoadStoreOpHint(
                    context, archStr, allowTmaValue, latency);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("arch"), nb::arg("allow_tma"),
          nb::arg("latency"), nb::arg("context") = nb::none());

  mlir_attribute_subclass(
      m, "MemoryOrderingSemanticsAttr",
      [](MlirAttribute attr) -> bool {
        return mlirCudaTileAttributeIsAMemoryOrderingSemanticsAttr(attr);
      })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileMemoryOrderingSemanticsAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr)) {
              // Fallback to default if invalid value
              MlirStringRef defaultStr = mlirStringRefCreateFromCString("weak");
              attr = mlirCudaTileMemoryOrderingSemanticsAttrGet(context,
                                                                defaultStr);
            }
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef =
            mlirCudaTileMemoryOrderingSemanticsAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });

  mlir_attribute_subclass(m, "MemoryScopeAttr",
                          [](MlirAttribute attr) -> bool {
                            return mlirCudaTileAttributeIsAMemoryScopeAttr(
                                attr);
                          })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileMemoryScopeAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid memory scope: " + value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef = mlirCudaTileMemoryScopeAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });

  mlir_attribute_subclass(m, "PaddingValueAttr",
                          [](MlirAttribute attr) -> bool {
                            return mlirCudaTileAttributeIsAPaddingValueAttr(
                                attr);
                          })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTilePaddingValueAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid padding value: " + value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef = mlirCudaTilePaddingValueAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });

  mlir_attribute_subclass(m, "AtomicRMWModeAttr",
                          [](MlirAttribute attr) -> bool {
                            return mlirCudaTileAttributeIsAAtomicRMWModeAttr(
                                attr);
                          })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileAtomicRMWModeAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid atomic RMW mode: " + value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef = mlirCudaTileAtomicRMWModeAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });

  mlir_attribute_subclass(m, "IntegerOverflowAttr",
                          [](MlirAttribute attr) -> bool {
                            return mlirCudaTileAttributeIsAIntegerOverflowAttr(
                                attr);
                          })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileIntegerOverflowAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid integer overflow: " + value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef = mlirCudaTileIntegerOverflowAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });

  mlir_attribute_subclass(m, "SignednessAttr",
                          [](MlirAttribute attr) -> bool {
                            return mlirCudaTileAttributeIsASignednessAttr(attr);
                          })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileSignednessAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid signedness: " + value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef = mlirCudaTileSignednessAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });
  mlir_attribute_subclass(
      m, "ComparisonOrderingAttr",
      [](MlirAttribute attr) -> bool {
        return mlirCudaTileAttributeIsAComparisonOrderingAttr(attr);
      })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileComparisonOrderingAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid comparison ordering: " +
                                          value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef =
            mlirCudaTileComparisonOrderingAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });
  mlir_attribute_subclass(
      m, "ComparisonPredicateAttr",
      [](MlirAttribute attr) -> bool {
        return mlirCudaTileAttributeIsAComparisonPredicateAttr(attr);
      })
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &value,
             MlirContext context) -> nb::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileComparisonPredicateAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw std::invalid_argument("Invalid comparison predicate: " +
                                          value);
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("value"), nb::arg("context") = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef =
            mlirCudaTileComparisonPredicateAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });
}
