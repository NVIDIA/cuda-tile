//===- DialectCudaTile.cpp - CUDA Tile dialect python bindings --*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile-c/Dialect/CudaTileDialect.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_cuda_tile, m) {
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
      py::arg("context") = py::none(), py::arg("load") = true);

  m.def("register_passes", []() { mlirCudaTileRegisterPasses(); });

  // Create a simple struct to avoid C++ symbol binding issues with
  // EMBED_CAPI_LINK_LIBS
  struct TileIROptimizationsOptsWrapper {
    int loop_split_threshold = -1;   // Default: disabled
    bool enable_cse = true;          // Default: enabled
    bool canonicalize_before = true; // Default: enabled
    bool canonicalize_after = true;  // Default: enabled
  };

  py::class_<TileIROptimizationsOptsWrapper>(m, "TileIROptimizationsOpts")
      .def(py::init<>())
      .def_readwrite("loop_split_threshold",
                     &TileIROptimizationsOptsWrapper::loop_split_threshold)
      .def_readwrite("enable_cse", &TileIROptimizationsOptsWrapper::enable_cse)
      .def_readwrite("canonicalize_before",
                     &TileIROptimizationsOptsWrapper::canonicalize_before)
      .def_readwrite("canonicalize_after",
                     &TileIROptimizationsOptsWrapper::canonicalize_after);

  // TODO: Add CudaTile python bindings tests for ir passes
  m.def(
      "applyTileIROptimizations",
      [](py::object &moduleOp, const TileIROptimizationsOptsWrapper &opts) {
        MlirOperation mlirOp = py::cast<MlirOperation>(moduleOp);
        return mlirCudaTileApplyOptimizations(
            mlirOp, opts.loop_split_threshold, opts.enable_cse,
            opts.canonicalize_before, opts.canonicalize_after);
      },
      py::arg("module"), py::arg("opts") = TileIROptimizationsOptsWrapper{},
      "Perform CUDA Tile IR optimizations using CAPI wrapper");

  m.def(
      "addLoopSplitThresholdAttr",
      [](py::object &Op, const int threshold) {
        MlirOperation mlirOp = py::cast<MlirOperation>(Op);
        MlirContext ctx = mlirOperationGetContext(mlirOp);
        MlirType i32Type = mlirCudaTileIntegerTypeGet(ctx, 32);
        MlirAttribute thresholdAttr =
            mlirCudaTileIntegerAttrGet(i32Type, threshold);
        MlirStringRef attrName =
            mlirStringRefCreateFromCString("loop_split_threshold");
        mlirCudaTileOperationSetDiscardableAttributeByName(mlirOp, attrName,
                                                           thresholdAttr);
      },
      py::arg("op"), py::arg("threshold"),
      "Set Loop Split optimization hint for operation using CAPI wrapper");

  m.def(
      "writeBytecode",
      [](const py::object &file_obj, const py::object &moduleOp) -> bool {
        // Convert the Python object to MLIR module
        MlirOperation mlirOp = py::cast<MlirOperation>(moduleOp);

        // Platform-independent approach: write to memory buffer via CAPI,
        // then let Python handle file I/O
        MlirStringRef bytecode_buffer =
            mlirCudaTileWriteBytecodeToBuffer(mlirOp);

        // Check for failure (empty buffer)
        if (bytecode_buffer.length == 0)
          return false;

        // Write buffer to Python file object
        py::bytes data(bytecode_buffer.data, bytecode_buffer.length);
        file_obj.attr("write")(data);
        if (py::hasattr(file_obj, "flush"))
          file_obj.attr("flush")();

          // Free the C-allocated buffer
        mlirCudaTileFreeBuffer(bytecode_buffer);

        return true;
      },
      py::arg("file"), py::arg("module"),
      "Write cuda_tile module to bytecode file object using CAPI wrapper");

  // TODO: Implement CudaTile C API wrappers using tablegen.
  // For now we implemented C-API wrappers manually.

  mlir_type_subclass(m, "PointerType",
                     [](MlirType type) -> bool {
                       return mlirCudaTileTypeIsAPointerType(type);
                     })
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirType pointeeType,
             MlirContext context) -> py::object {
            // Note: PointerType does not have a verifier, so `getCheckedType`
            // cannot be used.
            return cls(mlirCudaTilePointerTypeGet(context, pointeeType));
          },
          py::arg("cls"), py::arg("pointee_type"),
          py::arg("context") = py::none())
      .def_classmethod(
          "upcast_type",
          [](const py::object &cls, MlirType type) -> py::object {
            if (mlirCudaTileTypeIsAPointerType(type))
              return cls(type);
            return py::none();
          },
          py::arg("cls"), py::arg("type"))
      .def_property_readonly("pointee_type", [](MlirType self) -> MlirType {
        return mlirCudaTilePointerTypeGetPointeeType(self);
      });

  mlir_type_subclass(
      m, "TileType",
      [](MlirType type) -> bool { return mlirCudaTileTypeIsATileType(type); })
      .def_classmethod(
          "get",
          [](const py::object &cls, const std::vector<int64_t> &shape,
             MlirType elementType, MlirContext context) -> py::object {
            MlirType type = mlirCudaTileTileTypeGetChecked(
                context, shape.size(), shape.data(), elementType);
            if (mlirTypeIsNull(type))
              return py::none();
            return cls(type);
          },
          py::arg("cls"), py::arg("shape"), py::arg("element_type"),
          py::arg("context") = py::none())
      .def_classmethod(
          "upcast_type",
          [](const py::object &cls, MlirType type) -> py::object {
            if (mlirCudaTileTypeIsATileType(type))
              return cls(type);
            return py::none();
          },
          py::arg("cls"), py::arg("type"))
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
          [](const py::object &cls, MlirContext context) -> py::object {
            return cls(mlirCudaTileTokenTypeGet(context));
          },
          py::arg("cls"), py::arg("context") = py::none());

  mlir_type_subclass(m, "TensorViewType",
                     [](MlirType type) -> bool {
                       return mlirCudaTileTypeIsATensorViewType(type);
                     })
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirType elementType,
             const std::vector<std::optional<int64_t>> &shape,
             const std::vector<std::optional<int64_t>> &stride,
             MlirContext context) -> py::object {
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
              throw py::value_error(errorMsg);
            };

            std::vector<int64_t> shapeEncoded(shape.size());
            llvm::transform(shape, shapeEncoded.begin(), transformDynamic);
            std::vector<int64_t> strideEncoded(stride.size());
            llvm::transform(stride, strideEncoded.begin(), transformDynamic);

            MlirType type = mlirCudaTileTensorViewTypeGetChecked(
                context, elementType, shape.size(), shapeEncoded.data(),
                stride.size(), strideEncoded.data());
            if (mlirTypeIsNull(type))
              return py::none();
            return cls(type);
          },
          py::arg("cls"), py::arg("element_type"), py::arg("shape"),
          py::arg("stride"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::vector<int32_t> &tileShape,
             MlirType wrappedTensorViewType,
             const std::optional<std::vector<int32_t>> &dimMap,
             const py::object &paddingValue,
             MlirContext context) -> py::object {
            if (!mlirCudaTileTypeIsATensorViewType(wrappedTensorViewType)) {
              throw py::type_error("expected tensor_view type");
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
              paddingValueAttr = py::cast<MlirAttribute>(paddingValue);
            }

            // Create DenseI32ArrayAttr for tile shape
            MlirAttribute tileShapeAttr = mlirCudaTileDenseI32ArrayAttrGet(
                context, tileShape.size(), tileShape.data());

            MlirType type = mlirCudaTilePartitionViewTypeGetChecked(
                context, tileShapeAttr, wrappedTensorViewType,
                dimMapParam->size(), dimMapParam->data(), paddingValueAttr);
            if (mlirTypeIsNull(type))
              return py::none();
            return cls(type);
          },
          py::arg("cls"), py::arg("tile_shape"), py::arg("tensor_view_type"),
          py::arg("dim_map") = py::none(),
          py::arg("padding_value") = py::none(),
          py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
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
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &arch, const int &num_cta,
             const int &occupancy, MlirContext context) -> py::object {
            MlirStringRef archStr =
                mlirStringRefCreateFromCString(arch.c_str());
            MlirAttribute attr =
                mlirCudaTileOptimizationHintsAttrGetEntryOpHint(
                    context, archStr, num_cta, occupancy);
            return cls(attr);
          },
          py::arg("cls"), py::arg("arch"), py::arg("num_cta"),
          py::arg("occupancy"), py::arg("context") = py::none())
      .def_classmethod(
          "getLoadStoreOpHint",
          [](const py::object &cls, const std::string &arch,
             const py::object &allow_tma, const int &latency,
             MlirContext context) -> py::object {
            MlirStringRef archStr =
                mlirStringRefCreateFromCString(arch.c_str());
            // Convert Python None/True/False to -1/1/0
            int8_t allowTmaValue = -1; // default: not specified
            if (!allow_tma.is_none())
              allowTmaValue = py::cast<bool>(allow_tma) ? 1 : 0;
            MlirAttribute attr =
                mlirCudaTileOptimizationHintsAttrGetLoadStoreOpHint(
                    context, archStr, allowTmaValue, latency);
            return cls(attr);
          },
          py::arg("cls"), py::arg("arch"), py::arg("allow_tma"),
          py::arg("latency"), py::arg("context") = py::none());

  mlir_attribute_subclass(
      m, "MemoryOrderingSemanticsAttr",
      [](MlirAttribute attr) -> bool {
        return mlirCudaTileAttributeIsAMemoryOrderingSemanticsAttr(attr);
      })
      .def_classmethod(
          "get",
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
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
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileMemoryScopeAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid memory scope: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTilePaddingValueAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid padding value: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileAtomicRMWModeAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid atomic RMW mode: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileIntegerOverflowAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid integer overflow: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileSignednessAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid signedness: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileComparisonOrderingAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid comparison ordering: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
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
          [](const py::object &cls, const std::string &value,
             MlirContext context) -> py::object {
            MlirStringRef valueStr =
                mlirStringRefCreateFromCString(value.c_str());
            MlirAttribute attr =
                mlirCudaTileComparisonPredicateAttrGet(context, valueStr);
            if (mlirAttributeIsNull(attr))
              throw py::value_error("Invalid comparison predicate: " + value);
            return cls(attr);
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        MlirStringRef valueRef =
            mlirCudaTileComparisonPredicateAttrGetValue(self);
        return std::string(valueRef.data, valueRef.length);
      });
}
