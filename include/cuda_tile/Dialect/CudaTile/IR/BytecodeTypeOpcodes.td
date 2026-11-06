//===- BytecodeTypeOpcodes.td ------------------------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bytecode Type Tag Assignments for CudaTile Types
// This file defines the explicit type tag assignments to ensure backward
// compatibility across versions.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD

include "cuda_tile/Dialect/CudaTile/IR/Types.td"

//===----------------------------------------------------------------------===//
// Type Tag Assignment Class.
//===----------------------------------------------------------------------===//

/// Base class for type tag assignments.
class BytecodeTypeTag<string typeName, int tag> {
  string cppTypeName = typeName;
  int typeTagValue = tag;
}

/// Integer type tag.
class IntegerTypeTag<string name, int tag, int width>
    : BytecodeTypeTag<name, tag> {
  int integerBitWidth = width;
}

/// Float type tag.
class FloatTypeTag<string name, int tag, string floatType = "">
    : BytecodeTypeTag<name, tag> {
  string floatMlirTypeName = floatType;
}

/// CudaTile type tag.
class CudaTileTypeTag<string name, int tag>
  : BytecodeTypeTag<name, tag> {
}

/// Public-preview CudaTile type tag.
class PreviewCudaTileTypeTag<string name, int tag>
    : CudaTileTypeTag<name, tag>;

/// Type tag whose serializer and deserializer are written by hand rather than
/// generated, because the type is not a CudaTileTypeDef. The tag value is still
/// reserved here so it participates in the frozen tag enum and cannot be
/// reused. Every other tag must name a def; use this only when the hand-written
/// code actually exists in BytecodeWriter.cpp and BytecodeReader.cpp.
class HandSerializedTypeTag<string name, int tag>
    : BytecodeTypeTag<name, tag>;

/// Public-preview builtin singleton type tag. Names a nullary builtin MLIR
/// type that is reconstructed on read with `<cppType>::get(ctx)`
/// and matched on write with `isa<cppType>`.
class PreviewBuiltinSingletonTypeTag<string name, string cppType, int tag,
                                     CudaTilePreviewFeature feature>
    : BytecodeTypeTag<name, tag> {
  string builtinSingletonCppType = cppType;
  CudaTilePreviewFeature previewFeature = feature;
}

//===----------------------------------------------------------------------===//
// Explicit Type Tag Assignments - FROZEN for backward compatibility.
//===----------------------------------------------------------------------===//

// 13.1 types.
def : IntegerTypeTag<"Int1", 0, 1>;
def : IntegerTypeTag<"Int8", 1, 8>;
def : IntegerTypeTag<"Int16", 2, 16>;
def : IntegerTypeTag<"Int32", 3, 32>;
def : IntegerTypeTag<"Int64", 4, 64>;
def : FloatTypeTag<"Float16", 5, "Float16Type">;
def : FloatTypeTag<"BFloat16", 6, "BFloat16Type">;
def : FloatTypeTag<"Float32", 7, "Float32Type">;
def : FloatTypeTag<"TFloat32", 8, "FloatTF32Type">;
def : FloatTypeTag<"Float64", 9, "Float64Type">;
def : FloatTypeTag<"Float8E4M3FN", 10, "Float8E4M3FNType">;
def : FloatTypeTag<"Float8E5M2", 11, "Float8E5M2Type">;
def : CudaTileTypeTag<"PointerType", 12>;
def : CudaTileTypeTag<"TileType", 13>;
def : CudaTileTypeTag<"TensorViewType", 14>;
def : CudaTileTypeTag<"PartitionViewType", 15>;
def : HandSerializedTypeTag<"FunctionType", 16>;
def : CudaTileTypeTag<"TokenType", 17>;

// 13.2 types.
def : FloatTypeTag<"Float8E8M0FNU", 18, "Float8E8M0FNUType">;

// 13.3 types.
def : FloatTypeTag<"Float4E2M1FN", 19, "Float4E2M1FNType">;
def : CudaTileTypeTag<"GatherScatterViewType", 20>;
def : CudaTileTypeTag<"StridedViewType", 21>;
def : IntegerTypeTag<"Int4", 22, 4>;

// Public-preview types (0xD0 - 0xEF).
def : PreviewCudaTileTypeTag<"SymbolicType", 0xD1>;
def : PreviewCudaTileTypeTag<"FragmentType", 0xD2>;
def : PreviewCudaTileTypeTag<"DTypeParamType", 0xD3>;
def : PreviewCudaTileTypeTag<"ShapeParamType", 0xD4>;
def : PreviewCudaTileTypeTag<"StringParamType", 0xD5>;
def : PreviewBuiltinSingletonTypeTag<
          "IndexType", "::mlir::IndexType", 0xD6,
          CudaTile_SIMTPreviewFeature>;
def : PreviewBuiltinSingletonTypeTag<
          "NoneType", "::mlir::NoneType", 0xD7,
          CudaTile_SIMTPreviewFeature>;

#ifdef TILE_IR_INCLUDE_TESTS
// Testing type for parameter versioning evolution (version from TestingTypes.td).
def : CudaTileTypeTag<"BytecodeTestEvolvedType", 200>;
def : CudaTileTypeTag<"BytecodeTestSandwichedOptionalType", 201>;
def : CudaTileTypeTag<"BytecodeTestTypeContainerType", 202>;

// Testing public-preview type tags (0xD0 - 0xEF).
def : PreviewCudaTileTypeTag<"PreviewBytecodeTestType", 0xD0>;
#endif // TILE_IR_INCLUDE_TESTS

def : FloatTypeTag<"Float8E5M3FNU", 0x82, "Float8E5M3FNUType">;
#endif // CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD
