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
/// sinceVersion: The minimum bytecode version that supports this type.
///               This is the earliest version where the type is available.
class BytecodeTypeTag<string typeName, int tag, string version = "13.1"> {
  string cppTypeName = typeName;
  int typeTagValue = tag;
  string sinceVersion = version;
}

/// Integer type tag.
class IntegerTypeTag<string name, int tag, int width,
                     string version = "13.1">
    : BytecodeTypeTag<name, tag, version> {
  int integerBitWidth = width;
}

/// Float type tag.
class FloatTypeTag<string name, int tag, string floatType = "",
                   string version = "13.1">
    : BytecodeTypeTag<name, tag, version> {
  string floatMlirTypeName = floatType;
}

/// CudaTile type tag.
class CudaTileTypeTag<string name, int tag, string version = "13.1">
    : BytecodeTypeTag<name, tag, version>;

//===----------------------------------------------------------------------===//
// Explicit Type Tag Assignments - FROZEN for backward compatibility.
//===----------------------------------------------------------------------===//

// Integer types from 13.1.
def : IntegerTypeTag<"I1", 0, 1>;
def : IntegerTypeTag<"I8", 1, 8>;
def : IntegerTypeTag<"I16", 2, 16>;
def : IntegerTypeTag<"I32", 3, 32>;
def : IntegerTypeTag<"I64", 4, 64>;

// Float types from 13.1.
def : FloatTypeTag<"F16", 5, "Float16Type">;
def : FloatTypeTag<"BF16", 6, "BFloat16Type">;
def : FloatTypeTag<"F32", 7, "Float32Type">;
def : FloatTypeTag<"TF32", 8, "FloatTF32Type">;
def : FloatTypeTag<"F64", 9, "Float64Type">;
def : FloatTypeTag<"F8E4M3FN", 10, "Float8E4M3FNType">;
def : FloatTypeTag<"F8E5M2", 11, "Float8E5M2Type">;

// CudaTile types from 13.1 (auto-generated from CudaTileTypeDef).
def : CudaTileTypeTag<"PointerType", 12>;
def : CudaTileTypeTag<"TileType", 13>;
def : CudaTileTypeTag<"TensorViewType", 14>;
def : CudaTileTypeTag<"PartitionViewType", 15>;
def : CudaTileTypeTag<"FunctionType", 16>;
def : CudaTileTypeTag<"TokenType", 17>;

// Versioned float types from 13.2.
def : FloatTypeTag<"F8E8M0FNU", 18, "Float8E8M0FNUType", "13.2">;

#endif // CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD
