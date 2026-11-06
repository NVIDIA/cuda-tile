//===- BytecodeAttrOpcodes.td - Bytecode Attribute Tags ----*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bytecode Attribute Tag Assignments for CudaTile Attributes
// This file defines explicit attribute tag assignments to ensure backward
// compatibility across versions.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_ATTR_OPCODES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_ATTR_OPCODES_TD

include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"

//===----------------------------------------------------------------------===//
// Attribute Tag Assignment Class
//===----------------------------------------------------------------------===//

/// Bytecode attribute tag assignment.
/// Version information is looked up from CudaTileAttrAlias (for MLIR builtins)
/// or CudaTileAttrDef (for CudaTile-specific attrs) in AttrDefs.td.
class BytecodeAttrTag<string attrName, int tag> {
  string cppAttrName = attrName;
  int attrTagValue = tag;
}

/// Public-preview attribute tag assignment. Preview attributes are available
/// in all builds but only valid at the exact current bytecode version.
class PreviewBytecodeAttrTag<string attrName, int tag>
    : BytecodeAttrTag<attrName, tag>;

//===----------------------------------------------------------------------===//
// Attribute Tags - FROZEN for backward compatibility.
// Listed in tag value order.
//===----------------------------------------------------------------------===//

def : BytecodeAttrTag<"Integer", 1>;
def : BytecodeAttrTag<"Float", 2>;
def : BytecodeAttrTag<"Bool", 3>;
def : BytecodeAttrTag<"Type", 4>;
def : BytecodeAttrTag<"String", 5>;
def : BytecodeAttrTag<"Array", 6>;
def : BytecodeAttrTag<"DenseElements", 7>;
def : BytecodeAttrTag<"DivBy", 8>;
def : BytecodeAttrTag<"SameElements", 9>;
def : BytecodeAttrTag<"Dictionary", 10>;
def : BytecodeAttrTag<"OptimizationHints", 11>;
def : BytecodeAttrTag<"Bounded", 12>;
def : BytecodeAttrTag<"MultimemReductionKind", 13>;

#ifdef TILE_IR_INCLUDE_TESTS
// Testing Attribute Tags - Only available in testing builds.
def : BytecodeAttrTag<"BytecodeTestEvolved", 200>;
def : BytecodeAttrTag<"BytecodeTestSelfType", 201>;
def : BytecodeAttrTag<"BytecodeTestOptionalType", 202>;
def : BytecodeAttrTag<"BytecodeTestSandwichedOptional", 203>;
def : BytecodeAttrTag<"BytecodeTestEvolvedArrayRef", 204>;
def : BytecodeAttrTag<"BytecodeTestPriority", 205>;

// Public-preview attribute tags (0xD0 - 0xEF).
def : PreviewBytecodeAttrTag<"PreviewBytecodeTest", 0xD0>;
#endif // TILE_IR_INCLUDE_TESTS

// Public-preview symbolic expression and layout attributes.
def : PreviewBytecodeAttrTag<"SymbolicExprInt", 0xD1>;
def : PreviewBytecodeAttrTag<"SymbolicExprVar", 0xD2>;
def : PreviewBytecodeAttrTag<"SymbolicExprMul", 0xD3>;
def : PreviewBytecodeAttrTag<"SymbolicExprDiv", 0xD4>;
def : PreviewBytecodeAttrTag<"SymbolicExprAdd", 0xD5>;
def : PreviewBytecodeAttrTag<"SymbolicExprSub", 0xD6>;
def : PreviewBytecodeAttrTag<"SymbolicExprFloorDiv", 0xD7>;
def : PreviewBytecodeAttrTag<"SymbolicTuple", 0xD8>;
def : PreviewBytecodeAttrTag<"Layout", 0xD9>;
def : PreviewBytecodeAttrTag<"SymbolicExprCeilDiv", 0xDA>;
def : PreviewBytecodeAttrTag<"ConcreteType", 0xDB>;
def : PreviewBytecodeAttrTag<"SymbolicExprNeg", 0xDC>;
def : PreviewBytecodeAttrTag<"MemorySpace", 0xDD>;
def : PreviewBytecodeAttrTag<"Synchronization", 0xDE>;
def : PreviewBytecodeAttrTag<"Access", 0xDF>;
def : PreviewBytecodeAttrTag<"ScratchRequest", 0xE0>;
def : PreviewBytecodeAttrTag<"ConstParam", 0xE2>;
def : PreviewBytecodeAttrTag<"SymbolicExprBuiltin", 0xE3>;
def : PreviewBytecodeAttrTag<"SymbolicExprConstStr", 0xE4>;

#endif // CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_ATTR_OPCODES_TD
