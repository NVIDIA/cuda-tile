//===- BytecodeOpcodes.td - CUDA Tile Bytecode Opcodes -----*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bytecode Opcode Assignments for CudaTile Operations
// This file defines the explicit opcode assignments to ensure backward
// compatibility across versions.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_OPCODES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_OPCODES_TD

include "cuda_tile/Dialect/CudaTile/IR/Ops.td"

//===----------------------------------------------------------------------===//
// Opcode Assignment Class
//===----------------------------------------------------------------------===//

/// Base class for opcode assignments
class BytecodeOpcode<Op op, int opcode> {
  Op operation = op;
  int opcodeValue = opcode;
}

/// Public operations - available in all builds (0x0 - 0xFFF).
class PublicOpcode<Op op, int opcode> : BytecodeOpcode<op, opcode>;

/// Supported bytecode version definition.
class SupportedVersion<int major, int minor> {
  int majorVersion = major;
  int minorVersion = minor;
}

//===----------------------------------------------------------------------===//
// Supported Bytecode Versions.
//===----------------------------------------------------------------------===//

def : SupportedVersion<13, 1>;

// Testing versions - only available when TILE_IR_INCLUDE_TESTS is defined
#ifdef TILE_IR_INCLUDE_TESTS
def : SupportedVersion<250, 0>;
#endif // TILE_IR_INCLUDE_TESTS

//===----------------------------------------------------------------------===//
// Explicit Opcode Assignments - FROZEN for backward compatibility
//===----------------------------------------------------------------------===//

// PUBLIC OPERATIONS (0x0 - 0xFFF) - These are available in all builds
// and must never be renumbered for backward compatibility.
def : PublicOpcode<CudaTile_AbsFOp, 0x0>;
def : PublicOpcode<CudaTile_AbsIOp, 0x1>;
def : PublicOpcode<CudaTile_AddFOp, 0x2>;
def : PublicOpcode<CudaTile_AddIOp, 0x3>;
def : PublicOpcode<CudaTile_AndIOp, 0x4>;
def : PublicOpcode<CudaTile_AssertOp, 0x5>;
def : PublicOpcode<CudaTile_AssumeOp, 0x6>;
def : PublicOpcode<CudaTile_AtomicCASTkoOp, 0x7>;
def : PublicOpcode<CudaTile_AtomicRMWTkoOp, 0x8>;
def : PublicOpcode<CudaTile_BitcastOp, 0x9>;
def : PublicOpcode<CudaTile_BreakOp, 0xA>;
def : PublicOpcode<CudaTile_BroadcastOp, 0xB>;
def : PublicOpcode<CudaTile_CatOp, 0xC>;
def : PublicOpcode<CudaTile_CeilOp, 0xD>;
def : PublicOpcode<CudaTile_CmpFOp, 0xE>;
def : PublicOpcode<CudaTile_CmpIOp, 0xF>;
def : PublicOpcode<CudaTile_ConstantOp, 0x10>;
def : PublicOpcode<CudaTile_ContinueOp, 0x11>;
def : PublicOpcode<CudaTile_CosOp, 0x12>;
def : PublicOpcode<CudaTile_CosHOp, 0x13>;
def : PublicOpcode<CudaTile_DivFOp, 0x14>;
def : PublicOpcode<CudaTile_DivIOp, 0x15>;
def : PublicOpcode<CudaTile_EntryOp, 0x16>;
def : PublicOpcode<CudaTile_ExpOp, 0x17>;
def : PublicOpcode<CudaTile_Exp2Op, 0x18>;
def : PublicOpcode<CudaTile_ExtIOp, 0x25>;
def : PublicOpcode<CudaTile_ExtractOp, 0x26>;
def : PublicOpcode<CudaTile_FloorOp, 0x27>;
def : PublicOpcode<CudaTile_FmaOp, 0x28>;
def : PublicOpcode<CudaTile_ForOp, 0x29>;
def : PublicOpcode<CudaTile_FToFOp, 0x2A>;
def : PublicOpcode<CudaTile_FToIOp, 0x2B>;
def : PublicOpcode<CudaTile_GetGlobalOp, 0x2C>;
def : PublicOpcode<CudaTile_GetIndexSpaceShapeOp, 0x2D>;
def : PublicOpcode<CudaTile_GetNumTileBlocksOp, 0x2E>;
def : PublicOpcode<CudaTile_GetTensorShapeOp, 0x2F>;
def : PublicOpcode<CudaTile_GetTileBlockIdOp, 0x30>;
def : PublicOpcode<CudaTile_GlobalOp, 0x31>;
def : PublicOpcode<CudaTile_IfOp, 0x32>;
def : PublicOpcode<CudaTile_IntToPtrOp, 0x33>;
def : PublicOpcode<CudaTile_IotaOp, 0x3A>;
def : PublicOpcode<CudaTile_IToFOp, 0x3B>;
def : PublicOpcode<CudaTile_JoinTokensOp, 0x3C>;
def : PublicOpcode<CudaTile_LoadPtrTkoOp, 0x3D>;
def : PublicOpcode<CudaTile_LoadViewTkoOp, 0x3E>;
def : PublicOpcode<CudaTile_LogOp, 0x3F>;
def : PublicOpcode<CudaTile_Log2Op, 0x40>;
def : PublicOpcode<CudaTile_LoopOp, 0x41>;
def : PublicOpcode<CudaTile_MakePartitionViewOp, 0x42>;
def : PublicOpcode<CudaTile_MakeTensorViewOp, 0x43>;
def : PublicOpcode<CudaTile_MakeTokenOp, 0x44>;
def : PublicOpcode<CudaTile_MaxFOp, 0x45>;
def : PublicOpcode<CudaTile_MaxIOp, 0x46>;
def : PublicOpcode<CudaTile_MinFOp, 0x47>;
def : PublicOpcode<CudaTile_MinIOp, 0x48>;
def : PublicOpcode<CudaTile_MmaFOp, 0x49>;
def : PublicOpcode<CudaTile_MmaIOp, 0x4A>;
def : PublicOpcode<CudaTile_ModuleOp, 0x4B>;
def : PublicOpcode<CudaTile_MulFOp, 0x4C>;
def : PublicOpcode<CudaTile_MulhiIOp, 0x4D>;
def : PublicOpcode<CudaTile_MulIOp, 0x4E>;
def : PublicOpcode<CudaTile_NegFOp, 0x4F>;
def : PublicOpcode<CudaTile_NegIOp, 0x50>;
def : PublicOpcode<CudaTile_OffsetOp, 0x51>;
def : PublicOpcode<CudaTile_OrIOp, 0x52>;
def : PublicOpcode<CudaTile_PermuteOp, 0x53>;
def : PublicOpcode<CudaTile_PowOp, 0x54>;
def : PublicOpcode<CudaTile_PrintOp, 0x55>;
def : PublicOpcode<CudaTile_PtrToIntOp, 0x56>;
def : PublicOpcode<CudaTile_PtrToPtrOp, 0x57>;
def : PublicOpcode<CudaTile_ReduceOp, 0x58>;
def : PublicOpcode<CudaTile_RemFOp, 0x59>;
def : PublicOpcode<CudaTile_RemIOp, 0x5A>;
def : PublicOpcode<CudaTile_ReshapeOp, 0x5B>;
def : PublicOpcode<CudaTile_ReturnOp, 0x5C>;
def : PublicOpcode<CudaTile_RsqrtOp, 0x5D>;
def : PublicOpcode<CudaTile_ScanOp, 0x5E>;
def : PublicOpcode<CudaTile_SelectOp, 0x5F>;
def : PublicOpcode<CudaTile_ShLIOp, 0x60>;
def : PublicOpcode<CudaTile_ShRIOp, 0x61>;
def : PublicOpcode<CudaTile_SinOp, 0x62>;
def : PublicOpcode<CudaTile_SinHOp, 0x63>;
def : PublicOpcode<CudaTile_SqrtOp, 0x64>;
def : PublicOpcode<CudaTile_StorePtrTkoOp, 0x65>;
def : PublicOpcode<CudaTile_StoreViewTkoOp, 0x66>;
def : PublicOpcode<CudaTile_SubFOp, 0x67>;
def : PublicOpcode<CudaTile_SubIOp, 0x68>;
def : PublicOpcode<CudaTile_TanOp, 0x69>;
def : PublicOpcode<CudaTile_TanHOp, 0x6A>;
def : PublicOpcode<CudaTile_TruncIOp, 0x6B>;
def : PublicOpcode<CudaTile_XOrIOp, 0x6C>;
def : PublicOpcode<CudaTile_YieldOp, 0x6D>;

#ifdef TILE_IR_INCLUDE_TESTS
// TESTING OPERATIONS (0x3000+) - Only available when TILE_IR_INCLUDE_TESTS is defined.
def : PublicOpcode<CudaTile_Test_FuncOp, 0x3001>;
#endif // TILE_IR_INCLUDE_TESTS

#endif // CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_OPCODES_TD
