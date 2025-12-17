//===- BytecodeEnums.h - CUDA Tile Bytecode Enums ---------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_BYTECODE_ENUMS_H
#define CUDA_TILE_BYTECODE_ENUMS_H

#include <cstdint>

namespace mlir {
namespace cuda_tile {
namespace Bytecode {

//===----------------------------------------------------------------------===//
// General constants
//===----------------------------------------------------------------------===//

/// Enum representing different bytecode versions.
enum BytecodeConstants {
  // An arbitrary value used to fill alignment padding.
  kAlignmentByte = 0xCB,
};

/// Enum representing different section types in the bytecode.
namespace Section {
enum : uint8_t {
  EndOfBytecode = 0x00,
  String = 0x01,
  Func = 0x02,
  Debug = 0x03,
  Constant = 0x04,
  Type = 0x05,
  Global = 0x06,
  NumSections = 0x07
};
} // namespace Section

/// Enum representing different type tags in the bytecode.
enum class TypeTag : uint8_t {
  I1 = 0,
  I8 = 1,
  I16 = 2,
  I32 = 3,
  I64 = 4,
  F16 = 5,
  BF16 = 6,
  F32 = 7,
  TF32 = 8,
  F64 = 9,
  F8E4M3FN = 10,
  F8E5M2 = 11,
  Pointer = 12,
  Tile = 13,
  TensorView = 14,
  PartitionView = 15,
  Func = 16,
  Token = 17,
  Unknown = 18
};

enum class DebugTag : uint8_t {
  Unknown = 0,
  DICompileUnit = 1,
  DIFile = 2,
  DILexicalBlock = 3,
  DILoc = 4,
  DISubprogram = 5,
  CallSite = 6,
};

enum class DebugReserved : uint8_t {
  UnknownLoc = 0,
  SIZE = 1,
};

/// Enum representing function flags used in the bytecode.
enum class FunctionFlags : uint8_t {
  // Bit 0: Visibility Flag (0 = Public, 1 = Private)
  VisibilityPrivate = 0x01,
  // Bit 1: Function Kind Flag (0 = Device Function, 1 = Kernel Entry Point)
  KindKernel = 0x02,
  // Bit 2: Has Optimization Hints Flag (0 = No, 1 = Yes)
  HasOptimizationHints = 0x04,
};

/// Enum representing different attribute kinds in the bytecode.
enum class AttributeTag : uint8_t {
  Integer = 1,
  Float = 2,
  Bool = 3,
  Type = 4,
  String = 5,
  Array = 6,
  DenseElements = 7,
  DivBy = 8,
  SameElements = 9,
  Dictionary = 10,
  OptimizationHints = 11,
  Bounded = 12,
};

} // namespace Bytecode
} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_BYTECODE_ENUMS_H
