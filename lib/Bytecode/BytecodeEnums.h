//===- BytecodeEnums.h - CUDA Tile Bytecode Enums ---------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
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

/// Bit flags packed into the 16-bit version tag (`verTag`) of the bytecode
/// header. The tag lets a consumer cheaply detect properties of a buffer
/// before it parses any sections.
namespace VersionTag {
enum : uint16_t {
  // Set by the writer when the module contains any public-preview op/type/attr.
  // The reader uses this to (a) require a named preview-feature opt-in and
  // (b) enforce that the buffer's major.minor equals the current version,
  // because preview content is intentionally non-portable across versions.
  kContainsPreview = 0x1,
};

/// The bits defined above. Every released writer has only ever emitted a tag
/// within this mask, so a buffer setting anything else came from a producer
/// this reader cannot interpret and is rejected.
inline constexpr uint16_t kKnownVersionTagMask = kContainsPreview;
} // namespace VersionTag

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
  Producer = 0x07,
  NumSections = 0x08
};
} // namespace Section

/// Enum representing different type tags in the bytecode.
/// This enum is auto-generated from BytecodeTypeOpcodes.td.
#define GEN_TYPE_TAG_ENUM
#include "../Writer/TypeBytecode.inc"
#undef GEN_TYPE_TAG_ENUM

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
///
/// Layout for bytecode version >= kFunctionFlagsFormVersion (13.5):
///   bit 0     VisibilityPrivate
///   bit 1     KindKernel
///   bits 3-2  Form: 00 Normal | 01 ExternImpl | 10 Testing | 11 reserved
///   bits 7-4  reserved (must be 0)
///
///   Optimization hints are not flagged: every KindKernel entry always
///   carries an optimization_hints payload (empty when absent). Extern
///   FuncOp is FormNormal with body length 0 and an always-present
///   const_params payload.
///
/// Layout for bytecode version < kFunctionFlagsFormVersion:
///   bit 0 VisibilityPrivate, bit 1 KindKernel, bit 2 HasOptimizationHints.
///   Bits above 2 are rejected once past kFunctionFlagsExtendedVersion.
enum class FunctionFlags : uint8_t {
  // Bit 0: Visibility Flag (0 = Public, 1 = Private)
  VisibilityPrivate = 0x01,
  // Bit 1: Function Kind Flag (0 = Device Function, 1 = Kernel Entry Point)
  KindKernel = 0x02,
  // Bit 2: Has Optimization Hints (only for versions < 13.5)
  HasOptimizationHints = 0x04,

  // Bits [3:2]: Form field (13.5+). Reclaims bit 2 from HasOptimizationHints.
  FormMask = 0x0C,
  FormNormal = 0x00,
  FormExternImpl = 0x04, // 0b01
  FormTesting = 0x08,    // 0b10 -- test-only; reclaimable later
  FormReserved = 0x0C,   // 0b11 -- reject
};

/// Bits meaningful before the 13.5 Form layout.
inline constexpr uint8_t kFunctionFlagsPreFormMask =
    static_cast<uint8_t>(FunctionFlags::VisibilityPrivate) |
    static_cast<uint8_t>(FunctionFlags::KindKernel) |
    static_cast<uint8_t>(FunctionFlags::HasOptimizationHints);

/// Mask of flag bits recognized under the 13.5+ Form layout. FormTesting is
/// included so non-test builds can emit a dedicated diagnostic; acceptance is
/// still gated on TILE_IR_INCLUDE_TESTS at the Form decode site.
inline constexpr uint8_t kFunctionFlagsFormLayoutMask =
    static_cast<uint8_t>(FunctionFlags::VisibilityPrivate) |
    static_cast<uint8_t>(FunctionFlags::KindKernel) |
    static_cast<uint8_t>(FunctionFlags::FormMask);

/// Enum representing different attribute kinds in the bytecode.
/// This enum is auto-generated from BytecodeAttrOpcodes.td.
#define GEN_ATTR_TAG_ENUM
#include "../Writer/AttrBytecode.inc"
#undef GEN_ATTR_TAG_ENUM

} // namespace Bytecode
} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_BYTECODE_ENUMS_H
