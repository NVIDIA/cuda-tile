//===- BytecodeAttrAnalysis.h -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrAnalysis.h - Attribute Bytecode Analysis
//
// This file defines data structures and analysis functions for parsing
// TableGen attribute definitions into intermediate representations suitable
// for bytecode generation and validation.
//
// Structure mirrors BytecodeTypeAnalysis.h:
//   - BytecodeAttrParameter: Analyzed parameter.
//   - CudaTileSerializableAttr: Analyzed attr.
//   - BytecodeAttrStructure: Complete analysis result
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_ANALYSIS_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_ANALYSIS_H_

#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/AttrOrTypeDef.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/TableGen/Record.h"

#include <string>

namespace mlir::tblgen {

//===----------------------------------------------------------------------===//
// BytecodeAttrParameter - Analyzed parameter information
//===----------------------------------------------------------------------===//

/// Represents a single attribute parameter after TableGen analysis.
/// Fields:
///   name: Parameter name from TableGen.
///   accessorName: MLIR-generated getter.
///   cppType: Return type of getter.
///   cppStorageType: Storage type.
///   isOptional: True for OptionalParameter.
///   isSigned: True for signed integer types.
///   kind: Wire format classification.
///   sinceVersion: Version when parameter was introduced.
///   defaultValue: Default value for parameter.
struct BytecodeAttrParameter {
  enum class Kind {
    VarInt,
    APInt,
    DenseArray,
    EnumValue,
    TypeAttr,
    Type,
    NestedAttr,
    PolymorphicNestedAttr,
    AttributeArray,
    StringAttr,
    StringRef,
  };

  BytecodeAttrParameter(const AttrOrTypeParameter &param);

private:
  /// Classify parameter kind based on the parameter's TableGen record and,
  /// failing that, on its C++ type spelling.
  static Kind classifyParameter(const AttrOrTypeParameter &param);

public:
  std::string name;
  std::string accessorName;
  std::string cppType;
  std::string cppStorageType;
  bool isOptional;
  bool isSigned = false;
  Kind kind;
  std::string sinceVersion;
  std::string defaultValue;
  std::string bytecodeFallbackValue;
};

//===----------------------------------------------------------------------===//
// CudaTileSerializableAttr - Analyzed attribute for serialization
//===----------------------------------------------------------------------===//

/// Represents a CudaTile attribute that needs parameter-based bytecode
/// serialization. Mirrors CudaTileType structure.
/// Fields:
///   attrName: C++ class name.
///   tagName: AttributeTag enum name.
///   qualifiedCppName: Fully qualified C++ name.
///   tagValue: Wire format tag number.
///   sinceVersion: Version string.
///   parameters: Analyzed attribute parameters.
///   skipVersionCheck: True when version-gating is suppressed for this attr.
///   isPreview: True for public-preview attributes.
///   owningPreviewFeature: Named feature owning a preview attribute.
///   hasOptionalParams: True if attr has optional parameters.
///   requiredParamsFirst: True if required params come before flags
///   hasVerifier: True if attr has genVerifyDecl.
struct CudaTileSerializableAttr {
  CudaTileSerializableAttr(const AttrOrTypeDef &attrDef, unsigned tagValue,
                           StringRef version, bool skipVersionCheck,
                           bool isPreview);

  std::string attrName;
  std::string tagName;
  std::string qualifiedCppName;
  unsigned tagValue;
  std::string sinceVersion;
  SmallVector<BytecodeAttrParameter, 4> parameters;
  bool skipVersionCheck = false;
  bool isPreview = false;
  std::string owningPreviewFeature;
  bool hasOptionalParams = false;
  bool requiredParamsFirst = false;
  bool hasVerifier = false;
};

//===----------------------------------------------------------------------===//
// BytecodeAttr - Tag assignment and version info
//===----------------------------------------------------------------------===//

/// Represents an attribute with bytecode tag assignment and version info.
/// Version is looked up from CudaTileAttrAlias (for MLIR builtins) or
/// CudaTileAttrDef (for CudaTile-specific attrs) in AttrDefs.td.
struct BytecodeAttr {
  std::string attrName;
  unsigned tagValue;
  std::string sinceVersion;
  bool skipsVersionCheck = false;
  bool isPreview = false;
  std::string owningPreviewFeature;
};

/// Represents a single enum case with its version info.
struct EnumCase {
  std::string name;
  int value;
  std::string sinceVersion;
};

/// Represents a CudaTile enum attribute for bytecode serialization.
struct CudaTileEnumAttr {
  std::string enumName;
  std::string attrName;
  std::string sinceVersion;
  bool skipsVersionCheck = false;
  SmallVector<EnumCase, 4> cases;
};

/// Analyzed bytecode attribute structure for code generation.
/// Used by cuda-tile-tblgen to generate:
///   - AttributeTag enum (from bytecodeAttrs, sorted by tagValue)
///   - Version check functions (from bytecodeAttrs, enumAttrs)
///   - Type traits for enum attrs (from enumAttrs)
///   - Serializers/deserializers (from serializableAttrs)
/// Fields:
///   bytecodeAttrs: Attrs with bytecode tag assignments, sorted by tagValue.
///   enumAttrs: Enum attrs for type trait and version check generation.
///   serializableAttrs: CudaTile attrs with full parameter analysis.
struct BytecodeAttrStructure {
  SmallVector<BytecodeAttr, 0> bytecodeAttrs;
  SmallVector<CudaTileEnumAttr, 0> enumAttrs;
  SmallVector<CudaTileSerializableAttr, 0> serializableAttrs;
};

/// Parse and analyze bytecode attribute information from TableGen records.
/// Collects attribute tag assignments from BytecodeAttrOpcodes.td and
/// enum definitions from AttrDefs.td for bytecode code generation.
BytecodeAttrStructure analyzeBytecodeAttrs(const llvm::RecordKeeper &records);

/// Validate that all CudaTileAttrDef attributes have BytecodeAttrTag
/// assignments. Emits fatal errors for missing tags.
mlir::LogicalResult
validateAttrTagAssignments(const llvm::RecordKeeper &records,
                           const BytecodeAttrStructure &structure);

} // namespace mlir::tblgen

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_ANALYSIS_H_
