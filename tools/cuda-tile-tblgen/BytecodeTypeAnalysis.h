//===- BytecodeTypeAnalysis.h -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines data structures and analysis functions for parsing
// TableGen type definitions into intermediate representations suitable for
// bytecode generation.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODE_TYPE_ANALYSIS_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODE_TYPE_ANALYSIS_H_

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"

namespace mlir::tblgen {

//===----------------------------------------------------------------------===//
// BytecodeTypeParameter - Analyzed parameter information
//===----------------------------------------------------------------------===//

/// Represents a single type parameter after TableGen analysis.
/// Fields:
///   name: Parameter name from TableGen.
///   accessorName: MLIR-generated getter.
///   cppType: Return type of getter.
///   cppStorageType: Storage type.
///   isOptional: True for OptionalParameter<...>
///   enumTypeName: For OptionalEnum, underlying enum type.
///   kind: Classification for code generation.
struct BytecodeTypeParameter {
  enum class Kind {
    GenericType,
    SpecificType,
    Int64Array,
    Int32Array,
    DenseI32Array,
    OptionalEnum
  };

  BytecodeTypeParameter(const AttrOrTypeParameter &param);

private:
  /// Classify parameter kind based on types.
  static Kind classifyParameter(const AttrOrTypeParameter &param);

public:
  std::string name;
  std::string accessorName;
  std::string cppType;
  std::string cppStorageType;
  bool isOptional;
  std::string enumTypeName;
  Kind kind;
};

//===----------------------------------------------------------------------===//
// CudaTileType - Analyzed CudaTile type information
//===----------------------------------------------------------------------===//

/// Represents CudaTile type that needs parameter-based bytecode serialization.
/// Fields:
///   typeName: C++ class name and TypeTag enum name.
///   qualifiedTypeName: Fully qualified name.
///   typeTagValue: Wire format tag number.
///   sinceVersion: Version string.
///   parameters: Analyzed type parameters.
///   needsReverseOrder: True for TileType.
struct CudaTileType {
  CudaTileType(const AttrOrTypeDef &typeDef, unsigned tagValue,
               StringRef version);

  std::string typeName;
  std::string qualifiedTypeName;
  unsigned typeTagValue;
  std::string sinceVersion;
  SmallVector<BytecodeTypeParameter> parameters;
  bool needsReverseOrder;
};

//===----------------------------------------------------------------------===//
// BuiltinType - Analyzed built-in type information
//===----------------------------------------------------------------------===//

/// Represents built-in MLIR types for bytecode serialization.
/// Fields:
///   enumName: TypeTag enum value.
///   qualifiedTypeName: TypeSwitch dispatch type.
///   typeTagValue: Wire format tag number.
///   sinceVersion: Version string.
///   integerBitWidth: For integers (1,8,16,32,64); 0 for floats.
///   floatMlirTypeName: For floats ("Float16Type", etc.); empty for integers.
struct BuiltinType {
  BuiltinType(StringRef name, StringRef qualifiedType, unsigned tag,
              StringRef version, unsigned width = 0, StringRef floatType = "");

  bool isInteger() const { return integerBitWidth > 0; }
  bool isFloat() const { return !floatMlirTypeName.empty(); }

  std::string enumName;
  std::string qualifiedTypeName;
  unsigned typeTagValue;
  std::string sinceVersion;
  unsigned integerBitWidth;
  std::string floatMlirTypeName;
};

/// Complete analyzed bytecode type structure.
/// Contains all information needed for code generation.
/// Fields:
///   allTypeTags: All TypeTag enum entries.
///   builtinSerializableTypes: Integer and Float types for auto-generation.
///   cudaTileTypes: CudaTile types.
struct BytecodeTypeStructure {
  SmallVector<std::pair<std::string, unsigned>, 0> allTypeTags;
  SmallVector<BuiltinType, 0> builtinSerializableTypes;
  SmallVector<CudaTileType, 0> cudaTileTypes;
};

//===----------------------------------------------------------------------===//
// Analysis Entry Point.
//===----------------------------------------------------------------------===//

/// Parse and analyze all bytecode type information from TableGen records.
llvm::FailureOr<BytecodeTypeStructure>
analyzeBytecodeTypes(const llvm::RecordKeeper &records);

} // namespace mlir::tblgen

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODE_TYPE_ANALYSIS_H_
