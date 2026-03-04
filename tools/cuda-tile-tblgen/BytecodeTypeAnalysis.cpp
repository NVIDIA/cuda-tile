//===- BytecodeTypeAnalysis.cpp ---------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BytecodeTypeAnalysis.h"

#include "llvm/TableGen/Error.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// BytecodeTypeParameter Implementation.
//===----------------------------------------------------------------------===//

BytecodeTypeParameter::Kind
BytecodeTypeParameter::classifyParameter(const AttrOrTypeParameter &param) {
  StringRef cppType = param.getCppType();
  StringRef cppStorageType = param.getCppStorageType();

  // ArrayRefParameter sets cppStorageType to SmallVector.
  if (cppStorageType.contains("SmallVector<int64_t>"))
    return Kind::Int64Array;
  if (cppStorageType.contains("SmallVector<int32_t>"))
    return Kind::Int32Array;
  if (cppStorageType.contains("DenseI32ArrayAttr"))
    return Kind::DenseI32Array;

  // Check for Type parameters.
  if (cppType.contains("Type")) {
    if (cppType.contains("cuda_tile::"))
      return Kind::SpecificType;
    return Kind::GenericType;
  }

  // Check for optional enum attributes.
  if (param.isOptional() && cppType.contains("Attr"))
    return Kind::OptionalEnum;

  // Unsupported parameter type.
  PrintFatalError("Unsupported parameter type for bytecode generation: " +
                  cppType.str() + " (storage: " + cppStorageType.str() + ")");
}

BytecodeTypeParameter::BytecodeTypeParameter(const AttrOrTypeParameter &param)
    : name(param.getName().str()), accessorName(param.getAccessorName()),
      cppType(param.getCppType().str()),
      cppStorageType(param.getCppStorageType().str()),
      isOptional(param.isOptional()), kind(classifyParameter(param)) {
  // Extract enum type name for OptionalEnum kind.
  if (kind == Kind::OptionalEnum) {
    StringRef typeStr = param.getCppType();
    // Extract base name: "::mlir::cuda_tile::PaddingValueAttr" ->
    // "PaddingValue"
    auto split = typeStr.rsplit("::");
    if (split.second.empty() || !split.second.ends_with("Attr"))
      PrintFatalError("OptionalEnum parameter type must end with 'Attr': " +
                      typeStr.str());
    enumTypeName = split.second.drop_back(4).str();
  }
}

//===----------------------------------------------------------------------===//
// CudaTileType Implementation.
//===----------------------------------------------------------------------===//

CudaTileType::CudaTileType(const AttrOrTypeDef &typeDef, unsigned tagValue,
                           StringRef version)
    : typeName(typeDef.getCppClassName().str()),
      qualifiedTypeName(typeDef.getDialect().getCppNamespace().str() +
                        "::" + typeDef.getCppClassName().str()),
      typeTagValue(tagValue), sinceVersion(version.str()),
      needsReverseOrder(typeDef.getCppClassName() == "TileType") {

  // Analyze all parameters.
  llvm::append_range(parameters, typeDef.getParameters());
}

//===----------------------------------------------------------------------===//
// BuiltinType Implementation.
//===----------------------------------------------------------------------===//

BuiltinType::BuiltinType(StringRef name, StringRef qualifiedType, unsigned tag,
                         StringRef version, unsigned width, StringRef floatType)
    : enumName(name.str()), qualifiedTypeName(qualifiedType.str()),
      typeTagValue(tag), sinceVersion(version.str()), integerBitWidth(width),
      floatMlirTypeName(floatType.str()) {}

//===----------------------------------------------------------------------===//
// Analysis Entry Point.
//===----------------------------------------------------------------------===//

FailureOr<BytecodeTypeStructure>
mlir::tblgen::analyzeBytecodeTypes(const RecordKeeper &records) {
  BytecodeTypeStructure structure;

  // Build map of CudaTileTypeDef for matching.
  StringMap<const Record *> cudaTileTypeDefRecords;
  auto typeDefRecords = records.getAllDerivedDefinitions("CudaTileTypeDef");
  for (const Record *typeRecord : typeDefRecords)
    cudaTileTypeDefRecords[AttrOrTypeDef(typeRecord).getCppClassName()] =
        typeRecord;

  // Process all BytecodeTypeTag records.
  auto allTypeTagRecords = records.getAllDerivedDefinitions("BytecodeTypeTag");
  for (const Record *record : allTypeTagRecords) {
    StringRef enumName = record->getValueAsString("cppTypeName");
    unsigned tagValue = record->getValueAsInt("typeTagValue");
    StringRef version = record->getValueAsString("sinceVersion");

    // Add to enum.
    structure.allTypeTags.push_back({enumName.str(), tagValue});

    // Categorize and process based on subclass.
    if (record->isSubClassOf("IntegerTypeTag")) {
      structure.builtinSerializableTypes.emplace_back(
          enumName, "IntegerType", tagValue, version,
          record->getValueAsInt("integerBitWidth"));
    } else if (record->isSubClassOf("FloatTypeTag")) {
      structure.builtinSerializableTypes.emplace_back(
          enumName, "FloatType", tagValue, version, 0,
          record->getValueAsString("floatMlirTypeName"));
    } else if (record->isSubClassOf("CudaTileTypeTag")) {
      auto it = cudaTileTypeDefRecords.find(enumName);
      if (it != cudaTileTypeDefRecords.end())
        structure.cudaTileTypes.emplace_back(AttrOrTypeDef(it->second),
                                             tagValue, version);
    }
  }

  return structure;
}
