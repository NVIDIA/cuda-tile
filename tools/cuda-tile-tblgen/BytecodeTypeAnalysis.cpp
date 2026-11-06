//===- BytecodeTypeAnalysis.cpp ---------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BytecodeTypeAnalysis.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"

#include "BytecodeGenUtilities.h"
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

  // Dense array attrs use the existing dense-array bytecode path rather than
  // the new nested-attribute array encoding. Only DenseI32ArrayAttr has a
  // codegen path on the type side today.
  if (cppType.contains("DenseI32ArrayAttr") ||
      cppStorageType.contains("DenseI32ArrayAttr")) {
    return Kind::DenseI32Array;
  }
  static constexpr StringLiteral kUnsupportedDenseArrays[] = {
      "DenseI8ArrayAttr",  "DenseI16ArrayAttr", "DenseI64ArrayAttr",
      "DenseF32ArrayAttr", "DenseF64ArrayAttr", "DenseBoolArrayAttr",
  };
  for (StringRef name : kUnsupportedDenseArrays) {
    if (cppType.contains(name) || cppStorageType.contains(name)) {
      PrintFatalError("Type parameter uses unsupported dense-array attr '" +
                      name.str() +
                      "'. Extend BytecodeTypeAnalysis and the codegen to add "
                      "support for new dense-array element types.");
    }
  }

  // Container checks must precede the scalar TypeAttr check below: a param
  // like `ArrayRef<TypeAttr>` must be classified as AttributeArray, not as
  // a single TypeAttr.
  bool isContainer =
      cppType.contains("ArrayAttr") || cppStorageType.contains("ArrayAttr") ||
      (cppType.contains("ArrayRef") &&
       (cppType.contains("Attribute") || cppType.contains("Attr")));
  if (isContainer) {
    return Kind::AttributeArray;
  }

  if (cppType.contains("TypeAttr") || cppStorageType.contains("TypeAttr")) {
    return Kind::TypeAttr;
  }

  // Optional attribute parameters use the dedicated optional-enum bitfield
  // encoding.
  if (param.isOptional() && cppType.contains("Attr") &&
      !cppType.contains("Attribute")) {
    return Kind::OptionalEnum;
  }

  if (const auto *defInit = dyn_cast_if_present<DefInit>(param.getDef())) {
    if (defInit->getDef()->isSubClassOf("CudaTileAttrInterfaceParam")) {
      return Kind::PolymorphicNestedAttr;
    }
    if (defInit->getDef()->isSubClassOf(
            "CudaTilePreviewTypeExtensionParam")) {
      return Kind::PolymorphicNestedAttr;
    }
    // A required (non-optional) enum parameter declared via `EnumParameter<>`
    // carries the raw enum class as its C++ type. It is serialized as its
    // underlying integer value (see codegen).
    if (defInit->getDef()->isSubClassOf("EnumParameter")) {
      return Kind::Enum;
    }
  }
  if (cppType.contains("Attribute") || cppStorageType.contains("Attribute") ||
      cppType.ends_with("Attr") || cppStorageType.ends_with("Attr") ||
      cppType.contains("::Attr") || cppStorageType.contains("::Attr")) {
    return Kind::NestedAttr;
  }

  // ArrayRefParameter sets cppStorageType to SmallVector.
  if (cppStorageType.contains("SmallVector<int64_t>")) {
    return Kind::Int64Array;
  }
  if (cppStorageType.contains("SmallVector<int32_t>")) {
    return Kind::Int32Array;
  }

  // Check for uint32_t scalar
  if (cppType == "uint32_t" || cppStorageType == "uint32_t") {
    return Kind::UInt32Scalar;
  }

  // Check for Type parameters.
  if (cppType.contains("Type")) {
    if (cppType.contains("cuda_tile::")) {
      return Kind::SpecificType;
    }
    return Kind::GenericType;
  }

  // Unsupported parameter type.
  PrintFatalError("Unsupported parameter type for bytecode generation: " +
                  cppType.str() + " (storage: " + cppStorageType.str() + ")");
}

BytecodeTypeParameter::BytecodeTypeParameter(const AttrOrTypeParameter &param)
    : name(param.getName().str()), accessorName(param.getAccessorName()),
      cppType(param.getCppType().str()),
      cppStorageType(param.getCppStorageType().str()),
      isOptional(param.isOptional()), kind(classifyParameter(param)) {

  // Extract version and bytecode fallback from wrapped parameters.
  // CudaTileTypeParam adds sinceVersion and optionally bytecodeFallbackValue.
  // Use getValue() to check if field exists before accessing - raw parameters
  // like ArrayRefParameter don't have these fields.
  if (const auto *defInit = dyn_cast_if_present<DefInit>(param.getDef())) {
    const Record *paramRecord = defInit->getDef();
    if (paramRecord->getValue("sinceVersion") &&
        !paramRecord->isValueUnset("sinceVersion")) {
      sinceVersion = paramRecord->getValueAsString("sinceVersion").str();
    }
    if (paramRecord->getValue("bytecodeFallbackValue") &&
        !paramRecord->isValueUnset("bytecodeFallbackValue")) {
      bytecodeFallbackValue =
          paramRecord->getValueAsString("bytecodeFallbackValue").str();
    }
    if (paramRecord->getValue("bytecodePreviewExtension") &&
        !paramRecord->isValueUnset("bytecodePreviewExtension")) {
      isPreviewExtension =
          paramRecord->getValueAsBit("bytecodePreviewExtension");
    }
    if (paramRecord->getValue("previewFeature") &&
        !paramRecord->isValueUnset("previewFeature")) {
      owningPreviewFeature = getOwningPreviewFeatureName(paramRecord).str();
    }
    // The name is interpolated straight into a `PreviewFeature::` enumerator in
    // the generated classifier, so an empty one only fails when the generated
    // header is compiled.
    if (isPreviewExtension && owningPreviewFeature.empty()) {
      PrintFatalError(paramRecord->getLoc(),
                      "bytecode preview extension parameter '" + name +
                          "' is missing 'previewFeature'");
    }
  }

  // Extract default value if present.
  if (auto defValue = param.getDefaultValue()) {
    defaultValue = defValue->str();
  }

  // Extract enum type name for the enum kinds. A required enum parameter is
  // spelled as the enum itself; an optional one is wrapped in its attribute.
  if (kind == Kind::Enum) {
    enumTypeName = cppType;
  } else if (kind == Kind::OptionalEnum) {
    auto split = StringRef(cppType).rsplit("::");
    if (split.second.empty() || !split.second.ends_with("Attr")) {
      PrintFatalError("OptionalEnum parameter type must end with 'Attr': " +
                      cppType);
    }
    enumTypeName = split.second.drop_back(4).str();
  }

  // Compute if this parameter is optional with null default (Type, Enum, etc.).
  usesOptionalTypeFlags = isOptional && (defaultValue == cppStorageType + "()");
}

//===----------------------------------------------------------------------===//
// CudaTileType Implementation.
//===----------------------------------------------------------------------===//

CudaTileType::CudaTileType(const AttrOrTypeDef &typeDef, unsigned tagValue,
                           StringRef version, bool isPreview)
    : typeName(typeDef.getCppClassName().str()),
      qualifiedTypeName(typeDef.getDialect().getCppNamespace().str() +
                        "::" + typeDef.getCppClassName().str()),
      typeTagValue(tagValue), sinceVersion(version.str()),
      needsReverseOrder(typeDef.getCppClassName() == "TileType"),
      skipVersionCheck(isPreview), isPreview(isPreview) {
  const Record *typeRecord = typeDef.getDef();
  if (isPreview) {
    owningPreviewFeature = getOwningPreviewFeatureName(typeRecord).str();
  }
  hasCheckedBuilder = typeDef.genVerifyDecl() || typeDef.genVerifyInvariantsImpl();

  // Analyze and validate all parameters.
  for (const auto &attrParam : typeDef.getParameters()) {
    BytecodeTypeParameter param(attrParam);

    // Validate: All parameters must have version information.
    if (!skipVersionCheck && !param.isPreviewExtension &&
        param.sinceVersion.empty()) {
      PrintFatalError(
          "Parameter '" + param.name + "' in type '" + typeName +
          "' must be wrapped with CudaTileTypeParam or "
          "CudaTileConstrainedTypeParam to have version information.");
    }

    // Detect optional Type parameters with null defaults for flag-based
    // encoding.
    if (param.usesOptionalTypeFlags) {
      hasOptionalTypeParams = true;
      if (!skipVersionCheck &&
          (firstOptionalTypeParamVersion.empty() ||
           param.sinceVersion < firstOptionalTypeParamVersion)) {
        firstOptionalTypeParamVersion = param.sinceVersion;
      }
    }

    // Validate: Non-optional parameters introduced after type need a default
    // (either ODS defaultValue or bytecodeFallbackValue).
    if (!skipVersionCheck && !param.isPreviewExtension && !param.isOptional &&
        param.sinceVersion != sinceVersion) {
      if (param.defaultValue.empty() && param.bytecodeFallbackValue.empty()) {
        PrintFatalError(
            "Parameter '" + param.name + "' in type '" + typeName +
            "' was introduced in version " + param.sinceVersion +
            " after the type (version " + sinceVersion +
            "). It must have either an explicit bytecode fallback expression "
            "or an ODS default value for backward compatibility.");
      }
    }

    parameters.push_back(std::move(param));
  }

  unsigned previewExtensionCount = 0;
  for (auto [index, param] : llvm::enumerate(parameters)) {
    if (!param.isPreviewExtension)
      continue;
    ++previewExtensionCount;
    if (param.isOptional)
      PrintFatalError(typeRecord->getLoc(),
                      "preview extension parameter '" + param.name +
                          "' in type '" + typeName +
                          "' must not be optional");
    if (param.bytecodeFallbackValue.empty())
      PrintFatalError(typeRecord->getLoc(),
                      "preview extension parameter '" + param.name +
                          "' in type '" + typeName +
                          "' requires bytecodeFallbackValue");
    bool isLastOnWire = needsReverseOrder ? index == 0
                                          : index + 1 == parameters.size();
    if (!isLastOnWire)
      PrintFatalError(typeRecord->getLoc(),
                      "preview extension parameter '" + param.name +
                          "' in type '" + typeName +
                          "' must be last in bytecode serialization order");
  }
  if (previewExtensionCount > 1)
    PrintFatalError(typeRecord->getLoc(),
                    "type '" + typeName +
                        "' may have at most one preview extension parameter");
}

//===----------------------------------------------------------------------===//
// BuiltinType Implementation.
//===----------------------------------------------------------------------===//

BuiltinType::BuiltinType(StringRef name, StringRef qualifiedType, unsigned tag,
                         StringRef version, unsigned width, StringRef floatType,
                         StringRef singleton, bool preview,
                         StringRef owningPreviewFeature)
    : enumName(name.str()), qualifiedTypeName(qualifiedType.str()),
      typeTagValue(tag), sinceVersion(version.str()), integerBitWidth(width),
      floatMlirTypeName(floatType.str()), singletonCppType(singleton.str()),
      isPreview(preview), owningPreviewFeature(owningPreviewFeature.str()) {}

//===----------------------------------------------------------------------===//
// Analysis Entry Point.
//===----------------------------------------------------------------------===//

/// Validates all CudaTileTypeDef types have sinceVersion and BytecodeTypeTag.
static void validateAllTypesHaveBytecodeSupport(
    const StringMap<const Record *> &cudaTileTypeDefRecords,
    const llvm::DenseSet<StringRef> &typesWithTags) {
  for (const auto &entry : cudaTileTypeDefRecords) {
    StringRef typeName = entry.getKey();
    const Record *typeRecord = entry.getValue();

    bool isPreview = typeRecord->isSubClassOf("CudaTilePreviewTypeDef");

    if (!isPreview && typeRecord->getValueAsString("sinceVersion").empty()) {
      PrintFatalError(typeRecord->getLoc(), "CudaTileTypeDef '" +
                                                typeName.str() +
                                                "' is missing 'sinceVersion'");
    }

    if (!typesWithTags.count(typeName)) {
      PrintFatalError(typeRecord->getLoc(),
                      "CudaTileTypeDef '" + typeName.str() +
                          "' is missing BytecodeTypeTag assignment");
    }
  }
}

FailureOr<BytecodeTypeStructure>
mlir::tblgen::analyzeBytecodeTypes(const RecordKeeper &records) {
  BytecodeTypeStructure structure;

  // Build the index of every CudaTile enum-attribute wrapper's qualified C++
  // class name. CudaTileEnumAttr is the only wrapper class used for CudaTile
  // enum attribute wrappers (see Dialect.td); all enum infos are wrapped
  // through it. Both qualified forms ("ns::Foo" and "::ns::Foo") are indexed
  // so lookups don't depend on how OptionalParameter spells the name.
  llvm::StringSet<> enumAttrCppTypes;
  for (const Record *r : records.getAllDerivedDefinitions("CudaTileEnumAttr")) {
    AttrOrTypeDef def(r);
    std::string qualified =
        (def.getDialect().getCppNamespace() + "::" + def.getCppClassName())
            .str();
    enumAttrCppTypes.insert(qualified);
    enumAttrCppTypes.insert("::" + qualified);
  }

  // Build map of CudaTileTypeDef for matching.
  StringMap<const Record *> cudaTileTypeDefRecords;
  for (const Record *typeRecord :
       records.getAllDerivedDefinitions("CudaTileTypeDef")) {
    cudaTileTypeDefRecords[AttrOrTypeDef(typeRecord).getCppClassName()] =
        typeRecord;
  }

  // Build map of builtin type versions, stripping "CudaTile_" prefix.
  // Maps enum name (e.g., "Int32") to version string for efficient lookup.
  StringMap<StringRef> builtinTypeVersions;
  for (const Record *aliasRecord :
       records.getAllDerivedDefinitions("CudaTileTypeAlias")) {
    StringRef aliasName = aliasRecord->getName();
    StringRef version = aliasRecord->getValueAsString("sinceVersion");
    if (aliasName.starts_with("CudaTile_")) {
      builtinTypeVersions[aliasName.drop_front(9)] = version;
    }
  }

  // Helper to lookup version from tag enum name.
  auto lookupBuiltinVersion = [&](StringRef enumName) -> StringRef {
    StringRef version = builtinTypeVersions.lookup(enumName);
    if (version.empty()) {
      PrintFatalError("No version found for builtin type: " + enumName.str());
    }
    return version;
  };

  // Track which CudaTile types have BytecodeTypeTag assignments.
  llvm::DenseSet<StringRef> typesWithTags;

  // Process all BytecodeTypeTag records.
  for (const Record *record :
       records.getAllDerivedDefinitions("BytecodeTypeTag")) {
    StringRef enumName = record->getValueAsString("cppTypeName");
    unsigned tagValue = record->getValueAsInt("typeTagValue");
    // Preview tags are the CudaTile preview tag subclass or the preview builtin
    // singleton subclass.
    bool isPreviewTag =
        record->isSubClassOf("PreviewCudaTileTypeTag") ||
        record->isSubClassOf("PreviewBuiltinSingletonTypeTag");

    if (isPreviewTag && (tagValue < 0xD0 || tagValue > 0xEF))
      PrintFatalError(record->getLoc(),
                      "PreviewCudaTileTypeTag value must be in the reserved "
                      "range [0xD0, 0xEF]");
    if (!isPreviewTag && tagValue >= 0xD0 && tagValue <= 0xEF)
      PrintFatalError(record->getLoc(),
                      "type tags in the reserved range [0xD0, 0xEF] must use "
                      "PreviewCudaTileTypeTag");

    // Validate the tag name is non-empty.
    if (enumName.empty()) {
      PrintFatalError(record->getLoc(), "BytecodeTypeTag '" +
                                            record->getName().str() +
                                            "' has empty 'cppTypeName'");
    }

    // Add to enum.
    structure.allTypeTags.emplace_back(enumName.str(), tagValue);

    // Categorize and process based on subclass.
    if (record->isSubClassOf("IntegerTypeTag")) {
      structure.builtinSerializableTypes.emplace_back(
          enumName, "IntegerType", tagValue, lookupBuiltinVersion(enumName),
          record->getValueAsInt("integerBitWidth"));
    } else if (record->isSubClassOf("FloatTypeTag")) {
      structure.builtinSerializableTypes.emplace_back(
          enumName, "FloatType", tagValue, lookupBuiltinVersion(enumName), 0,
          record->getValueAsString("floatMlirTypeName"));
    } else if (record->isSubClassOf("PreviewBuiltinSingletonTypeTag")) {
      // Nullary preview builtin: versionless, matched via
      // isa<cppType> and reconstructed via cppType::get(ctx).
      StringRef cppType = record->getValueAsString("builtinSingletonCppType");
      structure.builtinSerializableTypes.emplace_back(
          enumName, cppType, tagValue, /*version=*/"", /*width=*/0,
          /*floatType=*/"", /*singleton=*/cppType, /*preview=*/true,
          getOwningPreviewFeatureName(record));
    } else if (record->isSubClassOf("HandSerializedTypeTag")) {
      // Reserves the tag value in the enum only; the serializer and
      // deserializer live in BytecodeWriter.cpp and BytecodeReader.cpp.
    } else if (record->isSubClassOf("CudaTileTypeTag")) {
      typesWithTags.insert(enumName);
      auto it = cudaTileTypeDefRecords.find(enumName);
      if (it == cudaTileTypeDefRecords.end()) {
        // The tag value is frozen wire format and was added to the enum above,
        // so accepting the unmatched record would consume it while generating
        // no serializer, deserializer or classifier. A type deliberately
        // serialized by hand must say so with HandSerializedTypeTag.
        PrintFatalError(record->getLoc(),
                        "BytecodeTypeTag '" + enumName.str() +
                            "' has no matching CudaTileTypeDef; use "
                            "HandSerializedTypeTag if it is serialized by hand");
      }
      AttrOrTypeDef typeDef(it->second);
      bool isPreviewType =
          typeDef.getDef()->isSubClassOf("CudaTilePreviewTypeDef");
      if (isPreviewType != isPreviewTag)
        PrintFatalError(record->getLoc(),
                        "preview type/tag classification mismatch for '" +
                            enumName.str() + "'");
      StringRef typeVersion = typeDef.getDef()->getValueAsString("sinceVersion");
      structure.cudaTileTypes.emplace_back(typeDef, tagValue, typeVersion,
                                           isPreviewType);
    } else {
      PrintFatalError(record->getLoc(),
                      "BytecodeTypeTag '" + enumName.str() +
                          "' must subclass IntegerTypeTag, FloatTypeTag, "
                          "PreviewBuiltinSingletonTypeTag, "
                          "HandSerializedTypeTag, or CudaTileTypeTag");
    }
  }
  for (const BuiltinType &bt : structure.builtinSerializableTypes) {
    if (bt.isSingleton() && !bt.isPreview)
      PrintFatalError("builtin singleton type '" + bt.enumName +
                      "' is not public-preview. Public singletons need a "
                      "version guard in the type serializer and a "
                      "getTypeMinVersion entry before they can be serialized; "
                      "add those, or expose it via PreviewBuiltinSingletonTypeTag.");
  }

  // Validate all CudaTileTypeDef types have bytecode support.
  validateAllTypesHaveBytecodeSupport(cudaTileTypeDefRecords, typesWithTags);

  // Validate every OptionalEnum-classified parameter is actually a
  // CudaTileEnumAttr-derived enum-attribute wrapper.
  for (const CudaTileType &type : structure.cudaTileTypes) {
    for (const BytecodeTypeParameter &p : type.parameters) {
      if (p.kind != BytecodeTypeParameter::Kind::OptionalEnum) {
        continue;
      }
      if (!enumAttrCppTypes.contains(p.cppType)) {
        PrintFatalError(
            "Optional attribute parameter '" + p.name + "' in type '" +
            type.typeName + "' has cppType '" + p.cppType +
            "' which is not a CudaTileEnumAttr-derived enum-attribute "
            "wrapper. Optional attribute params in CudaTile types are only "
            "supported for enum attributes today; either drop optionality, "
            "or extend BytecodeTypeAnalysis with a new optional-attr Kind.");
      }
    }
  }

  return structure;
}
