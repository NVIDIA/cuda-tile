//===- BytecodeAttrAnalysis.cpp ---------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrAnalysis.cpp - Attribute Bytecode Analysis Implementation
//
//===----------------------------------------------------------------------===//

#include "BytecodeAttrAnalysis.h"

#include "mlir/TableGen/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/TableGen/Error.h"

#include "BytecodeGenUtilities.h"
using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// Extracts the attribute tag name from a TableGen definition name.
static StringRef extractAttrTagName(StringRef defName) {
  defName.consume_front("CudaTile_");
  defName.consume_back("Attr");
  return defName;
}

//===----------------------------------------------------------------------===//
// BytecodeAttrParameter Implementation
//===----------------------------------------------------------------------===//

BytecodeAttrParameter::Kind
BytecodeAttrParameter::classifyParameter(const AttrOrTypeParameter &param) {
  StringRef cppType = param.getCppType();
  StringRef cppStorageType = param.getCppStorageType();

  auto containsAny = [](StringRef s, auto... patterns) {
    return (s.contains(patterns) || ...);
  };

  // Dense array attribute types -> DenseArray
  if (containsAny(cppType, "DenseI8ArrayAttr", "DenseI16ArrayAttr",
                  "DenseI32ArrayAttr", "DenseI64ArrayAttr", "DenseF32ArrayAttr",
                  "DenseF64ArrayAttr", "DenseBoolArrayAttr") ||
      containsAny(cppStorageType, "DenseI8ArrayAttr", "DenseI16ArrayAttr",
                  "DenseI32ArrayAttr", "DenseI64ArrayAttr", "DenseF32ArrayAttr",
                  "DenseF64ArrayAttr", "DenseBoolArrayAttr")) {
    return Kind::DenseArray;
  }

  if (const auto *defInit = dyn_cast_if_present<DefInit>(param.getDef())) {
    const Record *paramRecord = defInit->getDef();
    if (paramRecord->isSubClassOf("CudaTileI32EnumAttr") ||
        paramRecord->isSubClassOf("CudaTileI64EnumAttr") ||
        paramRecord->isSubClassOf("I32EnumAttr") ||
        paramRecord->isSubClassOf("I64EnumAttr") ||
        // MLIR's EnumAttr backs its single `$value` parameter with an
        // EnumParameter (derived from AttrParameter, not *EnumAttr), so match
        // it here to let a tagged EnumAttr serialize its value as a varint.
        paramRecord->isSubClassOf("EnumParameter") ||
        paramRecord->isSubClassOf("EnumAttr")) {
      return Kind::EnumValue;
    }
  }

  // Dictionary attributes -> NestedAttr (requires recursive serialization)
  if (cppType.contains("DictionaryAttr") ||
      cppStorageType.contains("DictionaryAttr")) {
    return Kind::NestedAttr;
  }

  // TypeAttr parameters are serialized as type-table indices.
  if (cppType.contains("TypeAttr") || cppStorageType.contains("TypeAttr")) {
    return Kind::TypeAttr;
  }

  // StringAttr parameters keep their MLIR wrapper on the C++ side.
  if (cppType.contains("StringAttr") || cppStorageType.contains("StringAttr")) {
    return Kind::StringAttr;
  }

  // StringRef parameters lower to raw string table indices.
  if (cppType.contains("StringRef") || cppStorageType.contains("std::string")) {
    return Kind::StringRef;
  }

  // Attribute arrays serialize as a length plus self-contained attributes.
  // Match ArrayRef<*Attr*>, SmallVector<*Attr*>, and ArrayAttr.
  auto isAttrContainer = [](StringRef t) {
    return t.contains("ArrayAttr") ||
           ((t.contains("ArrayRef") || t.contains("SmallVector")) &&
            (t.contains("Attr") || t.contains("Attribute")));
  };
  if (isAttrContainer(cppType) || isAttrContainer(cppStorageType)) {
    return Kind::AttributeArray;
  }

  // Nested attribute parameter. The wire format depends on whether the
  // declared type is a concrete attribute (inline payload) or an
  // `AttrInterface` (self-contained, with dispatch tag).
  if (const auto *defInit = dyn_cast_if_present<DefInit>(param.getDef())) {
    if (defInit->getDef()->isSubClassOf("CudaTileAttrInterfaceParam")) {
      return Kind::PolymorphicNestedAttr;
    }
  }
  if (cppType.contains("Attribute") || cppStorageType.contains("Attribute") ||
      cppType.ends_with("Attr") || cppStorageType.ends_with("Attr") ||
      cppType.contains("::Attr") || cppStorageType.contains("::Attr")) {
    return Kind::NestedAttr;
  }

  // Integer types (including optional<int>) -> VarInt
  // Signedness is handled separately via isSigned flag.
  if (containsAny(cppType, "int8_t", "int16_t", "int32_t", "int64_t") ||
      containsAny(cppStorageType, "int8_t", "int16_t", "int32_t", "int64_t")) {
    return Kind::VarInt;
  }

  // APInt parameters are stored as signed varints (restricted to 64-bit range
  // by the surrounding parser/verifier).
  if (cppType.contains("APInt") || cppStorageType.contains("APInt")) {
    return Kind::APInt;
  }

  // Raw `::mlir::Type` (or a concrete subclass such as `ShapedType`) is
  // serialized through the type table, mirroring the type-side
  // GenericType/SpecificType handling. This is what backs an attribute's self
  // type (MLIR's AttributeSelfTypeParameter, used to implement
  // TypedAttrInterface), but also covers any bare Type-valued parameter. The
  // earlier, more specific checks (TypeAttr, *Attr, etc.) have already run, so
  // a remaining "Type" spelling is an actual type parameter.
  if (cppType.contains("Type") || cppStorageType.contains("Type")) {
    return Kind::Type;
  }

  // Unsupported parameter type.
  PrintFatalError("Unsupported attribute parameter type for bytecode: " +
                  cppType.str() + " (storage: " + cppStorageType.str() + ")");
}

BytecodeAttrParameter::BytecodeAttrParameter(const AttrOrTypeParameter &param)
    : name(param.getName().str()), accessorName(param.getAccessorName()),
      cppType(param.getCppType().str()),
      cppStorageType(param.getCppStorageType().str()),
      isOptional(param.isOptional()), kind(classifyParameter(param)) {

  // Determine if this is a signed integer type (needs zigzag encoding).
  // Unsigned types contain "uint"; signed types contain "int" but not "uint".
  StringRef typeRef(cppType);
  StringRef storageRef(cppStorageType);
  isSigned = (typeRef.contains("int") && !typeRef.contains("uint")) ||
             (storageRef.contains("int") && !storageRef.contains("uint"));

  // Extract version and bytecode fallback from wrapped parameters.
  // CudaTileAttrParam adds sinceVersion and optionally bytecodeFallbackValue.
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
  }

  // Extract default value if present.
  if (auto defValue = param.getDefaultValue()) {
    defaultValue = defValue->str();
  }
}

//===----------------------------------------------------------------------===//
// CudaTileSerializableAttr Implementation
//===----------------------------------------------------------------------===//

CudaTileSerializableAttr::CudaTileSerializableAttr(const AttrOrTypeDef &attrDef,
                                                   unsigned tag,
                                                   StringRef version,
                                                   bool skipVersionCheck,
                                                   bool isPreview)
    : attrName(attrDef.getCppClassName().str()),
      qualifiedCppName(attrDef.getDialect().getCppNamespace().str() +
                       "::" + attrDef.getCppClassName().str()),
      tagValue(tag), sinceVersion(version.str()),
      skipVersionCheck(skipVersionCheck), isPreview(isPreview) {
  if (isPreview) {
    owningPreviewFeature = getOwningPreviewFeatureName(attrDef.getDef()).str();
  }
  // Derive tag name by stripping "Attr" suffix from C++ class name.
  StringRef tagNameRef(attrName);
  tagNameRef.consume_back("Attr");
  tagName = tagNameRef.str();

  // Legacy wire format shim: DivByAttr's pre-existing format writes the
  // required 'divisor' param before the flags byte, unlike the standard format
  // (flags first). This list is not expected to grow -- all new attrs should
  // use the standard format.
  requiredParamsFirst = (attrDef.getCppClassName() == "DivByAttr");

  // Check if attr has a verifier (genVerifyDecl = 1 in TableGen).
  // When true, getChecked() is available and should be used for better errors.
  hasVerifier = attrDef.genVerifyDecl();

  // Analyze and validate all parameters.
  for (const auto &attrParam : attrDef.getParameters()) {
    BytecodeAttrParameter param(attrParam);

    // Track optional parameters for flag-based encoding.
    if (param.isOptional) {
      hasOptionalParams = true;
    }

    // Validations.
    if (!skipVersionCheck) {
      if (param.sinceVersion.empty()) {
        PrintFatalError(
            "Parameter '" + param.name + "' in attribute '" + attrName +
            "' must be wrapped with CudaTileAttrParam to have version "
            "information.");
      }

      // Reject DefaultValuedParameter -- only OptionalParameter is supported.
      if (param.isOptional && !StringRef(param.defaultValue).ends_with("()") &&
          param.defaultValue != "std::nullopt") {
        PrintFatalError(
            "Parameter '" + param.name + "' in attribute '" + attrName +
            "' uses DefaultValuedParameter which is not supported for "
            "attribute serialization. Use OptionalParameter instead.");
      }

      // Evolved non-optional parameters need a fallback (either ODS
      // defaultValue or bytecodeFallbackValue).
      if (param.sinceVersion != sinceVersion && !param.isOptional &&
          param.defaultValue.empty() && param.bytecodeFallbackValue.empty()) {
        PrintFatalError(
            "Parameter '" + param.name + "' in attribute '" + attrName +
            "' was introduced in version " + param.sinceVersion +
            " after the attribute (version " + sinceVersion +
            "). It must have either an explicit bytecode fallback expression "
            "or an ODS default value for backward compatibility.");
      }
    }

    parameters.push_back(std::move(param));
  }
}

//===----------------------------------------------------------------------===//
// Analysis Implementation
//===----------------------------------------------------------------------===//

BytecodeAttrStructure
mlir::tblgen::analyzeBytecodeAttrs(const RecordKeeper &records) {
  BytecodeAttrStructure structure;

  // Build map of MLIR built-in attrs from CudaTileAttrAlias records.
  // These provide version info for MLIR built-in attributes we serialize.
  StringMap<StringRef> aliasVersions;
  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileAttrAlias")) {
    StringRef attrName = record->getValueAsString("attrName");
    StringRef version = record->getValueAsString("sinceVersion");
    if (attrName.empty())
      PrintFatalError(record->getLoc(), "CudaTileAttrAlias '" +
                                            record->getName().str() +
                                            "' has empty 'attrName'");
    if (version.empty())
      PrintFatalError(record->getLoc(), "CudaTileAttrAlias '" +
                                            record->getName().str() +
                                            "' has empty 'sinceVersion'");
    aliasVersions[attrName] = version;
  }

  // Build map of CudaTileAttrDef records, keyed by tag name.
  // Used for both tag version lookup and serializable attr matching.
  struct AttrDefInfo {
    const Record *record;
    StringRef sinceVersion;
    bool skipsVersionCheck;
    bool isPreview;
    StringRef owningPreviewFeature;
  };
  StringMap<AttrDefInfo> attrDefs;
  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileAttrDef")) {
    bool isPreview = record->isSubClassOf("CudaTilePreviewAttrDef");
    bool skips = isPreview;
    StringRef version;
    if (record->getValue("sinceVersion") &&
        !record->isValueUnset("sinceVersion")) {
      version = record->getValueAsString("sinceVersion");
    }

    if (!skips && version.empty())
      PrintFatalError(record->getLoc(),
                      "CudaTileAttrDef '" + record->getName().str() +
                          "' is missing or has empty 'sinceVersion'");

    StringRef owningPreviewFeature =
        isPreview ? getOwningPreviewFeatureName(record) : StringRef();
    attrDefs[extractAttrTagName(record->getName())] = {
        record, version, skips, isPreview, owningPreviewFeature};
  }

  // Add CudaTileEnumAttr to the attrDefs map, keyed by tag name. An MLIR
  // `EnumAttr` (e.g. one created via `CudaTileEnumAttr`) is an `AttrDef` with a
  // single `$value` EnumParameter, but it is *not* a `CudaTileAttrDef`, so the
  // scan above does not see it. A `BytecodeAttrTag` may still name one: once
  // tagged, it flows through the same serializable-attr machinery, serializing
  // its value as a single varint.
  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileEnumAttr")) {
    // Get the version from the EnumInfo, which must be a subclass of one of the
    // CudaTile*EnumAttr.
    const auto *enumInfo = record->getValueAsDef("enum");
    if (!enumInfo->isSubClassOf("CudaTileIntEnumAttr")) {
      PrintFatalError(record->getLoc(), "CudaTileEnumAttr has an unknown enum "
                                        "info, expected CudaTileIntEnumAttr");
    }

    StringRef version;
    if (enumInfo->getValue("sinceVersion") &&
        !enumInfo->isValueUnset("sinceVersion")) {
      version = enumInfo->getValueAsString("sinceVersion");
    }

    bool skips = false;
    if (!skips && version.empty()) {
      PrintFatalError(record->getLoc(),
                      "CudaTile '" + record->getName().str() +
                          "' is missing or has empty 'sinceVersion'");
    }

    attrDefs[extractAttrTagName(record->getName())] = {
        record, version, skips, /*isPreview=*/false,
        /*owningPreviewFeature=*/{}};
  }

  // Process BytecodeAttrTag records in a single pass: build bytecodeAttrs
  // (tag enum + version) and serializableAttrs (CudaTile attrs with params).
  for (const Record *record :
       records.getAllDerivedDefinitions("BytecodeAttrTag")) {
    StringRef attrName = record->getValueAsString("cppAttrName");
    unsigned tagValue = record->getValueAsInt("attrTagValue");
    bool isPreviewTag = record->isSubClassOf("PreviewBytecodeAttrTag");

    if (isPreviewTag && (tagValue < 0xD0 || tagValue > 0xEF))
      PrintFatalError(record->getLoc(),
                      "PreviewBytecodeAttrTag value must be in the reserved "
                      "range [0xD0, 0xEF]");
    if (!isPreviewTag && tagValue >= 0xD0 && tagValue <= 0xEF)
      PrintFatalError(record->getLoc(),
                      "attribute tags in the reserved range [0xD0, 0xEF] "
                      "must use PreviewBytecodeAttrTag");

    // Build BytecodeAttr for tag enum and version checking.
    BytecodeAttr attr;
    attr.attrName = attrName.str();
    attr.tagValue = tagValue;

    if (auto it = aliasVersions.find(attrName); it != aliasVersions.end()) {
      attr.sinceVersion = it->second.str();
    } else if (auto it = attrDefs.find(attrName); it != attrDefs.end()) {
      attr.sinceVersion = it->second.sinceVersion.str();
      attr.skipsVersionCheck = it->second.skipsVersionCheck;
      attr.isPreview = it->second.isPreview;
      attr.owningPreviewFeature = it->second.owningPreviewFeature.str();
      if (attr.isPreview != isPreviewTag)
        PrintFatalError(record->getLoc(),
                        "preview attribute/tag classification mismatch for '" +
                            attrName.str() + "'");
    } else {
      PrintFatalError(record->getLoc(),
                      "BytecodeAttrTag '" + attrName.str() +
                          "' has no matching CudaTileAttrAlias or "
                          "CudaTileAttrDef for version info");
    }
    structure.bytecodeAttrs.emplace_back(std::move(attr));

    // Check if this is a CudaTileAttrDef with parameters.
    auto it = attrDefs.find(attrName);
    if (it == attrDefs.end()) {
      continue;
    }

    AttrOrTypeDef attrDef(it->second.record);
    if (attrDef.getNumParameters() == 0) {
      continue;
    }

    structure.serializableAttrs.emplace_back(attrDef, tagValue,
                                             it->second.sinceVersion,
                                             it->second.skipsVersionCheck,
                                             it->second.isPreview);

    // Catch nested-attr params whose declared C++ type names an
    // AttrInterface that wasn't wrapped with `CudaTileAttrInterfaceParam`.
    for (const Record *iface :
         records.getAllDerivedDefinitions("AttrInterface")) {
      StringRef ifaceName = iface->getValueAsString("cppInterfaceName");
      std::string suffix = ("::" + ifaceName).str();
      for (const auto &param : structure.serializableAttrs.back().parameters) {
        if (param.kind == BytecodeAttrParameter::Kind::NestedAttr &&
            (param.cppType == ifaceName ||
             StringRef(param.cppType).ends_with(suffix))) {
          PrintFatalError(it->second.record->getLoc(),
                          "attribute '" + attrDef.getCppClassName().str() +
                              "' parameter '" + param.name +
                              "' has AttrInterface C++ type '" + param.cppType +
                              "'; wrap it with "
                              "CudaTileAttrInterfaceParam<\"" +
                              param.cppType + "\", \"<version>\">");
        }
      }
    }
  }

  // Process enum attributes.
  auto processEnumDefs = [&](StringRef baseClass, bool isPreviewEnum) {
    for (const Record *record : records.getAllDerivedDefinitions(baseClass)) {
      // A preview enum must keep genSpecializedAttr = 0. MLIR's auto-generated
      // attribute lives outside the CudaTilePreviewAttrDef machinery, so it
      // would carry neither preview bytecode metadata nor owning-feature
      // classification and would bypass the named preview gate. Expose preview
      // enums as attributes via CudaTilePreviewEnumAttr instead.
      if (isPreviewEnum) {
        if (const auto *gsa = record->getValue("genSpecializedAttr");
            gsa && record->getValueAsBit("genSpecializedAttr"))
          PrintFatalError(
              record->getLoc(),
              "preview enum '" + record->getName().str() +
                  "' sets genSpecializedAttr = 1; the auto-generated attribute "
                  "bypasses the preview gate. Set genSpecializedAttr = 0 and "
                  "expose it with CudaTilePreviewEnumAttr.");
      }
      // Skip utility enums that are not MLIR attributes (e.g.,
      // GpuArchitecture, HintKey).  These use specialized case classes
      // that indicate they are not meant for bytecode serialization.
      auto cases = record->getValueAsListOfDefs("enumerants");
      if (!cases.empty()) {
        const Record *firstCase = cases.front();
        if (firstCase->isSubClassOf("CudaTileGpuArchCase") ||
            firstCase->isSubClassOf("CudaTileHintKeyCase")) {
          continue;
        }
      }

      CudaTileEnumAttr enumAttr;
      StringRef className = record->getValueAsString("className");
      enumAttr.enumName = className.str();
      enumAttr.attrName = className.str() + "Attr";

      if (record->getValue("sinceVersion") &&
          !record->isValueUnset("sinceVersion")) {
        enumAttr.sinceVersion = record->getValueAsString("sinceVersion").str();
      }
      enumAttr.skipsVersionCheck = enumAttr.sinceVersion.empty();

      if (enumAttr.skipsVersionCheck) {
        structure.enumAttrs.emplace_back(std::move(enumAttr));
        continue;
      }

      for (const auto *caseInit : *record->getValueAsListInit("enumerants")) {
        const auto *caseRecord = cast<DefInit>(caseInit)->getDef();

        if (!caseRecord->getValue("sinceVersion") ||
            caseRecord->isValueUnset("sinceVersion") ||
            caseRecord->getValueAsString("sinceVersion").empty())
          PrintFatalError(caseRecord->getLoc(),
                          "Enum case '" + caseRecord->getName().str() +
                              "' is missing or has empty 'sinceVersion'");

        EnumCase enumCase;
        enumCase.name = caseRecord->getValueAsString("symbol").str();
        enumCase.value = caseRecord->getValueAsInt("value");
        enumCase.sinceVersion =
            caseRecord->getValueAsString("sinceVersion").str();
        enumAttr.cases.emplace_back(std::move(enumCase));
      }

      structure.enumAttrs.emplace_back(std::move(enumAttr));
    }
  };

  processEnumDefs("CudaTileI32EnumAttr", /*isPreviewEnum=*/false);
  processEnumDefs("CudaTileI64EnumAttr", /*isPreviewEnum=*/false);
  processEnumDefs("CudaTilePreviewI32EnumInfo", /*isPreviewEnum=*/true);

  // Sort by tag value for deterministic enum generation.
  llvm::sort(structure.bytecodeAttrs, [](const auto &a, const auto &b) {
    return a.tagValue < b.tagValue;
  });

  return structure;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

LogicalResult mlir::tblgen::validateAttrTagAssignments(
    const RecordKeeper &records, const BytecodeAttrStructure &structure) {
  DenseSet<StringRef> attrsWithTags;
  for (const auto &attr : structure.bytecodeAttrs)
    attrsWithTags.insert(attr.attrName);

  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileAttrDef")) {
    StringRef tagName = extractAttrTagName(record->getName());
    if (!attrsWithTags.contains(tagName))
      PrintFatalError(record->getLoc(), "CudaTileAttrDef '" +
                                            record->getName().str() +
                                            "' is missing BytecodeAttrTag in "
                                            "BytecodeAttrOpcodes.td");
  }

  return success();
}
