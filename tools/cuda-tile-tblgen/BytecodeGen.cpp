//===- BytecodeGen.cpp - CUDA Tile dialect bytecode generator ---*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TableGen backend for generating bytecode
// reader/writer functions for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "BytecodeAttrAnalysis.h"
#include "BytecodeAttrCodeGen.h"
#include "BytecodeGenUtilities.h"
#include "BytecodeTypeAnalysis.h"
#include "BytecodeTypeCodeGen.h"
#include <algorithm>
#include <map>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// Reads an OpcodeRange definition from TableGen records.
/// Returns {minValue, maxValue} or nullopt if not defined.
static std::optional<std::pair<unsigned, unsigned>>
getOpcodeRange(const RecordKeeper &records, StringRef name) {
  const Record *def = records.getDef(name);
  if (!def)
    return std::nullopt;
  return std::make_pair(static_cast<unsigned>(def->getValueAsInt("minValue")),
                        static_cast<unsigned>(def->getValueAsInt("maxValue")));
}

/// Validates that all CudaTileOpDef operations have bytecode opcode
/// assignments.
static void validateAllOperationsHaveOpcodes(
    const RecordKeeper &records,
    const llvm::DenseSet<StringRef> &opsWithOpcodes) {
  for (const Record *opDef :
       records.getAllDerivedDefinitions("CudaTileOpDef")) {
    if (!opsWithOpcodes.count(opDef->getName()))
      PrintFatalError(opDef->getLoc(),
                      "operation '" + Operator(opDef).getOperationName() +
                          "' is missing BytecodeOpcode assignment");
  }
}

/// Validates all opcode assignments for consistency and correctness.
/// Checks:
/// - Opcode values are within designated ranges for each opcode class.
/// - No duplicate opcode values across all assignments.
/// - No duplicate operation assignments.
/// - Opcode class matches operation type.
static void
validateAllOpcodeAssignments(const RecordKeeper &records,
                             ArrayRef<const Record *> opcodeRecords) {
  // Track seen opcode values and operations to detect duplicates.
  llvm::DenseMap<unsigned, const Record *> seenOpcodes;
  llvm::DenseMap<StringRef, const Record *> seenOperations;
  llvm::DenseSet<StringRef> opsWithOpcodes;

  for (const Record *record : opcodeRecords) {
    const Record *opRecord = record->getValueAsDef("operation");
    unsigned opcodeValue = record->getValueAsInt("opcodeValue");
    StringRef opName = opRecord->getName();

    // Check for duplicate opcode values.
    if (auto it = seenOpcodes.find(opcodeValue); it != seenOpcodes.end()) {
      PrintFatalError(
          record->getLoc(),
          "duplicate opcode value 0x" + llvm::utohexstr(opcodeValue) +
              " - already assigned to operation in " + it->second->getName());
    }
    seenOpcodes[opcodeValue] = record;

    // Check for duplicate operation assignments.
    if (auto it = seenOperations.find(opName); it != seenOperations.end()) {
      PrintFatalError(record->getLoc(),
                      "operation '" + opName +
                          "' has multiple opcode assignments - already "
                          "assigned in " +
                          it->second->getName());
    }
    seenOperations[opName] = record;
    opsWithOpcodes.insert(opName);

    // Validate opcode ranges and operation type consistency.
    if (record->isSubClassOf("PublicOpcode")) {
      auto publicRange = getOpcodeRange(records, "PublicOpcodeRange");
      auto testingRange = getOpcodeRange(records, "TestingOpcodeRange");

      // Public opcodes must be in public range or testing range.
      auto inRange = [&](auto &range) {
        return range && opcodeValue >= range->first &&
               opcodeValue <= range->second;
      };

      if (!inRange(publicRange) && !inRange(testingRange)) {
        std::string rangeMsg = "PublicOpcode value 0x" +
                               llvm::utohexstr(opcodeValue) +
                               " is out of range (must be ";
        if (publicRange)
          rangeMsg += "0x" + llvm::utohexstr(publicRange->first) + " - 0x" +
                      llvm::utohexstr(publicRange->second);
        if (publicRange && testingRange)
          rangeMsg += " or ";
        if (testingRange)
          rangeMsg += ">= 0x" + llvm::utohexstr(testingRange->first);
        rangeMsg += ")";
        PrintFatalError(record->getLoc(), rangeMsg);
      }

      // Public-preview operations must not use PublicOpcode.
      if (opRecord->isSubClassOf("CudaTilePreviewOpDef"))
        PrintFatalError(record->getLoc(),
                        "PublicOpcode cannot be used with public-preview "
                        "operation '" +
                            opName + "'. Use PreviewOpcode instead.");

      // Testing operation validation.
      if (testingRange && records.getClass("CudaTileTestingOpDef")) {
        bool inTestingRange = opcodeValue >= testingRange->first;
        // Testing operations must use testing opcode range.
        if (opRecord->isSubClassOf("CudaTileTestingOpDef")) {
          if (!inTestingRange)
            PrintFatalError(record->getLoc(),
                            "Testing operation '" + opName +
                                "' must use testing opcode range (>= 0x" +
                                llvm::utohexstr(testingRange->first) +
                                "), but has opcode 0x" +
                                llvm::utohexstr(opcodeValue));
        } else {
          // Non-testing operations must not use testing opcode range.
          if (inTestingRange)
            PrintFatalError(record->getLoc(),
                            "Non-testing operation '" + opName +
                                "' cannot use testing opcode range (>= 0x" +
                                llvm::utohexstr(testingRange->first) +
                                "). This range is reserved for testing.");
        }
      }
    }

    // Public-preview opcode validation (available in all builds).
    if (record->isSubClassOf("PreviewOpcode")) {
      auto range = getOpcodeRange(records, "PreviewOpcodeRange");
      if (range && (opcodeValue < range->first || opcodeValue > range->second))
        PrintFatalError(
            record->getLoc(),
            "PreviewOpcode value 0x" + llvm::utohexstr(opcodeValue) +
                " is out of range (must be 0x" + llvm::utohexstr(range->first) +
                " - 0x" + llvm::utohexstr(range->second) + ")");

      // Preview opcodes should only be used with public-preview operations.
      if (!opRecord->isSubClassOf("CudaTilePreviewOpDef"))
        PrintFatalError(record->getLoc(),
                        "PreviewOpcode should only be used with public-preview "
                        "operations (CudaTilePreviewOpDef), but '" +
                            opName + "' is not a preview operation");
    }

  }

  // Validate all operations have opcode assignments.
  validateAllOperationsHaveOpcodes(records, opsWithOpcodes);
}

/// Generates the opcode enum definition from TableGen records.
static void generateOpcodeEnumDefinition(const RecordKeeper &records,
                                         raw_ostream &os) {
  emitSourceFileHeader("Generated Opcode Enum Definition", os);

  // Get all BytecodeOpcode records.
  auto opcodeRecords = records.getAllDerivedDefinitions("BytecodeOpcode");

  // Validate all opcode assignments before generating code.
  validateAllOpcodeAssignments(records, opcodeRecords);

  os << "namespace mlir {\n"
     << "namespace cuda_tile {\n"
     << "namespace Bytecode {\n\n"
     << "/// FROZEN at current assignments for backward compatibility.\n"
     << "/// WARNING: NEVER CHANGE THESE VALUES - they must remain stable for "
        "backward\n"
     << "/// compatibility.\n"
     << "enum class Opcode {\n"
     << "  // === PUBLIC OPERATIONS ===\n"
     << "  // These are available in all builds and must never be "
        "renumbered.\n";

  // Generate public opcodes.
  for (const Record *record : opcodeRecords) {
    if (record->isSubClassOf("PublicOpcode")) {
      const Record *opRecord = record->getValueAsDef("operation");
      unsigned opcodeValue = record->getValueAsInt("opcodeValue");
      Operator op(opRecord);
      os << "  " << op.getCppClassName() << " = 0x"
         << llvm::format("%X", opcodeValue) << ",\n";
    }
  }
  os << "\n// Reserved range for future PUBLIC operations.\n\n";

  // Generate public-preview opcodes. These ship in all builds but are gated at
  // runtime by their named preview feature and are not part of the versioned
  // map.
  os << "  // === PUBLIC-PREVIEW OPERATIONS ===\n"
     << "  // Available in all builds; runtime-gated and version-locked.\n";
  for (const Record *record : opcodeRecords) {
    if (record->isSubClassOf("PreviewOpcode")) {
      const Record *opRecord = record->getValueAsDef("operation");
      unsigned opcodeValue = record->getValueAsInt("opcodeValue");
      Operator op(opRecord);
      os << "  " << op.getCppClassName() << " = 0x"
         << llvm::format("%X", opcodeValue) << ",\n";
    }
  }
  os << "\n";
  os << "};\n\n"
     << "} // namespace Bytecode\n"
     << "} // namespace cuda_tile\n"
     << "} // namespace mlir\n";
}

/// Generates the opcode map implementation from TableGen records
static void generateOpcodeMap(const RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Generated Opcode Map", os);

  // Get all BytecodeOpcode records.
  auto opcodeRecords = records.getAllDerivedDefinitions("BytecodeOpcode");

  os << "namespace mlir {\n"
     << "namespace cuda_tile {\n"
     << "namespace Bytecode {\n\n"
     << "const llvm::StringMap<Opcode> &getOpcodeMap() {\n"
     << "  static const llvm::StringMap<Opcode> opcodeMap = {\n"
     << "    // === PUBLIC OPERATIONS ===\n"
     << "    // These mappings are FROZEN and must never change for backward\n"
     << "    // compatibility.\n";

  // Generate public operation mappings.
  for (const Record *record : opcodeRecords) {
    if (record->isSubClassOf("PublicOpcode")) {
      const Record *opRecord = record->getValueAsDef("operation");
      Operator op(opRecord);
      os << "    {\"" << op.getOperationName()
         << "\", Opcode::" << op.getCppClassName() << "},\n";
    }
  }

  // Generate public-preview operation mappings (available in all builds).
  os << "\n    // === PUBLIC-PREVIEW OPERATIONS ===\n";
  for (const Record *record : opcodeRecords) {
    if (record->isSubClassOf("PreviewOpcode")) {
      const Record *opRecord = record->getValueAsDef("operation");
      Operator op(opRecord);
      os << "    {\"" << op.getOperationName()
         << "\", Opcode::" << op.getCppClassName() << "},\n";
    }
  }

  os << "  };\n"
     << "  return opcodeMap;\n"
     << "}\n\n"
     << "} // namespace Bytecode\n"
     << "} // namespace cuda_tile\n"
     << "} // namespace mlir\n";
}

/// Generates the C++ function signature for the 'write<OpName>' function,
/// which handles serialization for a specific cuda_tile operation.
/// Returns FailureOr<size_t> where the size_t is the number of results
/// that were serialized.
static void generateFunctionSignature(const Operator &op, raw_ostream &os) {
  StringRef opClassName = op.getCppClassName();
  StringRef dialectNamespace = op.getDialect().getCppNamespace();
  std::string qualifiedClassName =
      dialectNamespace.str() + "::" + opClassName.str();
  os << "FailureOr<size_t> write" << opClassName << "( " << qualifiedClassName
     << " op, \n"
     << "                                   EncodingWriter &writer, \n"
     << "                                   TypeManager &typeMgr, \n"
     << "                                   ConstantManager &constMgr, \n"
     << "                                   StringManager &strMgr, \n"
     << "                                   const BytecodeWriterConfig "
        "&config) {\n";
}

/// Generates the flags field serialization for optional attributes and
/// operands. Version checking is only done for optional attributes and
/// operands.
///
/// The flags field is a varint that uses individual bits to encode the presence
/// of optional attributes and operands. The bit layout is version-ordered to
/// ensure backward compatibility:
///   - Bits are assigned in version order (earliest versions first)
///   - Within each version: attributes first, then operands (declaration order)
///   - This prevents bit layout shifts when new optional fields are added.
///
/// Special case: UnitAttr presence is ONLY encoded in the flags field.
/// No actual attribute data is written to the stream for UnitAttr.
static void generateFlagsFieldSerialization(const Operator &op,
                                            raw_ostream &os) {
  // Versionless ops set optional-field flag bits without any version checks.
  bool isVersionless = isVersionlessOperation(op);

  // Get version-ordered bit assignments and earliest optional field version.
  auto [bitAssignments, minOptionalVersion] =
      getVersionOrderedBitAssignments(op);
  if (bitAssignments.empty())
    return;

  std::string opVersion = extractVersionFromOperation(op);
  os << "  // Write flags field for optional attributes/operands.\n"
     << "  uint64_t flags = 0;\n";

  // Set flags bits for optional attributes and validate their versions.
  for (const auto &namedAttr : op.getAttributes()) {
    if (isExternallyEncodedOpAttribute(namedAttr.name)) {
      continue;
    }
    if (namedAttr.attr.isOptional()) {
      StringRef attrName = namedAttr.name;
      std::string getterName = op.getGetterName(attrName);
      std::string storageGetterName = getterName + "Attr";
      size_t bitPos = bitAssignments.lookup(attrName);

      if (isVersionless) {
        os << llvm::formatv(R"(
  auto flagsAttrValue_{0} = op.{1}();
  if (flagsAttrValue_{0}) flags |= (1ULL << {2});
)",
                            attrName, storageGetterName, bitPos);
      } else {
        auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
        std::string version = (Twine(majorStr) + "." + minorStr).str();

        if (version == opVersion) {
          // Attribute from original operation - simple flag setting.
          os << llvm::formatv(R"(
  auto flagsAttrValue_{0} = op.{1}();
  if (flagsAttrValue_{0}) flags |= (1ULL << {2});
)",
                              attrName, storageGetterName, bitPos);
        } else {
          // Versioned attribute - validate version compatibility.
          os << llvm::formatv(R"(
  auto flagsAttrValue_{0} = op.{1}();
  if (flagsAttrValue_{0}) {{
    auto flagsRequiredVersionFor_{0} = BytecodeVersion::fromVersion({2}, {3}, 0);
    assert(flagsRequiredVersionFor_{0} && "TableGen should guarantee valid versions");
    if (config.getBytecodeVersion() < *flagsRequiredVersionFor_{0}) {{
      op.emitError() << "optional attribute '{0}' is provided but requires bytecode version {4}, targeting " << config.getBytecodeVersion().toString();
      return failure();
    }
    // Attribute provided and compatible - set flag.
    flags |= (1ULL << {5});
  }
  // Attribute not provided - don't set flag.
)",
                              attrName, storageGetterName, majorStr, minorStr,
                              version, bitPos);
        }
      }
    }
  }

  // Set flags bits for optional operands and validate them.
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    for (const auto &[operandIndex, odsOperand] :
         llvm::enumerate(op.getOperands())) {
      if (!odsOperand.isOptional() && !isVersionless) {
        // Validate that required operands were introduced with the operation
        // itself.
        auto [majorStr, minorStr] = extractVersionFromOperand(operandIndex, op);
        std::string version = (Twine(majorStr) + "." + minorStr).str();
        if (version != opVersion)
          PrintFatalError("Required operand '" + odsOperand.name.str() +
                          "' in operation '" + op.getOperationName() +
                          "' was introduced after the operation.");
      } else if (odsOperand.isOptional()) {
        StringRef operandName = odsOperand.name;
        size_t bitPos = bitAssignments.lookup(operandName);

        os << llvm::formatv(R"(
  auto operandGroup_{0} = op.getODSOperands({0});
)",
                            operandIndex);

        if (isVersionless) {
          os << llvm::formatv(R"(
  if (!operandGroup_{0}.empty()) flags |= (1ULL << {1});
)",
                              operandIndex, bitPos);
        } else {
          auto [majorStr, minorStr] =
              extractVersionFromOperand(operandIndex, op);
          std::string version = (Twine(majorStr) + "." + minorStr).str();

          if (version == opVersion) {
            // Operand from original operation - no version checking needed.
            os << llvm::formatv(R"(
  if (!operandGroup_{0}.empty()) flags |= (1ULL << {1});
)",
                                operandIndex, bitPos);
          } else {
            // Versioned operand - validate version compatibility.
            os << llvm::formatv(R"(
  if (!operandGroup_{0}.empty()) {{
    auto requiredVersionFor_{1} = BytecodeVersion::fromVersion({2}, {3}, 0);
    assert(requiredVersionFor_{1} && "TableGen should guarantee valid versions");
    if (config.getBytecodeVersion() < *requiredVersionFor_{1}) {{
      op.emitError() << "optional operand '{1}' is provided but requires bytecode version {4}, targeting " << config.getBytecodeVersion().toString();
      return failure();
    }
    // Operand provided and compatible - set flag.
    flags |= (1ULL << {5});
  }
  // Operand not provided - don't set flag.
)",
                                operandIndex, operandName, majorStr, minorStr,
                                version, bitPos);
          }
        }
      }
    }
  }

  // Backward Compatibility: Only generate version check if the first optional
  // field was added AFTER the operation's baseline version. This allows newer
  // writers (e.g., 13.2) to target older bytecode formats (e.g., 13.1 via
  // --bytecode-version=13.1) for compatibility with older readers. If optional
  // fields existed from the operation's baseline, flags field is always
  // written.
  std::string minOptionalVersionStr =
      minOptionalVersion
          ? (minOptionalVersion->first + "." + minOptionalVersion->second)
          : "";
  bool needsVersionCheck =
      minOptionalVersion && (minOptionalVersionStr != opVersion);

  if (needsVersionCheck) {
    auto [majorStr, minorStr] = *minOptionalVersion;
    os << llvm::formatv(
        R"(  // Only write flags if targeting version >= {0}.{1} (first optional field version)
  auto requiredVersionForFlags = BytecodeVersion::fromVersion({0}, {1}, 0);
  assert(requiredVersionForFlags && "TableGen should guarantee valid versions");
  if (config.getBytecodeVersion() >= *requiredVersionForFlags) {{
    writer.writeVarInt(flags);
  }

)",
        majorStr, minorStr);
  } else {
    // Flags field always exists for this operation.
    os << "  writer.writeVarInt(flags);\n\n";
  }
}

/// Helper function to generate common attribute serialization logic.
static void generateAttributeSerializationLogic(raw_ostream &os,
                                                StringRef getterName,
                                                StringRef attrName,
                                                StringRef indent = "  ") {
  os << llvm::formatv(R"({0}auto nativeAttrValue_{1} = op.{2}();
{0}if (failed(writeOpAttribute(op.getOperation(), "{1}", nativeAttrValue_{1}, writer, typeMgr, constMgr, strMgr)))
{0}  return failure();
)",
                      indent, attrName, getterName);
}

/// Helper function to generate common operand serialization logic.
static void generateOperandSerializationLogic(raw_ostream &os, unsigned index,
                                              bool isOptional,
                                              bool isVariadic) {
  if (isOptional) {
    os << llvm::formatv(R"(
  if (!operandGroup_{0}.empty())
    writeOperands(operandGroup_{0}, writer, /*encodeSize=*/false);
)",
                        index);
  } else {
    os << llvm::formatv("  writeOperands(op.getODSOperands({0}), writer, "
                        "/*encodeSize=*/{1});\n",
                        index, isVariadic ? "true" : "false");
  }
}

/// Generates C++ code within the 'write<OpName>' function to serialize the
/// attributes of the given operation by calling the writeOpAttribute helper.
static void generateAttributeSerialization(const Operator &op,
                                           raw_ostream &os) {
  if (op.getNumAttributes() == 0)
    return;
  os << "  // Serialize Attributes.\n";
  for (const auto &namedAttr : op.getAttributes()) {
    StringRef attrName = namedAttr.name;
    if (isExternallyEncodedOpAttribute(attrName)) {
      continue;
    }
    std::string getterName = op.getGetterName(attrName);
    bool isOptional = namedAttr.attr.isOptional();
    bool isUnitAttr =
        StringRef(namedAttr.attr.getStorageType()).contains("UnitAttr");
    if (isUnitAttr) {
      // UnitAttr: only flags field, no serialization needed.
      continue;
    } else if (isOptional && namedAttr.attr.hasDefaultValue()) {
      // The flags field records exact storage presence. Use the raw attribute
      // getter here too: a DefaultValuedOptionalAttr's native getter returns
      // its default even when the attribute is absent.
      os << llvm::formatv(R"(
  if (auto nativeAttrValue_{0} = op.{1}Attr()) {{
    if (failed(writeOpAttribute(op.getOperation(), "{0}", nativeAttrValue_{0}, writer, typeMgr, constMgr, strMgr)))
      return failure();
  }
)",
                          attrName, getterName);
    } else if (isOptional) {
      // Ordinary optional native getters return std::optional<T>, which both
      // preserves absence and converts attribute storage to the expected
      // native wire representation.
      generateAttributeSerializationLogic(os, getterName, attrName);
    } else {
      // Required attributes: need version checking and default value
      // validation.
      if (isVersionlessOperation(op)) {
        os << "  // Versionless operations: serialize directly without version "
              "checks.\n";
        generateAttributeSerializationLogic(os, getterName, attrName);
      } else {
        auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
        auto defaultValue = extractDefaultValue(namedAttr);
        auto encodingRequired =
            getAttributeEncodingRequiredPredicate(namedAttr, op);
        std::string version = (Twine(majorStr) + "." + minorStr).str();

        os << llvm::formatv(R"(
  auto requiredVersionFor_{0} = BytecodeVersion::fromVersion({1}, {2}, 0);
  assert(requiredVersionFor_{0} && "TableGen should guarantee valid versions");
  if (config.getBytecodeVersion() >= *requiredVersionFor_{0}) {{
)",
                            attrName, majorStr, minorStr);
        generateAttributeSerializationLogic(os, getterName, attrName, "    ");
        os << "  } else {\n";
        if (defaultValue.has_value()) {
          os << llvm::formatv(R"(
    // Check that attribute equals default value for older versions.
    if ({0}) {{
      op.emitError() << "attribute '{1}' requires bytecode version {2}+, but targeting " << config.getBytecodeVersion().toString();
      return failure();
    }
)",
                              *encodingRequired, attrName, version);
        } else if (encodingRequired.has_value()) {
          // Implicit default is all-`false` (the value the reader fixup
          // resynthesizes for legacy bytecode), so downgrade is lossless
          // iff the runtime value is itself all-`false`. Otherwise error.
          os << llvm::formatv(R"(
    if ({0}) {{
      op.emitError() << "operation requires bytecode version {1}+ because it carries a non-default '{2}' attribute (the legacy reader can only reconstruct an all-`false` vector), but targeting " << config.getBytecodeVersion().toString();
      return failure();
    }
)",
                              *encodingRequired, version, attrName);
        } else {
          // No default value available.
          std::string opVersion = extractVersionFromOperation(op);
          if (version != opVersion) {
            // Required attributes introduced after the operation must have
            // default value.
            PrintFatalError(
                "Versioned attribute '" + namedAttr.name + "' in operation '" +
                op.getOperationName() + "' (since " + version +
                ") was introduced after the operation itself (since " +
                opVersion +
                ") and must have a default value for backward compatibility");
          }
          // Note: Attributes introduced with the operation itself don't need
          // defaults.
        }
        os << "  }\n";
      }
    }
  }
  os << "\n";
}

/// Generates C++ code within the 'write<OpName>' function to serialize the
/// operands of the given operation.
static void generateOperandSerialization(const Operator &op, raw_ostream &os) {
  if (op.getNumOperands() == 0)
    return;

  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    os << "  // Serialize Operands (AttrSizedOperandSegments) - version "
          "validation done by flags field.\n";
    for (const auto &[index, odsOperand] : llvm::enumerate(op.getOperands()))
      generateOperandSerializationLogic(os, index, odsOperand.isOptional(),
                                        odsOperand.isVariadic());
  } else {
    bool opHasOptionalOperands =
        llvm::any_of(op.getOperands(),
                     [](const auto &operand) { return operand.isOptional(); });
    bool opHasVariadicOperands =
        llvm::any_of(op.getOperands(),
                     [](const auto &operand) { return operand.isVariadic(); });
    bool encodeSize = opHasVariadicOperands || opHasOptionalOperands;
    os << llvm::formatv(
        "  writeOperands(op->getOperands(), writer, /*encodeSize=*/{0});\n",
        encodeSize ? "true" : "false");
  }
  os << "\n";
}

/// Generates C++ code within the 'write<OpName>' function to serialize the
/// regions of the given operation, if it has any.
static void generateRegionSerialization(const Operator &op, raw_ostream &os) {
  // Only emit region code if this op can have regions
  if (!op.getNumRegions())
    return;

  os << "  // Serialize Regions\n"
     << "  writer.writeVarInt(op->getNumRegions());\n"
     << "  for (Region &region : op->getRegions()) {\n"
     << "    if (failed(writeRegion(region, writer)))\n"
     << "      return failure();\n"
     << "  }\n\n";
}

/// Generate result serialization without version checking.
static void generateSimpleResultSerialization(const Operator &op,
                                              raw_ostream &os) {
  // Track how many results are serialized.
  os << "  size_t numSerializedResults = op->getNumResults();\n";

  // If the op has variadic results, write the actual number of results.
  if (op.isVariadic())
    os << "  writer.writeVarInt(op->getNumResults());\n";

  // Write the result types of the operation.
  os << "  if (failed(writeResultTypes(op, writer, typeMgr)))\n"
     << "    return failure();\n\n";
}

/// Generate version-aware result serialization.
static void generateVersionAwareResultSerialization(const Operator &op,
                                                    raw_ostream &os) {
  os << "  // Public operations: version checking for results.\n";

  std::string opVersion = extractVersionFromOperation(op);

  // Single analysis pass - collect version info for all results.
  SmallVector<ResultVersionInfo> versionInfos;
  bool hasVersionedResults = false;

  for (int i = 0; i < op.getNumResults(); ++i) {
    ResultVersionInfo info(i, op.getResult(i), op, opVersion);
    if (info.requiresVersionCheck)
      hasVersionedResults = true;
    versionInfos.push_back(std::move(info));
  }

  if (!hasVersionedResults) {
    // All results from original operation - use simple serialization.
    generateSimpleResultSerialization(op, os);
    return;
  }

  // Usage validation, counting, and type collection in single phase.
  os << llvm::formatv(R"(
  uint64_t compatibleResults = 0;
  SmallVector<Type> compatibleResultTypes;
)");

  for (int i = 0; i < op.getNumResults(); ++i) {
    const auto &info = versionInfos[i];

    if (!info.requiresVersionCheck) {
      // Original result always compatible.
      os << llvm::formatv(R"(
  ++compatibleResults;
  compatibleResultTypes.push_back(op->getResult({0}).getType());
)",
                          i);
    } else {
      std::string version = info.majorStr + "." + info.minorStr;
      os << llvm::formatv(R"(
  auto requiredVersionFor_result_{0} = BytecodeVersion::fromVersion({1}, {2}, 0);
  assert(requiredVersionFor_result_{0} && "TableGen should guarantee valid versions");
  if (config.getBytecodeVersion() >= *requiredVersionFor_result_{0}) {{
    ++compatibleResults;
    compatibleResultTypes.push_back(op->getResult({3}).getType());
  } else {{
    Value resultValue = op->getResult({3});
    if (!resultValue.getUsers().empty()) {{
      op.emitError() << "result '{0}' requires bytecode version {4} but is being used and targeting "
                     << config.getBytecodeVersion().toString()
                     << ". Cannot serialize to older bytecode version when newer features are used.";
      return failure();
    }
  }
)",
                          info.name, info.majorStr, info.minorStr, i, version);
    }
  }

  if (op.isVariadic())
    os << "  writer.writeVarInt(compatibleResults);\n";

  // Write compatible result types.
  os << "  if (failed(writeResultTypes(compatibleResultTypes, writer, "
        "typeMgr)))\n"
     << "    return failure();\n";

  // Track number of serialized results for valueIndexMap updates.
  os << "  size_t numSerializedResults = compatibleResults;\n";

  os << "\n";
}

/// Generates C++ code within the 'write<OpName>' function to serialize the
/// result types of the given operation.
static void generateResultTypeSerialization(const Operator &op,
                                            raw_ostream &os) {
  // Check for unsupported AttrSizedResultSegments trait.
  if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments")) {
    os << "  op.emitError(\"operation '" << op.getOperationName()
       << "' has AttrSizedResultSegments, which is not supported by the "
          "bytecode writer.\");\n";
    os << "  return failure();\n";
  }

  // Versionless ops serialize result types without version conditioning.
  if (isVersionlessOperation(op))
    return generateSimpleResultSerialization(op, os);
  generateVersionAwareResultSerialization(op, os);
}

/// Generates the complete C++ function 'write<OpName>'.
static void generateOpWriter(const Operator &op, raw_ostream &os) {
  std::string opName = op.getOperationName();
  os << "// Writer for Op: " << opName << "\n";
  generateFunctionSignature(op, os);
  generateResultTypeSerialization(op, os);
  generateFlagsFieldSerialization(op, os);
  generateAttributeSerialization(op, os);
  generateOperandSerialization(op, os);
  generateRegionSerialization(op, os);
  // Return the number of serialized results for valueIndexMap updates.
  os << "  return numSerializedResults;\n"
     << "}\n\n";
}

/// Generates the implementations of the individual op writer functions.
static void generateOpWriterImplementations(const RecordKeeper &records,
                                            raw_ostream &os) {

  emitSourceFileHeader("Generated Bytecode Writers", os);
  auto opDefs = records.getAllDerivedDefinitions("Op");
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n"
     << "// Writer Functions\n"
     << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";
  for (const Record *opDef : opDefs)
    generateOpWriter(Operator(opDef), os);
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n"
     << "// End of generated functions.\n"
     << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
}

/// Generates the TypeSwitch statement for dispatching to op-specific writers.
/// Returns FailureOr<size_t> where size_t is the number of serialized results.
static void generateDispatchSwitch(const RecordKeeper &records,
                                   raw_ostream &os) {

  emitSourceFileHeader("Generated Bytecode Dispatch Switch", os);
  auto opDefs = records.getAllDerivedDefinitions("Op");
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n"
     << "// Dispatch Switch\n"
     << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";

  os << "return TypeSwitch<Operation *, FailureOr<size_t>>(op)\n";
  for (const Record *opDef : opDefs) {
    Operator op(opDef);
    StringRef opClassName = op.getCppClassName();
    StringRef dialectNamespace = op.getDialect().getCppNamespace();
    std::string qualifiedClassName =
        dialectNamespace.str() + "::" + opClassName.str();
    os << "                   .Case<" << qualifiedClassName
       << ">([&](auto concreteOp) {\n"
       << "                     return write" << opClassName
       << "(concreteOp, writer, typeMgr, constMgr, strMgr, config);\n"
       << "                   })\n";
  }
  os << "                   .Default([&](Operation *) -> FailureOr<size_t> {\n"
     << "                     return op->emitError(\n"
     << "                         \"unhandled operation type in bytecode "
        "writer\");\n"
     << "                   });\n\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n"
     << "// End of generated dispatch switch.\n"
     << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
}

/// Generate version constants based on actual opcode assignments
static void generateVersionConstants(const RecordKeeper &records,
                                     raw_ostream &os) {
  emitSourceFileHeader("Generated Version Constants", os);

  auto opcodeRecords = records.getAllDerivedDefinitions("BytecodeOpcode");

  // Track max opcode per version.
  llvm::DenseMap<std::pair<uint8_t, uint8_t>, uint32_t> versionToMaxOpcode;

  for (const Record *record : opcodeRecords) {
    unsigned opcode = record->getValueAsInt("opcodeValue");

    if (record->isSubClassOf("PublicOpcode")) {
      // Extract version from the operation definition.
      const Record *opRecord = record->getValueAsDef("operation");

      // Parse version string from operation definition (e.g., "13.1" -> {13,
      // 1})
      StringRef versionStr = opRecord->getValueAsString("operationVersion");
      auto dotPos = versionStr.find('.');
      if (dotPos == StringRef::npos) {
        PrintFatalError(record->getLoc(),
                        "operation version must be in format 'major.minor'");
      }

      unsigned majorVer, minorVer;
      if (versionStr.substr(0, dotPos).getAsInteger(10, majorVer) ||
          versionStr.substr(dotPos + 1).getAsInteger(10, minorVer)) {
        PrintFatalError(
            record->getLoc(),
            "invalid version format, expected 'major.minor' like '13.1'");
      }

      // Store opcode for its minimum version.
      auto versionKey = std::make_pair(uint8_t(majorVer), uint8_t(minorVer));
      versionToMaxOpcode[versionKey] =
          std::max(versionToMaxOpcode[versionKey], opcode);
    }
  }

  // Apply forward compatibility.
  auto versionRecords = records.getAllDerivedDefinitions("SupportedVersion");

  std::vector<std::pair<uint8_t, uint8_t>> knownVersions;
  for (const Record *record : versionRecords) {
    unsigned major = record->getValueAsInt("majorVersion");
    unsigned minor = record->getValueAsInt("minorVersion");
    knownVersions.emplace_back(uint8_t(major), uint8_t(minor));
  }

  std::sort(knownVersions.begin(), knownVersions.end());

  uint32_t prevMaxOpcode = 0;
  for (auto version : knownVersions) {
    uint32_t &maxOpcode = versionToMaxOpcode[version];
    if (maxOpcode == 0)
      maxOpcode = prevMaxOpcode;
    prevMaxOpcode = maxOpcode;
  }

  os << "namespace mlir {\nnamespace cuda_tile {\n\n";

  // Generate version-to-max-opcode map accessor function
  os << "// Auto-generated version to max opcode mapping\n";
  os << "static const llvm::DenseMap<std::pair<uint8_t, uint8_t>, uint32_t> "
        "&getVersionToMaxOpcodeMap() {\n";
  os << "  static const llvm::DenseMap<std::pair<uint8_t, uint8_t>, uint32_t> "
        "map = []() {\n";
  os << "    llvm::DenseMap<std::pair<uint8_t, uint8_t>, uint32_t> m;\n";
  for (const auto &[versionPair, maxOpcode] : versionToMaxOpcode) {
    os << "    m[{" << static_cast<int>(versionPair.first) << ", "
       << static_cast<int>(versionPair.second) << "}] = 0x"
       << llvm::format("%X", maxOpcode) << ";\n";
  }
  os << "    return m;\n";
  os << "  }();\n";
  os << "  return map;\n";
  os << "}\n\n";

  // Public-preview opcode range (available in all builds). Preview opcodes are
  // gated at runtime by their named preview feature and are only valid at the
  // exact current bytecode version.
  if (auto previewRange = getOpcodeRange(records, "PreviewOpcodeRange")) {
    os << "static constexpr uint32_t kPreviewOpcodesMin = 0x"
       << llvm::format("%X", previewRange->first) << ";\n";
    os << "static constexpr uint32_t kPreviewOpcodesMax = 0x"
       << llvm::format("%X", previewRange->second) << ";\n\n";
  }

  os << "} // namespace cuda_tile\n} // namespace mlir\n";
}

/// Generate version validation function from SupportedVersion records.
static void generateVersionValidation(const RecordKeeper &records,
                                      raw_ostream &os) {
  emitSourceFileHeader("Generated Version Validation", os);
  auto versionRecords = records.getAllDerivedDefinitions("SupportedVersion");
  // Group versions by major version.
  std::map<uint8_t, std::vector<uint8_t>> versionMap;

  for (const Record *record : versionRecords) {
    unsigned major = record->getValueAsInt("majorVersion");
    unsigned minor = record->getValueAsInt("minorVersion");
    versionMap[uint8_t(major)].push_back(uint8_t(minor));
  }

  os << "// Auto-generated version validation from SupportedVersion records\n";

  for (const auto &[major, minors] : versionMap) {
    bool isTestingVersion = (major == 250);

    if (isTestingVersion) {
      os << "#ifdef TILE_IR_INCLUDE_TESTS\n";
      os << "  // Testing versions - only available when TILE_IR_INCLUDE_TESTS "
            "is defined.\n";
    }

    os << "  if (verMajor == " << static_cast<int>(major) << ") {\n";

    auto minIt = std::min_element(minors.begin(), minors.end());
    auto maxIt = std::max_element(minors.begin(), minors.end());

    if (*minIt == *maxIt)
      os << "    if (verMinor == " << static_cast<int>(*minIt) << ")\n";
    else if (*minIt == 0)
      os << "    if (verMinor <= " << static_cast<int>(*maxIt) << ")\n";
    else
      os << "    if (verMinor >= " << static_cast<int>(*minIt)
         << " && verMinor <= " << static_cast<int>(*maxIt) << ")\n";

    os << "      return BytecodeVersion(verMajor, verMinor, verTag);\n";
    os << "  }\n";

    if (isTestingVersion)
      os << "#endif // TILE_IR_INCLUDE_TESTS\n";
  }

  os << "  return std::nullopt;\n";
}

/// Generate getSupportedVersions() function from SupportedVersion records.
static void generateSupportedVersionsList(const RecordKeeper &records,
                                          raw_ostream &os) {
  emitSourceFileHeader("Generated Supported Versions List", os);
  auto versionRecords = records.getAllDerivedDefinitions("SupportedVersion");

  // Collect and sort all versions.
  std::vector<std::pair<uint8_t, uint8_t>> versions;
  for (const Record *record : versionRecords) {
    unsigned major = record->getValueAsInt("majorVersion");
    unsigned minor = record->getValueAsInt("minorVersion");
    versions.emplace_back(uint8_t(major), uint8_t(minor));
  }
  llvm::sort(versions);

  std::string versionList;
  llvm::raw_string_ostream vos(versionList);
  llvm::interleaveComma(versions, vos, [&](const auto &v) {
    vos << llvm::formatv("*BytecodeVersion::fromVersion({0}, {1}, 0)",
                         static_cast<int>(v.first), static_cast<int>(v.second));
  });

  os << llvm::formatv(R"(// Auto-generated list of supported bytecode versions.
llvm::SmallVector<BytecodeVersion> mlir::cuda_tile::getSupportedVersions() {{
  return {{{0}};
}
)",
                      versionList);
}

//===----------------------------------------------------------------------===//
// Per-Operation Version Checking Generation
//===----------------------------------------------------------------------===//

/// Generates a per-operation version checking function.
/// This generates code like:
///   static BytecodeVersion getMinVersionFor_ForOp(
///       cuda_tile::ForOp op,
///       function_ref<BytecodeVersion(Type)> getTypeVersion,
///       function_ref<BytecodeVersion(Attribute)> getAttrVersion) { ... }
static void generateOpVersionChecker(const Operator &op, raw_ostream &os) {
  StringRef opClassName = op.getCppClassName();
  StringRef dialectNamespace = op.getDialect().getCppNamespace();
  std::string qualifiedClassName =
      dialectNamespace.str() + "::" + opClassName.str();

  std::string opVersion = extractVersionFromOperation(op);
  auto [opMajor, opMinor] = parseVersion(opVersion);

  os << "// Version checker for: " << op.getOperationName() << "\n";
  os << "static mlir::cuda_tile::BytecodeVersion getMinVersionFor_"
     << opClassName << "(\n";
  os << "    " << qualifiedClassName << " op,\n";
  os << "    llvm::function_ref<mlir::cuda_tile::BytecodeVersion(mlir::Type)> "
        "getTypeVersion,\n";
  os << "    "
        "llvm::function_ref<mlir::cuda_tile::BytecodeVersion(mlir::Attribute)> "
        "getAttrVersion) {\n";

  // Start with the operation's base version.
  os << "  mlir::cuda_tile::BytecodeVersion maxVersion = "
        "*mlir::cuda_tile::BytecodeVersion::fromVersion("
     << opMajor << ", " << opMinor << ", 0);\n";
  os << "  [[maybe_unused]] auto updateMax = "
        "[&](mlir::cuda_tile::BytecodeVersion v) "
        "{ maxVersion = std::max(maxVersion, v); };\n\n";

  // Check all operands: version bump if added after op, plus type check.
  if (op.getNumOperands() > 0) {
    os << "  // Check operands.\n";
    for (unsigned i = 0; i < static_cast<unsigned>(op.getNumOperands()); ++i) {
      auto [operandMajor, operandMinor] = extractVersionFromOperand(i, op);
      bool isVersioned =
          isVersionGreater(operandMajor, operandMinor, opMajor, opMinor);

      const auto &operand = op.getOperand(i);
      std::string getterName = op.getGetterName(operand.name);

      if (operand.isOptional()) {
        // Optional operand: check if present, bump version if versioned, check
        // type.
        os << "  if (auto operand = op." << getterName << "()) {\n";
        if (isVersioned) {
          os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
             << operandMajor << ", " << operandMinor << ", 0));\n";
        }
        os << "    updateMax(getTypeVersion(operand.getType()));\n";
        os << "  }\n";
      } else if (operand.isVariadic()) {
        // Variadic operand: bump version if versioned and non-empty, check all
        // types.
        if (isVersioned) {
          os << "  if (!op." << getterName << "().empty())\n";
          os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
             << operandMajor << ", " << operandMinor << ", 0));\n";
        }
        os << "  for (auto operand : op." << getterName << "())\n";
        os << "    updateMax(getTypeVersion(operand.getType()));\n";
      } else {
        // Required operand: bump version if versioned, always check type.
        if (isVersioned) {
          os << "  updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
             << operandMajor << ", " << operandMinor << ", 0));\n";
        }
        os << "  updateMax(getTypeVersion(op." << getterName
           << "().getType()));\n";
      }
    }
  }

  // Check all results: version bump if added after op (and used), check type.
  if (op.getNumResults() > 0) {
    os << "\n  // Check results.\n";
    for (unsigned i = 0; i < static_cast<unsigned>(op.getNumResults()); ++i) {
      auto [resultMajor, resultMinor] = extractVersionFromResult(i, op);
      bool isVersioned =
          isVersionGreater(resultMajor, resultMinor, opMajor, opMinor);

      const auto &result = op.getResult(i);
      std::string getterName = op.getGetterName(result.name);

      if (result.isVariadic()) {
        // Variadic result: bump version if versioned and non-empty, check all
        // types.
        if (isVersioned) {
          os << "  if (!op." << getterName << "().empty())\n";
          os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
             << resultMajor << ", " << resultMinor << ", 0));\n";
        }
        os << "  for (auto result : op." << getterName << "())\n";
        os << "    updateMax(getTypeVersion(result.getType()));\n";
      } else if (isVersioned) {
        // Versioned non-variadic result: only bump and check type if used.
        os << "  if (!op." << getterName << "().use_empty()) {\n";
        os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
           << resultMajor << ", " << resultMinor << ", 0));\n";
        os << "    updateMax(getTypeVersion(op." << getterName
           << "().getType()));\n";
        os << "  }\n";
      } else {
        // Base version non-variadic result: always check type.
        os << "  updateMax(getTypeVersion(op." << getterName
           << "().getType()));\n";
      }
    }
  }

  // Check all attributes: version bump if added after op, check attr version.
  // Track if we've emitted any attr checks (for section comment).
  bool hasAttrChecks = false;
  for (const auto &namedAttr : op.getAttributes()) {
    if (isExternallyEncodedOpAttribute(namedAttr.name)) {
      continue;
    }
    bool isUnitAttr =
        StringRef(namedAttr.attr.getStorageType()).contains("UnitAttr");

    auto [attrMajor, attrMinor] = extractVersionFromAttribute(namedAttr, op);
    bool isVersioned = isVersionGreater(attrMajor, attrMinor, opMajor, opMinor);

    // Base version UnitAttr: skip (no enum value, no version bump needed).
    if (isUnitAttr && !isVersioned) {
      continue;
    }

    // Emit section comment once, only if we have attr checks to emit.
    if (!hasAttrChecks) {
      os << "\n  // Check attributes.\n";
      hasAttrChecks = true;
    }

    std::string getterName = op.getGetterName(namedAttr.name);

    if (isUnitAttr) {
      // Versioned UnitAttr: presence indicates newer version required.
      os << "  if (op." << getterName << "())\n";
      os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
         << attrMajor << ", " << attrMinor << ", 0));\n";
    } else if (namedAttr.attr.isOptional()) {
      // Optional attr: check if present, bump version if versioned, check attr
      // version.
      os << "  if (auto attr = op." << getterName << "Attr()) {\n";
      if (isVersioned) {
        os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
           << attrMajor << ", " << attrMinor << ", 0));\n";
      }
      os << "    updateMax(getAttrVersion(attr));\n";
      os << "  }\n";
    } else {
      // Required attr: bump version if versioned, always check attr version.
      if (isVersioned) {
        // The writer drops an attribute introduced after its operation when the
        // value matches the fallback an older reader reconstructs, so it only
        // constrains the module when it would have to be encoded.
        auto encodingRequired =
            getAttributeEncodingRequiredPredicate(namedAttr, op);
        if (!encodingRequired) {
          PrintFatalError(
              op.getLoc(),
              "attribute '" + namedAttr.name + "' in operation '" +
                  op.getOperationName() +
                  "' was introduced after the operation itself and has no "
                  "default value, so its minimum version cannot be computed");
        }
        os << "  if (" << *encodingRequired << ") {\n";
        os << "    updateMax(*mlir::cuda_tile::BytecodeVersion::fromVersion("
           << attrMajor << ", " << attrMinor << ", 0));\n";
        os << "    updateMax(getAttrVersion(op." << getterName << "Attr()));\n";
        os << "  }\n";
      } else {
        os << "  updateMax(getAttrVersion(op." << getterName << "Attr()));\n";
      }
    }
  }

  // Check block argument types in regions.
  if (op.getNumRegions() > 0) {
    os << "\n  // Check block argument types.\n";
    os << "  for (auto &region : op->getRegions())\n";
    os << "    for (auto &block : region)\n";
    os << "      for (auto arg : block.getArguments())\n";
    os << "        updateMax(getTypeVersion(arg.getType()));\n";
  }

  os << "  return maxVersion;\n";
  os << "}\n\n";
}

/// Generates implementations of all per-operation version checking functions.
static void generateOpVersionCheckerImplementations(const RecordKeeper &records,
                                                    raw_ostream &os) {
  emitSourceFileHeader("Generated Operation Version Checkers", os);
  auto opDefs = records.getAllDerivedDefinitions("Op");

  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  os << "// Per-Operation Version Checking Functions\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";

  // Checkers compare against the default spelled in ODS, and those expressions
  // are relative to the dialect namespace (`RoundingMode::FULL`), so emit the
  // checkers there.
  os << "namespace mlir::cuda_tile {\n\n";

  for (const Record *opDef : opDefs) {
    Operator op(opDef);
    // Versionless ops have no version checker: their min version is handled
    // out-of-band (preview forces the current version).
    if (isVersionlessOperation(op)) {
      continue;
    }
    generateOpVersionChecker(op, os);
  }

  os << "} // namespace mlir::cuda_tile\n\n";

  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  os << "// End of generated version checking functions.\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
}

/// Generates the TypeSwitch dispatch for per-operation version checking.
static void generateVersionCheckDispatch(const RecordKeeper &records,
                                         raw_ostream &os) {
  emitSourceFileHeader("Generated Version Check Dispatch", os);
  auto opDefs = records.getAllDerivedDefinitions("Op");

  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  os << "// Version Check Dispatch\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";

  os << "BytecodeVersion opVersion = "
        "llvm::TypeSwitch<Operation *, BytecodeVersion>(op)\n";
  for (const Record *opDef : opDefs) {
    Operator op(opDef);
    StringRef opClassName = op.getCppClassName();
    StringRef dialectNamespace = op.getDialect().getCppNamespace();
    std::string qualifiedClassName =
        dialectNamespace.str() + "::" + opClassName.str();

    if (isVersionlessOperation(op)) {
      // Versionless ops carry no version metadata. Public-preview ops are still
      // version-locked to the current version, so emit a case reporting that.
      // Other versionless ops keep the default minimum.
      if (isPreviewOperation(op)) {
        os << "    .Case<" << qualifiedClassName << ">([&](auto) {\n"
           << "      return mlir::cuda_tile::BytecodeVersion::kCurrentVersion;\n"
           << "    })\n";
      }
      continue;
    }

    os << "    .Case<" << qualifiedClassName << ">([&](auto concreteOp) {\n"
       << "      return getMinVersionFor_" << opClassName
       << "(concreteOp, getTypeVersion, getAttrVersion);\n"
       << "    })\n";
  }
  os << "    .Default([&](mlir::Operation *) {\n"
     << "      return mlir::cuda_tile::BytecodeVersion::kMinSupportedVersion;\n"
     << "    });\n\n";

  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  os << "// End of generated dispatch.\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
}

/// Generates runtime classification of public-preview operations by feature.
static void generatePreviewOpClassification(const RecordKeeper &records,
                                            raw_ostream &os) {
  emitSourceFileHeader("Generated Preview Operation Classification", os);
  auto opDefs = records.getAllDerivedDefinitions("CudaTilePreviewOpDef");

  os << R"(
namespace mlir::cuda_tile::detail {

inline std::optional<PreviewFeature> getPreviewFeature(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<PreviewFeature>>(op)
)";
  for (const Record *opDef : opDefs) {
    Operator op(opDef);
    os << "      .Case<" << op.getDialect().getCppNamespace()
       << "::" << op.getCppClassName() << ">([](auto) { return PreviewFeature::"
       << getOwningPreviewFeatureName(opDef) << "; })\n";
  }
  os << R"(      .Default([](Operation *) { return std::nullopt; });
}

} // namespace mlir::cuda_tile::detail
)";
}

static SmallVector<const Record *>
getPreviewFeatureRecords(const RecordKeeper &records) {
  SmallVector<const Record *> features(
      records.getAllDerivedDefinitions("CudaTilePreviewFeature"));
  llvm::sort(features, [](const Record *lhs, const Record *rhs) {
    return lhs->getValueAsInt("value") < rhs->getValueAsInt("value");
  });

  llvm::DenseSet<int64_t> values;
  llvm::StringSet<> cppNames;
  llvm::StringSet<> spellings;
  for (const Record *feature : features) {
    int64_t value = feature->getValueAsInt("value");
    StringRef cppName = feature->getValueAsString("cppName");
    StringRef spelling = feature->getValueAsString("spelling");
    if (!values.insert(value).second) {
      PrintFatalError(feature->getLoc(),
                      "duplicate preview feature numeric value");
    }
    if (!cppNames.insert(cppName).second) {
      PrintFatalError(feature->getLoc(), "duplicate preview feature C++ name");
    }
    if (!spellings.insert(spelling).second) {
      PrintFatalError(feature->getLoc(),
                      "duplicate preview feature command-line spelling");
    }
  }
  return features;
}

static void
generatePreviewFeatureDeclarations(ArrayRef<const Record *> features,
                                   raw_ostream &os) {
  os << "enum class PreviewFeature {\n";
  for (const Record *feature : features) {
    os << "  " << feature->getValueAsString("cppName") << " = "
       << feature->getValueAsInt("value") << ",\n";
  }
  os << R"(};

/// Return the permanent command-line spelling for a preview feature.
llvm::StringRef stringifyPreviewFeature(PreviewFeature feature);

/// Parse any known command-line spelling, including retired features.
std::optional<PreviewFeature> symbolizePreviewFeature(llvm::StringRef name);

/// Return whether a known preview feature has been retired.
bool isRetiredPreviewFeature(PreviewFeature feature);

/// Return active feature names accepted by this build.
llvm::ArrayRef<PreviewFeature> getSupportedPreviewFeatures();

/// Return every feature known by this build, including retired features.
llvm::ArrayRef<PreviewFeature> getAllPreviewFeatures();
)";
}

static void generatePreviewFeatureDefinitions(ArrayRef<const Record *> features,
                                              raw_ostream &os) {
  os << R"(llvm::StringRef
mlir::cuda_tile::stringifyPreviewFeature(PreviewFeature feature) {
  switch (feature) {
)";
  for (const Record *feature : features) {
    os << "  case PreviewFeature::" << feature->getValueAsString("cppName")
       << ":\n"
       << "    return \"" << feature->getValueAsString("spelling") << "\";\n";
  }
  os << R"(  }
  llvm_unreachable("unknown preview feature");
}

std::optional<PreviewFeature>
mlir::cuda_tile::symbolizePreviewFeature(llvm::StringRef name) {
)";
  for (const Record *feature : features) {
    os << "  if (name == \"" << feature->getValueAsString("spelling") << "\")\n"
       << "    return PreviewFeature::" << feature->getValueAsString("cppName")
       << ";\n";
  }
  os << R"(  return std::nullopt;
}

bool mlir::cuda_tile::isRetiredPreviewFeature(PreviewFeature feature) {
  switch (feature) {
)";
  for (const Record *feature : features) {
    os << "  case PreviewFeature::" << feature->getValueAsString("cppName")
       << ":\n"
       << "    return "
       << (feature->getValueAsBit("retired") ? "true" : "false") << ";\n";
  }
  os << R"(  }
  llvm_unreachable("unknown preview feature");
}

llvm::ArrayRef<PreviewFeature>
mlir::cuda_tile::getSupportedPreviewFeatures() {
  static constexpr PreviewFeature features[] = {
)";
  for (const Record *feature : features) {
    if (!feature->getValueAsBit("retired")) {
      os << "      PreviewFeature::" << feature->getValueAsString("cppName")
         << ",\n";
    }
  }
  os << R"(  };
  return features;
}

llvm::ArrayRef<PreviewFeature>
mlir::cuda_tile::getAllPreviewFeatures() {
  static constexpr PreviewFeature features[] = {
)";
  for (const Record *feature : features) {
    os << "      PreviewFeature::" << feature->getValueAsString("cppName")
       << ",\n";
  }
  os << R"(  };
  return features;
}
)";
}

static bool generatePreviewFeatures(const RecordKeeper &records,
                                    raw_ostream &os) {
  SmallVector<const Record *> features = getPreviewFeatureRecords(records);
  auto typeStructure = analyzeBytecodeTypes(records);
  if (failed(typeStructure)) {
    PrintFatalError("Failed to analyze preview feature types");
    return true;
  }
  BytecodeAttrStructure attrStructure = analyzeBytecodeAttrs(records);

  os << "//===-- Begin Preview Feature Declarations --===//\n";
  os << "#ifdef GEN_PREVIEW_FEATURE_DECLS\n\n";
  generatePreviewFeatureDeclarations(features, os);
  os << "\n#undef GEN_PREVIEW_FEATURE_DECLS\n";
  os << "#endif // GEN_PREVIEW_FEATURE_DECLS\n";
  os << "//===-- End Preview Feature Declarations --===//\n\n";

  os << "//===-- Begin Preview Feature Definitions --===//\n";
  os << "#ifdef GEN_PREVIEW_FEATURE_DEFS\n\n";
  generatePreviewFeatureDefinitions(features, os);
  os << "\n#undef GEN_PREVIEW_FEATURE_DEFS\n";
  os << "#endif // GEN_PREVIEW_FEATURE_DEFS\n";
  os << "//===-- End Preview Feature Definitions --===//\n\n";

  os << "//===-- Begin Preview IR Classification --===//\n";
  os << "#ifdef GEN_PREVIEW_IR_CLASSIFICATION\n";
  generatePreviewOpClassification(records, os);
  generatePreviewTypeValueClassification(*typeStructure, os);
  generatePreviewAttrValueClassification(attrStructure, os);
  os << "\n#undef GEN_PREVIEW_IR_CLASSIFICATION\n";
  os << "#endif // GEN_PREVIEW_IR_CLASSIFICATION\n";
  os << "//===-- End Preview IR Classification --===//\n";
  return false;
}

/// The main entry point for the TableGen backend.
static bool generateBytecode(const RecordKeeper &records, raw_ostream &os) {
  os << "//===-- Begin Writer Implementations --===//\n";
  os << "#ifdef GEN_OP_WRITERS\n\n";
  generateOpWriterImplementations(records, os);
  os << "#undef GEN_OP_WRITERS\n";
  os << "#endif // GEN_OP_WRITERS\n";
  os << "//===-- End Writer Implementations --===//\n\n";

  os << "//===-- Begin Dispatch Switch --===//\n";
  os << "#ifdef GEN_OP_WRITER_DISPATCH\n\n";
  generateDispatchSwitch(records, os);
  os << "#undef GEN_OP_WRITER_DISPATCH\n";
  os << "#endif // GEN_OP_WRITER_DISPATCH\n";
  os << "//===-- End Dispatch Switch --===//\n\n";

  os << "//===-- Begin Operation Version Checkers --===//\n";
  os << "#ifdef GEN_OP_VERSION_CHECKERS\n\n";
  generateOpVersionCheckerImplementations(records, os);
  os << "#undef GEN_OP_VERSION_CHECKERS\n";
  os << "#endif // GEN_OP_VERSION_CHECKERS\n";
  os << "//===-- End Operation Version Checkers --===//\n\n";

  os << "//===-- Begin Version Check Dispatch --===//\n";
  os << "#ifdef GEN_OP_VERSION_CHECK_DISPATCH\n\n";
  generateVersionCheckDispatch(records, os);
  os << "#undef GEN_OP_VERSION_CHECK_DISPATCH\n";
  os << "#endif // GEN_OP_VERSION_CHECK_DISPATCH\n";
  os << "//===-- End Version Check Dispatch --===//\n\n";

  return false;
}

/// Generate opcode definitions in single file with ifdef guards
static bool generateOpcodes(const RecordKeeper &records, raw_ostream &os) {
  os << "//===-- Begin Opcode Enum --===//\n";
  os << "#ifdef GEN_OPCODE_ENUM\n\n";
  generateOpcodeEnumDefinition(records, os);
  os << "#undef GEN_OPCODE_ENUM\n";
  os << "#endif // GEN_OPCODE_ENUM\n";
  os << "//===-- End Opcode Enum --===//\n\n";

  os << "//===-- Begin Opcode Map --===//\n";
  os << "#ifdef GEN_OPCODE_MAP\n\n";
  generateOpcodeMap(records, os);
  os << "#undef GEN_OPCODE_MAP\n";
  os << "#endif // GEN_OPCODE_MAP\n";
  os << "//===-- End Opcode Map --===//\n\n";

  os << "//===-- Begin Version Constants --===//\n";
  os << "#ifdef GEN_VERSION_CONSTANTS\n\n";
  generateVersionConstants(records, os);
  os << "#undef GEN_VERSION_CONSTANTS\n";
  os << "#endif // GEN_VERSION_CONSTANTS\n";
  os << "//===-- End Version Constants --===//\n\n";

  os << "//===-- Begin Version Validation --===//\n";
  os << "#ifdef GEN_VERSION_VALIDATION\n\n";
  generateVersionValidation(records, os);
  os << "#undef GEN_VERSION_VALIDATION\n";
  os << "#endif // GEN_VERSION_VALIDATION\n";
  os << "//===-- End Version Validation --===//\n\n";

  os << "//===-- Begin Supported Versions List --===//\n";
  os << "#ifdef GEN_SUPPORTED_VERSIONS_LIST\n\n";
  generateSupportedVersionsList(records, os);
  os << "#undef GEN_SUPPORTED_VERSIONS_LIST\n";
  os << "#endif // GEN_SUPPORTED_VERSIONS_LIST\n";
  os << "//===-- End Supported Versions List --===//\n";

  return false;
}

/// Generate type bytecode functions.
static bool generateTypeBytecode(const RecordKeeper &records, raw_ostream &os) {
  // Phase 1: Analysis - parse TableGen records.
  auto structureOrError = analyzeBytecodeTypes(records);
  if (failed(structureOrError)) {
    PrintFatalError("Failed to analyze bytecode types");
    return true;
  }

  const auto &structure = *structureOrError;

  // Phase 2: Generation - use analyzed structure for all outputs.
  os << "//===-- Begin Type Tag Enum --===//\n";
  os << "#ifdef GEN_TYPE_TAG_ENUM\n\n";
  generateTypeTagEnum(structure, os);
  os << "#undef GEN_TYPE_TAG_ENUM\n";
  os << "#endif // GEN_TYPE_TAG_ENUM\n";
  os << "//===-- End Type Tag Enum --===//\n\n";

  os << "//===-- Begin Preview Type Classification --===//\n";
  os << "#ifdef GEN_PREVIEW_TYPE_CLASSIFICATION\n";
  generatePreviewTypeClassification(structure, os);
  os << "\n#undef GEN_PREVIEW_TYPE_CLASSIFICATION\n";
  os << "#endif // GEN_PREVIEW_TYPE_CLASSIFICATION\n";
  os << "//===-- End Preview Type Classification --===//\n\n";

  os << "//===-- Begin Type Writer Implementations --===//\n";
  os << "#ifdef GEN_TYPE_WRITERS\n\n";
  generateTypeSerializers(structure, os);
  os << "#undef GEN_TYPE_WRITERS\n";
  os << "#endif // GEN_TYPE_WRITERS\n";
  os << "//===-- End Type Writer Implementations --===//\n\n";

  os << "//===-- Begin Type Writer Dispatch --===//\n";
  os << "#ifdef GEN_TYPE_WRITER_DISPATCH\n\n";
  generateSerializerDispatch(structure, os);
  os << "#undef GEN_TYPE_WRITER_DISPATCH\n";
  os << "#endif // GEN_TYPE_WRITER_DISPATCH\n";
  os << "//===-- End Type Writer Dispatch --===//\n\n";

  os << "//===-- Begin Dependent Type Registration --===//\n";
  os << "#ifdef GEN_DEPENDENT_TYPE_REGISTRATION\n\n";
  generateDependentTypeRegistration(structure, os);
  os << "#undef GEN_DEPENDENT_TYPE_REGISTRATION\n";
  os << "#endif // GEN_DEPENDENT_TYPE_REGISTRATION\n";
  os << "//===-- End Dependent Type Registration --===//\n\n";

  os << "//===-- Begin Type Version Map --===//\n";
  os << "#ifdef GEN_TYPE_VERSION_MAP\n\n";
  generateTypeVersionMap(structure, os);
  os << "#undef GEN_TYPE_VERSION_MAP\n";
  os << "#endif // GEN_TYPE_VERSION_MAP\n";
  os << "//===-- End Type Version Map --===//\n";

  return false;
}

/// Generate attribute bytecode definitions including:
/// - AttributeTag enum
/// - Runtime version checking function
static bool generateAttrBytecode(const RecordKeeper &records, raw_ostream &os) {
  BytecodeAttrStructure structure = analyzeBytecodeAttrs(records);

  if (failed(validateAttrTagAssignments(records, structure)))
    return true;

  // Generate the AttributeTag enum.
  os << "//===-- Begin Attribute Tag Enum --===//\n";
  os << "#ifdef GEN_ATTR_TAG_ENUM\n\n";
  generateAttrTagEnum(structure, os);
  os << "\n#undef GEN_ATTR_TAG_ENUM\n";
  os << "#endif // GEN_ATTR_TAG_ENUM\n";
  os << "//===-- End Attribute Tag Enum --===//\n\n";

  // Generate the runtime version checking function.
  os << "//===-- Begin Attribute Version Check --===//\n";
  os << "#ifdef GEN_ATTR_VERSION_CHECK\n";
  generateAttrVersionCheck(structure, os);
  os << "\n#undef GEN_ATTR_VERSION_CHECK\n";
  os << "#endif // GEN_ATTR_VERSION_CHECK\n";
  os << "//===-- End Attribute Version Check --===//\n\n";

  // Generate public-preview attribute classifiers.
  os << "//===-- Begin Preview Attribute Classification --===//\n";
  os << "#ifdef GEN_PREVIEW_ATTR_CLASSIFICATION\n";
  generatePreviewAttrClassification(structure, os);
  os << "\n#undef GEN_PREVIEW_ATTR_CLASSIFICATION\n";
  os << "#endif // GEN_PREVIEW_ATTR_CLASSIFICATION\n";
  os << "//===-- End Preview Attribute Classification --===//\n\n";

  // Generate is_cuda_tile_enum type trait for BytecodeWriter.
  os << "//===-- Begin Enum Type Trait --===//\n";
  os << "#ifdef GEN_ENUM_TYPE_TRAIT\n";
  generateEnumTypeTrait(structure, os);
  os << "\n#undef GEN_ENUM_TYPE_TRAIT\n";
  os << "#endif // GEN_ENUM_TYPE_TRAIT\n";
  os << "//===-- End Enum Type Trait --===//\n\n";

  // Generate is_cuda_tile_enum_attr type trait + symbolizeEnum for
  // BytecodeReader.
  os << "//===-- Begin Enum Attr Type Trait --===//\n";
  os << "#ifdef GEN_ENUM_ATTR_TYPE_TRAIT\n";
  generateEnumAttrTypeTrait(structure, os);
  os << "\n#undef GEN_ENUM_ATTR_TYPE_TRAIT\n";
  os << "#endif // GEN_ENUM_ATTR_TYPE_TRAIT\n";
  os << "//===-- End Enum Attr Type Trait --===//\n\n";

  // Generate enum attr version checking for BytecodeReader.
  os << "//===-- Begin Enum Attr Version Check --===//\n";
  os << "#ifdef GEN_ENUM_ATTR_VERSION_CHECK\n";
  generateEnumAttrVersionCheck(structure, os);
  os << "\n#undef GEN_ENUM_ATTR_VERSION_CHECK\n";
  os << "#endif // GEN_ENUM_ATTR_VERSION_CHECK\n";
  os << "//===-- End Enum Attr Version Check --===//\n\n";

  // Generate per-value version checking for BytecodeWriter.
  os << "//===-- Begin Enum Value Version Check --===//\n";
  os << "#ifdef GEN_ENUM_VALUE_VERSION_CHECK\n";
  generateEnumValueVersionCheck(structure, os);
  os << "\n#undef GEN_ENUM_VALUE_VERSION_CHECK\n";
  os << "#endif // GEN_ENUM_VALUE_VERSION_CHECK\n";
  os << "//===-- End Enum Value Version Check --===//\n\n";

  // Generate attribute version map for MinVersionAnalyzer.
  os << "//===-- Begin Attr Version Map --===//\n";
  os << "#ifdef GEN_ATTR_VERSION_MAP\n";
  generateAttrVersionMap(structure, os);
  os << "\n#undef GEN_ATTR_VERSION_MAP\n";
  os << "#endif // GEN_ATTR_VERSION_MAP\n";
  os << "//===-- End Attr Version Map --===//\n\n";

  // Generate enum value version map for MinVersionAnalyzer.
  os << "//===-- Begin Enum Value Version Map --===//\n";
  os << "#ifdef GEN_ENUM_VALUE_VERSION_MAP\n";
  generateEnumValueVersionMap(structure, os);
  os << "\n#undef GEN_ENUM_VALUE_VERSION_MAP\n";
  os << "#endif // GEN_ENUM_VALUE_VERSION_MAP\n";
  os << "//===-- End Enum Value Version Map --===//\n\n";

  // Generate attribute parameter version map for MinVersionAnalyzer.
  os << "//===-- Begin Attr Param Version Map --===//\n";
  os << "#ifdef GEN_ATTR_PARAM_VERSION_MAP\n";
  generateAttrParamVersionMap(structure, os);
  os << "\n#undef GEN_ATTR_PARAM_VERSION_MAP\n";
  os << "#endif // GEN_ATTR_PARAM_VERSION_MAP\n";
  os << "//===-- End Attr Param Version Map --===//\n\n";

  // Generate attribute serializers.
  os << "//===-- Begin Attr Writer Implementations --===//\n";
  os << "#ifdef GEN_ATTR_WRITERS\n\n";
  generateAttrSerializers(structure, os);
  os << "#undef GEN_ATTR_WRITERS\n";
  os << "#endif // GEN_ATTR_WRITERS\n";
  os << "//===-- End Attr Writer Implementations --===//\n\n";

  // Generate is_cuda_tile_serializable_attr type trait for BytecodeReader.
  os << "//===-- Begin Serializable Attr Type Trait --===//\n";
  os << "#ifdef GEN_SERIALIZABLE_ATTR_TYPE_TRAIT\n";
  generateSerializableAttrTypeTrait(structure, os);
  os << "\n#undef GEN_SERIALIZABLE_ATTR_TYPE_TRAIT\n";
  os << "#endif // GEN_SERIALIZABLE_ATTR_TYPE_TRAIT\n";
  os << "//===-- End Serializable Attr Type Trait --===//\n\n";

  // Generate attribute deserializers.
  os << "//===-- Begin Attr Reader Implementations --===//\n";
  os << "#ifdef GEN_ATTR_READERS\n\n";
  generateAttrDeserializers(structure, os);
  os << "#undef GEN_ATTR_READERS\n";
  os << "#endif // GEN_ATTR_READERS\n";
  os << "//===-- End Attr Reader Implementations --===//\n\n";

  // Generate parseCudaTileAttrInline<T> specializations for BytecodeReader.
  os << "//===-- Begin Attr Inline Reader Dispatch --===//\n";
  os << "#ifdef GEN_ATTR_INLINE_READER_DISPATCH\n\n";
  generateAttrInlineReaderDispatch(structure, os);
  os << "#undef GEN_ATTR_INLINE_READER_DISPATCH\n";
  os << "#endif // GEN_ATTR_INLINE_READER_DISPATCH\n";
  os << "//===-- End Attr Inline Reader Dispatch --===//\n\n";

  // Generate attribute serializer dispatch.
  os << "//===-- Begin Attr Writer Dispatch --===//\n";
  os << "#ifdef GEN_ATTR_WRITER_DISPATCH\n\n";
  generateAttrSerializerDispatch(structure, os);
  os << "#undef GEN_ATTR_WRITER_DISPATCH\n";
  os << "#endif // GEN_ATTR_WRITER_DISPATCH\n";
  os << "//===-- End Attr Writer Dispatch --===//\n\n";

  // Generate attribute deserializer dispatch.
  os << "//===-- Begin Attr Reader Dispatch --===//\n";
  os << "#ifdef GEN_ATTR_READER_DISPATCH\n\n";
  generateAttrDeserializerDispatch(structure, os);
  os << "#undef GEN_ATTR_READER_DISPATCH\n";
  os << "#endif // GEN_ATTR_READER_DISPATCH\n";
  os << "//===-- End Attr Reader Dispatch --===//\n\n";

  // Generate case labels for the reader's self-contained attribute switch.
  os << "//===-- Begin Attr Reader Switch Cases --===//\n";
  os << "#ifdef GEN_ATTR_READER_SWITCH_CASES\n\n";
  generateAttrReaderSwitchCases(structure, os);
  os << "\n#undef GEN_ATTR_READER_SWITCH_CASES\n";
  os << "#endif // GEN_ATTR_READER_SWITCH_CASES\n";
  os << "//===-- End Attr Reader Switch Cases --===//\n";

  // Generate dependent type registration for CudaTile attributes.
  os << "//===-- Begin Dependent Attr Type Registration --===//\n";
  os << "#ifdef GEN_DEPENDENT_ATTR_TYPE_REGISTRATION\n\n";
  generateDependentAttrTypeRegistration(structure, os);
  os << "#undef GEN_DEPENDENT_ATTR_TYPE_REGISTRATION\n";
  os << "#endif // GEN_DEPENDENT_ATTR_TYPE_REGISTRATION\n";
  os << "//===-- End Dependent Attr Type Registration --===//\n\n";

  return false;
}

/// Register the generators.
static mlir::GenRegistration
    genCudaTileBytecode("gen-cuda-tile-bytecode",
                        "Generate cuda_tile bytecode writer implementations.",
                        [](const RecordKeeper &records, raw_ostream &os) {
                          return generateBytecode(records, os);
                        });

static mlir::GenRegistration
    genCudaTileOpcodes("gen-cuda-tile-opcodes",
                       "Generate cuda_tile opcode definitions.",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         return generateOpcodes(records, os);
                       });

static mlir::GenRegistration
    genCudaTileTypeBytecode("gen-cuda-tile-type-bytecode",
                            "Generate cuda_tile type bytecode implementations.",
                            [](const RecordKeeper &records, raw_ostream &os) {
                              return generateTypeBytecode(records, os);
                            });

static mlir::GenRegistration genCudaTileAttrBytecode(
    "gen-cuda-tile-attr-bytecode",
    "Generate cuda_tile attribute bytecode definitions.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return generateAttrBytecode(records, os);
    });

static mlir::GenRegistration genCudaTilePreviewFeatures(
    "gen-cuda-tile-preview-features",
    "Generate cuda_tile preview feature metadata and IR classification.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return generatePreviewFeatures(records, os);
    });
