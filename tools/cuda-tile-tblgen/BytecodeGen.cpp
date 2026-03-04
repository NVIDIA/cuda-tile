//===- BytecodeGen.cpp ------------------------------------------*- C++ -*-===//
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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "BytecodeGenUtilities.h"
#include "BytecodeTypeAnalysis.h"
#include "BytecodeTypeCodeGen.h"
#include <map>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// Generates the opcode enum definition from TableGen records.
static void generateOpcodeEnumDefinition(const RecordKeeper &records,
                                         raw_ostream &os) {
  emitSourceFileHeader("Generated Opcode Enum Definition", os);

  // Get all BytecodeOpcode records.
  auto opcodeRecords = records.getAllDerivedDefinitions("BytecodeOpcode");

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
    if (namedAttr.attr.isOptional()) {
      StringRef attrName = namedAttr.name;
      std::string getterName = op.getGetterName(attrName);
      size_t bitPos = bitAssignments.lookup(attrName);

        auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
        std::string version = majorStr + "." + minorStr;

        if (version == opVersion) {
          // Attribute from original operation - simple flag setting.
          os << llvm::formatv(R"(
  auto flagsAttrValue_{0} = op.{1}();
  if (flagsAttrValue_{0}) flags |= (1ULL << {2});
)",
                              attrName, getterName, bitPos);
        } else {
          // Versioned attribute - validate version compatibility.
          os << llvm::formatv(R"(
  auto flagsAttrValue_{0} = op.{1}();
  if (flagsAttrValue_{0}) {{
    auto flagsRequiredVersionFor_{0} = BytecodeVersion::fromVersion({2}, {3}, 0);
    assert(flagsRequiredVersionFor_{0} && "TableGen should guarantee valid versions");
    if (config.bytecodeVersion < *flagsRequiredVersionFor_{0}) {{
      op.emitError() << "optional attribute '{0}' is provided but requires bytecode version {4}, targeting " << config.bytecodeVersion.toString();
      return failure();
    }
    // Attribute provided and compatible - set flag.
    flags |= (1ULL << {5});
  }
  // Attribute not provided - don't set flag.
)",
                              attrName, getterName, majorStr, minorStr, version,
                              bitPos);
        }
    }
  }

  // Set flags bits for optional operands and validate them.
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    for (const auto &[operandIndex, odsOperand] :
         llvm::enumerate(op.getOperands())) {
      if (!odsOperand.isOptional()
      ) {
        // Validate that required operands were introduced with the operation
        // itself.
        auto [majorStr, minorStr] = extractVersionFromOperand(operandIndex, op);
        std::string version = majorStr + "." + minorStr;
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

          auto [majorStr, minorStr] =
              extractVersionFromOperand(operandIndex, op);
          std::string version = majorStr + "." + minorStr;

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
    if (config.bytecodeVersion < *requiredVersionFor_{1}) {{
      op.emitError() << "optional operand '{1}' is provided but requires bytecode version {4}, targeting " << config.bytecodeVersion.toString();
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
  if (config.bytecodeVersion >= *requiredVersionForFlags) {{
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
    std::string getterName = op.getGetterName(attrName);
    bool isOptional = namedAttr.attr.isOptional();
    bool isUnitAttr =
        StringRef(namedAttr.attr.getStorageType()).contains("UnitAttr");
    if (isUnitAttr) {
      // UnitAttr: only flags field, no serialization needed.
      continue;
    } else if (isOptional) {
      // Optional non-UnitAttr: validation done by flags field, just serialize.
      generateAttributeSerializationLogic(os, getterName, attrName);
    } else {
      // Required attributes: need version checking and default value
      // validation.
        auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
        auto defaultValue = extractDefaultValue(namedAttr);
        std::string version = majorStr + "." + minorStr;

        os << llvm::formatv(R"(
  auto requiredVersionFor_{0} = BytecodeVersion::fromVersion({1}, {2}, 0);
  assert(requiredVersionFor_{0} && "TableGen should guarantee valid versions");
  if (config.bytecodeVersion >= *requiredVersionFor_{0}) {{
)",
                            attrName, majorStr, minorStr);
        generateAttributeSerializationLogic(os, getterName, attrName, "    ");
        os << "  } else {\n";
        if (defaultValue.has_value()) {
          os << llvm::formatv(R"(
    // Check that attribute equals default value for older versions.
    auto nativeAttrValue_{0} = op.{1}();
    if (nativeAttrValue_{0} != {2}) {{
      op.emitError() << "attribute '{0}' requires bytecode version {3}+, but targeting " << config.bytecodeVersion.toString();
      return failure();
    }
)",
                              attrName, getterName, *defaultValue, version);
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
  if (config.bytecodeVersion >= *requiredVersionFor_result_{0}) {{
    ++compatibleResults;
    compatibleResultTypes.push_back(op->getResult({3}).getType());
  } else {{
    Value resultValue = op->getResult({3});
    if (!resultValue.getUsers().empty()) {{
      op.emitError() << "result '{0}' requires bytecode version {4} but is being used and targeting "
                     << config.bytecodeVersion.toString()
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

  return false;
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
  os << "//===-- End Version Validation --===//\n";

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
  os << "//===-- End Type Writer Dispatch --===//\n";

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
