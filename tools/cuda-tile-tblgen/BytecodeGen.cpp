//===- BytecodeGen.cpp - CUDA Tile dialect bytecode generator ---*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the TableGen backend for generating bytecode
// reader/writer functions for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

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
static void generateFunctionSignature(const Operator &op, raw_ostream &os) {
  std::string opClassName = op.getCppClassName().str();
  std::string dialectNamespace = op.getDialect().getCppNamespace().str();
  std::string qualifiedClassName = dialectNamespace + "::" + opClassName;
  os << "LogicalResult write" << opClassName << "( " << qualifiedClassName
     << " op, \n"
     << "                                   EncodingWriter &writer, \n"
     << "                                   TypeManager &typeMgr, \n"
     << "                                   ConstantManager &constMgr, \n"
     << "                                   StringManager &strMgr) {\n";
}

/// Generates the flags field serialization for optional attributes and
/// operands.
///
/// The flags field is a varint that uses individual bits to encode the presence
/// of optional attributes and operands. The bit layout is:
///   - Bits 0, 1, 2, ... : Optional attributes (in declaration order)
///   - Bits N, N+1, N+2, ... : Optional operands (in declaration order, where N
///                           = number of optional attributes)
///
/// Special case: UnitAttr presence is ONLY encoded in the flags field.
/// No actual attribute data is written to the stream for UnitAttr.
static void generateFlagsFieldSerialization(const Operator &op,
                                            raw_ostream &os) {
  size_t bitIndex = 0;
  bool hasAnyOptionalFields = false;

  auto checkGenerateFlagDeclFn = [&] {
    if (!hasAnyOptionalFields) {
      // Emit header on first optional field encountered.
      os << "  // Write flags field for optional attributes/operands.\n"
         << "  uint64_t flags = 0;\n";
      hasAnyOptionalFields = true;
    }
  };

  // Set flags bits for optional attributes.
  for (const auto &namedAttr : op.getAttributes()) {
    if (namedAttr.attr.isOptional()) {
      checkGenerateFlagDeclFn();

      std::string getterName = op.getGetterName(namedAttr.name);
      os << "  {\n"
         << "    auto nativeAttrValue = op." << getterName << "();\n"
         << "    if (nativeAttrValue) flags |= (1ULL << " << bitIndex << ");\n"
         << "  }\n";
      bitIndex++;
    }
  }

  // Set flags bits for optional operands.
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    for (const auto &[operandIndex, odsOperand] :
         llvm::enumerate(op.getOperands())) {
      if (odsOperand.isOptional()) {
        checkGenerateFlagDeclFn();

        os << "  {\n"
           << "    auto operandGroup = op.getODSOperands(" << operandIndex
           << ");\n"
           << "    if (!operandGroup.empty()) flags |= (1ULL << " << bitIndex
           << ");\n"
           << "  }\n";
        bitIndex++;
      }
    }
  }

  // Only write flags if we actually have optional fields.
  if (hasAnyOptionalFields)
    os << "  writer.writeVarInt(flags);\n\n";
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
    // UnitAttr presence is only encoded in flags field, no value written.
    // For all other cases (required attributes, optional non-UnitAttr),
    // let writeOpAttribute handle it (including std::optional)
    if (!(isOptional && isUnitAttr)) {
      os << "  {\n"
         << "    auto nativeAttrValue = op." << getterName << "();\n"
         << "    if (failed(writeOpAttribute(op.getOperation(), \"" << attrName
         << "\", nativeAttrValue, writer, typeMgr, constMgr, strMgr)))\n"
         << "      return failure();\n"
         << "  }\n";
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
    os << "  // Serialize Operands (AttrSizedOperandSegments).\n";

    for (const auto &[index, odsOperand] : llvm::enumerate(op.getOperands())) {
      if (odsOperand.isOptional()) {
        os << llvm::formatv(R"(
  // Optional ODS Operand: {1}
  auto operandGroup{0} = op.getODSOperands({0});
  if (!operandGroup{0}.empty())
    writeOperands(operandGroup{0}, writer, /*encodeSize=*/false);
)",
                            index, odsOperand.name);
      } else {
        // Not optional (either variadic or static).
        os << llvm::formatv("  writeOperands(op.getODSOperands({0}), writer, "
                            "/*encodeSize=*/{1});\n",
                            index, odsOperand.isVariadic() ? "true" : "false");
      }
    }
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

/// Generates C++ code within the 'write<OpName>' function to serialize the
/// result types of the given operation.
static void generateResultTypeSerialization(const Operator &op,
                                            raw_ostream &os) {
  // Check for unsupported AttrSizedResultSegments trait.
  if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments"))
    os << " return op.emitError(\"operation '" << op.getOperationName()
       << "' has AttrSizedResultSegments, which is not supported by the "
          "bytecode writer.\");\n";

  // If the op has variadic results, write the actual number of results
  // for this specific operation instance.
  if (op.isVariadic())
    os << "  writer.writeVarInt(op->getNumResults());\n";

  // Write the result types of the operation.
  os << "  if (failed(writeResultTypes(op, writer, typeMgr)))\n"
     << "    return failure();\n\n";
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
  os << "  return success();\n"
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

  os << "if (failed(TypeSwitch<Operation *, LogicalResult>(op)\n";
  for (const Record *opDef : opDefs) {
    Operator op(opDef);
    std::string opClassName = op.getCppClassName().str();
    std::string dialectNamespace = op.getDialect().getCppNamespace().str();
    std::string qualifiedClassName = dialectNamespace + "::" + opClassName;
    os << "                   .Case<" << qualifiedClassName
       << ">([&](auto concreteOp) {\n"
       << "                     return write" << opClassName
       << "(concreteOp, writer, typeMgr, constMgr, strMgr);\n"
       << "                   })\n";
  }
  os << "                   .Default([&](Operation *) {\n"
     << "                     return op->emitError(\n"
     << "                         \"unhandled operation type in bytecode "
        "writer\");\n"
     << "                   }))) {\n"
     << "  return failure();\n"
     << "}\n\n";
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
  os << "//===-- End Dispatch Switch --===//\n";

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
