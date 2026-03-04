//===- BytecodeReaderGen.cpp ------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TableGen backend for generating bytecode
// reader functions for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "BytecodeGenUtilities.h"
#include "BytecodeTypeAnalysis.h"
#include "BytecodeTypeCodeGen.h"
using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// The template for the C++ function signature for the 'parse<OpName>'
/// function.
/// {0}: The C++ class name of the operation.
static const char *const functionSignatureTemplate = R"(
  static LogicalResult parse{0}(EncodingReader &reader,
                                     OpBuilder &innerBuilder,
                                     Location loc,
                                     std::vector<Value> &valueIndexList,
                                     ArrayRef<ArrayRef<uint8_t>> constants,
                                     LazyTypeTable &types,
                                     DenseElementsAttrCache &constCache,
                                     DebugInfoReader::Iterator &diIterator,
                                     MLIRContext &context,
                                     const BytecodeVersion &bytecodeVersion) {{
)";

/// The template for generating operand deserialization code.
/// {0}: Argument for the number of operands to read (either a number or
/// "std::nullopt").
static const char *const operandDeserializationTemplate = R"(
  // --- Read Operands ---
  if (failed(parseOperands(reader, loc, valueIndexList, parsedOperands,
                           /*numOperandsToRead=*/{0})))
    return failure();
)";

/// Template for optional ODS operand segment deserialization using flags field.
/// {0}: ODS Operand Name string.
/// {1}: Operation Name string.
/// {2}: Index 'i' for unique variable name generation.
/// {3}: Bit index in the flags field.
static const char *const optionalOdsOperandSegmentTemplate = R"(
  // Optional operand '{0}' for operation '{1}'
  currentSegmentLengthOds_{2} = (flags & (1ULL << {3})) ? 1 : 0;
)";

/// Template for variadic (non-optional) ODS operand segment deserialization.
/// {0}: ODS Operand Name string.
/// {1}: Operation Name string.
/// {2}: Index 'i' for unique variable name generation.
static const char *const variadicOdsOperandSegmentTemplate = R"(
  uint64_t actualSegmentSizeFromStreamOds_{2};
  if (failed(reader.readVarInt(actualSegmentSizeFromStreamOds_{2})))
    return reader.emitError() << "failed to read actual size for variadic ODS segment '{0}' of op '{1}'";
  currentSegmentLengthOds_{2} = static_cast<int32_t>(actualSegmentSizeFromStreamOds_{2});
)";

/// Template for reading SSA value indices for an ODS operand segment.
/// {0}: Index 'i' for unique variable name generation.
/// {1}: ODS Operand Name string.
static const char *const odsOperandSSAReadTemplate = R"(
  readSegmentSizes.push_back(currentSegmentLengthOds_{0});
  if (parsedOperands.size() + static_cast<size_t>(currentSegmentLengthOds_{0})
      > std::numeric_limits<uint32_t>::max() - 1)
    return reader.emitError() << "failed to read operands for {1} segment, exceeds maximum supported capacity";
  if (currentSegmentLengthOds_{0} > 0) {{
    parsedOperands.reserve(parsedOperands.size() + static_cast<size_t>(currentSegmentLengthOds_{0}));
    for (int32_t j = 0; j < currentSegmentLengthOds_{0}; ++j) {{
      uint64_t operandIdxOds_{0}_j;
      if (failed(reader.readVarInt(operandIdxOds_{0}_j)))
        return reader.emitError() << "failed to read operand index for {1} segment, element " << j;
      if (operandIdxOds_{0}_j >= valueIndexList.size())
        return reader.emitError() << "operand index " << operandIdxOds_{0}_j << " out of bounds (size=" << valueIndexList.size() << ") for {1} segment, element " << j;
      parsedOperands.push_back(valueIndexList[operandIdxOds_{0}_j]);
    }
  }
)";

/// Template for optional attribute parsing with parseOpAttribute.
/// {0}: Variable name for the attribute.
/// {1}: C++ type string for temp variable.
/// {2}: Expected type argument.
/// {3}: Attribute name string.
static const char *const optionalAttrParseTemplate = R"(
      {1} tempValue;
      if (failed(parseOpAttribute(reader, context, types, constants, constCache, tempValue, {2})))
        return reader.emitError() << "failed to parse optional attribute '" << "{3}" << "'";
      {0} = tempValue;
)";

/// Template for required attribute parsing with parseOpAttribute.
/// {0}: Variable name for the attribute.
/// {1}: Expected type argument.
/// {2}: Attribute name string.
static const char *const requiredAttrParseTemplate = R"(
  if (failed(parseOpAttribute(reader, context, types, constants, constCache, {0}, {1})))
    return reader.emitError() << "failed to parse attribute '" << "{2}" << "'";
)";

/// Helper function to generate common attribute parsing logic.
static void generateAttributeParsingLogic(raw_ostream &os, StringRef varName,
                                          StringRef expectedTypeArg,
                                          StringRef attrName,
                                          StringRef baseCppTypeStr,
                                          bool isOptional, bool isUnitAttr,
                                          int bitIndex) {
  if (isOptional) {
    // Optional attribute - check flags field.
    os << "    if (flags & (1ULL << " << bitIndex << ")) {\n";
    if (isUnitAttr)
      os << llvm::formatv(R"(      {0} = UnitAttr::get(&context);)", varName);
    else {
      os << "      ";
      os << llvm::formatv(optionalAttrParseTemplate, varName, baseCppTypeStr,
                          expectedTypeArg, attrName);
    }
    os << "    }\n";
  } else {
    // Required attribute - read directly.
    os << llvm::formatv(requiredAttrParseTemplate, varName, expectedTypeArg,
                        attrName);
  }
}

/// The template for generating result type deserialization code.
/// {0}: Number of results.
/// {1}: C++ class name of the operation.
static const char *const resultTypeDeserializationTemplate = R"(
  SmallVector<Type> resultTypes;
  uint64_t numResultsToRead = {0};
  if (numResultsToRead > 0) {
    resultTypes.reserve(numResultsToRead);
    for (unsigned i = 0; i != numResultsToRead; ++i) {
      Type resultType = types.readAndGetType(reader);
      if (!resultType)
        return reader.emitError() << "failed to get result type " << i << " for {1}";
      resultTypes.push_back(resultType);
    }
  }
)";

/// The template for generating the final operation creation code.
/// {0}: The MLIR operation name (e.g. "cuda_tile.addf").
/// {1}: The number of results to add to valueIndexList.
static const char *const operationDeserializationTemplate = R"(
  // --- Create Operation ---
  if (failed(createOperationGeneric(innerBuilder, loc, "{0}",
                                  resultTypes, parsedOperands, attributes,
                                  valueIndexList, parsedRegions, {1})))
    return failure();
)";

/// The template for generating a case in the opcode dispatch switch statement.
/// {0}: The C++ class name of the operation (e.g., CudaTile_AddIOp).
static const char *const dispatchCaseTemplate = R"(
  case Opcode::{0}:
    if (failed(parse{0}(reader, innerBuilder, loc, valueIndexList, constants, types, constCache, diIterator, context, bytecodeVersion)))
      return failure();
    break;
)";

/// The template for generating region deserialization code.
/// {0}: The MLIR operation name (e.g., "cuda_tile.if").
/// {1}: Number of expected regions for op.
static const char *const regionDeserializationTemplate = R"(
  // --- Read Regions ---
  uint64_t numRegionsToParse;
  if (failed(reader.readVarInt(numRegionsToParse, std::numeric_limits<uint32_t>::max() - 1)))
    return reader.emitError() << "failed to read number of regions to parse.";
  if (numRegionsToParse != {1})
    return reader.emitError() << "{0} op expected {1} regions, got " << numRegionsToParse;
  if (numRegionsToParse > 0) {{
    parsedRegions.reserve(numRegionsToParse);
    for (uint64_t i = 0; i < numRegionsToParse; ++i) {{
      auto region = std::make_unique<Region>();
      if (failed(parseRegion(reader, innerBuilder, loc, valueIndexList, constants, types, constCache, diIterator, context, *region, bytecodeVersion)))
        return reader.emitError() << "failed to parse region " << i;
      parsedRegions.push_back(std::move(region));
    }
  }
)";

/// Reads the flags field that encodes the presence of optional attributes
/// and operands using individual bits.
static void generateFlagsFieldDeserialization(
    const Operator &op, raw_ostream &os,
    const StringMap<size_t> &bitAssignments,
    const std::optional<std::pair<std::string, std::string>>
        &minOptionalVersion) {
  size_t totalOptionalFields = bitAssignments.size();

  // Always declare flags variable for use in conditional logic below.
  if (totalOptionalFields > 0) {
    os << "  // Read flags field for optional attributes/operands.\n"
       << "  uint64_t flags = 0;\n";

    // Forward Compatibility: Only generate version check if the first optional
    // field was added AFTER the operation's baseline version. This allows newer
    // readers (e.g., 13.2) to read older bytecode (e.g., 13.1) that was written
    // before optional fields existed. If optional fields existed from the
    // operation's baseline, flags field is always present and no check needed.
    std::string opVersion = extractVersionFromOperation(op);
    std::string minOptionalVersionStr =
        minOptionalVersion
            ? (minOptionalVersion->first + "." + minOptionalVersion->second)
            : "";
    bool needsVersionCheck =
        minOptionalVersion && (minOptionalVersionStr != opVersion);

    if (needsVersionCheck) {
      auto [majorStr, minorStr] = *minOptionalVersion;
      os << llvm::formatv(
          R"(  // Only read flags if bytecode version supports it (>= {0}.{1})
  auto requiredVersionForFlags = BytecodeVersion::fromVersion({0}, {1}, 0);
  assert(requiredVersionForFlags && "TableGen should guarantee valid versions");
  if (bytecodeVersion >= *requiredVersionForFlags) {{
    if (failed(reader.readVarInt(flags)))
      return reader.emitError() << "failed to read flags field";
  }
  // Else: flags stays 0 (no optional fields in older bytecode versions)

)",
          majorStr, minorStr);
    } else {
      // Flags field always exists for this operation.
      os << "  if (failed(reader.readVarInt(flags)))\n"
         << "    return reader.emitError() << \"failed to read flags "
            "field\";\n\n";
    }
  }
}

/// Generates the C++ function signature for the 'parse<OpName>' function,
/// which handles deserialization for a specific cuda_tile operation.
static void generateFunctionSignature(const Operator &op, raw_ostream &os) {
  os << llvm::formatv(functionSignatureTemplate, op.getCppClassName());
}

/// Generates C++ code within the 'parse<OpName>' function to deserialize the
/// operands of the given operation.
static void
generateOperandDeserialization(const Operator &opDef, raw_ostream &os,
                               const StringMap<size_t> &bitAssignments) {
  os << "  SmallVector<Value, 0> parsedOperands;\n";
  if (opDef.getNumOperands() == 0)
    return;

  if (opDef.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    os << "  // --- Deserialize Operands (AttrSizedOperandSegments) ---\n";
    os << "  SmallVector<int32_t, 4> readSegmentSizes;\n";

    std::string opName = opDef.getOperationName();

    for (unsigned i = 0; i < static_cast<unsigned>(opDef.getNumOperands());
         ++i) {
      const auto &odsOperand = opDef.getOperand(i);
      StringRef odsOperandName = odsOperand.name;
      bool isOptional = odsOperand.isOptional();
      bool isVariadic = odsOperand.isVariableLength();

      // Make variable names unique within the generated function by embedding
      // the index 'i'.
      os << "  // Parsing ODS Operand Segment: " << odsOperandName << "\n"
         << "  int32_t currentSegmentLengthOds_" << i << " = 0;\n";

      if (isOptional) {
        size_t operandBitIndex = bitAssignments.lookup(odsOperandName);
       // Public operations: check operand version compatibility.
          auto [majorStr, minorStr] = extractVersionFromOperand(i, opDef);
          std::string version = majorStr + "." + minorStr;
          std::string opVersion = extractVersionFromOperation(opDef);

          if (version == opVersion) {
            // Operand from original operation - simple flag reading.
            os << llvm::formatv(optionalOdsOperandSegmentTemplate,
                                odsOperandName, opName, i, operandBitIndex);
          } else {
            // Versioned operand - validate flag consistency.
            std::string templateCode =
                llvm::formatv(optionalOdsOperandSegmentTemplate, odsOperandName,
                              opName, i, operandBitIndex)
                    .str();

            os << llvm::formatv(R"(
  auto requiredVersionFor_{0} = BytecodeVersion::fromVersion({1}, {2}, 0);
  assert(requiredVersionFor_{0} && "TableGen should guarantee valid versions");
  if (bytecodeVersion >= *requiredVersionFor_{0}) {{
    // Operand supported - read flag normally.
    {3}  
  } else {{
    // Operand not supported in this bytecode version - validate consistency.
    if (flags & (1ULL << {4}))
      return reader.emitError() << "malformed bytecode: flag set for operand '{0}' which requires {5} but reading "
                                << bytecodeVersion.toString() << " bytecode";
    currentSegmentLengthOds_{6} = 0;
  }
)",
                                odsOperandName, majorStr, minorStr,
                                templateCode, operandBitIndex, version, i);
          }
      } else if (isVariadic) {
        // Read variadic operand size from stream.
        os << llvm::formatv(variadicOdsOperandSegmentTemplate, odsOperandName,
                            opName, i);
      } else {
        // Required operand: always 1 element.
        os << "  currentSegmentLengthOds_" << i << " = 1;\n";
      }
      // Code to read SSA value indices based on currentSegmentLengthOds_i.
      os << llvm::formatv(odsOperandSSAReadTemplate, i, odsOperandName);
    }

    os << "  "
          "attributes.emplace_back(innerBuilder.getStringAttr(\"operand_"
          "segment_sizes\"), "
          "mlir::DenseI32ArrayAttr::get(&context, readSegmentSizes));\n";

  } else {
    os << "  // Standard operand deserialization for ops without "
          "AttrSizedOperandSegments.\n";
    bool opHasOptionalOperands =
        llvm::any_of(opDef.getOperands(),
                     [](const auto &operand) { return operand.isOptional(); });
    bool opHasVariadicOperands =
        llvm::any_of(opDef.getOperands(),
                     [](const auto &operand) { return operand.isVariadic(); });
    bool readSizeFromStream = opHasVariadicOperands || opHasOptionalOperands;
    os << llvm::formatv(operandDeserializationTemplate,
                        readSizeFromStream
                            ? "std::nullopt"
                            : std::to_string(opDef.getNumOperands()));
  }
}

/// Generates C++ code within the 'parse<OpName>' function to deserialize the
/// attributes of the given operation by calling the parseOpAttribute helper.
static void
generateAttributeDeserialization(const Operator &op, raw_ostream &os,
                                 const StringMap<size_t> &bitAssignments) {
  os << R"(
  // --- Deserialize Attributes ---
  SmallVector<NamedAttribute> attributes;
  )";

  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    std::string attrName = namedAttr.name.str();
    std::string varName = "parsed_" + attrName;
    StringRef baseCppTypeStr = namedAttr.attr.getStorageType();
    bool isOptional = namedAttr.attr.isOptional();
    bool isUnitAttr = baseCppTypeStr.contains("UnitAttr");

    // Declare the attribute variable
    os << llvm::formatv(R"(  {0} {1};)", baseCppTypeStr, varName) << "\n";

    // Determine expectedType for parseOpAttribute
    bool isElements = baseCppTypeStr.contains("ElementsAttr");
    std::string expectedTypeArg = "nullptr";
    std::string expectedTypeDeclaration;

    if (isElements && op.getNumResults() > 0) {
      expectedTypeArg = "resultTypes.empty() ? nullptr : resultTypes[0]";
    } else if (baseCppTypeStr == "::mlir::IntegerAttr") {
      StringRef attrDefName = namedAttr.attr.getAttrDefName();
      static const llvm::StringMap<unsigned> attrWidthMap = {
          {"I1Attr", 1},   {"I8Attr", 8},   {"I16Attr", 16},
          {"I32Attr", 32}, {"I64Attr", 64},
      };
      auto it = attrWidthMap.find(attrDefName);
      if (it != attrWidthMap.end()) {
        unsigned width = it->second;
        std::string typeVarName = varName + "_expectedType";
        expectedTypeDeclaration =
            formatv("  Type {0} = IntegerType::get(&context, {1});\n",
                    typeVarName, width);
        expectedTypeArg = typeVarName;
      } else {
        os << formatv(
            R"(  return reader.emitError() << "could not determine width for inline IntegerAttr '{0}' with definition '{1}'";)"
            "\n",
            attrName, attrDefName);
      }
    }
    // Emit the expected type declaration if needed.
    os << expectedTypeDeclaration;

      os << "  // Public operations: use version checking\n";
      // For public operations, add version checking.
      auto [majorStr, minorStr] = extractVersionFromAttribute(namedAttr, op);
      auto defaultValue = extractDefaultValue(namedAttr);
      std::string version = majorStr + "." + minorStr;

      os << llvm::formatv(R"(
  auto requiredVersionFor_{0} = BytecodeVersion::fromVersion({1}, {2}, 0);
  assert(requiredVersionFor_{0} && "TableGen should guarantee valid versions");
  if (bytecodeVersion >= *requiredVersionFor_{0}) {{
)",
                          attrName, majorStr, minorStr);

      // Generate parsing logic within version check.
      int bitPos =
          isOptional ? static_cast<int>(bitAssignments.lookup(attrName)) : -1;
      generateAttributeParsingLogic(os, varName, expectedTypeArg, attrName,
                                    baseCppTypeStr, isOptional, isUnitAttr,
                                    bitPos);

      os << "  } else {\n"
         << "    // For older bytecode versions, use default value\n";
      if (defaultValue.has_value()) {
        // Handle different attribute types with their specific construction
        // patterns
        if (baseCppTypeStr.contains("UnitAttr")) {
          // UnitAttr with false default means don't create the attribute
          // (nullptr).
          if (*defaultValue == "false")
            os << "    " << varName << " = nullptr;\n";
          else
            os << "    " << varName << " = ::mlir::UnitAttr::get(&context);\n";
        } else if (baseCppTypeStr.contains("IntegerAttr")) {
          // IntegerAttr needs a type.
          os << "    " << varName << " = ::mlir::IntegerAttr::get("
             << expectedTypeArg << ", " << *defaultValue << ");\n";
        } else if (baseCppTypeStr.contains("cuda_tile::")) {
          // Custom cuda_tile attributes follow the standard pattern.
          os << "    " << varName << " = " << baseCppTypeStr
             << "::get(&context, " << *defaultValue << ");\n";
        } else {
          PrintFatalError("Versioned attribute type '" + baseCppTypeStr.str() +
                          "' is not supported. Please add explicit handling in "
                          "BytecodeReaderGen.cpp for operation '" +
                          op.getOperationName() + "'");
        }
      } else {
        // No default value available.
        std::string opVersion = extractVersionFromOperation(op);
        if (version != opVersion) {
          // For attributes introduced after the operation itself
          if (namedAttr.attr.isOptional()) {
            // Optional attributes should be nullptr (missing) for older
            // versions
            os << "    " << varName << " = nullptr;\n";
          } else {
            // Required attributes introduced after the operation must have
            // default value
            PrintFatalError(
                "Versioned attribute '" + namedAttr.name + "' in operation '" +
                op.getOperationName() + "' (since " + version +
                ") was introduced after the operation itself (since " +
                opVersion +
                ") and must have a default value for backward compatibility");
          }
        }
        // Note: Attributes introduced with the operation itself don't need
        // defaults.
      }
      os << "  }\n";
    // Generate attribute addition to the attributes vector.
    os << formatv(R"(  if ({0}) {{
    attributes.emplace_back(innerBuilder.getStringAttr("{1}"), {2});})",
                  varName, attrName, varName);
  }
  os << "\n";
}

/// Generate result deserialization without version checking.
static void generateSimpleResultDeserialization(const Operator &op,
                                                raw_ostream &os) {
  StringRef opClassName = op.getCppClassName();
  os << "  // Deserialize results directly.\n";

  std::string numResultsStr;
  if (op.isVariadic()) {
    os << "  uint64_t numActualResults;\n"
       << "  if (failed(reader.readVarInt(numActualResults,\n"
       << "             std::numeric_limits<uint32_t>::max() - 1)))\n"
       << "    return reader.emitError() << \"failed to read number of result "
          "types for "
       << opClassName << "\";\n";
    numResultsStr = "numActualResults";
  } else {
    numResultsStr = std::to_string(op.getNumResults());
  }
  os << "  // --- Read Result Types ---\n";
  os << llvm::formatv(resultTypeDeserializationTemplate, numResultsStr,
                      opClassName);
  // For simple deserialization, all results were serialized.
  os << "  std::optional<size_t> numResultsForValueIndex = std::nullopt;\n";
}

/// Generate version-aware result deserialization.
static void generateVersionAwareResultDeserialization(const Operator &op,
                                                      raw_ostream &os) {
  StringRef opClassName = op.getCppClassName();
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
    // All results from original operation - use simple deserialization.
    generateSimpleResultDeserialization(op, os);
    return;
  }

  os << llvm::formatv(R"(
  // --- Read Result Types (Version-Aware) ---
  SmallVector<Type> resultTypes;
)");

  if (op.isVariadic()) {
    os << llvm::formatv(R"(
  uint64_t numSerializedResults;
  if (failed(reader.readVarInt(numSerializedResults,
             std::numeric_limits<uint32_t>::max() - 1)))
    return reader.emitError() << "failed to read number of result types for {0}";
)",
                        opClassName);
  }

  os << llvm::formatv(R"(
  uint64_t expectedResults = 0;
  uint64_t resultIndex = 0;
)");

  for (int i = 0; i < op.getNumResults(); ++i) {
    const auto &info = versionInfos[i];

    if (!info.requiresVersionCheck) {
      // Original result always compatible.
      os << llvm::formatv(R"(
  ++expectedResults;
  Type resultType = types.readAndGetType(reader);
  if (!resultType)
    return reader.emitError() << "failed to get result type " << resultIndex << " for {0}";
  resultTypes.push_back(resultType);
  ++resultIndex;
)",
                          opClassName);
    } else {
      // Add default result type based on actual type constraint.
      const auto &result = op.getResult(i);
      const auto &constraint = result.constraint;

      std::string defaultTypeCode;
      std::optional<StringRef> builderCall = constraint.getBuilderCall();
      if (builderCall.has_value()) {
        FmtContext fctx;
        fctx.withBuilder("innerBuilder");
        fctx.addSubst("_ctxt", "&context");
        defaultTypeCode = "    resultTypes.push_back(" +
                          tgfmt(*builderCall, &fctx).str() + ");\n";
      } else {
        PrintFatalError("Result '" + result.name.str() + "' in operation '" +
                        op.getOperationName() +
                        "' has non-buildable type constraint '" +
                        constraint.getDefName().str() +
                        "'. Results introduced after operation version must "
                        "have buildable types.");
      }

      os << llvm::formatv(
          R"(
  auto requiredVersionFor_result_{0} = BytecodeVersion::fromVersion({1}, {2}, 0);
  assert(requiredVersionFor_result_{0} && "TableGen should guarantee valid versions");
  if (bytecodeVersion >= *requiredVersionFor_result_{0}) {{
    ++expectedResults;
    Type resultType = types.readAndGetType(reader);
    if (!resultType)
      return reader.emitError() << "failed to get result type " << resultIndex << " for {3}";
    resultTypes.push_back(resultType);
    ++resultIndex;
  } else {{
    // Result introduced in newer version.
{4}  }
)",
          info.name, info.majorStr, info.minorStr, opClassName,
          defaultTypeCode);
    }
  }

  if (op.isVariadic()) {
    os << llvm::formatv(R"(
  if (numSerializedResults != expectedResults)
    return reader.emitError() << "result count mismatch for {0}: expected " << expectedResults 
                              << " compatible results but got " << numSerializedResults;
)",
                        opClassName);
  }

  // For version-aware deserialization, only add serialized results to
  // valueIndexList. Results introduced in newer versions (with default types)
  // should not be added to valueIndexList to preserve SSA value numbering.
  os << "  std::optional<size_t> numResultsForValueIndex = expectedResults;\n";
}

/// Generates C++ code to deserialize the result types of the operation.
static void generateResultTypeDeserialization(const Operator &op,
                                              raw_ostream &os) {
  generateVersionAwareResultDeserialization(op, os);
}

/// Generates C++ code within the 'parse<OpName>' function to deserialize the
/// regions of the given operation, if it has any.
static void generateRegionDeserialization(const Operator &op, raw_ostream &os) {
  os << R"(  // --- Read Regions ---
  SmallVector<std::unique_ptr<Region>> parsedRegions;
)";
  if (op.getNumRegions() != 0)
    os << llvm::formatv(regionDeserializationTemplate, op.getOperationName(),
                        op.getNumRegions());
}

/// Generates C++ code within the 'parse<OpName>' function to deserialize the
/// operation.
static void generateOperationDeserialization(const Operator &op,
                                             raw_ostream &os) {
  std::string opName = op.getOperationName();
  os << llvm::formatv(operationDeserializationTemplate, opName,
                      "numResultsForValueIndex");
}

/// Generates the complete C++ function 'parse<OpName>'.
static void generateOpReader(const Operator &op, raw_ostream &os) {
  std::string opName = op.getOperationName();
  auto [bitAssignments, minOptionalVersion] =
      getVersionOrderedBitAssignments(op);
  os << "// Reader for Op: " << opName << "\n";

  generateFunctionSignature(op, os);
  generateResultTypeDeserialization(op, os);
  generateFlagsFieldDeserialization(op, os, bitAssignments, minOptionalVersion);
  generateAttributeDeserialization(op, os, bitAssignments);
  generateOperandDeserialization(op, os, bitAssignments);
  generateRegionDeserialization(op, os);
  generateOperationDeserialization(op, os);
  os << "  return success();\n"
     << "}\n\n";
}

/// Generates the implementations of the individual op reader functions.
static void generateOpReaderImplementations(const RecordKeeper &records,
                                            raw_ostream &os) {

  emitSourceFileHeader("Generated Bytecode Reader Functions", os);
  auto opDefs = records.getAllDerivedDefinitions("Op");
  for (const Record *opDef : opDefs)
    generateOpReader(Operator(opDef), os);
  os << R"(//===----------------------------------------------------------------------===//
// End of generated functions.
//===----------------------------------------------------------------------===//
)";
}

/// Generates the C++ switch statement to dispatch based on opcode.
static void generateOpReaderDispatch(const RecordKeeper &records,
                                     raw_ostream &os) {
  emitSourceFileHeader("Generated Bytecode Opcode Dispatcher", os);
  auto opDefs = records.getAllDerivedDefinitions("Op");
  os << R"(switch (static_cast<Opcode>(opcode)) {)";
  for (const Record *opDef : opDefs) {
    Operator op(opDef);
    StringRef opClassName = op.getCppClassName();
    os << llvm::formatv(dispatchCaseTemplate, opClassName);
  }
  os << R"(  default:
    return reader.emitError() << "unknown or unimplemented opcode: " << static_cast<int>(opcode);})";
  os << R"(//===-------------------------------------------------------------------//
// End of generated dispatcher.
//===-------------------------------------------------------------------//
)";
}

/// The main entry point for the TableGen backend.
static bool generateBytecodeReader(const RecordKeeper &records,
                                   raw_ostream &os) {
  os << "//===-- Begin Reader Implementations --===//\n"
     << "#ifdef GEN_OP_READERS\n\n";
  generateOpReaderImplementations(records, os);
  os << "#undef GEN_OP_READERS\n"
     << "#endif // GEN_OP_READERS\n"
     << "//===-- End Reader Implementations --===//\n\n";

  os << "//===-- Begin Opcode Dispatcher --===//\n"
     << "#ifdef GEN_OP_READER_DISPATCH\n\n";
  generateOpReaderDispatch(records, os);
  os << "#undef GEN_OP_READER_DISPATCH\n"
     << "#endif // GEN_OP_READER_DISPATCH\n"
     << "//===-- End Opcode Dispatcher --===//\n\n";

  return false;
}

/// Generate type reader bytecode functions.
static bool generateTypeReaderBytecode(const RecordKeeper &records,
                                       raw_ostream &os) {
  // Phase 1: Analysis - parse TableGen records.
  auto structureOrError = analyzeBytecodeTypes(records);
  if (failed(structureOrError)) {
    PrintFatalError("Failed to analyze bytecode types");
    return true;
  }
  const auto &structure = *structureOrError;

  // Phase 2: Generation - use analyzed structure for all outputs.
  os << "//===-- Begin Type Reader Implementations --===//\n";
  os << "#ifdef GEN_TYPE_READERS\n\n";
  generateTypeDeserializers(structure, os);
  os << "#undef GEN_TYPE_READERS\n";
  os << "#endif // GEN_TYPE_READERS\n";
  os << "//===-- End Type Reader Implementations --===//\n\n";

  os << "//===-- Begin Type Reader Dispatch --===//\n";
  os << "#ifdef GEN_TYPE_READER_DISPATCH\n\n";
  generateDeserializerDispatch(structure, os);
  os << "#undef GEN_TYPE_READER_DISPATCH\n";
  os << "#endif // GEN_TYPE_READER_DISPATCH\n";
  os << "//===-- End Type Reader Dispatch --===//\n";

  return false;
}

/// Register the generator.
static mlir::GenRegistration genCudaTileBytecodeReader(
    "gen-cuda-tile-bytecode-reader",
    "Generate cuda_tile bytecode reader implementations.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return generateBytecodeReader(records, os);
    });

static mlir::GenRegistration genCudaTileTypeBytecodeReader(
    "gen-cuda-tile-type-bytecode-reader",
    "Generate cuda_tile type bytecode reader implementations.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return generateTypeReaderBytecode(records, os);
    });
