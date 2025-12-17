//===- BytecodeReaderGen.cpp - CUDA Tile Bytecode Reader Gen ----*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the TableGen backend for generating bytecode
// reader functions for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

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
                                     MLIRContext &context) {{
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
static const char *const operationDeserializationTemplate = R"(
  // --- Create Operation ---
  if (failed(createOperationGeneric(innerBuilder, loc, "{0}",
                                  resultTypes, parsedOperands, attributes,
                                  valueIndexList, parsedRegions)))
    return failure();
)";

/// The template for generating a case in the opcode dispatch switch statement.
/// {0}: The Opcode enum name (e.g., AddI).
/// {1}: The C++ class name of the operation (e.g., CudaTile_AddIOp).
static const char *const dispatchCaseTemplate = R"(
  case Opcode::{0}:
    if (failed(parse{1}(reader, innerBuilder, loc, valueIndexList, constants, types, constCache, diIterator, context)))
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
      if (failed(parseRegion(reader, innerBuilder, loc, valueIndexList, constants, types, constCache, diIterator, context, *region)))
        return reader.emitError() << "failed to parse region " << i;
      parsedRegions.push_back(std::move(region));
    }
  }
)";

/// Collects information about optional attributes and operands for flags field.
static void
collectOptionalFields(const Operator &op,
                      SmallVector<std::string> &optionalAttrNames,
                      SmallVector<std::string> &optionalOperandNames) {
  // Collect optional attributes.
  for (const auto &namedAttr : op.getAttributes())
    if (namedAttr.attr.isOptional())
      optionalAttrNames.push_back(namedAttr.name.str());

  // Collect optional operands (for AttrSizedOperandSegments).
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments"))
    for (const auto &[index, odsOperand] : llvm::enumerate(op.getOperands()))
      if (odsOperand.isOptional())
        optionalOperandNames.push_back(odsOperand.name.str());
}

/// Reads the flags field that encodes the presence of optional attributes
/// and operands using individual bits.
static void
generateFlagsFieldDeserialization(const Operator &op, raw_ostream &os,
                                  ArrayRef<std::string> optionalAttrNames,
                                  ArrayRef<std::string> optionalOperandNames) {
  size_t totalOptionalFields =
      optionalAttrNames.size() + optionalOperandNames.size();

  if (totalOptionalFields > 0)
    os << "  // Read flags field for optional attributes/operands.\n"
       << "  uint64_t flags = 0;\n"
       << "  if (failed(reader.readVarInt(flags)))\n"
       << "    return reader.emitError() << \"failed to read flags "
          "field\";\n\n";
}

/// Generates the C++ function signature for the 'parse<OpName>' function,
/// which handles deserialization for a specific cuda_tile operation.
static void generateFunctionSignature(const Operator &op, raw_ostream &os) {
  std::string opClassName = op.getCppClassName().str();
  os << llvm::formatv(functionSignatureTemplate, opClassName);
}

/// Generates C++ code within the 'parse<OpName>' function to deserialize the
/// operands of the given operation.
static void
generateOperandDeserialization(const Operator &opDef, raw_ostream &os,
                               ArrayRef<std::string> optionalAttrNames) {
  os << "  SmallVector<Value, 0> parsedOperands;\n";
  if (opDef.getNumOperands() == 0)
    return;

  if (opDef.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    os << "  // --- Deserialize Operands (AttrSizedOperandSegments) ---\n";
    os << "  SmallVector<int32_t, 4> readSegmentSizes;\n";

    std::string opName = opDef.getOperationName();
    size_t operandBitIndex = optionalAttrNames.size();

    for (unsigned i = 0; i < static_cast<unsigned>(opDef.getNumOperands());
         ++i) {
      const auto &odsOperand = opDef.getOperand(i);
      std::string odsOperandName = odsOperand.name.str();
      bool isOptional = odsOperand.isOptional();
      bool isVariadic = odsOperand.isVariableLength();

      // Make variable names unique within the generated function by embedding
      // the index 'i'.
      os << "  // Parsing ODS Operand Segment: " << odsOperandName << "\n"
         << "  int32_t currentSegmentLengthOds_" << i << ";\n";

      if (isOptional) {
        os << llvm::formatv(optionalOdsOperandSegmentTemplate, odsOperandName,
                            opName, i, operandBitIndex);
        operandBitIndex++;
      } else if (isVariadic) {
        os << llvm::formatv(variadicOdsOperandSegmentTemplate, odsOperandName,
                            opName, i);
      } else {
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
static void generateAttributeDeserialization(const Operator &op,
                                             raw_ostream &os) {
  os << R"(  // --- Deserialize Attributes ---
  SmallVector<NamedAttribute> attributes;)";

  int bitIndex = 0;
  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    std::string attrName = namedAttr.name.str();
    std::string varName = "parsed_" + attrName;
    std::string baseCppTypeStr = namedAttr.attr.getStorageType().str();
    bool isOptional = namedAttr.attr.isOptional();
    bool isUnitAttr = StringRef(baseCppTypeStr).contains("UnitAttr");

    // Declare the attribute variable
    os << llvm::formatv(R"(  {0} {1};)", baseCppTypeStr, varName);

    // Determine expectedType for parseOpAttribute
    bool isElements = StringRef(baseCppTypeStr).contains("ElementsAttr");
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
    // Generate parsing logic.
    if (isOptional) {
      // Optional attribute - check flags field.
      os << "  if (flags & (1ULL << " << bitIndex << ")) {\n";
      if (isUnitAttr)
        os << llvm::formatv(R"(    {0} = UnitAttr::get(&context);)", varName);
      else
        os << llvm::formatv(optionalAttrParseTemplate, varName, baseCppTypeStr,
                            expectedTypeArg, attrName);
      os << "  }\n";
      bitIndex++;
    } else {
      // Required attribute - read directly.
      os << llvm::formatv(requiredAttrParseTemplate, varName, expectedTypeArg,
                          attrName);
    }

    // Generate attribute addition to the attributes vector.
    os << formatv(R"(  if ({0}) {{
    attributes.emplace_back(innerBuilder.getStringAttr("{1}"), {2});})",
                  varName, attrName, varName);
  }
  os << "\n";
}

/// Generates C++ code to deserialize the result types of the operation.
static void generateResultTypeDeserialization(const Operator &op,
                                              raw_ostream &os) {
  std::string opClassName = op.getCppClassName().str();
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
  os << llvm::formatv(operationDeserializationTemplate, opName);
}

/// Generates the complete C++ function 'parse<OpName>'.
static void generateOpReader(const Operator &op, raw_ostream &os) {
  std::string opName = op.getOperationName();
  os << "// Reader for Op: " << opName << "\n";

  // Collect optional fields.
  SmallVector<std::string> optionalAttrNames;
  SmallVector<std::string> optionalOperandNames;
  collectOptionalFields(op, optionalAttrNames, optionalOperandNames);

  generateFunctionSignature(op, os);
  generateResultTypeDeserialization(op, os);
  generateFlagsFieldDeserialization(op, os, optionalAttrNames,
                                    optionalOperandNames);
  generateAttributeDeserialization(op, os);
  generateOperandDeserialization(op, os, optionalAttrNames);
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
    std::string opClassName = op.getCppClassName().str();
    StringRef enumName = opClassName;
    os << llvm::formatv(dispatchCaseTemplate, enumName, opClassName);
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

/// Register the generator.
static mlir::GenRegistration genCudaTileBytecodeReader(
    "gen-cuda-tile-bytecode-reader",
    "Generate cuda_tile bytecode reader implementations.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return generateBytecodeReader(records, os);
    });
