//===- BytecodeTypeCodeGen.cpp ----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BytecodeTypeCodeGen.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Code Generation Templates.
//===----------------------------------------------------------------------===//

/// Template for integer type serializer function.
/// {0}: Integer type checks.
static const char *const integerSerializerTemplate = R"(
// Auto-generated integer type serialization with version checking
LogicalResult serializeIntegerType(IntegerType type, EncodingWriter &writer,
                                   const BytecodeWriterConfig &config) {{
  unsigned width = type.getWidth();
{0}
  return emitError(UnknownLoc::get(type.getContext()), "unsupported integer type width");
}
)";

/// Template for float type serializer function.
/// {0}: Float type checks.
static const char *const floatSerializerTemplate = R"(
// Auto-generated float type serialization with version checking
LogicalResult serializeFloatType(FloatType type, EncodingWriter &writer,
                                 const BytecodeWriterConfig &config) {{
{0}
  return emitError(UnknownLoc::get(type.getContext()), "unsupported float type: ") << type;
}
)";

/// Template for CudaTile type serializer function signature.
/// {0}: Type name, {1}: Qualified type name.
static const char *const cudaTileSerializerSignatureTemplate = R"(
// Writer for Type: {0}
LogicalResult serialize{0}({1} type,
                                   EncodingWriter &writer,
                                   const BytecodeWriterConfig &config) {{
)";

//===----------------------------------------------------------------------===//
// Parameter Serialization/Deserialization Templates.
//===----------------------------------------------------------------------===//

/// {0}: Getter call.
static const char *const serializeArrayTemplate =
    "  writer.writeLEVarSize({0});\n";
static const char *const serializeDenseI32ArrayTemplate =
    "  writer.writeLEVarSize({0}.asArrayRef());\n";
static const char *const serializeTypeTemplate = R"(
  if (failed(writeTypeIndex({0}, writer)))
    return failure();
)";
/// {0}: Getter call, {1}: Enum type
static const char *const serializeOptionalEnumTemplate = R"(
  writer.writeByte({0} != nullptr);
  if ({0})
    writer.writeVarInt(static_cast<std::underlying_type_t<{1}>>({0}.getValue()));
)";

/// {0}: Variable name.
static const char *const deserializeInt64ArrayTemplate = R"(
  SmallVector<int64_t, 4> {0};
  if (failed(reader.readLEVarSize({0})))
    return reader.emitError() << "failed to read {0}";
)";
static const char *const deserializeInt32ArrayTemplate = R"(
  SmallVector<int32_t, 4> {0};
  if (failed(reader.readLEVarSize({0})))
    return reader.emitError() << "failed to read {0}";
)";
static const char *const deserializeDenseI32ArrayTemplate = R"(
  SmallVector<int32_t, 4> {0}_data;
  if (failed(reader.readLEVarSize({0}_data)))
    return reader.emitError() << "failed to read {0} data";
  auto {0} = DenseI32ArrayAttr::get(&context, {0}_data);
)";
static const char *const deserializeGenericTypeTemplate = R"(
  Type {0} = readAndGetType(reader);
  if (!{0})
    return reader.emitError() << "failed to get {0} type";
)";

/// {0}: Variable name, {1}: C++ type
static const char *const deserializeSpecificTypeTemplate = R"(
  Type {0}_generic = readAndGetType(reader);
  if (!{0}_generic)
    return reader.emitError() << "failed to get {0} type";
  auto {0} = ::mlir::dyn_cast<{1}>({0}_generic);
  if (!{0})
    return reader.emitError() << "expected {1} but got " << {0}_generic;
)";
static const char *const deserializeOptionalEnumTemplate = R"(
  {1} {0};
  if (reader.readLE<uint8_t>())
    if (failed(parseGenericEnumAttr(reader, context, {0})))
      return failure();
)";

//===----------------------------------------------------------------------===//
// Helper Functions.
//===----------------------------------------------------------------------===//

/// Get parameters in serialization order.
static auto getSerializationOrder(const CudaTileType &type) {
  if (type.needsReverseOrder)
    return llvm::to_vector(llvm::reverse(type.parameters));
  return type.parameters;
}

/// Generate version check with proper indentation.
static std::string generateVersionCheck(unsigned indent, StringRef version,
                                        StringRef typeName) {
  auto dotPos = version.find('.');
  std::string indentStr(indent, ' ');
  return formatv("{0}auto requiredVersion = BytecodeVersion::fromVersion({1}, "
                 "{2}, 0);\n"
                 "{0}if (config.bytecodeVersion < *requiredVersion)\n"
                 "{0}  return emitError(UnknownLoc::get(type.getContext()),\n"
                 "{0}               \"type '{3}' requires bytecode version "
                 "{4}+, targeting \") << config.bytecodeVersion.toString();\n",
                 indentStr, version.substr(0, dotPos),
                 version.substr(dotPos + 1), typeName, version)
      .str();
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Tag Enum.
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateTypeTagEnum(const BytecodeTypeStructure &structure,
                                       raw_ostream &os) {
  emitSourceFileHeader("Generated TypeTag Enum", os);
  os << "/// FROZEN at current assignments for backward compatibility.\n"
     << "/// WARNING: NEVER CHANGE THESE VALUES - they must remain stable.\n"
     << "enum class TypeTag : uint8_t {\n";
  // Generate all type tags.
  for (const auto &[enumName, tagValue] : structure.allTypeTags)
    os << "  " << enumName << " = " << tagValue << ",\n";
  os << "};\n";
}

//===----------------------------------------------------------------------===//
// C++ Generator - Parameter Serialization.
//===----------------------------------------------------------------------===//

static void generateParameterSerialization(const BytecodeTypeParameter &param,
                                           raw_ostream &os) {
  std::string getterCall = "type." + param.accessorName + "()";

  switch (param.kind) {
  case BytecodeTypeParameter::Kind::Int64Array:
  case BytecodeTypeParameter::Kind::Int32Array:
    os << formatv(serializeArrayTemplate, getterCall);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    os << formatv(serializeDenseI32ArrayTemplate, getterCall);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
  case BytecodeTypeParameter::Kind::SpecificType:
    os << formatv(serializeTypeTemplate, getterCall);
    break;
  case BytecodeTypeParameter::Kind::OptionalEnum:
    os << formatv(serializeOptionalEnumTemplate, getterCall,
                  param.enumTypeName);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Parameter Deserialization.
//===----------------------------------------------------------------------===//

static void generateParameterDeserialization(const BytecodeTypeParameter &param,
                                             raw_ostream &os) {
  switch (param.kind) {
  case BytecodeTypeParameter::Kind::Int64Array:
    os << formatv(deserializeInt64ArrayTemplate, param.name);
    break;
  case BytecodeTypeParameter::Kind::Int32Array:
    os << formatv(deserializeInt32ArrayTemplate, param.name);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    os << formatv(deserializeDenseI32ArrayTemplate, param.name);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
    os << formatv(deserializeGenericTypeTemplate, param.name);
    break;
  case BytecodeTypeParameter::Kind::SpecificType:
    os << formatv(deserializeSpecificTypeTemplate, param.name, param.cppType);
    break;
  case BytecodeTypeParameter::Kind::OptionalEnum:
    os << formatv(deserializeOptionalEnumTemplate, param.name, param.cppType);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Built-in Type Serializers.
//===----------------------------------------------------------------------===//

/// Generate serializers for all built-in types.
static void
generateBuiltinTypeSerializers(const BytecodeTypeStructure &structure,
                               raw_ostream &os) {
  std::string intChecks, floatChecks;

  for (const auto &bt : structure.builtinSerializableTypes) {
    std::string condition =
        bt.isInteger() ? formatv("width == {0}", bt.integerBitWidth).str()
                       : formatv("isa<{0}>(type)", bt.floatMlirTypeName).str();
    std::string check =
        formatv(R"(
  if ({0}) {{
{1}
    writer.writeVarInt(Bytecode::TypeTag::{2});
    return success();
  })",
                condition,
                generateVersionCheck(4, bt.sinceVersion, bt.enumName),
                bt.enumName)
            .str();
    if (bt.isInteger())
      intChecks += check;
    if (bt.isFloat())
      floatChecks += check;
  }
  if (!intChecks.empty())
    os << formatv(integerSerializerTemplate, intChecks);
  if (!floatChecks.empty())
    os << formatv(floatSerializerTemplate, floatChecks);
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Serializers.
//===----------------------------------------------------------------------===//

static void generateCudaTileTypeSerializer(const CudaTileType &type,
                                           raw_ostream &os) {
  // Function signature
  os << formatv(cudaTileSerializerSignatureTemplate, type.typeName,
                type.qualifiedTypeName);

  // Version checking.
  os << generateVersionCheck(2, type.sinceVersion, type.typeName);
  // Write type tag.
  os << "  writer.writeVarInt(Bytecode::TypeTag::" << type.typeName << ");\n";
  // Serialize parameters.
  for (const auto &param : getSerializationOrder(type))
    generateParameterSerialization(param, os);
  os << "  return success();\n}\n\n";
}

void mlir::tblgen::generateTypeSerializers(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Serialization Functions", os);
  generateBuiltinTypeSerializers(structure, os);
  for (const auto &type : structure.cudaTileTypes)
    generateCudaTileTypeSerializer(type, os);
}

//===----------------------------------------------------------------------===//
// C++ Generator - Built-in Type Deserializers.
//===----------------------------------------------------------------------===//

/// Generate deserializers for all built-in types.
static void
generateBuiltinTypeDeserializers(const BytecodeTypeStructure &structure,
                                 raw_ostream &os) {
  std::string intChecks, floatChecks;

  for (const auto &bt : structure.builtinSerializableTypes) {
    std::string typeCreation =
        bt.isInteger()
            ? formatv("IntegerType::get(&context, {0})", bt.integerBitWidth)
                  .str()
            : formatv("{0}::get(&context)", bt.floatMlirTypeName).str();
    std::string check = formatv(R"(
  if (typeTag == {0}) {{  
    result = {1};
    return success();
  })",
                                bt.typeTagValue, typeCreation)
                            .str();

    if (bt.isInteger())
      intChecks += check;
    if (bt.isFloat())
      floatChecks += check;
  }

  if (!intChecks.empty())
    os << formatv(R"(// Auto-generated integer type deserialization
LogicalResult parseIntegerType(uint8_t typeTag, Type &result, MLIRContext &context) {{
{0}  
  return ::emitError(UnknownLoc::get(&context)) << "invalid integer type tag: " << static_cast<int>(typeTag);
})",
                  intChecks);

  if (!floatChecks.empty())
    os << formatv(R"(// Auto-generated float type deserialization
LogicalResult parseFloatType(uint8_t typeTag, Type &result, MLIRContext &context) {{
{0}  
  return ::emitError(UnknownLoc::get(&context)) << "unsupported float type tag: " << static_cast<int>(typeTag);
})",
                  floatChecks);
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Deserializers.
//===----------------------------------------------------------------------===//

static void generateCudaTileTypeDeserializer(const CudaTileType &type,
                                             raw_ostream &os) {
  os << formatv(R"(// Reader for Type: {0}
LogicalResult parse{0}(EncodingReader &reader, Type &result) {{
)",
                type.typeName);

  // Deserialize parameters.
  for (const auto &param : getSerializationOrder(type))
    generateParameterDeserialization(param, os);

  // Build constructor arguments.
  std::string args;
  for (const auto &param : type.parameters) {
    StringRef paramTypeRef(param.cppStorageType);
    if (paramTypeRef.contains("SmallVector<int64_t>"))
      args += ", ArrayRef<int64_t>(" + param.name + ")";
    else if (paramTypeRef.contains("SmallVector<int32_t>"))
      args += ", ArrayRef<int32_t>(" + param.name + ")";
    else
      args += ", " + param.name;
  }

  os << formatv(R"(  result = {0}::getChecked(
      [&]() {{ return reader.emitError(); }, &context{1});
  return success(result);
}

)",
                type.qualifiedTypeName, args);
}

void mlir::tblgen::generateTypeDeserializers(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Deserialization Functions", os);
  generateBuiltinTypeDeserializers(structure, os);
  for (const auto &type : structure.cudaTileTypes)
    generateCudaTileTypeDeserializer(type, os);
}

//===----------------------------------------------------------------------===//
// C++ Generator - Dispatch.
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateSerializerDispatch(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Serialization Dispatch", os);

  os << R"(return TypeSwitch<Type, LogicalResult>(type)
)";

  // Built-in types.
  if (llvm::any_of(structure.builtinSerializableTypes,
                   [](auto &t) { return t.isInteger(); }))
    os << R"(    .Case<IntegerType>([&](auto concreteType) {
      return serializeIntegerType(concreteType, writer, config);
    })
)";

  if (llvm::any_of(structure.builtinSerializableTypes,
                   [](auto &t) { return t.isFloat(); }))
    os << R"(    .Case<FloatType>([&](auto concreteType) {
      return serializeFloatType(concreteType, writer, config);
    })
)";

  // CudaTile types.
  for (const auto &type : structure.cudaTileTypes)
    os << formatv(R"(    .Case<{0}>([&](auto concreteType) {{
      return serialize{1}(concreteType, writer, config);
    })
)",
                  type.qualifiedTypeName, type.typeName);

  // FunctionType and default case.
  os << R"(    .Case<FunctionType>([&](auto concreteType) {
      return serializeFunctionType(concreteType, writer);
    })
    .Default([&](Type) {
      return emitError(UnknownLoc::get(type.getContext()),
                       "unsupported type in bytecode writer");
    });
)";
}

void mlir::tblgen::generateDeserializerDispatch(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Deserialization Dispatch", os);

  os << R"(switch (static_cast<TypeTag>(typeTag)) {
)";

  // Built-in types.
  for (const auto &builtinType : structure.builtinSerializableTypes) {
    os << "case Bytecode::TypeTag::" << builtinType.enumName << ":\n";
    if (builtinType.isInteger())
      os << "  return parseIntegerType(typeTag, result, context);\n";
    else if (builtinType.isFloat())
      os << "  return parseFloatType(typeTag, result, context);\n";
  }

  // CudaTile types.
  for (const auto &type : structure.cudaTileTypes)
    os << formatv(R"(case Bytecode::TypeTag::{0}:
  return parse{0}(reader, result);
)",
                  type.typeName);

  // FunctionType and default.
  os << R"(case Bytecode::TypeTag::FunctionType:
  return parseFunctionType(reader, result);
default:
  return ::emitError(UnknownLoc::get(&context))
         << "unknown type tag: " << static_cast<int>(typeTag);
}
)";
}
