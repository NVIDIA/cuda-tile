//===- BytecodeTypeCodeGen.cpp ----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BytecodeTypeCodeGen.h"

#include "mlir/Support/IndentedOstream.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "BytecodeGenUtilities.h"

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
    "writer.writeLEVarSize({0});\n";
static const char *const serializeDenseI32ArrayTemplate =
    "writer.writeLEVarSize({0}.asArrayRef());\n";
static const char *const serializeTypeTemplate = R"(
if (failed(writeTypeIndex({0}, writer)))
  return failure();
)";
static const char *const serializeTypeAttrTemplate = R"(
if (failed(writeTypeIndex({0}.getValue(), writer)))
  return failure();
)";
static const char *const serializeNestedAttrTemplate = R"(
if (failed(writeSelfContainedTypeAttribute({0}, writer)))
  return failure();
)";
/// Polymorphic nested-attr serialization template (self-contained encoding,
/// used when the param's declared C++ type is an AttrInterface).
/// {0}: getter call.
static const char *const serializePolymorphicNestedAttrTemplate = R"(
if (failed(writeSelfContainedTypeAttribute({0}, writer)))
  return failure();
)";
/// Template for AttributeArray serialization on the type side.
/// {0}: Getter call (size source), {1}: range expression to iterate.
static const char *const serializeAttrArrayTemplate = R"(
writer.writeVarInt({0}.size());
for (Attribute elementAttr : {1}) {{
  if (failed(writeSelfContainedTypeAttribute(elementAttr, writer)))
    return failure();
}
)";
/// Runtime template for OptionalEnum (includes both formats). A case can be
/// introduced later than the parameter holding it, so the value is checked
/// against the target version too: writing it as a bare integer would leave an
/// older reader to symbolize a case it does not know.
/// {0}: Getter call, {1}: Enum type, {2}: Runtime flag variable,
/// {3}: Parameter name.
static const char *const serializeOptionalEnumTemplate = R"(
if ({0} && !isEnumValueAvailableInVersion<{1}>({0}.getValue(),
                                               config.getBytecodeVersion()))
  return ::emitError(UnknownLoc::get(type.getContext()))
         << "enum value for type parameter '{3}' is not available in bytecode "
         << "version " << config.getBytecodeVersion().toString();
if ({2}) {{
  if ({0})
    writer.writeVarInt(static_cast<std::underlying_type_t<{1}>>({0}.getValue()));
} else {{
  writer.writeByte({0} != nullptr);
  if ({0})
    writer.writeVarInt(static_cast<std::underlying_type_t<{1}>>({0}.getValue()));
}
)";
/// Templates for deserialization of original parameters.
/// {0}: Variable name.
static const char *const readArrayTemplate = R"(
if (failed(reader.readLEVarSize({0})))
 return reader.emitError() << "failed to read {0}";
)";

/// Template for a required enum parameter serialization. The enum is written
/// as its underlying integer value, once the case is known to exist in the
/// targeted version.
/// {0}: Getter call, {1}: Enum type, {2}: Parameter name.
static const char *const serializeEnumTemplate = R"(
if (!isEnumValueAvailableInVersion<{1}>({0}, config.getBytecodeVersion()))
  return ::emitError(UnknownLoc::get(type.getContext()))
         << "enum value for type parameter '{2}' is not available in bytecode "
         << "version " << config.getBytecodeVersion().toString();
writer.writeVarInt(static_cast<uint64_t>({0}));
)";

/// Template for a required enum parameter deserialization. The integer value is
/// read back and validated against the enum's symbolizer and against the file's
/// version, since a case can postdate the parameter carrying it.
/// {0}: Variable name, {1}: enum C++ type, {2}: symbolizer function.
static const char *const readEnumTemplate = R"(
uint64_t {0}_raw;
if (failed(reader.readVarInt({0}_raw)))
  return reader.emitError() << "failed to read {0}";
auto {0}_opt = {2}(static_cast<uint32_t>({0}_raw));
if (!{0}_opt)
  return reader.emitError() << "invalid {1} enum value " << {0}_raw;
if (!isEnumValueAvailableInVersion<{1}>(*{0}_opt, fileVersion))
  return reader.emitError() << "enum value " << {0}_raw
                            << " is not available in bytecode version "
                            << fileVersion.toString();
{0} = *{0}_opt;
)";

/// Template for uint32_t scalar serialization.
/// {0}: Getter call.
static const char *const serializeUInt32ScalarTemplate =
    "writer.writeVarInt(static_cast<uint64_t>({0}));\n";

/// Template for uint32_t scalar deserialization.
/// {0}: Variable name.
static const char *const readUInt32ScalarTemplate = R"(
uint64_t {0}_raw;
if (failed(reader.readVarInt({0}_raw)))
  return reader.emitError() << "failed to read {0}";
if ({0}_raw > UINT32_MAX)
  return reader.emitError() << "{0} value " << {0}_raw << " exceeds uint32_t range";
{0} = static_cast<uint32_t>({0}_raw);
)";

/// {0}: Variable name.
static const char *const readDenseI32ArrayTemplate = R"(
SmallVector<int32_t, 4> {0}_data;
if (failed(reader.readLEVarSize({0}_data)))
 return reader.emitError() << "failed to read {0} data";
{0} = DenseI32ArrayAttr::get(&context, {0}_data);
)";
static const char *const readTypeAttrTemplate = R"(
Type {0}_type = readAndGetType(reader);
if (!{0}_type)
  return reader.emitError() << "failed to get {0} type";
{0} = TypeAttr::get({0}_type);
)";
static const char *const readNestedAttrTemplate = R"(
if (failed(readNestedAttribute(reader, {0})))
  return failure();
)";
/// Polymorphic nested-attr deserialization template. Reads a generic
/// Attribute via tag dispatch then narrows to the declared interface type.
/// {0}: variable name. {1}: declared C++ interface type.
static const char *const readPolymorphicNestedAttrTemplate = R"(
{
  Attribute {0}_self_contained;
  if (failed(readNestedAttribute(reader, {0}_self_contained)))
    return failure();
  {0} = ::llvm::dyn_cast<{1}>({0}_self_contained);
  if (!{0})
    return reader.emitError() << "expected attribute implementing '{1}' "
                              << "for parameter '{0}', got "
                              << {0}_self_contained;
}
)";
/// Template for AttributeArray deserialization on the type side.
/// Reads N self-contained attributes into an ArrayAttr stored in {0}.
/// {0}: Variable name.
static const char *const readAttrArrayTemplate = R"(
uint64_t {0}_size;
if (failed(reader.readVarInt({0}_size,
                             std::numeric_limits<uint32_t>::max() - 1)))
  return reader.emitError() << "failed to read {0} size";
SmallVector<Attribute, 4> {0}_storage;
{0}_storage.reserve({0}_size);
for (uint64_t i = 0; i < {0}_size; ++i) {{
  Attribute {0}_element;
  if (failed(readNestedAttribute(reader, {0}_element)))
    return reader.emitError() << "failed to read {0} element " << i;
  {0}_storage.push_back({0}_element);
}
{0} = ArrayAttr::get(&context, {0}_storage);
)";

/// {0}: Variable name.
static const char *const readGenericTypeTemplate = R"(
{0} = readAndGetType(reader);
if (!{0})
  return reader.emitError() << "failed to get {0} type";
)";

/// {0}: Variable name, {1}: C++ type.
static const char *const readSpecificTypeTemplate = R"(
Type {0}_generic = readAndGetType(reader);
if (!{0}_generic)
  return reader.emitError() << "failed to get {0} type";
{0} = ::mlir::dyn_cast<{1}>({0}_generic);
if (!{0})
  return reader.emitError() << "expected {1} but got " << {0}_generic;
)";
/// Runtime template for OptionalEnum deserialization (includes both formats).
/// The helper rejects a case that postdates the file's own version, so a forged
/// buffer cannot smuggle one past a reader that happens to know it.
/// {0}: Variable name, {1}: Runtime flag variable
static const char *const readOptionalEnumTemplate = R"(
if ({1}) {{
  if (failed(parseGenericEnumAttr(reader, context, fileVersion, {0})))
    return failure();
} else {{
  uint8_t presence;
  if (failed(reader.readLE(presence)))
    return reader.emitError() << "failed to read OptionalEnum presence flag";
  if (presence)
    if (failed(parseGenericEnumAttr(reader, context, fileVersion, {0})))
      return failure();
}
)";

//===----------------------------------------------------------------------===//
// Helper Functions.
//===----------------------------------------------------------------------===//

/// Get parameters in serialization order.
static SmallVector<BytecodeTypeParameter, 4>
getSerializationOrder(const CudaTileType &type) {
  if (type.needsReverseOrder) {
    return llvm::to_vector<4>(llvm::reverse(type.parameters));
  }
  return type.parameters;
}

/// Derive the MLIR-generated symbolizer function for an enum from its fully
/// qualified C++ type. E.g. "::mlir::cuda_tile::MemorySpace" yields
/// "::mlir::cuda_tile::symbolizeMemorySpace".
static std::string enumSymbolizer(StringRef enumCppType) {
  auto [ns, name] = enumCppType.rsplit("::");
  if (name.empty()) {
    // No namespace qualifier; symbolize the bare name.
    return ("symbolize" + enumCppType).str();
  }
  return (ns + "::symbolize" + name).str();
}

static bool needsAttributeTypeRegistration(BytecodeTypeParameter::Kind kind) {
  return kind == BytecodeTypeParameter::Kind::TypeAttr ||
         kind == BytecodeTypeParameter::Kind::NestedAttr ||
         kind == BytecodeTypeParameter::Kind::PolymorphicNestedAttr ||
         kind == BytecodeTypeParameter::Kind::AttributeArray;
}

/// Returns true if the parameter holds an attribute whose own bytecode version
/// must be scored when computing a type's recursive minimum version.
static bool isAttributeVersionParameter(BytecodeTypeParameter::Kind kind) {
  return kind == BytecodeTypeParameter::Kind::OptionalEnum ||
         kind == BytecodeTypeParameter::Kind::NestedAttr ||
         kind == BytecodeTypeParameter::Kind::PolymorphicNestedAttr ||
         kind == BytecodeTypeParameter::Kind::AttributeArray;
}

static bool needsUnifiedBitfieldSerializationFlag(const CudaTileType &type) {
  return llvm::any_of(
      getSerializationOrder(type), [](const BytecodeTypeParameter &param) {
        return param.kind == BytecodeTypeParameter::Kind::OptionalEnum;
      });
}

/// Check if version string is >= 13.3 (unified bitfield format).
/// Types introduced at 13.3+ always use unified format.
static bool isUnifiedBitfieldVersion(StringRef version) {
  auto [majorStr, minorStr] = parseVersion(version);
  unsigned major, minor;
  if (majorStr.getAsInteger(10, major) || minorStr.getAsInteger(10, minor)) {
    PrintFatalError("invalid version string: " + version.str());
  }
  return (major > 13) || (major == 13 && minor >= 3);
}

/// Generate version check with proper indentation.
static std::string
generateVersionCheck(unsigned indent, StringRef version, StringRef typeName,
                     StringRef versionVar = "config.getBytecodeVersion()",
                     StringRef contextExpr = "type.getContext()",
                     StringRef errorSuffix = "targeting ") {
  auto [majorStr, minorStr] = parseVersion(version);
  std::string indentStr(indent, ' ');
  return formatv("{0}if ({1} < *BytecodeVersion::fromVersion({2}, {3}, 0))\n"
                 "{0}  return ::emitError(UnknownLoc::get({4}))\n"
                 "{0}         << \"type '{5}' requires bytecode version {6}+, "
                 "{7}\" << {1}.toString();\n",
                 indentStr, versionVar, majorStr, minorStr, contextExpr,
                 typeName, version, errorSuffix)
      .str();
}

/// Returns true if the provided parameter of the type was added after the
/// type's original version.
static bool isEvolvedTypeParam(const CudaTileType &type,
                               const BytecodeTypeParameter &param) {
  return !type.skipVersionCheck && !param.isPreviewExtension &&
         param.sinceVersion != type.sinceVersion;
}

/// Generates the opening of a version check. The content of the generated if
/// statement will be ran if the bytecode version is higher or equal to the
/// provided version.
static void generateOpenBytecodeVersionCondition(StringRef version,
                                                 StringRef versionVarName,
                                                 raw_ostream &os) {
  auto [majorStr, minorStr] = parseVersion(version);
  os << formatv(R"(
  if ({2} >= *BytecodeVersion::fromVersion({0}, {1}, 0)) {{
)",
                majorStr, minorStr, versionVarName);
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
  for (const auto &[enumName, tagValue] : structure.allTypeTags) {
    os << "  " << enumName << " = " << tagValue << ",\n";
  }
  os << "};\n";
}

static std::string
resolvePreviewExtensionFallback(const CudaTileType &type,
                                const BytecodeTypeParameter &param,
                                StringRef typeVar, StringRef contextExpr) {
  SmallVector<ParamSubst> paramSubsts;
  for (const auto &sibling : type.parameters)
    paramSubsts.emplace_back(
        sibling.name,
        StringRef(formatv("{0}.{1}()", typeVar, sibling.accessorName)));
  return replaceTableGenPlaceholders(param.bytecodeFallbackValue, contextExpr,
                                     paramSubsts);
}

//===----------------------------------------------------------------------===//
// Code Generation: Public-Preview Type Classification
//===----------------------------------------------------------------------===//

void mlir::tblgen::generatePreviewTypeClassification(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  os << R"(
namespace mlir::cuda_tile::detail {

/// Return the feature owning a public-preview type tag.
inline std::optional<PreviewFeature> getPreviewFeatureForTypeTag(uint64_t tag) {
  if (tag > 0xFF)
    return std::nullopt;
  switch (static_cast<Bytecode::TypeTag>(tag)) {
)";

  for (const auto &type : structure.cudaTileTypes) {
    if (!type.isPreview)
      continue;
    os << "  case Bytecode::TypeTag::" << type.typeName << ":\n"
       << "    return PreviewFeature::" << type.owningPreviewFeature << ";\n";
  }
  // Preview builtin singletons.
  for (const auto &bt : structure.builtinSerializableTypes) {
    if (!bt.isPreview)
      continue;
    os << "  case Bytecode::TypeTag::" << bt.enumName << ":\n"
       << "    return PreviewFeature::" << bt.owningPreviewFeature << ";\n";
  }

  os << R"(  default:
    return std::nullopt;
  }
}

inline bool isPreviewTypeTag(uint64_t tag) {
  return getPreviewFeatureForTypeTag(tag).has_value();
}

} // namespace mlir::cuda_tile::detail
)";
}

void mlir::tblgen::generatePreviewTypeValueClassification(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  os << R"(
namespace mlir::cuda_tile::detail {

/// Return the feature owning a public-preview type.
inline std::optional<PreviewFeature> getPreviewFeature(Type type) {
  return llvm::TypeSwitch<Type, std::optional<PreviewFeature>>(type)
)";

  // Preview builtin singletons. Only singletons expose a concrete cppType to
  // switch on here; a future non-singleton preview builtin would need its own
  // handling (and this guard keeps it from emitting `.Case<>` if one is added).
  for (const auto &bt : structure.builtinSerializableTypes) {
    if (!bt.isPreview || !bt.isSingleton())
      continue;
    os << "      .Case<" << bt.singletonCppType
       << ">([](auto) { return PreviewFeature::" << bt.owningPreviewFeature
       << "; })\n";
  }

  // Preview cuda tile types.
  for (const auto &type : structure.cudaTileTypes) {
    if (type.isPreview) {
      os << "      .Case<" << type.qualifiedTypeName
         << ">([](auto) { return PreviewFeature::" << type.owningPreviewFeature
         << "; })\n";
      continue;
    }
    auto extension = llvm::find_if(
        type.parameters,
        [](const BytecodeTypeParameter &p) { return p.isPreviewExtension; });
    if (extension == type.parameters.end())
      continue;
    std::string fallback = resolvePreviewExtensionFallback(
        type, *extension, "type", "type.getContext()");
    os << formatv(R"(      .Case<{0}>([]({0} type)
                                       -> std::optional<PreviewFeature> {{
        if (type.{1}() != {2})
          return PreviewFeature::{3};
        return std::nullopt;
      })
)",
                  type.qualifiedTypeName, extension->accessorName, fallback,
                  extension->owningPreviewFeature);
  }

  os << R"(      .Default([](Type) { return std::nullopt; });
}

} // namespace mlir::cuda_tile::detail
)";
}

//===----------------------------------------------------------------------===//
// C++ Generator - Parameter Serialization.
//===----------------------------------------------------------------------===//

static void generateParameterSerialization(const BytecodeTypeParameter &param,
                                           raw_ostream &os,
                                           StringRef indent = "",
                                           StringRef runtimeFlag = "") {
  std::string getterCall = "type." + param.accessorName + "()";
  std::string attributeArrayRange =
      (param.cppType.find("ArrayAttr") != std::string::npos ||
       param.cppStorageType.find("ArrayAttr") != std::string::npos)
          ? getterCall + ".getValue()"
          : getterCall;
  mlir::raw_indented_ostream ios(os);

  bool isOptionalEnum = param.kind == BytecodeTypeParameter::Kind::OptionalEnum;
  bool onlyIfNotNull = param.usesOptionalTypeFlags && !isOptionalEnum;

  // The enum type names the per-case version table to consult, so an
  // unresolved one would only surface as a compile error in the generated
  // serializer.
  if ((isOptionalEnum || param.kind == BytecodeTypeParameter::Kind::Enum) &&
      param.enumTypeName.empty()) {
    PrintFatalError("type parameter '" + param.name +
                    "' is an enum but its enum type could not be resolved");
  }

  std::string effectiveIndent = indent.str();

  if (onlyIfNotNull) {
    ios.printReindented(formatv("if ({0}) {{\n", getterCall).str(), indent);
    effectiveIndent.append("  ");
  }

  switch (param.kind) {
  case BytecodeTypeParameter::Kind::TypeAttr:
    ios.printReindented(formatv(serializeTypeAttrTemplate, getterCall).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::NestedAttr:
    ios.printReindented(formatv(serializeNestedAttrTemplate, getterCall).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::PolymorphicNestedAttr:
    ios.printReindented(
        formatv(serializePolymorphicNestedAttrTemplate, getterCall).str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::AttributeArray:
    ios.printReindented(
        formatv(serializeAttrArrayTemplate, getterCall, attributeArrayRange)
            .str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::Int64Array:
  case BytecodeTypeParameter::Kind::Int32Array:
    ios.printReindented(formatv(serializeArrayTemplate, getterCall).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::UInt32Scalar:
    ios.printReindented(
        formatv(serializeUInt32ScalarTemplate, getterCall).str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    ios.printReindented(
        formatv(serializeDenseI32ArrayTemplate, getterCall).str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
  case BytecodeTypeParameter::Kind::SpecificType:
    ios.printReindented(formatv(serializeTypeTemplate, getterCall).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::OptionalEnum:
    ios.printReindented(formatv(serializeOptionalEnumTemplate, getterCall,
                                param.enumTypeName, runtimeFlag, param.name)
                            .str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::Enum:
    ios.printReindented(formatv(serializeEnumTemplate, getterCall,
                                param.enumTypeName, param.name)
                            .str(),
                        indent);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }

  if (onlyIfNotNull) {
    effectiveIndent.resize(effectiveIndent.size() - 2);
    ios.printReindented("}\n", effectiveIndent);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Parameter Deserialization.
//===----------------------------------------------------------------------===//

static void
generateParameterVariableDeclaration(const BytecodeTypeParameter &param,
                                     raw_ostream &os, StringRef indent = "") {
  switch (param.kind) {
  case BytecodeTypeParameter::Kind::TypeAttr:
    os << indent << formatv("TypeAttr {0}", param.name);
    break;
  case BytecodeTypeParameter::Kind::NestedAttr:
  case BytecodeTypeParameter::Kind::PolymorphicNestedAttr:
  case BytecodeTypeParameter::Kind::SpecificType:
  case BytecodeTypeParameter::Kind::OptionalEnum:
  case BytecodeTypeParameter::Kind::Enum:
    os << indent << formatv("{0} {1}", param.cppType, param.name);
    break;
  case BytecodeTypeParameter::Kind::AttributeArray:
    os << indent << formatv("ArrayAttr {0}", param.name);
    break;
  case BytecodeTypeParameter::Kind::Int64Array:
    os << indent << formatv("SmallVector<int64_t, 4> {0}", param.name);
    break;
  case BytecodeTypeParameter::Kind::Int32Array:
    os << indent << formatv("SmallVector<int32_t, 4> {0}", param.name);
    break;
  case BytecodeTypeParameter::Kind::UInt32Scalar:
    os << indent << formatv("uint32_t {0}", param.name);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    os << indent << formatv("DenseI32ArrayAttr {0}", param.name);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
    os << indent << formatv("Type {0}", param.name);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }

  os << ";\n";
}

static void generateParameterDeserialization(const BytecodeTypeParameter &param,
                                             raw_ostream &os,
                                             StringRef indent = "",
                                             StringRef runtimeFlag = "") {
  mlir::raw_indented_ostream ios(os);

  std::string effectiveIndent = indent.str();

  if (param.usesOptionalTypeFlags) {
    ios.printReindented("if (!useUnifiedBitfield || (optionalFlags & (1ULL << "
                        "nextFlagToUseIndex))) {\n",
                        effectiveIndent);
    effectiveIndent.append("  ");
  }

  switch (param.kind) {
  case BytecodeTypeParameter::Kind::TypeAttr:
    ios.printReindented(formatv(readTypeAttrTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::NestedAttr:
    ios.printReindented(formatv(readNestedAttrTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::PolymorphicNestedAttr:
    ios.printReindented(
        formatv(readPolymorphicNestedAttrTemplate, param.name, param.cppType)
            .str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::AttributeArray:
    ios.printReindented(formatv(readAttrArrayTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::Int64Array:
  case BytecodeTypeParameter::Kind::Int32Array:
    ios.printReindented(formatv(readArrayTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::UInt32Scalar:
    ios.printReindented(formatv(readUInt32ScalarTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    ios.printReindented(formatv(readDenseI32ArrayTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
    ios.printReindented(formatv(readGenericTypeTemplate, param.name).str(),
                        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::SpecificType:
    ios.printReindented(
        formatv(readSpecificTypeTemplate, param.name, param.cppType).str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::OptionalEnum:
    ios.printReindented(
        formatv(readOptionalEnumTemplate, param.name, runtimeFlag).str(),
        effectiveIndent);
    break;
  case BytecodeTypeParameter::Kind::Enum:
    ios.printReindented(formatv(readEnumTemplate, param.name, param.cppType,
                                enumSymbolizer(param.cppType))
                            .str(),
                        indent);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }

  if (param.usesOptionalTypeFlags) {
    effectiveIndent.resize(effectiveIndent.size() - 2);
    ios.printReindented(R"(
      }
      nextFlagToUseIndex++;
      assert(nextFlagToUseIndex < 64);
)",
                        effectiveIndent);
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
    if (bt.isInteger()) {
      intChecks += check;
    }
    if (bt.isFloat()) {
      floatChecks += check;
    }
  }
  if (!intChecks.empty()) {
    os << formatv(integerSerializerTemplate, intChecks);
  }
  if (!floatChecks.empty()) {
    os << formatv(floatSerializerTemplate, floatChecks);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Optional Parameter Flags (Writer).
//===----------------------------------------------------------------------===//
//
// Optional Parameter Handling:
//
// Types can have optional parameters (OptionalParameter<T>) that may be null.
// These use a bitfield-based encoding for efficient serialization.
//
// Format Evolution:
//   Version <13.3 (Legacy):
//     - OptionalEnum: Uses inline flag byte per parameter
//     - Optional Type: Not supported as it was added in later versions.
//
//   Version ≥13.3 (Unified Bitfield):
//     - ALL optional params (Type, Enum, etc.): Single unified bitfield
//     - No inline flags - bitfield indicates presence
//
// Bitfield Layout:
//   - Bit 0: First optional param (in serialization order)
//   - Bit N: Nth optional param
//   - Bit set (1) = present, Bit clear (0) = null/absent
//   - Parameters added in a further version only occupy a bit when targeting
//     this version or newer.
//
// Version Checking:
//   Flags written if:
//     1. config.getBytecodeVersion() >= firstOptionalParamVersion (type supports it)
//     2. config.getBytecodeVersion() >= 13.3 (unified format exists)
//   This allows targeting older versions for backward compatibility.
//
//===----------------------------------------------------------------------===//

/// Generates flags field serialization for optional type parameters.
static void generateOptionalParamFlags(const CudaTileType &type,
                                       raw_ostream &os) {
  if (!type.hasOptionalTypeParams) {
    return;
  }

  if (needsUnifiedBitfieldSerializationFlag(type)) {
    if (type.skipVersionCheck) {
      os << R"(
  bool useUnifiedBitfield = true;
  (void)useUnifiedBitfield;
)";
    } else {
      os << R"(
  bool useUnifiedBitfield = config.getBytecodeVersion() >= BytecodeVersion::kUnifiedBitfieldVersion;
  (void)useUnifiedBitfield;
)";
    }
  }

  os << R"(  uint64_t optionalFlags = 0;

  // Index of the next flag to use for optional serialization, based on
  // bytecode version.
  uint64_t nextFlagToUseIndex = 0;

)";

  // Build flags for all optional params.
  for (const auto &param : getSerializationOrder(type)) {
    if (param.usesOptionalTypeFlags) {
      StringRef indent = "  ";
      bool evolved = isEvolvedTypeParam(type, param);
      if (evolved) {
        generateOpenBytecodeVersionCondition(param.sinceVersion,
                                             "config.getBytecodeVersion()", os);
        indent = "    ";
      }

      os << indent
         << formatv("if (type.{0}()) {{ optionalFlags |= (1ULL << "
                    "nextFlagToUseIndex); }\n",
                    param.accessorName);
      os << indent << "nextFlagToUseIndex++;\n";

      if (evolved) {
        os << "  }\n";
      }
    }
  }

  // Accumulates conditions that must be satisfied to actually write the
  // flags.
  SmallVector<std::string, 2> checks;

  if (!type.skipVersionCheck) {
    std::string minOptionalVersionStr = type.firstOptionalTypeParamVersion;
    bool needsVersionCheck = !minOptionalVersionStr.empty() &&
                             (minOptionalVersionStr != type.sinceVersion);

    // Only write the flags in versions where the type has at least one
    // optional parameter.
    if (needsVersionCheck) {
      auto [majorStr, minorStr] = parseVersion(minOptionalVersionStr);
      checks.push_back(formatv("config.getBytecodeVersion() >= "
                               "*BytecodeVersion::fromVersion({0}, {1}, 0)",
                               majorStr, minorStr));
    }

    // If the type was introduced before unified bitfields, only use flags if
    // the target version uses bitfields.
    bool typeIsUnified = isUnifiedBitfieldVersion(type.sinceVersion);
    if (!typeIsUnified) {
      checks.push_back("config.getBytecodeVersion() >= "
                       "BytecodeVersion::kUnifiedBitfieldVersion");
    }
  }

  // Generate the write.
  if (checks.empty()) {
    os << "  writer.writeVarInt(optionalFlags);\n";
  } else {
    os << "  if (" << llvm::join(checks, " &&\n      ") << ") {\n";
    os << "    writer.writeVarInt(optionalFlags);\n";
    os << "  }\n";
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Serializers.
//===----------------------------------------------------------------------===//
//
// Generates serialization function for each CudaTile type.
// For types with optional parameters, uses unified bitfield (13.3+) or
// legacy inline flags (<13.3). See "Optional Parameter Flags" section above.
//
//===----------------------------------------------------------------------===//

static void generateCudaTileTypeSerializer(const CudaTileType &type,
                                           raw_ostream &os) {
  // Function signature
  os << formatv(cudaTileSerializerSignatureTemplate, type.typeName,
                type.qualifiedTypeName);

  if (!type.skipVersionCheck) {
    os << generateVersionCheck(2, type.sinceVersion, type.typeName);
  }

  // Write type tag.
  os << "  writer.writeVarInt(Bytecode::TypeTag::" << type.typeName << ");\n";

  // Declare format flag and write flags only if type has optional params.
  generateOptionalParamFlags(type, os);

  // Serialize parameters conditionally based on target bytecode version.
  // The substitution map is built incrementally — each param is added after
  // processing so evolved params can reference all preceding params.
  SmallVector<ParamSubst, 4> writerParamSubsts;
  for (const auto &param : getSerializationOrder(type)) {
    if (param.isPreviewExtension) {
      std::string fallbackValue = replaceTableGenPlaceholders(
          param.bytecodeFallbackValue, "type.getContext()", writerParamSubsts);
      os << formatv(R"(  if (type.{0}() != {1}) {{
)",
                    param.accessorName, fallbackValue);
      generateParameterSerialization(param, os, "    ",
                                     "useUnifiedBitfield");
      os << "  }\n";
      writerParamSubsts.push_back(
          {param.name, "type." + param.accessorName + "()"});
      continue;
    }
    if (!isEvolvedTypeParam(type, param)) {
      generateParameterSerialization(param, os, "  ", "useUnifiedBitfield");
      writerParamSubsts.push_back(
          {param.name, "type." + param.accessorName + "()"});
      continue;
    }

    // Evolved parameters - version-guarded with validation.

    // Resolve effective bytecode fallback: bytecodeFallbackValue takes
    // priority, falling back to ODS defaultValue. The analysis guarantees
    // every evolved parameter has one or the other.
    StringRef fallbackExpr = param.bytecodeFallbackValue.empty()
                                 ? StringRef(param.defaultValue)
                                 : StringRef(param.bytecodeFallbackValue);
    if (fallbackExpr.empty()) {
      PrintFatalError("Evolved parameter '" + param.name + "' in type '" +
                      type.typeName +
                      "' is missing a bytecode fallback expression or "
                      "ODS default value.");
    }
    std::string fallbackValue = replaceTableGenPlaceholders(
        fallbackExpr, "type.getContext()", writerParamSubsts);

    generateOpenBytecodeVersionCondition(param.sinceVersion,
                                         "config.getBytecodeVersion()", os);

    generateParameterSerialization(param, os, "    ", "useUnifiedBitfield");

    // Validation when targeting an older version: the parameter must equal
    // the fallback value.
    std::string getterCall = "type." + param.accessorName + "()";
    std::string validationCheck =
        formatv("if ({0} != {1})", getterCall, fallbackValue).str();
    os << formatv(R"(  } else {{
      // Validate: parameter must equal the fallback value when targeting older version.
      {0}
      return ::emitError(UnknownLoc::get(type.getContext()))
             << "type parameter '{1}' requires bytecode version {2}+ "
             << "to differ from fallback or default value, "
             << "but targeting " << config.getBytecodeVersion().toString();
  }
)",
                  validationCheck, param.name, param.sinceVersion);

    writerParamSubsts.push_back(
        {param.name, "type." + param.accessorName + "()"});
  }

  os << "  return success();\n}\n\n";
}

void mlir::tblgen::generateTypeSerializers(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Serialization Functions", os);
  generateBuiltinTypeSerializers(structure, os);
  for (const auto &type : structure.cudaTileTypes) {
    generateCudaTileTypeSerializer(type, os);
  }
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

    // Generate version check.
    std::string versionCheck =
        generateVersionCheck(4, bt.sinceVersion, bt.enumName, "fileVersion",
                             "&context", "file version is ");

    std::string check = formatv(R"(
  if (typeTag == {0}) {{
{1}    result = {2};
    return success();
  })",
                                bt.typeTagValue, versionCheck, typeCreation)
                            .str();

    if (bt.isInteger()) {
      intChecks += check;
    }
    if (bt.isFloat()) {
      floatChecks += check;
    }
  }

  if (!intChecks.empty()) {
    os << formatv(R"(// Auto-generated integer type deserialization
LogicalResult parseIntegerType(uint8_t typeTag, Type &result, MLIRContext &context,
                               const BytecodeVersion &fileVersion) {{
{0}  
  return ::emitError(UnknownLoc::get(&context)) << "invalid integer type tag: " << static_cast<int>(typeTag);
})",
                  intChecks);
  }

  if (!floatChecks.empty()) {
    os << formatv(R"(// Auto-generated float type deserialization
LogicalResult parseFloatType(uint8_t typeTag, Type &result, MLIRContext &context,
                             const BytecodeVersion &fileVersion) {{
{0}  
  return ::emitError(UnknownLoc::get(&context)) << "unsupported float type tag: " << static_cast<int>(typeTag);
})",
                  floatChecks);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Optional Parameter Flags (Reader).
//===----------------------------------------------------------------------===//
//
// Reads the optional parameter bitfield written by the writer.
// See "Optional Parameter Flags (Writer)" section above for format details.
//
// Reading Strategy:
//   - Determine format from fileVersion (≥13.3 = unified, <13.3 = legacy)
//   - Read bitfield if: fileVersion >= firstOptionalParamVersion && unified
//   - For legacy format: OptionalEnum uses inline flags (read by templates)
//
//===----------------------------------------------------------------------===//

/// Generates flags field deserialization for optional type parameters.
static void generateOptionalParamFlagsReader(const CudaTileType &type,
                                             raw_ostream &os) {
  if (!type.hasOptionalTypeParams) {
    return;
  }

  if (type.skipVersionCheck) {
    os << "  bool useUnifiedBitfield = true;\n";
  } else {
    os << "  bool useUnifiedBitfield = fileVersion >= "
          "BytecodeVersion::kUnifiedBitfieldVersion;\n";
  }

  os << R"(
  (void)useUnifiedBitfield;
  uint64_t optionalFlags = 0;

  // Index of the next flag to use for optional deserialization, based on
  // bytecode version.
  uint64_t nextFlagToUseIndex = 0;
)";

  if (type.skipVersionCheck) {
    os << R"(
  if (failed(reader.readVarInt(optionalFlags)))
    return reader.emitError() << "failed to read optional parameter flags";
  )";
  } else {
    std::string minOptionalVersionStr = type.firstOptionalTypeParamVersion;
    auto [majorStr, minorStr] = parseVersion(minOptionalVersionStr);

    os << formatv(R"(
  if (fileVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0) && useUnifiedBitfield) {{
    if (failed(reader.readVarInt(optionalFlags)))
      return reader.emitError() << "failed to read optional parameter flags";
  }
)",
                  majorStr, minorStr);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Deserializers.
//===----------------------------------------------------------------------===//
//
// Generates deserialization function for each CudaTile type.
// Reads optional parameter bitfield (13.3+) or inline flags (<13.3).
// See "Optional Parameter Flags" sections above for format details.
//
//===----------------------------------------------------------------------===//

static void generateCudaTileTypeDeserializer(const CudaTileType &type,
                                             raw_ostream &os) {
  os << formatv(R"(// Reader for Type: {0}
LogicalResult parse{0}(EncodingReader &reader, Type &result) {{
)",
                type.typeName);

  // Version check for the type itself.
  if (!type.skipVersionCheck) {
    os << generateVersionCheck(2, type.sinceVersion, type.typeName,
                               "fileVersion", "&context", "file version is ");
  }

  // Declare format flag and read flags only if type has optional params.
  generateOptionalParamFlagsReader(type, os);

  // Deserialize parameters conditionally based on file bytecode version.
  // The substitution map is built incrementally — each param is added after
  // processing so evolved params can reference all preceding params.
  SmallVector<ParamSubst, 4> readerParamSubsts;
  for (const auto &param : getSerializationOrder(type)) {
    bool isEvolved = !type.skipVersionCheck && !param.isPreviewExtension &&
                     param.sinceVersion != type.sinceVersion;

    generateParameterVariableDeclaration(param, os, "  ");

    if (param.isPreviewExtension) {
      std::string fallbackValue = replaceTableGenPlaceholders(
          param.bytecodeFallbackValue, "&context", readerParamSubsts);
      os << R"(  if (reader.remaining() != 0) {
    if (!(fileVersion.getTag() &
          Bytecode::VersionTag::kContainsPreview))
      return reader.emitError()
             << "malformed bytecode: preview type extension requires the "
                "header to advertise preview content";
)";
      generateParameterDeserialization(param, os, "    ",
                                       "useUnifiedBitfield");
      os << formatv(R"(    if (reader.remaining() != 0)
      return reader.emitError()
             << "unexpected trailing data after preview extension parameter '{0}'";
  } else {
    {0} = {1};
  }
)",
                    param.name, fallbackValue);
      readerParamSubsts.push_back({param.name, param.name});
      continue;
    }

    // Original parameters - always deserialize.
    if (!isEvolved) {
      generateParameterDeserialization(param, os, "  ", "useUnifiedBitfield");
      readerParamSubsts.push_back({param.name, param.name});
      continue;
    }

    // Evolved parameters - version-guarded deserialization.
    // We first check whether the file version is more recent than the
    // parameter or not.
    generateOpenBytecodeVersionCondition(param.sinceVersion, "fileVersion", os);

    // If the version is recent enough, read normally.
    generateParameterDeserialization(param, os, "    ", "useUnifiedBitfield");

    os << R"(
  } else {
)";

    // Else, resolve effective bytecode fallback: either from
    // bytecodeFallbackValue or ODS default value, with the bytecode fallback
    // value taking priority. The analysis guarantees every evolved parameter
    // has one or the other.
    StringRef fallbackExpr = param.bytecodeFallbackValue.empty()
                                 ? StringRef(param.defaultValue)
                                 : StringRef(param.bytecodeFallbackValue);
    if (fallbackExpr.empty()) {
      PrintFatalError("Evolved parameter '" + param.name + "' in type '" +
                      type.typeName +
                      "' is missing a bytecode fallback expression or "
                      "ODS default value.");
    }
    std::string fallbackValue = replaceTableGenPlaceholders(
        fallbackExpr, "&context", readerParamSubsts);

    // Write fallback value for old bytecode versions.
    os << formatv("    {0} = {1};\n", param.name, fallbackValue);

    os << R"(
  }
)";

    readerParamSubsts.push_back({param.name, param.name});
  }

  if (type.hasOptionalTypeParams) {
    os << R"(
  if (nextFlagToUseIndex < 64 &&
      (optionalFlags & (~uint64_t(0) << nextFlagToUseIndex)))
    return reader.emitError()
           << "unused bits set in optional parameter flags";
)";
  }

  // Build constructor arguments.
  std::string args;
  for (const auto &param : type.parameters) {
    StringRef paramTypeRef(param.cppStorageType);
    if (paramTypeRef.contains("SmallVector<int64_t>")) {
      args += ", ArrayRef<int64_t>(" + param.name + ")";
    } else if (paramTypeRef.contains("SmallVector<int32_t>")) {
      args += ", ArrayRef<int32_t>(" + param.name + ")";
    } else {
      args += ", " + param.name;
    }
  }

  if (type.hasCheckedBuilder) {
    os << formatv(R"(  
  result = {0}::getChecked(
      [&]() {{ return reader.emitError(); }, &context{1});
  return success(result);
}
)",
                  type.qualifiedTypeName, args);
  } else {
    os << formatv(R"(  
  result = {0}::get(&context{1});
  return success();
}
)",
                  type.qualifiedTypeName, args);
  }
}

void mlir::tblgen::generateTypeDeserializers(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Deserialization Functions", os);
  generateBuiltinTypeDeserializers(structure, os);
  for (const auto &type : structure.cudaTileTypes) {
    generateCudaTileTypeDeserializer(type, os);
  }
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
                   [](auto &t) { return t.isInteger(); })) {
    os << R"(    .Case<IntegerType>([&](auto concreteType) {
      return serializeIntegerType(concreteType, writer, config);
    })
)";
  }

  if (llvm::any_of(structure.builtinSerializableTypes,
                   [](auto &t) { return t.isFloat(); })) {
    os << R"(    .Case<FloatType>([&](auto concreteType) {
      return serializeFloatType(concreteType, writer, config);
    })
)";
  }

  // Nullary preview builtins: just emit the tag; there are no
  // parameters, and preview content is version-locked by the writer.
  for (const auto &bt : structure.builtinSerializableTypes) {
    if (!bt.isSingleton())
      continue;
    os << formatv(R"(    .Case<{0}>([&](auto) {{
      writer.writeVarInt(Bytecode::TypeTag::{1});
      return success();
    })
)",
                  bt.singletonCppType, bt.enumName);
  }

  // CudaTile types.
  for (const auto &type : structure.cudaTileTypes) {
    os << formatv(R"(    .Case<{0}>([&](auto concreteType) {{
      return serialize{1}(concreteType, writer, config);
    })
)",
                  type.qualifiedTypeName, type.typeName);
  }

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

void mlir::tblgen::generateDependentTypeRegistration(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Dependent Type Registration", os);

  for (const auto &type : structure.cudaTileTypes) {
    // Check if type has nested type-bearing parameters that need
    // registration.
    if (!llvm::any_of(type.parameters, [](const BytecodeTypeParameter &p) {
          return isTypeParameter(p.kind) ||
                 needsAttributeTypeRegistration(p.kind);
        })) {
      continue;
    }

    // Generate registration for this type.
    os << formatv("if (auto concreteType = dyn_cast<{0}>(type)) {{\n",
                  type.qualifiedTypeName);
    for (const auto &param : type.parameters) {
      if (isTypeParameter(param.kind)) {
        os << formatv(R"(  if (auto paramType = concreteType.{0}())
    getTypeIndex(paramType);
)",
                      param.accessorName);
        continue;
      }

      if (needsAttributeTypeRegistration(param.kind)) {
        if (param.isPreviewExtension) {
          std::string fallback = resolvePreviewExtensionFallback(
              type, param, "concreteType", "concreteType.getContext()");
          os << formatv(
              R"(  if (concreteType.{0}() != {1})
    registerDependentAttributeTypes(concreteType.{0}());
)",
              param.accessorName, fallback);
          continue;
        }
        os << formatv(
            R"(  registerDependentAttributeTypes(concreteType.{0}());
)",
            param.accessorName);
      }
    }

    os << "  return;\n}\n";
  }
}

void mlir::tblgen::generateDeserializerDispatch(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Deserialization Dispatch", os);

  os << R"(switch (static_cast<TypeTag>(typeTag)) {
)";

  // Built-in types.
  for (const auto &builtinType : structure.builtinSerializableTypes) {
    os << "case Bytecode::TypeTag::" << builtinType.enumName << ":\n";
    if (builtinType.isInteger()) {
      os << "  return parseIntegerType(typeTag, result, context, "
            "fileVersion);\n";
    } else if (builtinType.isFloat()) {
      os << "  return parseFloatType(typeTag, result, context, "
            "fileVersion);\n";
    } else if (builtinType.isSingleton()) {
      // Nullary preview builtin: reconstruct directly.
      os << formatv("  result = {0}::get(&context);\n  return success();\n",
                    builtinType.singletonCppType);
    }
  }

  // CudaTile types.
  for (const auto &type : structure.cudaTileTypes) {
    os << formatv(R"(case Bytecode::TypeTag::{0}:
  return parse{0}(reader, result);
)",
                  type.typeName);
  }

  // FunctionType and default.
  os << R"(case Bytecode::TypeTag::FunctionType:
  return parseFunctionType(reader, result);
default:
  return ::emitError(UnknownLoc::get(&context))
         << "unknown type tag: " << static_cast<int>(typeTag);
}
)";
}

void mlir::tblgen::generateTypeVersionMap(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Version Map", os);

  // Build integer switch cases and float .Case<> entries.
  // Skip minimum version entries - they don't raise the min version.
  std::string integerCasesStr;
  std::string floatCasesStr;
  llvm::raw_string_ostream integerCases(integerCasesStr);
  llvm::raw_string_ostream floatCases(floatCasesStr);
  for (const auto &bt : structure.builtinSerializableTypes) {
    auto [majorStr, minorStr] = parseVersion(bt.sinceVersion);
    if (isMinimumVersion(majorStr, minorStr)) {
      continue;
    }

    if (bt.isInteger()) {
      integerCases << formatv(R"(        case {0}:
          return BytecodeVersion::fromVersion({1}, {2}, 0);
)",
                              bt.integerBitWidth, majorStr, minorStr);
    } else if (bt.isFloat()) {
      floatCases << formatv(R"(      .Case<mlir::{0}>([](auto) {{
        return BytecodeVersion::fromVersion({1}, {2}, 0);
      })
)",
                            bt.floatMlirTypeName, majorStr, minorStr);
    }
  }

  os << R"(namespace mlir::cuda_tile::detail {

/// Returns the minimum bytecode version required for an MLIR Type.
/// Returns nullopt for unsupported types or those at minimum version.
inline std::optional<BytecodeVersion>
getTypeMinVersion(mlir::Type type) {
  return llvm::TypeSwitch<Type, std::optional<BytecodeVersion>>(type)
)";

  // Only generate IntegerType case if we have non-minimum version integers.
  if (!integerCasesStr.empty()) {
    os << R"(      .Case<mlir::IntegerType>([](mlir::IntegerType intType)
                              -> std::optional<BytecodeVersion> {
        switch (intType.getWidth()) {
)" << integerCasesStr
       << R"(        default:
          return std::nullopt;
        }
      })
)";
  }

  // Only generate float cases if we have non-minimum version floats.
  if (!floatCasesStr.empty()) {
    os << floatCasesStr;
  }

  // Single pass: generate getTypeMinVersion cases and accumulate
  // getTypeMinVersionRecursive cases.
  std::string recursiveCasesStr;
  llvm::raw_string_ostream recursiveCases(recursiveCasesStr);

  for (const auto &type : structure.cudaTileTypes) {
    if (type.isPreview) {
      os << formatv(R"(      .Case<{0}>([](auto) {{
        return BytecodeVersion::kCurrentVersion;
      })
)",
                    type.qualifiedTypeName);
      bool hasRecursiveParam = llvm::any_of(
          type.parameters, [](const BytecodeTypeParameter &p) {
            return isTypeParameter(p.kind) ||
                   isAttributeVersionParameter(p.kind);
          });
      if (hasRecursiveParam) {
        recursiveCases << formatv(R"(  if (auto t = dyn_cast<{0}>(type)) {{
)",
                                    type.qualifiedTypeName);
        for (const auto &param : type.parameters) {
          if (isTypeParameter(param.kind)) {
            recursiveCases << formatv(
                R"(    if (auto paramVal = t.{0}())
      updateMax(getTypeMinVersionRecursive(paramVal));
)",
                param.accessorName);
          } else if (param.kind ==
                     BytecodeTypeParameter::Kind::AttributeArray) {
            recursiveCases << formatv(
                R"(    for (Attribute elementAttr : t.{0}())
      updateMax(getAttrMinVersionRecursive(elementAttr));
)",
                param.accessorName);
          } else if (isAttributeVersionParameter(param.kind)) {
            recursiveCases << formatv(
                R"(    if (auto paramVal = t.{0}())
      updateMax(getAttrMinVersionRecursive(paramVal));
)",
                param.accessorName);
          }
        }
        recursiveCases << "  }\n";
      }
      continue;
    }
    if (type.skipVersionCheck) {
      continue;
    }

    // For getTypeMinVersion: account for both the type's introduction and
    // evolved parameters whose values differ from their bytecode fallback.
    auto [majorStr, minorStr] = parseVersion(type.sinceVersion);
    bool hasEvolvedParams = llvm::any_of(
        type.parameters, [&](const BytecodeTypeParameter &param) {
          return isEvolvedTypeParam(type, param);
        });
    bool hasPreviewExtension = llvm::any_of(
        type.parameters, [](const BytecodeTypeParameter &param) {
          return param.isPreviewExtension;
        });
    if (!isMinimumVersion(majorStr, minorStr) || hasEvolvedParams ||
        hasPreviewExtension) {
      SmallVector<ParamSubst> fallbackParamSubsts;
      for (const auto &param : type.parameters) {
        fallbackParamSubsts.emplace_back(
            param.name, StringRef(formatv("t.{0}()", param.accessorName)));
      }

      os << formatv(R"(      .Case<{0}>([]({0} t) {{
        std::optional<BytecodeVersion> minVersion;
        auto updateMax = [&](std::optional<BytecodeVersion> version) {{
          if (version && (!minVersion || *version > *minVersion))
            minVersion = version;
        };
)",
                    type.qualifiedTypeName);

      if (!isMinimumVersion(majorStr, minorStr)) {
        os << formatv(
            "        updateMax(BytecodeVersion::fromVersion({0}, {1}, 0));\n",
            majorStr, minorStr);
      }

      for (const auto &param : type.parameters) {
        if (param.isPreviewExtension) {
          std::string fallbackValue = resolvePreviewExtensionFallback(
              type, param, "t", "t.getContext()");
          os << formatv(R"(        if (t.{0}() != {1})
          updateMax(BytecodeVersion::kCurrentVersion);
)",
                        param.accessorName, fallbackValue);
          continue;
        }
        if (!isEvolvedTypeParam(type, param))
          continue;

        StringRef fallbackExpr = param.bytecodeFallbackValue.empty()
                                     ? StringRef(param.defaultValue)
                                     : StringRef(param.bytecodeFallbackValue);
        if (fallbackExpr.empty()) {
          PrintFatalError("Evolved parameter '" + param.name + "' in type '" +
                          type.typeName +
                          "' is missing a bytecode fallback expression or "
                          "ODS default value.");
        }

        std::string fallbackValue = replaceTableGenPlaceholders(
            fallbackExpr, "t.getContext()", fallbackParamSubsts);
        auto [paramMajorStr, paramMinorStr] =
            parseVersion(param.sinceVersion);
        os << formatv(R"(        if (t.{0}() != {1})
          updateMax(BytecodeVersion::fromVersion({2}, {3}, 0));
)",
                      param.accessorName, fallbackValue, paramMajorStr,
                      paramMinorStr);
      }

      os << R"(        return minVersion;
      })
)";
    }

    // For getTypeMinVersionRecursive: descend into every parameter that can
    // carry its own bytecode version. Type parameters contribute their nested
    // type's version; attribute parameters contribute the parameter's own
    // `sinceVersion` plus the version of the attribute value they hold.
    bool hasRecursiveParam =
        llvm::any_of(type.parameters, [](const BytecodeTypeParameter &p) {
          return isTypeParameter(p.kind) || isAttributeVersionParameter(p.kind);
        });
    if (hasRecursiveParam) {
      // Substitutions for fallback expressions that reference sibling
      // parameters. The type is fully built here, so every accessor is valid
      // regardless of parameter order.
      SmallVector<ParamSubst> fallbackParamSubsts;
      for (const auto &param : type.parameters) {
        fallbackParamSubsts.emplace_back(
            param.name, StringRef(formatv("t.{0}()", param.accessorName)));
      }

      recursiveCases << formatv(R"(  if (auto t = dyn_cast<{0}>(type)) {{
)",
                                type.qualifiedTypeName);
      for (const auto &param : type.parameters) {
        bool isTypeParam = isTypeParameter(param.kind);
        bool isAttrParam = isAttributeVersionParameter(param.kind);
        if (!isTypeParam && !isAttrParam) {
          continue;
        }

        // An evolved parameter (introduced after its enclosing type) only
        // contributes when its value differs from the fallback the reader
        // would synthesize when targeting an older version. This mirrors the
        // writer-side gate in generateCudaTileTypeSerializer.
        std::string evolvedCheck;
        if (param.isPreviewExtension) {
          std::string fallbackValue = resolvePreviewExtensionFallback(
              type, param, "t", "t.getContext()");
          evolvedCheck = formatv("; paramVal != {0}", fallbackValue).str();
        } else if (!type.skipVersionCheck &&
                   param.sinceVersion != type.sinceVersion) {
          StringRef fallbackExpr = param.bytecodeFallbackValue.empty()
                                       ? StringRef(param.defaultValue)
                                       : StringRef(param.bytecodeFallbackValue);
          if (fallbackExpr.empty()) {
            PrintFatalError("Evolved parameter '" + param.name + "' in type '" +
                            type.typeName +
                            "' is missing a bytecode fallback expression or "
                            "ODS default value.");
          }
          std::string fallbackValue = replaceTableGenPlaceholders(
              fallbackExpr, "t.getContext()", fallbackParamSubsts);
          evolvedCheck = formatv("; paramVal != {0}", fallbackValue).str();
        }

        // The recursive contribution differs per kind: a type parameter
        // recurses into the nested type, an attribute-array parameter scores
        // each element, and any other attribute parameter scores the single
        // attribute value it holds.
        std::string recurseBody;
        if (isTypeParam) {
          recurseBody =
              "      updateMax(getTypeMinVersionRecursive(paramVal));\n";
        } else if (param.kind == BytecodeTypeParameter::Kind::AttributeArray) {
          recurseBody =
              "      for (Attribute elementAttr : paramVal)\n"
              "        updateMax(getAttrMinVersionRecursive(elementAttr));\n";
        } else {
          recurseBody =
              "      updateMax(getAttrMinVersionRecursive(paramVal));\n";
        }

        if (param.isPreviewExtension) {
          recursiveCases << formatv(
              R"(    if (auto paramVal = t.{0}(){1}) {{
      updateMax(BytecodeVersion::kCurrentVersion);
{2}    }
)",
              param.accessorName, evolvedCheck, recurseBody);
        } else {
          auto [majorStr, minorStr] = parseVersion(param.sinceVersion);
          recursiveCases << formatv(
              R"(    if (auto paramVal = t.{0}(){1}) {{
      updateMax(BytecodeVersion::fromVersion({2}, {3}, 0));
{4}    }
)",
              param.accessorName, evolvedCheck, majorStr, minorStr,
              recurseBody);
        }
      }
      recursiveCases << "  }\n";
    }
  }

  os << R"(      .Default([](Type) -> std::optional<BytecodeVersion> {
        return std::nullopt;
      });
}

/// Process nested types recursively and return the maximum version required.
/// This handles composite types like TileType, TensorViewType, etc., including
/// attributes embedded in a type parameter, whose version is obtained via
/// getAttrMinVersionRecursive.
inline std::optional<BytecodeVersion>
getTypeMinVersionRecursive(Type type) {
  std::optional<BytecodeVersion> maxVersion;

  auto updateMax = [&](std::optional<BytecodeVersion> v) {
    if (v && (!maxVersion || *v > *maxVersion))
      maxVersion = v;
  };

  // Get this type's version.
  updateMax(getTypeMinVersion(type));

  // FunctionType: has input and result types that need recursive processing.
  if (auto funcType = dyn_cast<FunctionType>(type)) {
    for (Type input : funcType.getInputs())
      updateMax(getTypeMinVersionRecursive(input));
    for (Type result : funcType.getResults())
      updateMax(getTypeMinVersionRecursive(result));
  }

  // Process CudaTile types with nested Type parameters.
)" << recursiveCasesStr
     << R"(
  return maxVersion;
}

} // namespace mlir::cuda_tile::detail
)";
}
