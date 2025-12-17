//===- BytecodeReader.cpp - CUDA Tile Bytecode Reader -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implements the BytecodeReader for the cuda_tile dialect, enabling
// deserialization of bytecode into a cuda_tile module.
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopeExit.h"

#include "../BytecodeEnums.h"
#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include <optional>

using namespace mlir;
using namespace mlir::cuda_tile;
using namespace mlir::cuda_tile::Bytecode;

//===----------------------------------------------------------------------===//
// Bytecode Header Utilities
//===----------------------------------------------------------------------===//

static const uint8_t kTileIRBytecodeMagic[8] = {
    0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00,
};

bool cuda_tile::isTileIRBytecode(llvm::MemoryBufferRef bytecodeBuffer) {
  // Check if the bytecode buffer starts with the expected magic number.
  if (bytecodeBuffer.getBufferSize() < sizeof(kTileIRBytecodeMagic))
    return false;
  return memcmp(bytecodeBuffer.getBufferStart(), kTileIRBytecodeMagic,
                sizeof(kTileIRBytecodeMagic)) == 0;
}
bool cuda_tile::isTileIRBytecode(const char *bytecodeBuffer) {
  if (!bytecodeBuffer)
    return false;

  // Use strlen size because the magic number is null-terminated.
  size_t strSize =
      strnlen(bytecodeBuffer, sizeof(kTileIRBytecodeMagic) - 1) + 1;
  return isTileIRBytecode(
      llvm::MemoryBufferRef(StringRef(bytecodeBuffer, strSize), StringRef()));
}

//===----------------------------------------------------------------------===//
// Bytecode Format Overview
//===----------------------------------------------------------------------===//
// The bytecode format consists of a header followed by a sequence of sections.
// Each section has a specific format and purpose.
//
// bytecode =:
//   header
//   section*
//
// header =:
//   magic[8 bytes: 0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00]
//   version[varint]
//
// section =:
//   sectionId[byte]   // The lower 7 bits represent the ID, the high bit
//                     //   indicates alignment presence.
//   length[varint]    // The length of the section in bytes.
//   alignment[varint] // Optional: This field is only present
//                     //   if the high bit of sectionId is set.
//   padding[bytes]    // Optional: These are alignment padding bytes (0xCF).
//   data[bytes]       // The section-specific data format.

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EncodingReader: A helper class for reading encoded data from a byte buffer.
//===----------------------------------------------------------------------===//
namespace {
class EncodingReader {
public:
  EncodingReader(ArrayRef<uint8_t> data, MLIRContext &context)
      : data(data), offset(0), context(context) {}

  LogicalResult readVarInt(uint64_t &result, uint64_t max = 0) {
    if (offset >= data.size())
      return failure();
    result = 0;
    uint64_t shift = 0;
    uint8_t byte;
    do {
      if (offset >= data.size() || shift > 63)
        return failure();
      byte = data[offset++];
      uint64_t value = byte & 0x7F;
      result |= (value << shift);
      shift += 7;
    } while (byte & 0x80);
    if (max && result > max)
      return emitError() << "varint value exceeds maximum supported"
                         << " capacity. (expected value less than " << max
                         << ", got " << result << ").";
    return success();
  }

  /// Parse a signed variable length encoded integer from the byte stream. A
  /// signed varint is encoded as a normal varint with zigzag encoding applied,
  /// i.e. the low bit of the value is used to indicate the sign.
  LogicalResult readSignedVarInt(uint64_t &result) {
    if (failed(readVarInt(result)))
      return failure();
    // Essentially (but using unsigned): (x >> 1) ^ -(x & 1).
    result = (result >> 1) ^ (~(result & 1) + 1);
    return success();
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, LogicalResult> readLE(T &value) {
    if (offset + sizeof(T) > data.size())
      return failure();
    value = 0;
    for (size_t i = 0; i < sizeof(T); ++i)
      value |= static_cast<T>(data[offset++]) << (8 * i);
    return success();
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, T> readLE() {
    T value = 0;
    if (failed(readLE(value)))
      return 0;
    return value;
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, LogicalResult>
  readLE(size_t count, SmallVectorImpl<T> &result) {
    // Validate size to prevent excessive memory allocation.
    if (count > (std::numeric_limits<uint32_t>::max() - 1))
      return emitError() << "array size in bytecode (" << count
                         << ") exceeds maximum supported capacity";
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      T value;
      if (failed(readLE(value)))
        return failure();
      result.push_back(value);
    }
    return success();
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, LogicalResult>
  readLEVarSize(SmallVectorImpl<T> &result) {
    uint64_t size;
    if (failed(readVarInt(size)))
      return failure();
    return readLE(static_cast<size_t>(size), result);
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, LogicalResult>
  readLE(T &value) {
    static_assert(std::numeric_limits<T>::is_iec559, "IEEE 754 required");
    using IntType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    IntType intValue;
    if (failed(readLE(intValue)))
      return failure();
    std::memcpy(&value, &intValue, sizeof(T));
    return success();
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, T> readLE() {
    T value = 0;
    if (failed(readLE(value)))
      return 0;
    return value;
  }

  LogicalResult skip(size_t bytes) {
    if (offset + bytes > data.size())
      return failure();
    offset += bytes;
    return success();
  }

  size_t remaining() const { return data.size() - offset; }

  LogicalResult readBytes(size_t length, ArrayRef<uint8_t> &result) {
    if (offset + length > data.size()) {
      result = ArrayRef<uint8_t>();
      return failure();
    }
    result = data.slice(offset, length);
    offset += length;
    return success();
  }

  ArrayRef<uint8_t> readBytes(size_t length) {
    ArrayRef<uint8_t> result;
    if (failed(readBytes(length, result)))
      return ArrayRef<uint8_t>();
    return result;
  }

  const char *getCurrentPtr() const {
    if (offset >= data.size()) {
      return nullptr;
    }
    return reinterpret_cast<const char *>(data.data() + offset);
  }

  LogicalResult getString(uint64_t index, StringRef &result,
                          MLIRContext &context) const {
    if (index >= stringOffsets.size())
      return ::emitError(UnknownLoc::get(&context))
             << "string index " << index << " out of bounds";
    uint32_t start = stringOffsets[index];
    uint32_t end = (index + 1 < stringOffsets.size())
                       ? stringOffsets[index + 1]
                       : static_cast<uint32_t>(stringData.size());
    result = stringData.substr(start, end - start);
    return success();
  }

  /// Reads a string index and returns the corresponding StringRef.
  LogicalResult readAndGetString(StringRef &result) {
    uint64_t stringIndex;
    if (failed(readVarInt(stringIndex)))
      return failure();
    return getString(stringIndex, result, context);
  }

  void setStringTable(StringRef data, ArrayRef<uint32_t> offsets) {
    stringData = data;
    stringOffsets = offsets;
  }

  size_t currentOffset() const { return offset; }

  LogicalResult skipPadding(uint64_t alignment) {
    if (alignment < 2)
      return success();
    if (remaining() == 0)
      return failure();
    size_t offset_position = this->currentOffset();
    size_t padding = (alignment - (offset_position % alignment)) % alignment;
    if (remaining() < padding)
      return failure();
    return skip(padding);
  }

  // Emits an error message associated with the current reader offset.
  // TODO: Generate a location based on the current offset instead of
  // UnknownLoc.
  InFlightDiagnostic emitError() const {
    return ::emitError(UnknownLoc::get(&context))
           << "error at offset " << offset << ": ";
  }

  void inheritStringTableFrom(const EncodingReader &masterReader) {
    this->stringData = masterReader.stringData;
    this->stringOffsets = masterReader.stringOffsets;
  }

private:
  ArrayRef<uint8_t> data;
  size_t offset;
  StringRef stringData;
  ArrayRef<uint32_t> stringOffsets;
  MLIRContext &context;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Header Parsing
//===----------------------------------------------------------------------===//
namespace {
struct SectionHeader {
  uint8_t sectionID = Section::EndOfBytecode;
  uint64_t length = 0;
  bool hasAlignment = false;
  uint64_t alignment = 1;
};
} // end anonymous namespace

/// Parses and validates the bytecode header, including the magic number and
/// version.
static LogicalResult parseHeader(EncodingReader &reader, MLIRContext &context) {
  // Read and verify the magic number.
  for (int i = 0, e = std::size(kTileIRBytecodeMagic); i < e; ++i) {
    uint8_t byte = reader.readLE<uint8_t>();
    if (byte != kTileIRBytecodeMagic[i])
      return reader.emitError()
             << "invalid magic number at position " << i << ", got "
             << static_cast<int>(byte) << " expected "
             << static_cast<int>(kTileIRBytecodeMagic[i]);
  }
  /// Read and verify the version number.
  uint8_t verMajor, verMinor;
  uint16_t tag;
  if (failed(reader.readLE(verMajor)) || failed(reader.readLE(verMinor)) ||
      failed(reader.readLE(tag)))
    return failure();
  // Check if the version is supported.
  std::optional<BytecodeVersion> versionInfo =
      BytecodeVersion::fromVersion(verMajor, verMinor, tag);
  if (!versionInfo || *versionInfo < BytecodeVersion::kMinSupportedVersion) {
    return reader.emitError()
           << "unsupported Tile version " << verMajor << "." << verMinor << "."
           << tag << ", this reader supports versions ["
           << BytecodeVersion::kMinSupportedVersion.toString() << ", "
           << BytecodeVersion::kCurrentVersion.toString() << "]";
  }
  return success();
}

/// Parses the section header from the bytecode.
static LogicalResult parseSectionHeader(EncodingReader &reader,
                                        SectionHeader &header,
                                        MLIRContext &context) {
  if (reader.remaining() < 1)
    return reader.emitError()
           << "unexpected end of data while reading section header";
  uint8_t idAndIsAligned;
  if (failed(reader.readLE(idAndIsAligned)))
    return reader.emitError() << "failed to read section ID and alignment flag";
  header.sectionID = idAndIsAligned & 0x7F;
  header.hasAlignment = (idAndIsAligned & 0x80) != 0;

  // If this is the end section marker, return success.
  if (header.sectionID == Section::EndOfBytecode) {
    if (header.hasAlignment)
      return reader.emitError()
             << "end section should not have alignment flag set";
    return success();
  }
  if (header.sectionID >= Section::NumSections)
    return reader.emitError() << "unknown section ID: " << header.sectionID;

  // Read the section length.
  if (failed(reader.readVarInt(header.length)))
    return reader.emitError() << "failed to read section length";
  if (header.length > reader.remaining())
    return reader.emitError()
           << "section length " << header.length
           << " exceeds remaining data size " << reader.remaining();

  // If the section is aligned, read the alignment value and adjust the buffer.
  if (header.hasAlignment) {
    if (failed(reader.readVarInt(header.alignment)))
      return failure();
    if (header.alignment == 0 || !llvm::isPowerOf2_64(header.alignment))
      return reader.emitError()
             << "invalid alignment value: " << header.alignment;
    if (failed(reader.skipPadding(header.alignment)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// String Section
//===----------------------------------------------------------------------===//
// string-section =:
//   numStrings[varint]
//   padding[bytes]            // Align to 4 bytes
//   stringOffsets[uint32_t]   // Array of offsets, one per string
//   stringData[bytes]         // Concatenated string data
//
/// Parses the string section and sets up the string table for lazy loading.
static LogicalResult parseStringSection(ArrayRef<uint8_t> payload,
                                        EncodingReader &reader,
                                        MLIRContext &context) {
  EncodingReader sectionReader(payload, context);
  uint64_t numStrings;
  if (failed(sectionReader.readVarInt(numStrings)))
    return failure();

  // Handle empty string table case.
  if (numStrings == 0) {
    reader.setStringTable(StringRef(), ArrayRef<uint32_t>());
    return success();
  }

  if (numStrings >
      (payload.size() - sectionReader.currentOffset()) / sizeof(uint32_t)) {
    return sectionReader.emitError()
           << "number of strings (" << numStrings << ") exceeds the maximum of "
           << (payload.size() - sectionReader.currentOffset()) /
                  sizeof(uint32_t)
           << " that can fit in the remaining payload of "
           << (payload.size() - sectionReader.currentOffset()) << " bytes.";
  }

  // Ensure 4-byte alignment for the start indices array.
  if (failed(sectionReader.skipPadding(alignof(uint32_t))))
    return failure();
  // Read the string offsets directly from the payload.
  const uint32_t *startIndicesPtr =
      reinterpret_cast<const uint32_t *>(sectionReader.getCurrentPtr());
  if (!startIndicesPtr)
    return failure();
  ArrayRef<uint32_t> stringOffsets(startIndicesPtr, numStrings);
  if (failed(sectionReader.skip(numStrings * sizeof(uint32_t))))
    return failure();
  // Get the string data
  StringRef stringData(
      reinterpret_cast<const char *>(sectionReader.getCurrentPtr()),
      sectionReader.remaining());
  // Set up the string table in the main reader.
  reader.setStringTable(stringData, stringOffsets);
  return success();
}

//===----------------------------------------------------------------------===//
// Enum Parsing
//===----------------------------------------------------------------------===//

// Include generated opcode enum definition
#define GEN_OPCODE_ENUM
#include "StaticOpcodes.inc"

// Generic template for symbolizing enums from an integer value.
template <typename EnumType>
static std::optional<EnumType> symbolizeEnum(uint32_t value);

// Specializations for CUDA tile enum types.
template <>
std::optional<cuda_tile::RoundingMode>
symbolizeEnum<cuda_tile::RoundingMode>(uint32_t value) {
  return cuda_tile::symbolizeRoundingMode(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::ComparisonPredicate>
symbolizeEnum<cuda_tile::ComparisonPredicate>(uint32_t value) {
  return cuda_tile::symbolizeComparisonPredicate(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::ComparisonOrdering>
symbolizeEnum<cuda_tile::ComparisonOrdering>(uint32_t value) {
  return cuda_tile::symbolizeComparisonOrdering(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::AtomicRMWMode>
symbolizeEnum<cuda_tile::AtomicRMWMode>(uint32_t value) {
  return cuda_tile::symbolizeAtomicRMWMode(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::MemoryOrderingSemantics>
symbolizeEnum<cuda_tile::MemoryOrderingSemantics>(uint32_t value) {
  return cuda_tile::symbolizeMemoryOrderingSemantics(
      static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::MemoryScope>
symbolizeEnum<cuda_tile::MemoryScope>(uint32_t value) {
  return cuda_tile::symbolizeMemoryScope(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::IntegerOverflow>
symbolizeEnum<cuda_tile::IntegerOverflow>(uint32_t value) {
  return cuda_tile::symbolizeIntegerOverflow(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::PaddingValue>
symbolizeEnum<cuda_tile::PaddingValue>(uint32_t value) {
  return cuda_tile::symbolizePaddingValue(static_cast<int32_t>(value));
}
template <>
std::optional<cuda_tile::Signedness>
symbolizeEnum<cuda_tile::Signedness>(uint32_t value) {
  return cuda_tile::symbolizeSignedness(static_cast<int32_t>(value));
}

/// Generic helper to parse an enum attribute.
template <typename AttrType>
static LogicalResult parseGenericEnumAttr(EncodingReader &reader,
                                          MLIRContext &context,
                                          AttrType &nativeValue) {
  uint64_t rawEnumValueU64;
  if (failed(reader.readVarInt(rawEnumValueU64)))
    return reader.emitError() << "failed to read VarInt for enum attribute.";
  uint32_t rawEnumValue = static_cast<uint32_t>(rawEnumValueU64);

  using EnumType = decltype(std::declval<AttrType>().getValue());
  static_assert(!std::is_void_v<EnumType>,
                "EnumType cannot be void for enum attribute.");
  std::optional<EnumType> enumOpt = symbolizeEnum<EnumType>(rawEnumValue);
  if (!enumOpt)
    return reader.emitError()
           << "invalid integer value for enum type: " << rawEnumValue;
  nativeValue = AttrType::get(&context, enumOpt.value());
  return success();
}

//===----------------------------------------------------------------------===//
// LazyTypeTable: Manages lazy parsing and caching of types from the type
// section.
//===----------------------------------------------------------------------===//
// type-section =:
//   numTypes[varint]
//   padding[bytes]          // Align to 4 bytes
//   typeOffsets[uint32_t]   // Array of offsets, one per type
//   typeData[bytes]         // Concatenated type data
//
// type-data =:
//   typeTag[byte]           // Indicates the kind of type
//   type-specific-data      // Format depends on typeTag
//
namespace {
class LazyTypeTable {
public:
  LazyTypeTable(MLIRContext &context) : context(context) {}

  void initialize(ArrayRef<uint8_t> payloadData, ArrayRef<uint32_t> indices) {
    payload = payloadData;
    typeStartIndices = indices;
    typeCache.resize(indices.size());
  }

  Type getType(uint64_t typeIndex) {
    if (typeIndex >= typeCache.size())
      return Type();
    if (typeCache[typeIndex])
      return typeCache[typeIndex];

    // Check for recursion.
    if (currentlyParsing.count(typeIndex))
      return Type();
    // Mark this type as currently being parsed.
    currentlyParsing.insert(typeIndex);
    auto removeIndex =
        llvm::make_scope_exit([&] { currentlyParsing.erase(typeIndex); });
    // Calculate the boundaries for the type data.
    uint32_t start = typeStartIndices[typeIndex];
    uint32_t end = (typeIndex + 1 < typeStartIndices.size())
                       ? typeStartIndices[typeIndex + 1]
                       : payload.size();
    if (end < start || end > payload.size())
      return Type();
    // Parse the type from its specific byte slice.
    EncodingReader typeReader(payload.slice(start, end - start), context);
    uint64_t typeTag;
    if (failed(typeReader.readVarInt(typeTag)))
      return nullptr;
    ArrayRef<uint8_t> payloadBytes;
    if (typeReader.remaining() > 0)
      payloadBytes = typeReader.readBytes(typeReader.remaining());
    Type parsedType;
    if (failed(parseTypeImpl(typeTag, payloadBytes, parsedType)))
      return Type();

    // Cache the result.
    typeCache[typeIndex] = parsedType;
    return parsedType;
  }

  size_t size() const { return typeStartIndices.size(); }

  /// Reads a type index using the provided reader and retrieves the
  /// corresponding Type. Emits an error and returns a null Type on failure.
  Type readAndGetType(EncodingReader &reader) {
    uint64_t typeIndex;
    if (failed(reader.readVarInt(typeIndex))) {
      return Type();
    }
    // getType already emits an error if the index is bad or parsing fails.
    return getType(typeIndex);
  }

private:
  // integer-type =: typeTag[I1/I8/I16/I32/I64]
  LogicalResult parseIntegerType(uint8_t typeTag, Type &result) {

    switch (static_cast<TypeTag>(typeTag)) {
    case TypeTag::I1:
      result = IntegerType::get(&context, 1);
      return success();
    case TypeTag::I8:
      result = IntegerType::get(&context, 8);
      return success();
    case TypeTag::I16:
      result = IntegerType::get(&context, 16);
      return success();
    case TypeTag::I32:
      result = IntegerType::get(&context, 32);
      return success();
    case TypeTag::I64:
      result = IntegerType::get(&context, 64);
      return success();
    default:
      return ::emitError(UnknownLoc::get(&context))
             << "invalid integer type tag: " << static_cast<int>(typeTag);
    }
  }

  // tile-type =:
  //   typeTag[Tile]
  //   elementTypeIndex[varint]
  //   rank[varint]
  //   dimensions[int64_t*rank]
  LogicalResult parseTileType(EncodingReader &reader, Type &result) {
    Type elementType = readAndGetType(reader);
    if (!elementType)
      return reader.emitError() << "failed to get element type";
    // Validate that the element type is a valid scalar type for tiles.
    if (!elementType.isIntOrFloat() &&
        !isa<cuda_tile::PointerType>(elementType))
      return reader.emitError() << "tile element type must be integer, float, "
                                   "or valid pointer, got: "
                                << elementType;
    // Read dimensions (size is implicitly read by readLEVarSize)
    SmallVector<int64_t, 4> shape;
    if (failed(reader.readLEVarSize(shape)))
      return reader.emitError() << "failed to read tile shape";
    result = cuda_tile::TileType::getChecked(
        [&]() { return reader.emitError(); }, &context,
        ArrayRef<int64_t>(shape), elementType);
    return success(result);
  }

  // function-type =:
  //   typeTag[Func]
  //   numInputs[varint]
  //   inputTypeIndices[varint*numInputs]
  //   numResults[varint]
  //   resultTypeIndices[varint*numResults]
  LogicalResult parseFunctionType(EncodingReader &reader, Type &result) {
    uint64_t numParams, numResults;
    // Read the number of parameters (VarInt as per specification).
    if (failed(reader.readVarInt(numParams,
                                 std::numeric_limits<uint32_t>::max() - 1)))
      return reader.emitError() << "failed to read number of parameters";

    // Read parameter types
    SmallVector<Type, 4> paramTypes;
    paramTypes.reserve(numParams);
    for (uint64_t i = 0; i < numParams; ++i) {
      Type paramType = readAndGetType(reader);
      if (!paramType)
        return reader.emitError() << "failed to get parameter type";
      paramTypes.push_back(paramType);
    }
    //  Read the number of results (VarInt as per specification).
    if (failed(reader.readVarInt(numResults,
                                 std::numeric_limits<uint32_t>::max() - 1)))
      return reader.emitError() << "failed to read number of results";
    // Read result types
    SmallVector<Type, 4> resultTypes;
    resultTypes.reserve(numResults);
    for (uint64_t i = 0; i < numResults; ++i) {
      Type resultType = readAndGetType(reader);
      if (!resultType)
        return reader.emitError() << "failed to get result type";
      resultTypes.push_back(resultType);
    }
    result = FunctionType::get(&context, paramTypes, resultTypes);
    return success();
  }

  // float-type =: typeTag[F16/BF16/F32/TF32/F64/F8E4M3FN/F8E5M2]
  LogicalResult parseFloatType(uint8_t typeTag, Type &result) {
    if (typeTag == static_cast<uint8_t>(TypeTag::F16)) {
      result = Float16Type::get(&context);
      return success();
    }
    if (typeTag == static_cast<uint8_t>(TypeTag::BF16)) {
      result = BFloat16Type::get(&context);
      return success();
    }
    if (typeTag == static_cast<uint8_t>(TypeTag::F32)) {
      result = Float32Type::get(&context);
      return success();
    }
    if (typeTag == static_cast<uint8_t>(TypeTag::TF32)) {
      result = FloatTF32Type::get(&context);
      return success();
    }
    if (typeTag == static_cast<uint8_t>(TypeTag::F64)) {
      result = Float64Type::get(&context);
      return success();
    }
    if (typeTag == static_cast<uint8_t>(TypeTag::F8E4M3FN)) {
      result = Float8E4M3FNType::get(&context);
      return success();
    }
    if (typeTag == static_cast<uint8_t>(TypeTag::F8E5M2)) {
      result = Float8E5M2Type::get(&context);
      return success();
    }
    return ::emitError(UnknownLoc::get(&context))
           << "unsupported float type tag: " << static_cast<int>(typeTag);
  }

  // pointer-type =:
  //   typeTag[Pointer]
  //   pointeeTypeIndex[varint]
  LogicalResult parsePointerType(EncodingReader &reader, Type &result) {
    Type pointeeType = readAndGetType(reader);
    if (!pointeeType)
      return reader.emitError() << "failed to get pointer type";
    result = cuda_tile::PointerType::getChecked(
        [&]() { return reader.emitError(); }, &context, pointeeType);
    return success(result);
  }

  // memref-type =:
  //   typeTag[MemRef]
  //   elementTypeIndex[varint]
  //   shapeSize[varint]
  //   shape[int64_t*shapeSize]
  //   stridesSize[varint]
  //   strides[int64_t*stridesSize]
  //   indexBitwidth[varint]
  LogicalResult parseTensorViewType(EncodingReader &reader, Type &result) {
    // Read the element type index.
    Type elementType = readAndGetType(reader);
    if (!elementType || !elementType.isIntOrFloat())
      return reader.emitError()
             << "tensor_view element type must be integer or float, got: "
             << elementType;
    // Read the shape size.
    SmallVector<int64_t, 4> shape;
    if (failed(reader.readLEVarSize(shape)))
      return reader.emitError() << "failed to read tensor_view shape";
    // Read the strides size.
    SmallVector<int64_t, 4> strides;
    if (failed(reader.readLEVarSize(strides)))
      return reader.emitError() << "failed to read tensor_view strides";
    result = cuda_tile::TensorViewType::getChecked(
        [&]() { return reader.emitError(); }, &context, elementType,
        ArrayRef<int64_t>(shape), ArrayRef<int64_t>(strides));
    return success(result);
  }

  // tile-partition-view-type =:
  //   typeTag[PartitionView]
  //   tileShapeSize[varint]
  //   tileShape[int32_t*tileShapeSize]
  //   memrefTypeIndex[varint]
  //   dimMapSize[varint]
  //   dimMap[int32_t*dimMapSize]
  //   hasPadding[byte]
  //   paddingValueEnum[varint]?
  LogicalResult parsePartitionViewType(EncodingReader &reader, Type &result) {
    // Read the tile shape size.
    SmallVector<int32_t, 4> tileShape;
    if (failed(reader.readLEVarSize(tileShape)))
      return reader.emitError()
             << "failed to read tile partition view tile shape";
    // Validate tile shape values.
    for (int32_t val : tileShape) {
      if (LLVM_UNLIKELY(val == 0x7fffffff || val == (-0x7fffffff - 1)))
        return reader.emitError()
               << "tile shape contains unsupported value " << val;
    }

    // Read the tensor_view type index.
    Type tensorViewType = readAndGetType(reader);
    if (!tensorViewType)
      return reader.emitError()
             << "failed to get tensor_view type for tile partition view";
    auto typedTensorViewType =
        mlir::dyn_cast<cuda_tile::TensorViewType>(tensorViewType);
    if (!typedTensorViewType)
      return reader.emitError()
             << "invalid tensor_view type for tile partition view";
    // Read the dimension map size.
    SmallVector<int32_t, 4> dimMap;
    if (failed(reader.readLEVarSize(dimMap)))
      return reader.emitError()
             << "failed to read tile partition view dimension map";

    PaddingValueAttr paddingValueAttr;
    if (reader.readLE<uint8_t>())
      if (failed(parseGenericEnumAttr(reader, context, paddingValueAttr)))
        return failure();

    //  Create an attribute for the tile shape.
    auto tileShapeAttr = DenseI32ArrayAttr::get(&context, tileShape);
    result = cuda_tile::PartitionViewType::getChecked(
        [&]() { return reader.emitError(); }, &context, tileShapeAttr,
        typedTensorViewType, ArrayRef<int32_t>(dimMap), paddingValueAttr);

    return success(result);
  }

  LogicalResult parseTypeImpl(uint8_t typeTag, ArrayRef<uint8_t> payloadBytes,
                              Type &result) {
    EncodingReader reader(payloadBytes, context);
    switch (static_cast<TypeTag>(typeTag)) {
    case TypeTag::I1:
    case TypeTag::I8:
    case TypeTag::I16:
    case TypeTag::I32:
    case TypeTag::I64:
      return parseIntegerType(typeTag, result);
    case TypeTag::F16:
    case TypeTag::BF16:
    case TypeTag::F32:
    case TypeTag::TF32:
    case TypeTag::F64:
    case TypeTag::F8E4M3FN:
    case TypeTag::F8E5M2:
      return parseFloatType(typeTag, result);
    case TypeTag::Tile:
      return parseTileType(reader, result);
    case TypeTag::Pointer:
      return parsePointerType(reader, result);
    case TypeTag::TensorView:
      return parseTensorViewType(reader, result);
    case TypeTag::PartitionView:
      return parsePartitionViewType(reader, result);
    case TypeTag::Func:
      return parseFunctionType(reader, result);
    case TypeTag::Token:
      result = cuda_tile::TokenType::get(&context);
      return success();
    default:
      return ::emitError(UnknownLoc::get(&context))
             << "unknown type tag: " << static_cast<int>(typeTag);
    }
  }

  MLIRContext &context;
  ArrayRef<uint8_t> payload;
  ArrayRef<uint32_t> typeStartIndices;
  std::vector<Type> typeCache;
  DenseSet<uint64_t> currentlyParsing;
};
} // end anonymous namespace

/// Parses the type section and initializes the lazy type table
static LogicalResult parseTypeSection(ArrayRef<uint8_t> payload,
                                      LazyTypeTable &types,
                                      MLIRContext &context) {
  EncodingReader reader(payload, context);
  uint64_t numTypes;
  if (failed(reader.readVarInt(numTypes)))
    return failure();

  // Handle empty type table case.
  if (numTypes == 0) {
    types.initialize(ArrayRef<uint8_t>(), ArrayRef<uint32_t>());
    return success();
  }

  if (numTypes > (payload.size() - reader.currentOffset()) / sizeof(uint32_t)) {
    return reader.emitError()
           << "number of types (" << numTypes << ") exceeds the maximum of "
           << (payload.size() - reader.currentOffset()) / sizeof(uint32_t)
           << " that can fit in the remaining payload of "
           << (payload.size() - reader.currentOffset()) << " bytes.";
  }

  // Ensure 4-byte alignment for the start indices array
  if (failed(reader.skipPadding(alignof(uint32_t))))
    return failure();
  // Read type start indices as a contiguous array
  const uint32_t *startIndicesPtr =
      reinterpret_cast<const uint32_t *>(reader.getCurrentPtr());
  if (!startIndicesPtr)
    return failure();
  ArrayRef<uint32_t> typeStartIndices(startIndicesPtr, numTypes);
  if (failed(reader.skip(numTypes * sizeof(uint32_t))))
    return failure();
  // Initialize the lazy type table with the payload and indices
  ArrayRef<uint8_t> typeData = payload.slice(reader.currentOffset());
  types.initialize(typeData, typeStartIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Section
//===----------------------------------------------------------------------===//
// constant-section =:
//   numConstants[varint]
//   padding[bytes]             // Align to 8 bytes
//   constantOffsets[uint64_t]  // Array of offsets, one per constant
//   constantData[bytes]        // Concatenated constant data
//
// constant-data format depends on the attribute type
// scalar-constant =: raw binary representation of the scalar value
//
namespace {
///  A cache for deduplicating constant attributes during parsing.
class DenseElementsAttrCache {
public:
  FailureOr<DenseElementsAttr> getOrCreate(Type type, ArrayRef<uint8_t> data,
                                           MLIRContext &context) {
    // The key is a combination of the expected type and the raw data blob.
    std::pair<Type, ArrayRef<uint8_t>> key = {type, data};
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;

    // Create a reader for the constant data blob.
    EncodingReader reader(data, context);

    // Cast to TileType to get element type and shape info.
    if (!type)
      return reader.emitError() << "provided type is null";
    auto tileType = mlir::dyn_cast<cuda_tile::TileType>(type);
    if (!tileType || !tileType.getElementType().isIntOrFloat())
      return reader.emitError()
             << "expect Cuda Tile integer or float type but got: " << tileType;

    // Read the size of the raw data buffer.
    uint64_t rawDataSize;
    if (failed(reader.readVarInt(rawDataSize)))
      return reader.emitError() << "failed to read the size of the data buffer";

    // Read the raw byte data.
    ArrayRef<uint8_t> rawUint8Data;
    if (failed(reader.readBytes(rawDataSize, rawUint8Data)))
      return reader.emitError() << "failed to read the raw byte data";

    // Convert ArrayRef<uint8_t> to ArrayRef<char>.
    ArrayRef<char> rawData(reinterpret_cast<const char *>(rawUint8Data.data()),
                           rawUint8Data.size());
    // Validate the buffer size and format.
    bool isSplat = false;
    if (!DenseElementsAttr::isValidRawBuffer(tileType, rawData, isSplat))
      return reader.emitError() << "failed to validate buffer size and format";

    DenseElementsAttr attr = nullptr;
    // Handle endianness conversion.
    if (llvm::endianness::native == llvm::endianness::big) {
      // Convert endianess.
      SmallVector<char, 64> outDataVec(rawData.size());
      MutableArrayRef<char> convRawData(outDataVec);
      DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
          rawData, convRawData, tileType);
      attr = DenseElementsAttr::getFromRawBuffer(tileType, convRawData);
    } else {
      attr = DenseElementsAttr::getFromRawBuffer(tileType, rawData);
    }

    if (attr)
      cache.insert({key, attr});
    return attr;
  }

private:
  DenseMap<std::pair<Type, ArrayRef<uint8_t>>, DenseElementsAttr> cache;
};
} // namespace

/// Parses the constant section and populates the constant table
static LogicalResult
parseConstantSection(ArrayRef<uint8_t> payload,
                     std::vector<ArrayRef<uint8_t>> &constants,
                     MLIRContext &context) {
  EncodingReader reader(payload, context);
  uint64_t numConstants;
  if (failed(reader.readVarInt(numConstants,
                               std::numeric_limits<uint32_t>::max() - 1)))
    return failure();
  // Handle empty constant section case
  if (numConstants == 0)
    return success();
  // Ensure 8-byte alignment for the start indices array
  if (failed(reader.skipPadding(alignof(uint64_t))))
    return failure();
  // Check if we have enough data to read the indices
  if (reader.remaining() / sizeof(uint64_t) < numConstants)
    return reader.emitError() << "insufficient data for constant indices";

  // Read constant start indices as a contiguous array
  const uint64_t *startIndicesPtr =
      reinterpret_cast<const uint64_t *>(reader.getCurrentPtr());
  if (!startIndicesPtr)
    return failure();

  ArrayRef<uint64_t> constantStartIndices(startIndicesPtr, numConstants);
  if (failed(reader.skip(constantStartIndices.size() * sizeof(uint64_t))))
    return failure();
  ArrayRef<uint8_t> constantData = payload.slice(reader.currentOffset());
  // Populate constants based on constantStartIndices
  constants.reserve(numConstants);
  for (uint64_t i = 0; i < numConstants; ++i) {
    uint64_t start = constantStartIndices[i];
    uint64_t end = (i + 1 < numConstants) ? constantStartIndices[i + 1]
                                          : constantData.size();
    if (end < start)
      return reader.emitError()
             << "invalid constant start indices: end (" << end
             << ") is less than start (" << start << ") for constant " << i;
    size_t constantSize = end - start;
    if (constantSize + start > constantData.size())
      return reader.emitError()
             << "constant " << i << " extends beyond available data: "
             << "size=" << constantSize << ", start=" << start
             << ", total data size=" << constantData.size();
    constants.push_back(constantData.slice(start, constantSize));
  }
  return success();
}

namespace {

//===----------------------------------------------------------------------===//
// DebugInfo Section
//===----------------------------------------------------------------------===//

/// This class manages reading debug info attributes from bytecode format.
class DebugInfoReader {
public:
  DebugInfoReader(MLIRContext &context, EncodingReader &masterReader)
      : context(context), masterReader(masterReader) {}

  class Iterator {
  public:
    Iterator(DebugInfoReader &reader, uint64_t opIndex)
        : reader(reader), opIndex(opIndex) {}

    /// Return the next debug info attribute for the current operation.
    template <typename T>
    T next() {
      // Check if the index is reserved for special debug info attributes.
      if (opIndex == static_cast<uint64_t>(Bytecode::DebugReserved::UnknownLoc))
        return dyn_cast<T>(UnknownLoc::get(&reader.context));

      // Adjust the index to account for reserved indices.
      auto actualOpIndex =
          opIndex - static_cast<uint64_t>(Bytecode::DebugReserved::SIZE);

      // Calculate the offset for the current operation index.
      if (actualOpIndex >= reader.diIndexOffsets.size())
        return T();
      auto offset = reader.diIndexOffsets[actualOpIndex];

      // Validate size to prevent excessive memory allocation.
      if (reader.diIndices.size() > (std::numeric_limits<uint32_t>::max() - 1))
        return T();
      if (offset > (std::numeric_limits<uint32_t>::max() - 1))
        return T();
      if (offset + opIndexOffset >= reader.diIndices.size())
        return T();
      offset += opIndexOffset++;

      // Return the next debug info attribute for the current operation.
      return reader.getDebugInfo<T>(reader.diIndices[offset]);
    }

  private:
    DebugInfoReader &reader;
    uint64_t opIndex;
    uint64_t opIndexOffset = 0;
  };

  Iterator getIterator(uint64_t opIndex) { return Iterator(*this, opIndex); }

  /// This method initializes the debug info reader after construction.
  void initialize(ArrayRef<uint64_t> indices, ArrayRef<uint32_t> indexOffsets,
                  ArrayRef<uint8_t> data, ArrayRef<uint32_t> offsets) {
    diIndices = indices;
    diIndexOffsets = indexOffsets;
    diData = data;
    diOffsets = offsets;
    diCache.resize(offsets.size());
  }

private:
  /// This method returns a debug info attribute for a given index.
  template <typename T>
  T getDebugInfo(uint64_t diIndex) {
    // Check if the index is reserved for special debug info attributes.
    if (diIndex == static_cast<uint64_t>(Bytecode::DebugReserved::UnknownLoc))
      return dyn_cast<T>(UnknownLoc::get(&context));

    // Adjust the index to account for reserved indices.
    diIndex -= static_cast<uint64_t>(Bytecode::DebugReserved::SIZE);

    if (diIndex >= diCache.size())
      return T();
    if (diCache[diIndex])
      return dyn_cast_or_null<T>(diCache[diIndex]);

    return dyn_cast_or_null<T>(getDebugInfo(diIndex));
  }

  /// This method reads an index and converts it to a debug info attribute.
  template <typename T>
  T readAndGetDebugInfo(EncodingReader &reader) {
    uint64_t diIndex;
    if (failed(reader.readVarInt(diIndex)))
      return T();

    return getDebugInfo<T>(diIndex);
  }

  Attribute getDebugInfo(uint64_t diIndex) {
    // Check for bounds
    if (diIndex >= diCache.size())
      return Attribute();
    if (diCache[diIndex])
      return diCache[diIndex];

    uint32_t start = diOffsets[diIndex];
    uint32_t end = (diIndex + 1 < diOffsets.size()) ? diOffsets[diIndex + 1]
                                                    : diData.size();

    if (end < start || end > diData.size())
      return Attribute();

    // Check for recursion.
    if (currentlyParsing.count(diIndex))
      return Attribute();
    // Mark this index as currently being parsed.
    currentlyParsing.insert(diIndex);
    auto removeIndex =
        llvm::make_scope_exit([&] { currentlyParsing.erase(diIndex); });

    // Slice the payload to get the data for this debug info attribute.
    EncodingReader diReader(diData.slice(start, end - start), context);
    uint64_t diTag;
    if (failed(diReader.readVarInt(diTag)))
      return nullptr;
    ArrayRef<uint8_t> diData;
    if (diReader.remaining() > 0)
      diData = diReader.readBytes(diReader.remaining());

    // Parse the debug info attribute based on the tag.
    Attribute diParsed;
    if (failed(parseDebugInfo(diTag, diData, diParsed)))
      return Attribute();

    // Cache the result.
    diCache[diIndex] = diParsed;
    return diParsed;
  }

  // di-compile-unit =:
  //   DebugTag[DICompileUnit]
  //   diFileIndex[varint] - DIFileAttr
  LogicalResult parseDICompileUnit(EncodingReader &reader,
                                   Attribute &diCompileUnit) {
    auto file = readAndGetDebugInfo<DIFileAttr>(reader);
    if (!file)
      return reader.emitError()
             << "failed to read file attribute when parsing DICompileUnitAttr";

    diCompileUnit = DICompileUnitAttr::get(&context, file);
    return success();
  }

  // di-file =:
  //   DebugTag[DIFile]
  //   fileNameIndex[varint] - StringAttr
  //   directoryIndex[varint] - StringAttr
  LogicalResult parseDIFile(EncodingReader &reader, Attribute &diFile) {
    StringRef nameStr;
    if (failed(reader.readAndGetString(nameStr)))
      return reader.emitError()
             << "failed to read file name attribute when parsing DIFileAttr";
    StringAttr name = StringAttr::get(&context, nameStr);

    StringRef directoryStr;
    if (failed(reader.readAndGetString(directoryStr)))
      return reader.emitError()
             << "failed to read directory attribute when parsing DIFileAttr";
    StringAttr directory = StringAttr::get(&context, directoryStr);

    diFile = DIFileAttr::get(&context, name, directory);
    return success();
  }

  // di-lexical-block =:
  //   DebugTag[DILexicalBlock]
  //   diScopeIndex[varint] - DILocalScopeAttr
  //   diFileIndex[varint] - DIFileAttr
  //   lineNumber[varint] - unsigned
  //   columnNumber[varint] - unsigned
  LogicalResult parseDILexicalBlock(EncodingReader &reader,
                                    Attribute &diLexicalBlock) {
    auto scope = readAndGetDebugInfo<DILocalScopeAttr>(reader);
    if (!scope)
      return reader.emitError() << "failed to read scope attribute when "
                                   "parsing DILexicalBlockAttr";

    auto file = readAndGetDebugInfo<DIFileAttr>(reader);
    if (!file)
      return reader.emitError()
             << "failed to read file attribute when parsing DILexicalBlockAttr";

    uint64_t line;
    if (failed(reader.readVarInt(line)))
      return reader.emitError()
             << "failed to read line number when parsing DILexicalBlockAttr";

    uint64_t column;
    if (failed(reader.readVarInt(column)))
      return reader.emitError()
             << "failed to read column number when parsing DILexicalBlockAttr";

    diLexicalBlock =
        DILexicalBlockAttr::get(&context, scope, file, line, column);
    return success();
  }

  // di-loc =:
  //   DebugTag[DILoc]
  //   diScopeIndex[varint] - DILocalScopeAttr
  //   fileNameIndex[varint] - StringAttr
  //   lineNumber[varint] - unsigned
  //   columnNumber[varint] - unsigned
  LogicalResult parseDILoc(EncodingReader &reader, Attribute &diLoc) {
    auto scope = readAndGetDebugInfo<DILocalScopeAttr>(reader);
    if (!scope)
      return reader.emitError()
             << "failed to read scope attribute when parsing DILocAttr";

    StringRef filenameStr;
    if (failed(reader.readAndGetString(filenameStr)))
      return reader.emitError() << "failed to read file name attribute when "
                                   "parsing FileLineColLoc";
    StringAttr filename = StringAttr::get(&context, filenameStr);

    uint64_t line;
    if (failed(reader.readVarInt(line)))
      return reader.emitError()
             << "failed to read line number when parsing FileLineColLoc";

    uint64_t column;
    if (failed(reader.readVarInt(column)))
      return reader.emitError()
             << "failed to read column number when parsing FileLineColLoc";

    auto fileLineCol = FileLineColLoc::get(&context, filename, line, column);
    diLoc = DILocAttr::get(&context, fileLineCol, scope);
    return success();
  }

  // di-subprogram =:
  //  DebugTag[DISubprogram]
  //  diFileIndex[varint] - DIFileAttr
  //  lineNumber[varint] - unsigned
  //  nameIndex[varint] - StringAttr
  //  linkageNameIndex[varint] - StringAttr
  //  diCompileUnitIndex[varint] - DICompileUnitAttr
  //  scopeLine[varint] - unsigned
  LogicalResult parseDISubprogram(EncodingReader &reader,
                                  Attribute &diSubprogram) {
    auto file = readAndGetDebugInfo<DIFileAttr>(reader);
    if (!file)
      return reader.emitError()
             << "failed to read file attribute when parsing DISubprogramAttr";

    uint64_t line;
    if (failed(reader.readVarInt(line)))
      return reader.emitError()
             << "failed to read line number when parsing DISubprogramAttr";

    StringRef nameStr;
    if (failed(reader.readAndGetString(nameStr)))
      return reader.emitError()
             << "failed to read name attribute when parsing DISubprogramAttr";
    StringAttr name = StringAttr::get(&context, nameStr);

    StringRef linkageNameStr;
    if (failed(reader.readAndGetString(linkageNameStr)))
      return reader.emitError() << "failed to read linkage name attribute when "
                                   "parsing DISubprogramAttr";
    StringAttr linkageName = StringAttr::get(&context, linkageNameStr);

    auto compileUnit = readAndGetDebugInfo<DICompileUnitAttr>(reader);
    if (!compileUnit)
      return reader.emitError() << "failed to read compile unit attribute when "
                                   "parsing DISubprogramAttr";

    uint64_t scopeLine;
    if (failed(reader.readVarInt(scopeLine)))
      return reader.emitError() << "failed to read scope line number when "
                                   "parsing DISubprogramAttr";

    diSubprogram = DISubprogramAttr::get(&context, file, line, name,
                                         linkageName, compileUnit, scopeLine);
    return success();
  }

  // call-site =:
  //  DebugTag[CallSite]
  //  diCalleeIndex[varint] - LocationAttr
  //  diCallerIndex[varint] - LocationAttr
  LogicalResult parseCallSite(EncodingReader &reader, Attribute &callSite) {
    auto callee = readAndGetDebugInfo<LocationAttr>(reader);
    if (!callee)
      return reader.emitError()
             << "failed to read callee attribute when parsing CallSiteLoc";

    auto caller = readAndGetDebugInfo<LocationAttr>(reader);
    if (!caller)
      return reader.emitError()
             << "failed to read caller attribute when parsing CallSiteLoc";

    callSite = CallSiteLoc::get(callee, caller);
    return success();
  }

  // unknown =:
  //   DebugTag[Unknown]
  LogicalResult parseUnknown(EncodingReader &reader, Attribute &unknown) {
    unknown = UnknownLoc::get(&context);
    return success();
  }

  LogicalResult parseDebugInfo(uint8_t diTag, ArrayRef<uint8_t> diData,
                               Attribute &diParsed) {
    EncodingReader reader(diData, context);
    reader.inheritStringTableFrom(masterReader);
    switch (static_cast<DebugTag>(diTag)) {
    case DebugTag::DICompileUnit:
      return parseDICompileUnit(reader, diParsed);
    case DebugTag::DIFile:
      return parseDIFile(reader, diParsed);
    case DebugTag::DILexicalBlock:
      return parseDILexicalBlock(reader, diParsed);
    case DebugTag::DILoc:
      return parseDILoc(reader, diParsed);
    case DebugTag::DISubprogram:
      return parseDISubprogram(reader, diParsed);
    case DebugTag::CallSite:
      return parseCallSite(reader, diParsed);
    default:
      return parseUnknown(reader, diParsed);
    }
  }

  MLIRContext &context;
  ArrayRef<uint64_t> diIndices;
  ArrayRef<uint32_t> diIndexOffsets;
  ArrayRef<uint8_t> diData;
  ArrayRef<uint32_t> diOffsets;
  std::vector<Attribute> diCache;
  EncodingReader &masterReader;
  DenseSet<uint64_t> currentlyParsing;
};
} // namespace

//===----------------------------------------------------------------------===//
// InstructionParser: Parses individual instructions within a function body.
//===----------------------------------------------------------------------===//
// instruction =:
//   opcode[varint]
//   op-specific-data          // Format depends on the opcode
//

namespace {

// Type trait to check if T is one of the specified CUDA tile enum attribute
// types.
template <typename T>
struct is_cuda_tile_enum_attr
    : std::disjunction<std::is_same<T, cuda_tile::RoundingModeAttr>,
                       std::is_same<T, cuda_tile::ComparisonPredicateAttr>,
                       std::is_same<T, cuda_tile::ComparisonOrderingAttr>,
                       std::is_same<T, cuda_tile::AtomicRMWModeAttr>,
                       std::is_same<T, cuda_tile::MemoryOrderingSemanticsAttr>,
                       std::is_same<T, cuda_tile::MemoryScopeAttr>,
                       std::is_same<T, cuda_tile::IntegerOverflowAttr>,
                       std::is_same<T, cuda_tile::PaddingValueAttr>,
                       std::is_same<T, cuda_tile::SignednessAttr>> {};

class InstructionParser {
  //===----------------------------------------------------------------------===//
  // Helper for Operation Creation and Result Handling
  //===----------------------------------------------------------------------===//
  /// Creates an operation using OperationState and pushes its results to the
  /// valueIndexList.
  static LogicalResult createOperationGeneric(
      OpBuilder &builder, Location loc, StringRef opNameStr,
      ArrayRef<Type> resultTypes, ArrayRef<Value> operands,
      ArrayRef<NamedAttribute> attributes, std::vector<Value> &valueIndexList,
      SmallVectorImpl<std::unique_ptr<Region>> &parsedRegions) {
    OperationState state(loc, opNameStr, operands, resultTypes, attributes);

    // Add parsed regions to the operation state.
    for (auto &region_ptr : parsedRegions)
      state.addRegion(std::move(region_ptr));

    Operation *op = builder.create(state);
    // Operation creation using OperationState can fail if verification fails.
    // Emit an error noting the failure.
    if (!op)
      return ::emitError(loc) << "failed to create operation '" << opNameStr
                              << "'due to verification error.";
    // Add results to the value index list.
    llvm::append_range(valueIndexList, op->getResults());

    return success();
  }

  /// Parses operand indices and returns the corresponding Values from the
  /// valueIndexList. If numOperandsToRead is std::nullopt, it first reads the
  /// number of operands as a VarInt. Otherwise, it uses the provided count.
  static LogicalResult
  parseOperands(EncodingReader &reader, Location loc,
                ArrayRef<Value> valueIndexList, SmallVectorImpl<Value> &results,
                std::optional<uint64_t> numOperandsToRead = std::nullopt) {
    uint64_t numOperands;
    if (numOperandsToRead.has_value())
      numOperands = *numOperandsToRead;
    else if (failed(reader.readVarInt(
                 numOperands, std::numeric_limits<uint32_t>::max() - 1)))
      return reader.emitError() << "failed to read operand count";

    results.reserve(numOperands);
    for (uint64_t i = 0; i < numOperands; ++i) {
      uint64_t operandIdx;
      if (failed(reader.readVarInt(operandIdx)))
        return reader.emitError() << "failed to read operand index " << i;
      if (operandIdx >= valueIndexList.size())
        return reader.emitError()
               << "operand index " << operandIdx
               << " out of bounds (size=" << valueIndexList.size()
               << ") for operand " << i;
      results.push_back(valueIndexList[operandIdx]);
    }
    return success();
  }

  /// Helper function to parse a given block during deserialization.
  static LogicalResult
  parseBlock(EncodingReader &reader, OpBuilder &builder, Location loc,
             Block &targetBlock, std::vector<Value> &valueIndexList,
             ArrayRef<ArrayRef<uint8_t>> constants, LazyTypeTable &types,
             DenseElementsAttrCache &constCache,
             DebugInfoReader::Iterator &diIterator, MLIRContext &context) {
    // Read number of block arguments
    uint64_t numBlockArgs;
    if (failed(reader.readVarInt(numBlockArgs)))
      return reader.emitError() << "failed to read block argument count.";

    // Record the current size of valueIndexList. Block arguments and operations
    // defined within this block will be added, and then the list will be
    // resized back to this original size upon exiting the block.
    size_t originalValueIndexListSize = valueIndexList.size();

    // Read argument types and create block arguments in the targetBlock.
    for (uint64_t i = 0; i < numBlockArgs; ++i) {
      Type argType = types.readAndGetType(reader);
      if (!argType)
        return reader.emitError()
               << "failed to read block argument type: " << i;
      Value arg = targetBlock.addArgument(argType, loc);
      valueIndexList.push_back(arg);
    }

    // Read number of operations in the block.
    uint64_t numOps;
    if (failed(reader.readVarInt(numOps)))
      return reader.emitError() << "failed to read block operation count.";

    // Set insertion point to the end of the targetBlock for parsing operations.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&targetBlock);

    // Parse operations in the block using the valueIndexList.
    for (uint64_t i = 0; i < numOps; ++i) {
      if (failed(InstructionParser::parseOperation(
              reader, builder, valueIndexList, constants, types, constCache,
              diIterator, context)))
        return reader.emitError()
               << "failed to parse operation " << i << " in block.";
    }

    // Validate block structure: ensure block has terminator.
    if (!targetBlock.empty()) {
      Operation *lastOp = &targetBlock.back();
      if (!lastOp->hasTrait<OpTrait::IsTerminator>())
        return reader.emitError()
               << "invalid block structure: block is expected to have a "
                  "terminator "
               << "operation, but the last operation '" << lastOp->getName()
               << "' is not a terminator.";
    } else {
      return reader.emitError()
             << "invalid block structure: block is expected to have a "
                "terminator operation, but it is empty";
    }

    // Restore the valueIndexList to its original size, removing arguments
    // and operation results defined within this block.
    valueIndexList.resize(originalValueIndexListSize);
    return success();
  }

  /// Helper function to parse a region during deserialization.
  static LogicalResult
  parseRegion(EncodingReader &reader, OpBuilder &builder, Location loc,
              std::vector<Value> &parentValueIndexList,
              ArrayRef<ArrayRef<uint8_t>> constants, LazyTypeTable &types,
              DenseElementsAttrCache &constCache,
              DebugInfoReader::Iterator &diIterator, MLIRContext &context,
              Region &regionToPopulate) {
    // Read number of blocks in the region.
    uint64_t numBlocks;
    if (failed(reader.readVarInt(numBlocks)))
      return reader.emitError() << "failed to read region block count";

    // Parse each block in the region.
    for (uint64_t i = 0; i < numBlocks; ++i) {
      Block &currentBlock = regionToPopulate.emplaceBlock();
      // The value context for this block's arguments and operations starts
      // with values defined in the parent scope.
      if (failed(parseBlock(reader, builder, loc, currentBlock,
                            parentValueIndexList, constants, types, constCache,
                            diIterator, context)))
        return reader.emitError()
               << "failed to parse block " << i << " in region.";
    }

    return success();
  }

  // ===----------------------------------------------------------------------===//
  // Helper Functions for Attribute Deserialization
  // ===----------------------------------------------------------------------===//

  /// Parses an APInt from the bytecode stream.
  static LogicalResult parseAPInt(EncodingReader &reader, unsigned bitWidth,
                                  APInt &apIntResult) {
    // Small values are encoded using a single byte.
    if (bitWidth <= 8) {
      uint8_t value;
      if (failed(reader.readLE(value)))
        return reader.emitError()
               << "failed to read byte for APInt (<= 8 bits).";
      // Validate that the value fits in the specified bit width.
      if (!llvm::isUIntN(bitWidth, value))
        return reader.emitError()
               << "value " << static_cast<unsigned>(value)
               << " does not fit in " << bitWidth << " bits.";
      apIntResult = APInt(bitWidth, value);
      return success();
    }

    // Large values up to 64 bits are encoded using a single varint.
    if (bitWidth <= 64) {
      uint64_t value;
      if (failed(reader.readSignedVarInt(value)))
        return reader.emitError()
               << "failed to read signed varint for APInt (<= 64 bits).";
      // Validate that the value fits in the specified bit width.
      if (!llvm::isUIntN(bitWidth, value))
        return reader.emitError() << "value " << value << " does not fit in "
                                  << bitWidth << " bits";
      apIntResult = APInt(bitWidth, value);
      return success();
    }

    // Otherwise, for really big values we encode the array of active words in
    // the value.
    uint64_t numActiveWords;
    if (failed(reader.readVarInt(numActiveWords)))
      return reader.emitError()
             << "failed to read numActiveWords for APInt (> 64 bits).";
    // Validate that numActiveWords makes sense for the given bitWidth.
    uint64_t expectedMaxWords = (bitWidth + 63) / 64;
    if (numActiveWords > expectedMaxWords)
      return reader.emitError()
             << "numActiveWords " << numActiveWords << " exceeds maximum of "
             << expectedMaxWords << " for " << bitWidth << " bit";
    if (numActiveWords == 0)
      return reader.emitError()
             << "numActiveWords cannot be zero for multi-word APInt";

    SmallVector<uint64_t, 4> words(numActiveWords);
    for (uint64_t i = 0; i < numActiveWords; ++i)
      if (failed(reader.readSignedVarInt(words[i])))
        return reader.emitError()
               << "failed to read word " << i << " for multi-word APInt.";
    apIntResult = APInt(bitWidth, words);
    return success();
  }

  /// Parses a scalar attribute that was serialized directly (inline).
  /// Currently supports:
  /// - IntegerAttr (i1 through i64)
  /// - FloatAttr (all standard float types)
  static LogicalResult parseScalarAttributeInline(EncodingReader &reader,
                                                  MLIRContext &context,
                                                  Type expectedType,
                                                  Attribute &result) {
    if (auto intType = dyn_cast_or_null<IntegerType>(expectedType)) {
      unsigned width = intType.getWidth();
      if (width == 0 || width > 64)
        return reader.emitError()
               << "unsupported width for inline integer attribute: " << width;

      if (width == 1) {
        uint8_t byte;
        if (failed(reader.readLE(byte)))
          return reader.emitError()
                 << "failed to read byte for inline bool (i1)";
        result = BoolAttr::get(&context, byte != 0);
        return success();
      }

      uint64_t value;
      if (failed(reader.readVarInt(value)))
        return reader.emitError()
               << "failed to read VarInt for inline integer (width=" << width
               << ")";
      // Validate that the value fits in the specified bit width.
      if (!llvm::isUIntN(width, value))
        return reader.emitError()
               << "value " << value << " does not fit in " << width << " bits";

      APInt apValue(width, value);
      result = IntegerAttr::get(expectedType, apValue);
      return success();
    } else if (auto floatType = dyn_cast_or_null<FloatType>(expectedType)) {
      APInt parsedAPInt;
      unsigned bitWidth = APFloat::getSizeInBits(floatType.getFloatSemantics());
      if (failed(parseAPInt(reader, bitWidth, parsedAPInt)))
        return failure();
      APFloat apFloat(floatType.getFloatSemantics(), parsedAPInt);
      result = FloatAttr::get(floatType, apFloat);
      return success();
    }
    return reader.emitError()
           << "unsupported type for inline scalar parsing: " << expectedType;
  }

  // Parses a DenseElementsAttr (reads an index into the constant pool).
  // `expectedType` is the MLIR Type of the constant (e.g., TileType).
  static LogicalResult parseConstantAttrIndex(
      EncodingReader &reader, MLIRContext &context, Type expectedType,
      ArrayRef<ArrayRef<uint8_t>> constants, DenseElementsAttrCache &constCache,
      Attribute &result) {
    uint64_t constantIndex;
    if (failed(reader.readVarInt(constantIndex)))
      return reader.emitError() << "failed to read constant index";
    if (constantIndex >= constants.size())
      return reader.emitError()
             << "constant index " << constantIndex << " out of bounds";
    FailureOr<Attribute> attributeOrFailure =
        constCache.getOrCreate(expectedType, constants[constantIndex], context);
    if (failed(attributeOrFailure))
      return failure();
    result = *attributeOrFailure;
    return success();
  }

  /// Parses a DivByAttr attribute.
  static LogicalResult parseDivByAttr(EncodingReader &reader,
                                      MLIRContext &context,
                                      cuda_tile::DivByAttr &nativeValue) {
    uint64_t divisor;
    if (failed(reader.readVarInt(divisor)))
      return reader.emitError() << "failed to read divisor for DivByAttr";

    uint8_t flagsByte;
    if (failed(reader.readLE(flagsByte)))
      return reader.emitError() << "failed to read flags byte for DivByAttr";

    bool has_every = (flagsByte & 0x01) != 0;
    bool has_along = (flagsByte & 0x02) != 0;

    std::optional<int64_t> every_opt;
    if (has_every) {
      uint64_t val;
      if (failed(reader.readSignedVarInt(val)))
        return reader.emitError()
               << "failed to read value for 'every' in DivByAttr";
      every_opt = val;
    }

    std::optional<int64_t> along_opt;
    if (has_along) {
      uint64_t val;
      if (failed(reader.readSignedVarInt(val)))
        return reader.emitError()
               << "failed to read value for 'along' in DivByAttr";
      along_opt = val;
    }

    nativeValue =
        cuda_tile::DivByAttr::get(&context, divisor, every_opt, along_opt);
    return success();
  }

  /// Base template: Parse attribute and convert to native type T
  /// Note about expectedType:
  /// - REQUIRED for inline IntegerAttr to determine the bit width.
  /// - REQUIRED for DenseElementsAttr when parsing constant indices.
  /// - Passed recursively for nested structures like std::optional.
  /// - Optional/nullptr otherwise.
  template <typename T>
  static LogicalResult
  parseOpAttribute(EncodingReader &reader, MLIRContext &context,
                   LazyTypeTable &types, ArrayRef<ArrayRef<uint8_t>> constants,
                   DenseElementsAttrCache &constCache, T &nativeValue,
                   Type expectedType = nullptr) {
    Attribute parsedAttr;
    // The logic here determines how to read the attribute based on the
    // *expected C++ type T*, because the bytecode format doesn't explicitly
    // store how each attribute was encoded (inline vs index).
    if constexpr (std::is_same_v<T, UnitAttr>) {
      // UnitAttr presence is stored as inline bool (i1).
      BoolAttr presentAttr;
      if (failed(parseScalarAttributeInline(
              reader, context, IntegerType::get(&context, 1), presentAttr)))
        return failure();
      // Convert the parsed BoolAttr to UnitAttr (or nullptr if false)
      nativeValue = presentAttr.getValue() ? UnitAttr::get(&context) : nullptr;
      return success();
    } else if constexpr (std::is_same_v<T, BoolAttr>) {
      // BoolAttr is stored as inline bool (i1).
      return parseScalarAttributeInline(
          reader, context, IntegerType::get(&context, 1), nativeValue);
    } else if constexpr (std::is_same_v<T, IntegerAttr>) {
      if (!expectedType) {
        expectedType = types.readAndGetType(reader);
        if (!isa_and_nonnull<IntegerType>(expectedType))
          return reader.emitError()
                 << "failed to read valid IntegerType for IntegerAttr";
      }
      if (failed(parseScalarAttributeInline(reader, context, expectedType,
                                            parsedAttr)))
        return failure();
      nativeValue = dyn_cast_or_null<IntegerAttr>(parsedAttr);
      if (!nativeValue)
        return reader.emitError()
               << "failed to cast parsed attribute to IntegerAttr";
      return success();
    } else if constexpr (std::is_same_v<T, FloatAttr>) {
      if (!expectedType) {
        expectedType = types.readAndGetType(reader);
        if (!isa_and_nonnull<FloatType>(expectedType))
          return reader.emitError()
                 << "failed to read valid FloatType for FloatAttr";
      }
      if (failed(parseScalarAttributeInline(reader, context, expectedType,
                                            parsedAttr)))
        return failure();
      nativeValue = dyn_cast_or_null<FloatAttr>(parsedAttr);
      if (!nativeValue)
        return reader.emitError()
               << "failed to cast parsed attribute to FloatAttr";
      return success();
    } else if constexpr (std::is_same_v<T, TypeAttr>) {
      // TypeAttr is stored as an index into the type table.
      Type referencedType = types.readAndGetType(reader);
      if (!referencedType)
        return reader.emitError()
               << "failed to get referenced type for TypeAttr";
      nativeValue = TypeAttr::get(referencedType);
      return success();
    } else if constexpr (std::is_same_v<T, StringAttr>) {
      // StringAttr is stored as an index into the string table.
      StringRef strRef;
      if (failed(reader.readAndGetString(strRef)))
        return reader.emitError() << "failed to read StringAttr.";
      nativeValue = StringAttr::get(&context, strRef);
      return success();
    } else if constexpr (std::is_same_v<T, DenseI32ArrayAttr>) {
      SmallVector<int32_t, 4> values;
      if (failed(reader.readLEVarSize(values)))
        return reader.emitError() << "failed to read DenseI32ArrayAttr values.";
      // Validate array values.
      for (int32_t val : values)
        if (LLVM_UNLIKELY(val == 0x7fffffff || val == (-0x7fffffff - 1))) {
          return reader.emitError()
                 << "array contains unsupported value " << val;
        }

      nativeValue = DenseI32ArrayAttr::get(&context, values);
      return success();
    } else if constexpr (std::is_same_v<T, DenseI64ArrayAttr>) {
      SmallVector<int64_t, 4> values;
      if (failed(reader.readLEVarSize(values)))
        return reader.emitError() << "failed to read DenseI64ArrayAttr values.";
      // Validate array values.
      for (int64_t val : values)
        if (LLVM_UNLIKELY(val == 0x7fffffffffffffffLL ||
                          val == (-0x7fffffffffffffffLL - 1))) {
          return reader.emitError()
                 << "array contains unsupported value " << val;
        }

      nativeValue = DenseI64ArrayAttr::get(&context, values);
      return success();
    } else if constexpr (std::is_same_v<std::decay_t<T>,
                                        mlir::FlatSymbolRefAttr>) {
      StringRef strRef;
      if (failed(reader.readAndGetString(strRef)))
        return reader.emitError()
               << "failed to read string for FlatSymbolRefAttr.";
      nativeValue = mlir::FlatSymbolRefAttr::get(&context, strRef);
      return success();
    } else if constexpr (is_cuda_tile_enum_attr<T>::value) {
      return parseGenericEnumAttr(reader, context, nativeValue);
    } else if constexpr (std::is_base_of_v<DenseElementsAttr, T> ||
                         std::is_same_v<T, DenseElementsAttr>) {
      if (!expectedType) {
        Type denseMLIRType = types.readAndGetType(reader);
        expectedType = dyn_cast_or_null<TileType>(denseMLIRType);
        if (!expectedType)
          return reader.emitError() << "failed to read valid MLIR Type for "
                                       "self-contained DenseElementsAttr";
      }

      if (std::is_same_v<T, DenseIntOrFPElementsAttr>) {
        if (failed(parseConstantAttrIndex(reader, context, expectedType,
                                          constants, constCache, parsedAttr)))
          return failure();
        nativeValue = dyn_cast_or_null<T>(parsedAttr);
        if (!nativeValue)
          return reader.emitError() << "parsed constant attribute is not the "
                                       "expected type derived "
                                       "from DenseElementsAttr";
      } else if (std::is_same_v<T, DenseElementsAttr>) {
        uint64_t numStringAttrs;
        if (failed(reader.readVarInt(numStringAttrs)))
          return reader.emitError() << "failed to read number of string attrs "
                                       "in DenseElementsAttr";

        llvm::SmallVector<StringRef> strs;
        for (unsigned i = 0; i != numStringAttrs; ++i) {
          StringRef strRef;
          if (failed(reader.readAndGetString(strRef)))
            return reader.emitError()
                   << "failed to read string in DenseElementsAttr";
          strs.push_back(strRef);
        }

        auto stringType = cuda_tile::StringType::get(&context);
        auto tensorType = mlir::RankedTensorType::get(
            {static_cast<int64_t>(strs.size())}, stringType);
        nativeValue = dyn_cast<T>(DenseElementsAttr::get(tensorType, strs));

      } else
        return reader.emitError() << "unknown DenseElementsAttr based Attr";
      return success();
    } else if constexpr (std::is_same_v<
                             T,
                             mlir::cuda_tile::AssumePredicateAttrInterface>) {
      Attribute parsedAttr;
      if (failed(parseSelfContainedOpAttribute(
              reader, context, types, constants, constCache, parsedAttr)))
        return reader.emitError() << "failed to parse self-contained attribute "
                                     "for AssumePredicateAttrInterface";
      nativeValue = dyn_cast_or_null<T>(parsedAttr);
      if (!nativeValue)
        return reader.emitError() << "failed to cast parsed attribute to "
                                     "AssumePredicateAttrInterface";
      return success();
    } else if constexpr (std::is_same_v<T, cuda_tile::DivByAttr>) {
      return parseDivByAttr(reader, context, nativeValue);
    } else if constexpr (std::is_same_v<T, cuda_tile::SameElementsAttr>) {
      DenseI64ArrayAttr valuesAttr;
      if (failed(parseOpAttribute(reader, context, types, constants, constCache,
                                  valuesAttr)))
        return reader.emitError()
               << "failed to read DenseI64ArrayAttr for SameElementsAttr";
      nativeValue = cuda_tile::SameElementsAttr::get(&context, valuesAttr);
      return success();
    } else if constexpr (std::is_same_v<T, ArrayAttr>) {
      // ArrayAttr parsing.
      uint64_t arraySize;
      if (failed(reader.readVarInt(arraySize,
                                   std::numeric_limits<uint32_t>::max() - 1)))
        return reader.emitError() << "failed to read size for ArrayAttr.";

      SmallVector<Attribute> elements;
      elements.reserve(arraySize);
      for (uint64_t i = 0; i < arraySize; ++i) {
        Attribute parsedElement;
        if (failed(parseSelfContainedOpAttribute(
                reader, context, types, constants, constCache, parsedElement)))
          return reader.emitError()
                 << "failed to parse ArrayAttr element " << i;
        if (!isa_and_nonnull<TypedAttr>(parsedElement))
          return reader.emitError()
                 << "ArrayAttr contains non-TypedAttr elements or is null";
        elements.push_back(parsedElement);
      }
      nativeValue = ArrayAttr::get(&context, elements);
      return success();
    } else if constexpr (std::is_same_v<T, DictionaryAttr>) {
      uint64_t dictSize;
      if (failed(reader.readVarInt(dictSize,
                                   std::numeric_limits<uint32_t>::max() - 1)))
        return reader.emitError() << "failed to read size for DictionaryAttr";
      SmallVector<NamedAttribute> elements;
      elements.reserve(dictSize);
      for (uint64_t i = 0; i < dictSize; ++i) {
        StringRef key;
        if (failed(reader.readAndGetString(key)))
          return reader.emitError()
                 << "failed to read key for DictionaryAttr element " << i;
        // Validate that the attribute name is not empty.
        if (key.empty())
          return reader.emitError()
                 << "invalid empty attribute name for DictionaryAttr element "
                 << i;
        Attribute value;
        if (failed(parseSelfContainedOpAttribute(reader, context, types,
                                                 constants, constCache, value)))
          return reader.emitError()
                 << "failed to parse DictionaryAttr value for key " << key;
        elements.emplace_back(StringAttr::get(&context, key), value);
      }
      if (auto duplicate = DictionaryAttr::findDuplicate(elements, false))
        return reader.emitError()
               << "failed to parse DictionaryAttr, duplicate key found: "
               << duplicate.value().getName();
      nativeValue = DictionaryAttr::get(&context, elements);
      return success();
    } else if constexpr (std::is_same_v<T, cuda_tile::OptimizationHintsAttr>) {
      // OptimizationHintsAttr contains a DictionaryAttr.
      DictionaryAttr dictAttr;
      if (failed(parseOpAttribute(reader, context, types, constants, constCache,
                                  dictAttr, expectedType)))
        return failure();
      nativeValue = cuda_tile::OptimizationHintsAttr::getChecked(
          [&]() {  return reader.emitError(); }, &context, dictAttr);
      if (!nativeValue)
        return reader.emitError() << "failed to parse OptimizationHintsAttr";
      return success();
    } else if constexpr (std::is_same_v<T, cuda_tile::BoundedAttr>) {
      uint8_t flagsByte;
      if (failed(reader.readLE(flagsByte)))
        return reader.emitError()
               << "failed to read flags byte for BoundedAttr";
      bool hasLb = (flagsByte & 0x01) != 0;
      bool hasUb = (flagsByte & 0x02) != 0;
      std::optional<int64_t> lb = std::nullopt;
      std::optional<int64_t> ub = std::nullopt;
      if (hasLb) {
        uint64_t lbVal = 0;
        if (failed(reader.readSignedVarInt(lbVal)))
          return reader.emitError()
                 << "failed to read lower bound for BoundedAttr";
        lb = lbVal;
      }
      if (hasUb) {
        uint64_t ubVal = 0;
        if (failed(reader.readSignedVarInt(ubVal)))
          return reader.emitError()
                 << "failed to read upper bound for BoundedAttr";
        ub = ubVal;
      }
      nativeValue = cuda_tile::BoundedAttr::get(&context, lb, ub);
      return success();
    } else {
      // Add specific cases above for any other attribute types needed.
      return reader.emitError()
             << "attribute deserialization not implemented for "
                "the requested C++ type";
    }
  }

  /// Specialization for std::optional<T>
  template <typename T>
  static LogicalResult
  parseOpAttribute(EncodingReader &reader, MLIRContext &context,
                   LazyTypeTable &types, ArrayRef<ArrayRef<uint8_t>> constants,
                   DenseElementsAttrCache &constCache,
                   std::optional<T> &nativeValue, Type expectedType = nullptr) {
    uint8_t isPresent;
    if (failed(reader.readLE(isPresent)))
      return reader.emitError()
             << "failed to read presence byte for optional attribute";
    if (isPresent == 0x01) {
      T value;
      // Call the non-optional version to parse the actual attribute value.
      if (failed(parseOpAttribute(reader, context, types, constants, constCache,
                                  value, expectedType)))
        return failure();
      nativeValue = value;
    } else if (isPresent == 0x00) {
      nativeValue = std::nullopt;
    } else {
      return reader.emitError()
             << "invalid presence byte for optional attribute: "
             << static_cast<int>(isPresent);
    }
    return success();
  }

public:
  /// Parses a self-contained attribute, including its tag and data.
  static LogicalResult parseSelfContainedOpAttribute(
      EncodingReader &reader, MLIRContext &context, LazyTypeTable &types,
      ArrayRef<ArrayRef<uint8_t>> constants, DenseElementsAttrCache &constCache,
      Attribute &resultAttr) {
    uint64_t attributeTag;
    if (failed(reader.readVarInt(attributeTag)))
      return reader.emitError()
             << "failed to read AttributeTag for self-contained attribute.";

    auto parseAttr = [&](auto &attr) {
      if (failed(parseOpAttribute(reader, context, types, constants, constCache,
                                  attr, nullptr)))
        return failure();
      resultAttr = attr;
      return success();
    };

    switch (static_cast<Bytecode::AttributeTag>(attributeTag)) {
    case Bytecode::AttributeTag::Integer: {
      IntegerAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::Float: {
      FloatAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::Bool: {
      BoolAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::Type: {
      TypeAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::String: {
      StringAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::Array: {
      ArrayAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::DenseElements: {
      DenseElementsAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::DivBy: {
      cuda_tile::DivByAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::SameElements: {
      cuda_tile::SameElementsAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::Dictionary: {
      DictionaryAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::OptimizationHints: {
      cuda_tile::OptimizationHintsAttr elem;
      return parseAttr(elem);
    }
    case Bytecode::AttributeTag::Bounded: {
      cuda_tile::BoundedAttr elem;
      return parseAttr(elem);
    }
    default:
      return reader.emitError() << "unsupported AttributeTag " << attributeTag
                                << " for self-contained attribute";
    }
  }

// Contains generated implementations of the operation-specific
// bytecode reading functions.
#define GEN_OP_READERS
#include "BytecodeReader.inc"

  static LogicalResult
  parseOperation(EncodingReader &reader, OpBuilder &innerBuilder,
                 std::vector<Value> &valueIndexList,
                 ArrayRef<ArrayRef<uint8_t>> constants, LazyTypeTable &types,
                 DenseElementsAttrCache &constCache,
                 DebugInfoReader::Iterator &diIterator, MLIRContext &context) {
    uint64_t opcode;
    if (failed(reader.readVarInt(opcode)))
      return reader.emitError() << "failed to read operation opcode.";

    // Get the location for this operation.
    auto loc = diIterator.next<LocationAttr>();
    if (!loc)
      return reader.emitError() << "failed to read operation location.";

      // Includes the generated switch statement for dispatching to the
      // appropriate 'parse<OpName>' function based on the opcode.
#define GEN_OP_READER_DISPATCH
#include "BytecodeReader.inc"

    return success();
  }
};

} // namespace

// debuginfo-section =:
//   diOpsNum[varint]          // Total number of operations with debug info
//   padding[bytes]            // Align to 4 bytes
//   diIndexOffsets[uint32_t]  // Per op offset into the debug info indices
//   diIndicesNum[varint]      // Total number of debug info indices
//   padding[bytes]            // Align to 8 bytes
//   diIndices[uint64_t]       // Array of debug indices to debug info
//   attributes diAttrNum[varint]         // Total number of debug info
//   attributes padding[bytes]            // Align to 4 bytes
//   diOffsets[uint32_t]       // Per debug info attribute offset into the debug
//   info data diData[bytes]             // Data for each debug info attribute
//
// diData =:
//   DebugTag[byte]            // Indicates the debug info attribute type
//   debuginfo-encoding        // Format depends on DebugTag
static LogicalResult parseDebugSection(ArrayRef<uint8_t> payload,
                                       DebugInfoReader &debuginfo,
                                       MLIRContext &context) {
  EncodingReader reader(payload, context);

  // Read the total number of operations with debug info.
  uint64_t diOpsNum;
  if (failed(reader.readVarInt(diOpsNum)))
    return reader.emitError()
           << "failed to read total number of operations with debug info";

  // Align to 4 bits for the uint32_t diIndexOffsetsPtr.
  auto alignment = alignof(uint32_t);
  if (failed(reader.skipPadding(alignment)))
    return reader.emitError()
           << "failed to skip padding for debug info index offset pointer";

  // Read the per op offset into the debug info indices.
  const uint32_t *diIndexOffsetsPtr =
      reinterpret_cast<const uint32_t *>(reader.getCurrentPtr());
  if (!diIndexOffsetsPtr)
    return reader.emitError()
           << "failed to read debug info index offset pointer.";

  ArrayRef<uint32_t> diIndexOffsets(diIndexOffsetsPtr, diOpsNum);
  if (failed(reader.skip(diOpsNum * sizeof(uint32_t))))
    return reader.emitError() << "failed to skip debug info index offsets";

  // Read the total number of debug info indices.
  uint64_t diIndicesNum = 0;
  if (failed(reader.readVarInt(diIndicesNum)))
    return reader.emitError()
           << "failed to read total number of debug info indices";

  // Align to 8 bytes for the uint64_t diIndicesPtr.
  auto uint64Alignment = alignof(uint64_t);
  if (failed(reader.skipPadding(uint64Alignment)))
    return reader.emitError()
           << "failed to skip padding for debug info indices pointer";

  // Read the array of debug indices to debug info attributes.
  const uint64_t *diIndicesPtr =
      reinterpret_cast<const uint64_t *>(reader.getCurrentPtr());
  if (!diIndicesPtr)
    return reader.emitError() << "failed to read debug info indices pointer";

  ArrayRef<uint64_t> diIndices(diIndicesPtr, diIndicesNum);
  if (failed(reader.skip(diIndicesNum * sizeof(uint64_t))))
    return reader.emitError() << "failed to skip debug info indices";

  // Read the total number of debug info attributes.
  uint64_t diAttrNum;
  if (failed(reader.readVarInt(diAttrNum)))
    return reader.emitError()
           << "failed to read total number of debug info attributes";

  if (diAttrNum >
      (payload.size() - reader.currentOffset()) / sizeof(uint32_t)) {
    return reader.emitError()
           << "number of debug info attributes (" << diAttrNum
           << ") exceeds the maximum of "
           << (payload.size() - reader.currentOffset()) / sizeof(uint32_t)
           << " that can fit in the remaining payload of "
           << (payload.size() - reader.currentOffset()) << " bytes.";
  }

  // Align to 4 bits for the uint32_t diOffsetsPtr.
  if (failed(reader.skipPadding(alignment)))
    return reader.emitError()
           << "failed to skip padding for debug info offset pointer";

  // Read per debug info attribute offset into the debug info data.
  const uint32_t *diOffsetsPtr =
      reinterpret_cast<const uint32_t *>(reader.getCurrentPtr());
  if (!diOffsetsPtr)
    return reader.emitError() << "failed to read debug info offset pointer";

  ArrayRef<uint32_t> diOffsets(diOffsetsPtr, diAttrNum);
  if (failed(reader.skip(diAttrNum * sizeof(uint32_t))))
    return reader.emitError() << "failed to skip debug info offsets";

  // Read data for each debug info attribute.
  ArrayRef<uint8_t> diData = payload.slice(reader.currentOffset());

  debuginfo.initialize(diIndices, diIndexOffsets, diData, diOffsets);
  return success();
}

//===----------------------------------------------------------------------===//
// Function Section
//===----------------------------------------------------------------------===//
// function-table-section =:
//   numFunctions[varint]
//   function-entry*
//
// function-entry =:
//   nameIndex[varint]         // Index into the string table.
//   signatureIndex[varint]    // Index into the type table.
//   functionLocIndex[varint]  // Index into the location table for the function
//   instruction location info. bodyLength[varint]      // Length of the
//   function body in bytes. functionBody[bytes]       // The function body data
//   itself.
//
// function-body =:
//   instruction*
//
namespace {
struct FunctionInfo {
  uint64_t nameIndex;
  uint64_t signatureIndex;
  uint64_t functionLocIndex;
  uint8_t entryFlag;
  uint64_t lengthOfFunction;
  StringRef functionBody;
  Attribute optimizationHints;
};
} // end anonymous namespace

/// Parses the function table section and creates metadata for each function.
static LogicalResult parseFunctionTableSection(
    ArrayRef<uint8_t> payload, std::vector<FunctionInfo> &functionInfoList,
    const EncodingReader &reader, LazyTypeTable &types,
    ArrayRef<ArrayRef<uint8_t>> constants, DenseElementsAttrCache &constCache,
    MLIRContext &context) {
  EncodingReader sectionReader(payload, context);
  sectionReader.inheritStringTableFrom(reader);
  uint64_t numFunctions;
  if (failed(sectionReader.readVarInt(numFunctions)))
    return failure();
  if (numFunctions > payload.size())
    return sectionReader.emitError()
           << "number of functions (" << numFunctions
           << ") exceeds payload size (" << payload.size() << ")";
  // Read each function's metadata
  functionInfoList.reserve(numFunctions);
  for (uint64_t i = 0; i < numFunctions; ++i) {
    FunctionInfo funcInfo;
    // Read the name index as a varint.
    if (failed(sectionReader.readVarInt(funcInfo.nameIndex)))
      return failure();
    // Read the signature index as a varint.
    if (failed(sectionReader.readVarInt(funcInfo.signatureIndex)))
      return failure();
    // Read the entry flag byte.
    if (failed(sectionReader.readLE(funcInfo.entryFlag)))
      return failure();
    // Read the function location index as a varint.
    if (failed(sectionReader.readVarInt(funcInfo.functionLocIndex)))
      return failure();

    // Read optimization hints if the flag is set for EntryOp.
    bool isEntry =
        (funcInfo.entryFlag &
         static_cast<uint8_t>(Bytecode::FunctionFlags::KindKernel)) != 0;
    bool hasOptHints =
        (funcInfo.entryFlag &
         static_cast<uint8_t>(Bytecode::FunctionFlags::HasOptimizationHints)) !=
        0;
    if (isEntry && hasOptHints) {
      if (failed(InstructionParser::parseSelfContainedOpAttribute(
              sectionReader, context, types, constants, constCache,
              funcInfo.optimizationHints)))
        return failure();
    }

    // Read the length of the function as a varint.
    if (failed(sectionReader.readVarInt(funcInfo.lengthOfFunction)))
      return failure();

    // Validate function length.
    if (funcInfo.lengthOfFunction > std::numeric_limits<size_t>::max())
      return sectionReader.emitError()
             << "function body length " << funcInfo.lengthOfFunction
             << " exceeds maximum addressable size ("
             << std::numeric_limits<size_t>::max() << " bytes)";

    // Check that we have enough remaining bytes.
    if (funcInfo.lengthOfFunction > sectionReader.remaining())
      return sectionReader.emitError()
             << "function body length " << funcInfo.lengthOfFunction
             << " exceeds remaining bytecode data ("
             << sectionReader.remaining() << " bytes)";

    // Read the function body as raw bytes.
    ArrayRef<uint8_t> bodyBytes;
    if (failed(sectionReader.readBytes(funcInfo.lengthOfFunction, bodyBytes)))
      return failure();

    if (bodyBytes.empty() && funcInfo.lengthOfFunction > 0)
      return sectionReader.emitError()
             << "failed to read " << funcInfo.lengthOfFunction
             << " bytes for function body";

    funcInfo.functionBody =
      StringRef(reinterpret_cast<const char *>(bodyBytes.data()),
                    funcInfo.lengthOfFunction);

    functionInfoList.emplace_back(funcInfo);
  }
  return success();
}

/// Parses the function body bytecode and creates the corresponding operations.
static LogicalResult
parseFunctionBody(ArrayRef<uint8_t> bodyBytes, OpBuilder &innerBuilder,
                  std::vector<Value> &valueIndexList,
                  DebugInfoReader::Iterator &diIterator,
                  ArrayRef<ArrayRef<uint8_t>> constants, LazyTypeTable &types,
                  DenseElementsAttrCache &constCache, MLIRContext &context,
                  const EncodingReader &mainFileStreamReader) {
  EncodingReader bodyReader(bodyBytes, context);
  // Inherit the string table from the main file stream reader.
  bodyReader.inheritStringTableFrom(mainFileStreamReader);

  while (bodyReader.remaining() > 0)
    if (failed(InstructionParser::parseOperation(
            bodyReader, innerBuilder, valueIndexList, constants, types,
            constCache, diIterator, context)))
      return failure();
  return success();
}

/// Creates a function based on the parsed FunctionInfo.
static LogicalResult
createFunction(const FunctionInfo &funcInfo, OpBuilder &builder,
               OpBuilder &funcBuilder, const EncodingReader &reader,
               LazyTypeTable &types, DebugInfoReader &debuginfo,
               ArrayRef<ArrayRef<uint8_t>> constants,
               DenseElementsAttrCache &constCache,
               std::vector<Value> &valueIndexList, MLIRContext &context) {
  StringRef funcNameStr;
  if (failed(reader.getString(funcInfo.nameIndex, funcNameStr, context)))
    return reader.emitError() << "failed to get function name string at index "
                              << funcInfo.nameIndex;
  StringAttr funcName = builder.getStringAttr(funcNameStr);
  // Get the function type lazily from the type table.
  if (funcInfo.signatureIndex >= types.size())
    return reader.emitError()
           << "function signature index " << funcInfo.signatureIndex
           << " out of bounds for function '" << funcNameStr << "'";
  Type signatureType = types.getType(funcInfo.signatureIndex);
  if (!signatureType)
    return reader.emitError()
           << "failed to parse type at index " << funcInfo.signatureIndex
           << " for function '" << funcNameStr;
  FunctionType funcType = mlir::dyn_cast<FunctionType>(signatureType);
  if (!funcType)
    return reader.emitError()
           << "function signature index " << funcInfo.signatureIndex
           << " does not refer to a function type for function '" << funcNameStr
           << "', got type: " << signatureType;
  auto diIterator = debuginfo.getIterator(funcInfo.functionLocIndex);
  auto funcLoc = diIterator.next<LocationAttr>();
  if (!funcLoc)
    return reader.emitError()
           << "failed to read function location for '" << funcNameStr << "'";

  // Determine if it's an EntryOp based on the flag
  bool isEntry =
      (funcInfo.entryFlag &
       static_cast<uint8_t>(Bytecode::FunctionFlags::KindKernel)) != 0;

  // TODO: Handle visibility flag (Bit 0) when supported.

  SmallVector<Attribute, 4> argAttrs;
  argAttrs.reserve(funcType.getNumInputs());
  for (size_t i = 0; i < funcType.getNumInputs(); ++i)
    argAttrs.emplace_back(builder.getDictionaryAttr({}));
  SmallVector<Attribute, 4> retAttrs;
  retAttrs.reserve(funcType.getNumResults());
  ArrayAttr funcArgAttrs = builder.getArrayAttr(argAttrs);
  for (size_t i = 0; i < funcType.getNumResults(); ++i)
    retAttrs.emplace_back(builder.getDictionaryAttr({}));
  ArrayAttr funcRetAttrs = builder.getArrayAttr(retAttrs);

  // Create the appropriate operation type
  mlir::FunctionOpInterface funcOpIFace;
  if (isEntry) {
    // Use optimization hints from bytecode or create default empty hints
    OptimizationHintsAttr funcOptHintAttr =
        funcInfo.optimizationHints
            ? dyn_cast<OptimizationHintsAttr>(funcInfo.optimizationHints)
            : OptimizationHintsAttr::get(&context,
                                         builder.getDictionaryAttr({}));

    if (!funcOptHintAttr) {
      return reader.emitError()
             << "invalid optimization hints attribute for function '"
             << funcNameStr << "'";
    }

    funcOpIFace = funcBuilder.create<cuda_tile::EntryOp>(
        funcLoc, funcName, TypeAttr::get(funcType), funcArgAttrs, funcRetAttrs,
        funcOptHintAttr);
  } else {
    return reader.emitError()
           << "un-expected non-entry function '" << funcNameStr << "'";
  }

  auto &entryBlock = *funcOpIFace.addEntryBlock();
  OpBuilder innerBuilder(&entryBlock, entryBlock.begin());
  valueIndexList.clear();
  for (BlockArgument arg : entryBlock.getArguments())
    valueIndexList.emplace_back(arg);
  ArrayRef<uint8_t> bodyBytes(
      reinterpret_cast<const uint8_t *>(funcInfo.functionBody.data()),
      funcInfo.functionBody.size());
  // Parse the function body  instructions.
  if (failed(parseFunctionBody(bodyBytes, innerBuilder, valueIndexList,
                               diIterator, constants, types, constCache,
                               context, reader)))
    return reader.emitError() << "failed to parse function body for function '"
                              << funcNameStr << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// Global Section
//===----------------------------------------------------------------------===//
// global-section =:
//   numGlobals[varint]
//   padding[bytes]             // Align to 8 bytes.
//   global-entry*
//
// global-entry =:
//   symbolNameIndex[varint]    // Index into the string table.
//   valueTypeIndex[varint]     // Index into the type table.
//   constantValueIndex[varint] // Index into the constant table.
//   alignment[varint]          // Alignment of the global variable.
//
namespace {
struct GlobalInfo {
  uint64_t symbolNameIndex;
  uint64_t valueTypeIndex;
  uint64_t constantValueIndex;
  uint64_t alignment;
};
} // end anonymous namespace

/// Parses the global section and creates metadata for each global variable.
static LogicalResult parseGlobalSection(ArrayRef<uint8_t> payload,
                                        const EncodingReader &mainReader,
                                        std::vector<GlobalInfo> &globalInfoList,
                                        MLIRContext &context) {
  EncodingReader sectionReader(payload, context);
  sectionReader.inheritStringTableFrom(mainReader);

  uint64_t numGlobals;
  if (failed(sectionReader.readVarInt(numGlobals)))
    return sectionReader.emitError() << "failed to read number of global.";

  // A global entry has at least 4 varints, each at least 1 byte.
  static constexpr size_t kMinGlobalInfoSize = 4;
  if (numGlobals >
      (payload.size() - sectionReader.currentOffset()) / kMinGlobalInfoSize) {
    return sectionReader.emitError()
           << "number of globals (" << numGlobals << ") exceeds the maximum of "
           << (payload.size() - sectionReader.currentOffset()) /
                  kMinGlobalInfoSize
           << " that can fit in the remaining payload of "
           << (payload.size() - sectionReader.currentOffset()) << " bytes.";
  }

  globalInfoList.reserve(numGlobals);

  for (uint64_t i = 0; i < numGlobals; ++i) {
    GlobalInfo globalInfo;
    // 1. Read symbol name index.
    if (failed(sectionReader.readVarInt(globalInfo.symbolNameIndex)))
      return sectionReader.emitError()
             << "failed to read global symbol name string.";

    // 2. Read type index of the value.
    if (failed(sectionReader.readVarInt(globalInfo.valueTypeIndex)))
      return sectionReader.emitError() << "failed to read global value type";

    // 3. Read constant index for the value.
    if (failed(sectionReader.readVarInt(globalInfo.constantValueIndex)))
      return sectionReader.emitError()
             << "failed to read global value constant index";

    // 4. Read alignment.
    if (failed(sectionReader.readVarInt(globalInfo.alignment)))
      return sectionReader.emitError() << "failed to read global alignment";

    globalInfoList.emplace_back(globalInfo);
  }
  return success();
}

/// Creates a global (cuda_tile::GlobalOp) based on the parsed GlobalInfo.
static LogicalResult
createGlobal(const GlobalInfo &globalInfo, OpBuilder &builder,
             const EncodingReader &reader, LazyTypeTable &types,
             ArrayRef<ArrayRef<uint8_t>> constants,
             DenseElementsAttrCache &constCache, DebugInfoReader &debuginfo,
             MLIRContext &context) {
  if (globalInfo.constantValueIndex >= constants.size())
    return reader.emitError() << "global value constant index out of bounds";

  StringRef symNameStr;
  if (failed(reader.getString(globalInfo.symbolNameIndex, symNameStr, context)))
    return reader.emitError()
           << "failed to get global symbol name string for index "
           << globalInfo.symbolNameIndex;

  FailureOr<Attribute> valueAttr =
      constCache.getOrCreate(types.getType(globalInfo.valueTypeIndex),
                             constants[globalInfo.constantValueIndex], context);
  if (failed(valueAttr))
    return failure();

  auto denseValueAttr = dyn_cast<DenseIntOrFPElementsAttr>(*valueAttr);
  if (!denseValueAttr)
    return reader.emitError()
           << "parsed global constant attribute is not the expected type "
              "derived from DenseIntOrFPElementsAttr";

  // Global variables must not have DILocAttr location type because CudaTile
  // supports only local scope. Therefore, global variables must have UnknownLoc
  // location type - the only other legal location type.
  builder.create<cuda_tile::GlobalOp>(UnknownLoc::get(&context), symNameStr,
                                      denseValueAttr, globalInfo.alignment);
  return success();
}

//===----------------------------------------------------------------------===//
// readBytecode Function Implementation
// Implements the core functionality of reading bytecode from a memory buffer
// and constructing the corresponding cuda_tile::ModuleOp.
//===----------------------------------------------------------------------===//

std::optional<size_t> cuda_tile::getBytecodeSize(const char *bytecodeBuffer) {
  if (!isTileIRBytecode(bytecodeBuffer))
    return std::nullopt;

  auto charBuffer = reinterpret_cast<const unsigned char*>(bytecodeBuffer);
  if (charBuffer[sizeof(kTileIRBytecodeMagic)] == 0)
    return std::nullopt;

  // Build a buffer assuming we have the maximum size of the bytecode, we'll
  // infer the actual size as we parse the bytecode.
  ArrayRef<uint8_t> bytecodeData(
      reinterpret_cast<const uint8_t *>(bytecodeBuffer),
      SIZE_MAX - reinterpret_cast<uintptr_t>(bytecodeBuffer));

  // Set up the reader and context.
  MLIRContext context(MLIRContext::Threading::DISABLED);
  EncodingReader reader(bytecodeData, context);
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &diag) {
    // Ignore all errors.
  });

  // Parse the header of the bytecode.
  if (failed(parseHeader(reader, context)))
    return std::nullopt;

  // Parse the sections until we reach the end of the bytecode. We don't
  // actually try to reason about the section data, we just want to know the
  // sizes.
  std::array<bool, Section::NumSections> seenSections;
  seenSections.fill(false);
  while (true) {
    // Parse the next section.
    SectionHeader header;
    if (failed(parseSectionHeader(reader, header, context)) ||
        failed(reader.skip(header.length)) ||
        std::exchange(seenSections[header.sectionID], true))
      return std::nullopt;

    // Check for the end of the bytecode stream.
    if (header.sectionID == Section::EndOfBytecode)
      return reader.currentOffset();
  }
}

OwningOpRef<cuda_tile::ModuleOp>
cuda_tile::readBytecode(llvm::MemoryBufferRef bytecodeBuffer,
                        MLIRContext &context) {
  ArrayRef<uint8_t> bytecodeData(
      reinterpret_cast<const uint8_t *>(bytecodeBuffer.getBufferStart()),
      bytecodeBuffer.getBufferSize());

  EncodingReader reader(bytecodeData, context);
  DebugInfoReader debuginfo(context, reader);

  if (failed(parseHeader(reader, context)))
    return reader.emitError() << "failed to parse bytecode header", nullptr;

  // Store section payloads to allow parsing in a specific order later.
  std::array<std::optional<ArrayRef<uint8_t>>, Section::NumSections + 1>
      sectionPayloads;

  // Discover all sections and store their payloads.
  while (true) {
    SectionHeader header;
    if (failed(parseSectionHeader(reader, header, context)))
      return reader.emitError() << "failed to parse section header", nullptr;

    // Check for the end of the bytecode stream.
    if (header.sectionID == Section::EndOfBytecode) {
      if (reader.remaining() != 0) {
        reader.emitError() << "end section is not the last section";
        return nullptr;
      }
      break;
    }

    auto validateSectionAlignment = [&](uint64_t requiredAlignment) -> bool {
      if (!header.hasAlignment || (header.alignment % requiredAlignment) != 0) {
        reader.emitError() << "section " << static_cast<int>(header.sectionID)
                           << " must have alignment that is a multiple of "
                           << requiredAlignment << " bytes, but "
                           << (header.hasAlignment
                                   ? ("has alignment of " +
                                      std::to_string(header.alignment))
                                   : "has no alignment flag set");
        return false;
      }
      return true;
    };

    switch (header.sectionID) {
    case Section::String:
    case Section::Type:
      if (!validateSectionAlignment(alignof(uint32_t)))
        return nullptr;
      break;
    case Section::Constant:
    case Section::Debug:
    case Section::Func:
      if (!validateSectionAlignment(alignof(uint64_t)))
        return nullptr;
      break;
    case Section::Global:
      // Global section has variable alignment requirements, skip validation
      [[fallthrough]];
    default:
      // Unknown sections or sections with variable alignment requirements
      break;
    }

    // Read the section payload.
    if (header.length > reader.remaining()) {
      reader.emitError() << "section length " << header.length
                         << " exceeds remaining bytecode data. "
                         << reader.remaining();
      return nullptr;
    }
    ArrayRef<uint8_t> payload;
    if (failed(reader.readBytes(header.length, payload)))
      return reader.emitError() << "failed to read section payload", nullptr;

    sectionPayloads[header.sectionID] = payload;
  }

  // Initialize data structures for parsed sections.
  LazyTypeTable types(context);
  std::vector<ArrayRef<uint8_t>> constants;
  std::vector<FunctionInfo> functionInfoList;
  std::vector<Value> valueIndexList;
  DenseElementsAttrCache globalConstCache;
  std::vector<GlobalInfo> globalInfoList;

  // Process sections in dependency order using their stored payloads.
  // Parse String Section.
  if (sectionPayloads[Section::String].has_value()) {
    if (failed(parseStringSection(*sectionPayloads[Section::String], reader,
                                  context)))
      return reader.emitError() << "failed to parse string section", nullptr;
  } else {
    return reader.emitError() << "string section is mandatory", nullptr;
  }
  // Parse Type Section.
  if (sectionPayloads[Section::Type].has_value())
    if (failed(
            parseTypeSection(*sectionPayloads[Section::Type], types, context)))
      return reader.emitError() << "failed to parse type section", nullptr;
  // Parse Constant Section.
  if (sectionPayloads[Section::Constant].has_value())
    if (failed(parseConstantSection(*sectionPayloads[Section::Constant],
                                    constants, context)))
      return reader.emitError() << "failed to parse constant section", nullptr;
  // Parse Global Section.
  if (sectionPayloads[static_cast<uint8_t>(Bytecode::Section::Global)]
          .has_value())
    if (failed(parseGlobalSection(
            *sectionPayloads[static_cast<uint8_t>(Bytecode::Section::Global)],
            reader, globalInfoList, context)))
      return reader.emitError() << "failed to parse global section", nullptr;
  // Parse Function Section.
  if (sectionPayloads[Section::Func].has_value()) {
    if (failed(parseFunctionTableSection(*sectionPayloads[Section::Func],
                                         functionInfoList, reader, types,
                                         constants, globalConstCache, context)))
      return reader.emitError() << "failed to parse function section", nullptr;
  } else {
    return reader.emitError() << "function section is mandatory", nullptr;
  }
  // Parse Debug Section.
  if (sectionPayloads[Section::Debug].has_value())
    if (failed(parseDebugSection(*sectionPayloads[Section::Debug], debuginfo,
                                 context)))
      return reader.emitError() << "failed to parse debug section", nullptr;

  OwningOpRef<cuda_tile::ModuleOp> cudaTileModule(
      OpBuilder(&context).create<cuda_tile::ModuleOp>(UnknownLoc::get(&context),
                                                      "kernels"));
  OpBuilder builder(cudaTileModule->getBody());
  OpBuilder funcBuilder(&cudaTileModule->getBody().front(),
                        cudaTileModule->getBody().front().begin());
  for (const auto &globalInfo : globalInfoList)
    if (failed(createGlobal(globalInfo, builder, reader, types, constants,
                            globalConstCache, debuginfo, context)))
      return reader.emitError() << "failed to create global from bytecode",
             nullptr;
  for (const auto &funcInfo : functionInfoList)
    if (failed(createFunction(funcInfo, builder, funcBuilder, reader, types,
                              debuginfo, constants, globalConstCache,
                              valueIndexList, context)))
      return reader.emitError() << "failed to create function from bytecode",
             nullptr;
  if (failed(verify(cudaTileModule.get()))) {
    ::emitError(UnknownLoc::get(&context))
        << "verification failed for deserialized cuda_tile::ModuleOp";
    return nullptr;
  }
  return cudaTileModule;
}
