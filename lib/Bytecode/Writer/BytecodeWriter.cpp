//===- BytecodeWriter.cpp - CUDA Tile Bytecode Writer -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the BytecodeWriter for the cuda_tile dialect, enabling
// serialization of a cuda_tile module into a custom bytecode format.
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include "../BytecodeEnums.h"
#include "../Common/VersionUtils.h"
#include "cuda_tile/Bytecode/Common/BytecodeVerification.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/PreviewFeatures.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::cuda_tile;

static constexpr const char *kOptimizationHints = "optimization_hints";

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
//   sectionId[byte]   // Lower 7 bits = ID, high bit = hasAlignment
//   length[varint]    // Length of section in bytes
//   alignment[varint] // Optional: only present if high bit of sectionId is set
//   padding[bytes]    // Optional: alignment padding bytes (0xCF)
//   data[bytes]       // Section-specific data format
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EncodingWriter
// Provides utilities for writing encoded data to a stream.
//===----------------------------------------------------------------------===//
namespace {
class EncodingWriter {
public:
  EncodingWriter(raw_ostream &stream, uint64_t alignment = 1)
      : stream(stream), requiredAlignment(alignment) {}

  void writeByte(uint8_t byte) { stream.write(static_cast<char>(byte)); }

  template <typename Enum, std::enable_if_t<std::is_enum<Enum>::value, int> = 0>
  void writeByte(Enum value) {
    writeByte(static_cast<uint8_t>(value));
  }

  void writeVarInt(uint64_t value) {
    uint8_t bytes[10]; // Supports up to 64 bits
    size_t index = 0;

    do {
      uint8_t byte = value & 0x7F; // Lower 7 bits
      value >>= 7;
      if (value != 0) {
        byte |= 0x80; // Set continuation bit
      }
      bytes[index++] = byte;
    } while (value != 0 && index < sizeof(bytes));

    stream.write(reinterpret_cast<char *>(bytes), index);
  }

  template <typename Enum, std::enable_if_t<std::is_enum<Enum>::value, int> = 0>
  void writeVarInt(Enum value) {
    writeVarInt(static_cast<uint64_t>(value));
  }

  /// Emit a signed variable length integer. Signed varints are encoded using
  /// a varint with zigzag encoding, meaning that we use the low bit of the
  /// value to indicate the sign of the value. This allows for more efficient
  /// encoding of negative values by limiting the number of active bits
  void writeSignedVarInt(uint64_t value) {
    writeVarInt((value << 1) ^ (uint64_t)((int64_t)value >> 63));
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, void> writeLE(T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
      writeByte(value & 0xFF);
      // Only shift if there are more bytes to process
      if (sizeof(T) > 1 && i < sizeof(T) - 1) {
        value >>= 8;
      }
    }
  }

  template <typename T>
  void writeLE(ArrayRef<T> values) {
    for (T value : values) {
      writeLE<T>(value);
    }
  }

  template <typename T>
  void writeLEVarSize(ArrayRef<T> values) {
    writeVarInt(values.size());
    writeLE(values);
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void> writeLE(T value) {
    static_assert(std::numeric_limits<T>::is_iec559, "IEEE 754 required");
    using IntType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    IntType intValue;
    std::memcpy(&intValue, &value, sizeof(T));
    for (size_t i = 0; i < sizeof(T); ++i) {
      writeByte(intValue & 0xFF);
      intValue >>= 8;
    }
  }

  void write(const char *data, size_t size) { stream.write(data, size); }

  void write(char c) { writeByte(static_cast<uint8_t>(c)); }

  void write(StringRef str) { write(str.data(), str.size()); }

  uint64_t tell() const { return stream.tell(); }

  void alignTo(uint64_t alignment,
               uint8_t paddingByte = Bytecode::kAlignmentByte) {
    if (alignment < 2) {
      return;
    }
    uint64_t currentPos = tell();
    uint64_t padding = (alignment - (currentPos % alignment)) % alignment;
    for (uint64_t i = 0; i < padding; ++i) {
      writeByte(paddingByte);
    }
    // Update the required alignment
    requiredAlignment = std::max(requiredAlignment, alignment);
  }
  uint64_t getRequiredAlignment() const { return requiredAlignment; }

private:
  raw_ostream &stream;
  uint64_t requiredAlignment = 1;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Header Writer
//===----------------------------------------------------------------------===//

static LogicalResult writeHeader(raw_ostream &stream, Operation *op,
                                 const BytecodeWriterConfig &config) {
  // Validate the bytecode version.
  if (config.getBytecodeVersion() < BytecodeVersion::kMinSupportedVersion) {
    return op->emitError() << "unsupported version requested "
                           << config.getBytecodeVersion().toString()
                           << ", must be in range ["
                           << BytecodeVersion::kMinSupportedVersion.toString()
                           << ", "
                           << BytecodeVersion::kCurrentVersion.toString()
                           << ']';
  }

  EncodingWriter writer(stream);
  const char magicNumber[8] = {0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00};
  writer.write(magicNumber, sizeof(magicNumber));
  writer.writeLE(config.getBytecodeVersion().getMajor());
  writer.writeLE(config.getBytecodeVersion().getMinor());
  writer.writeLE(config.getBytecodeVersion().getTag());

  return success();
}

//===----------------------------------------------------------------------===//
// Section Header Writer
//===----------------------------------------------------------------------===//

static void writeSectionHeader(raw_ostream &stream, uint8_t sectionID,
                               uint64_t length, uint64_t alignment) {
  EncodingWriter writer(stream);
  uint8_t idAndIsAligned = sectionID & 0x7F;
  if (alignment > 1) {
    idAndIsAligned |= 0x80;
  }
  writer.writeByte(idAndIsAligned);
  writer.writeVarInt(length);
  if (alignment > 1) {
    writer.writeVarInt(alignment);
    writer.alignTo(alignment);
  }
}

/// Helper function to serialize an APInt.
static void writeAPInt(const APInt &apInt, EncodingWriter &writer) {
  unsigned bitWidth = apInt.getBitWidth();
  if (bitWidth <= 8) {
    writer.writeByte(static_cast<uint8_t>(apInt.getLimitedValue()));
  } else if (bitWidth <= 64) {
    writer.writeSignedVarInt(apInt.getLimitedValue());
  } else {
    unsigned numActiveWords = apInt.getActiveWords();
    writer.writeVarInt(numActiveWords);
    const uint64_t *rawValueData = apInt.getRawData();
    for (unsigned i = 0; i < numActiveWords; ++i) {
      writer.writeSignedVarInt(rawValueData[i]);
    }
  }
}

/// Helper function to serialize the APFloat representation of a FloatAttr.
static void writeAPFloatRepresentation(const APFloat &apFloat,
                                       EncodingWriter &writer) {
  writeAPInt(apFloat.bitcastToAPInt(), writer);
}

//===----------------------------------------------------------------------===//
// String Section Management
//===----------------------------------------------------------------------===//
// string-section =:
//   numStrings[varint]
//   padding[bytes]            // Align to 4 bytes
//   stringOffsets[uint32_t]   // Array of offsets, one per string
//   stringData[bytes]         // Concatenated string data
//
namespace {
struct StringManager {
  uint64_t getStringIndex(StringRef str) {
    auto it = stringIndexMap.find(str);
    if (it != stringIndexMap.end()) {
      return it->second;
    }
    uint64_t index = stringIndexMap.size();
    stringIndexMap[str] = index;
    return index;
  }

  LogicalResult writeStringSection(raw_ostream &stream) {
    SmallVector<char> buffer;
    llvm::raw_svector_ostream sectionStream(buffer);
    EncodingWriter sectionWriter(sectionStream);
    sectionWriter.writeVarInt(stringIndexMap.size());
    // Align the string section
    uint64_t alignmentNeeded = alignof(uint32_t);
    sectionWriter.alignTo(alignmentNeeded);

    // Save the current position to fix up offsets later.
    auto offsetsPtr = sectionWriter.tell();

    // Reserve space for the offset table (filled later).
    for (size_t i = 0; i < stringIndexMap.size(); ++i) {
      sectionWriter.writeLE<uint32_t>(0);
    }

    // Write each string and record its starting offset.
    SmallVector<uint32_t> finalOffsets;
    finalOffsets.reserve(stringIndexMap.size());

    uint32_t running = 0;
    for (const auto &str : stringIndexMap) {
      finalOffsets.push_back(running);
      sectionWriter.write(str.first);
      running += str.first.size();
    }

    // Copy the pre-computed offsets into the reserved slot.
    std::copy_n(finalOffsets.begin(), finalOffsets.size(),
                reinterpret_cast<uint32_t *>(buffer.data() + offsetsPtr));

    writeSectionHeader(stream, Bytecode::Section::String, buffer.size(),
                       sectionWriter.getRequiredAlignment());
    stream.write(buffer.data(), buffer.size());
    return success();
  }

private:
  llvm::MapVector<StringRef, uint64_t> stringIndexMap;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Type Section Management
// Collects and writes all unique types used in the module.
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
// integer-type =: typeTag[I1/I32/I64]  // No additional data
// float-type =: typeTag[F32]           // No additional data
//
// tile-type =:
//   typeTag[Tile]
//   elementTypeIndex[varint]
//   rank[varint]
//   dimensions[int64_t*rank]
//
// function-type =:
//   typeTag[Func]
//   numInputs[varint]
//   inputTypeIndices[varint*numInputs]
//   numResults[varint]
//   resultTypeIndices[varint*numResults]
//
namespace {
struct ConstantManager;
struct FunctionTableWriter;

/// Runtime version check for enum values, used by the generated type writers
/// below and by the generated attribute writers further down.
#define GEN_ENUM_VALUE_VERSION_CHECK
#include "AttrBytecode.inc"

struct TypeManager {
public:
  TypeManager(const BytecodeWriterConfig &config, StringManager &strMgr)
      : config(config), strMgr(strMgr) {}

  void setConstantManager(ConstantManager &manager) { constMgr = &manager; }
  void setFunctionWriter(FunctionTableWriter *writer) { funcWriter = writer; }

  /// Returns the MLIRContext captured from the first registered type, or
  /// nullptr if no types have been registered yet. Used by error paths that
  /// lack an operation handle (e.g. when serializing attrs nested in type
  /// payloads).
  MLIRContext *getContext() const { return cachedContext; }

  // Gets or creates an index for a type in the type table.
  uint64_t getTypeIndex(Type type) {
    if (!cachedContext) {
      cachedContext = type.getContext();
    }
    // Use the type's memory address as a unique key for lookup
    const void *key = type.getAsOpaquePointer();
    auto it = typeIndexMap.find(key);
    if (it != typeIndexMap.end()) {
      return it->second;
    }
    // Ensure dependent/nested types are registered before the type itself
    registerDependentTypes(type);
    uint64_t index = typeList.size();
    typeIndexMap[key] = index;
    typeList.push_back(type);
    return index;
  }

  LogicalResult writeTypeSection(raw_ostream &stream) {
    SmallVector<char> buffer;
    llvm::raw_svector_ostream sectionStream(buffer);
    EncodingWriter sectionWriter(sectionStream);

    sectionWriter.writeVarInt(typeList.size());
    // Align the type section
    uint64_t alignmentNeeded = alignof(uint32_t);
    sectionWriter.alignTo(alignmentNeeded);

    // Save the current position to fix up offsets later.
    auto offsetsPtr = sectionWriter.tell();

    // Reserve space for the offset table (filled later).
    for (size_t i = 0; i < typeList.size(); ++i) {
      sectionWriter.writeLE<uint32_t>(0);
    }

    // Write each type and record its starting offset.
    SmallVector<uint32_t> finalOffsets;
    finalOffsets.reserve(typeList.size());

    uint32_t running = 0;
    for (Type type : typeList) {
      finalOffsets.push_back(running);
      auto before = sectionWriter.tell();
      if (failed(serializeType(type, sectionWriter))) {
        return failure();
      }
      running += static_cast<uint32_t>(sectionWriter.tell() - before);
    }

    // Copy the pre-computed offsets into the reserved slot.
    std::copy_n(finalOffsets.begin(), finalOffsets.size(),
                reinterpret_cast<uint32_t *>(buffer.data() + offsetsPtr));

    writeSectionHeader(stream, Bytecode::Section::Type, buffer.size(),
                       sectionWriter.getRequiredAlignment());
    stream.write(buffer.data(), buffer.size());
    return success();
  }

  /// Helper function to write the index of a given type to the writer.
  LogicalResult writeTypeIndex(Type type, EncodingWriter &writer) {
    // Ensure type is registered and get its index.
    uint64_t index = getTypeIndex(type);
    writer.writeVarInt(index);
    return success();
  }

private:
  // Include generated type serialization functions.
#define GEN_TYPE_WRITERS
#include "TypeBytecode.inc"

  LogicalResult serializeType(Type type, EncodingWriter &writer){
  // Generated type serialization dispatch.
#define GEN_TYPE_WRITER_DISPATCH
#include "TypeBytecode.inc"
  }

  LogicalResult
      serializeFunctionType(FunctionType type, EncodingWriter &writer) {
    // Write the function type with tag
    writer.writeVarInt(Bytecode::TypeTag::FunctionType);
    // Using VarInt for numParams per spec
    uint64_t numInputs = type.getNumInputs();
    writer.writeVarInt(numInputs);
    // Serialize input types
    for (Type input : type.getInputs()) {
      if (failed(writeTypeIndex(input, writer))) {
        return failure();
      }
    }
    // Using VarInt for numResults per spec
    uint64_t numResults = type.getNumResults();
    writer.writeVarInt(numResults);
    // Serialize result types
    for (Type result : type.getResults()) {
      if (failed(writeTypeIndex(result, writer))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult writeSelfContainedTypeAttribute(Attribute attrValue,
                                                EncodingWriter &writer);

  // Helper to recursively register dependent types before the main type.
  void registerDependentTypes(Type type) {
    // Check if the type itself is already registered or being registered
    if (typeIndexMap.count(type.getAsOpaquePointer())) {
      return;
    }

    // Auto-generated dependent type registration for CudaTile types.
#define GEN_DEPENDENT_TYPE_REGISTRATION
#include "TypeBytecode.inc"

    // FunctionType is not a CudaTile type, handle it manually.
    if (auto funcType = dyn_cast<FunctionType>(type)) {
      for (Type input : funcType.getInputs()) {
        getTypeIndex(input);
      }
      for (Type result : funcType.getResults()) {
        getTypeIndex(result);
      }
    }
  }

  // Helper to recursively register any types referenced by a given attribute.
  void registerDependentAttributeTypes(Attribute attr) {
    if (!attr) {
      return;
    }
    if (!visitedAttrs.insert(attr).second) {
      return;
    }

    // MLIR builtin attrs that carry types or contain other attributes.
    if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
      getTypeIndex(typeAttr.getValue());
      return;
    }
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      getTypeIndex(denseAttr.getType());
      return;
    }
    if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
      for (Attribute elementAttr : arrayAttr) {
        registerDependentAttributeTypes(elementAttr);
      }
      return;
    }
    if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
      for (NamedAttribute namedAttr : dictAttr) {
        registerDependentAttributeTypes(namedAttr.getValue());
      }
      return;
    }

    // Auto-generated dispatch for CudaTile attrs with type-bearing parameters.
#define GEN_DEPENDENT_ATTR_TYPE_REGISTRATION
#include "AttrBytecode.inc"
  }

  llvm::MapVector<const void *, uint64_t> typeIndexMap;
  SmallVector<Type> typeList;
  const BytecodeWriterConfig &config;
  StringManager &strMgr;
  ConstantManager *constMgr = nullptr;
  FunctionTableWriter *funcWriter = nullptr;
  MLIRContext *cachedContext = nullptr;
  llvm::DenseSet<Attribute> visitedAttrs;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Constant Section Management
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
struct ConstantManager {
  explicit ConstantManager(const BytecodeWriterConfig &config)
      : config(config) {}

  LogicalResult addConstant(Attribute attr, uint64_t &index) {
    auto it = constantsMap.find(attr);
    if (it != constantsMap.end()) {
      index = std::distance(constantsMap.begin(), it);
      return success();
    }
    SmallVector<char> data;
    llvm::raw_svector_ostream dataStream(data);
    EncodingWriter writer(dataStream);
    if (failed(serializeAttribute(attr, writer))) {
      return emitError(UnknownLoc::get(attr.getContext()),
                       "failed to serialize attribute");
    }
    index = constantsMap.size();
    constantsMap[attr] = std::move(data);
    return success();
  }

  // Look up a constant by attribute without adding it
  LogicalResult getConstantIndex(Attribute attr, uint64_t &index) const {
    auto it = constantsMap.find(attr);
    if (it != constantsMap.end()) {
      index = std::distance(constantsMap.begin(), it);
      return success();
    }
    return failure();
  }

  // Provide access to the constant map
  const llvm::MapVector<Attribute, SmallVector<char>> &getConstantsMap() const {
    return constantsMap;
  }

  /// Serializes a single MLIR attribute into its raw byte representation.
  /// This function handles different attribute types, focusing on scalar
  /// and dense element attributes suitable for the constant pool.
  LogicalResult serializeAttribute(Attribute attr, EncodingWriter &writer) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      // i1 dense attributes use a canonical bit-packed encoding so the
      // bytes don't depend on whichever MLIR version is linked at build
      // time. Upstream MLIR's `getRawData()` for i1 has shifted between
      // bit-packed and one-byte-per-element across LLVM revisions; both
      // forms appear in legacy bytecode. We pick bit-packed as the
      // forever-canonical wire format for v13.4+ and unpack inside the
      // reader.
      auto tileType = dyn_cast<cuda_tile::TileType>(denseAttr.getType());
      if (tileType && tileType.getElementType().isInteger(1)) {
        if (config.getBytecodeVersion() <
                cuda_tile::BytecodeVersion::kCanonicalI1Version &&
            denseAttr.isSplat()) {
          char b = denseAttr.getSplatValue<bool>() ? static_cast<char>(0xff)
                                                   : static_cast<char>(0x00);
          writer.writeVarInt(1);
          writer.write(&b, 1);
          return success();
        }
        size_t numEls = tileType.getNumElements();
        SmallVector<char> packed(llvm::divideCeil(numEls, 8), 0);
        size_t i = 0;
        for (bool val : denseAttr.getValues<bool>()) {
          if (val) {
            packed[i / 8] |= static_cast<char>(1u << (i % 8));
          }
          ++i;
        }
        writer.writeVarInt(packed.size());
        writer.write(packed.data(), packed.size());
        return success();
      }
      // Non-i1 path: ride MLIR's raw buffer contract.
      ArrayRef<char> rawData = denseAttr.getRawData();
      // Write the size of the raw buffer.
      writer.writeVarInt(rawData.size());
      // Write the raw buffer content.
      writer.write(rawData.data(), rawData.size());
      return success();
    } else if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      APInt value = intAttr.getValue();
      writer.writeLE<uint64_t>(value.getZExtValue());
      return success();
    } else if (auto boolAttr = dyn_cast<BoolAttr>(attr)) {
      uint8_t boolValue = boolAttr.getValue() ? 0x01 : 0x00;
      writer.writeByte(boolValue);
      return success();
    } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      writeAPFloatRepresentation(floatAttr.getValue(), writer);
      return success();
    }
    return emitError(UnknownLoc::get(attr.getContext()),
                     "unsupported attribute type in constant data");
  }

  LogicalResult writeConstantSection(raw_ostream &stream) {
    // If there are no constants, skip writing this section entirely
    if (constantsMap.empty()) {
      return success();
    }
    SmallVector<char> buffer;
    llvm::raw_svector_ostream sectionStream(buffer);
    EncodingWriter sectionWriter(sectionStream);
    // Write numConstants
    sectionWriter.writeVarInt(constantsMap.size());
    // Align the constant section
    uint64_t alignmentNeeded = alignof(uint64_t);
    sectionWriter.alignTo(alignmentNeeded);

    // Save the current position to fix up offsets later.
    auto offsetsPtr = sectionWriter.tell();

    // Reserve space for the offset table (filled later).
    for (size_t i = 0; i < constantsMap.size(); ++i) {
      sectionWriter.writeLE<uint64_t>(0);
    }

    // Write each constant and record its starting offset.
    SmallVector<uint64_t> finalOffsets;
    finalOffsets.reserve(constantsMap.size());

    uint64_t running = 0;
    for (const auto &pair : constantsMap) {
      if (pair.second.empty()) {
        return emitError(UnknownLoc::get(pair.first.getContext()))
               << "constant has empty serialized representation, which is "
                  "invalid";
      }
      finalOffsets.push_back(running);
      sectionWriter.write(pair.second.data(), pair.second.size());
      running += pair.second.size();
    }

    // Copy the pre-computed offsets into the reserved slot.
    std::copy_n(finalOffsets.begin(), finalOffsets.size(),
                reinterpret_cast<uint64_t *>(buffer.data() + offsetsPtr));

    writeSectionHeader(stream, Bytecode::Section::Constant, buffer.size(),
                       sectionWriter.getRequiredAlignment());
    // Write the section content
    stream.write(buffer.data(), buffer.size());
    return success();
  }

private:
  llvm::MapVector<Attribute, SmallVector<char>> constantsMap;
  const BytecodeWriterConfig &config;
};

//===----------------------------------------------------------------------===//
// DebugInfo Section
//===----------------------------------------------------------------------===//

/// This class manages writing debug info attributes to bytecode format.
class DebugInfoWriter {
public:
  DebugInfoWriter(StringManager &strMgr, const BytecodeWriterConfig &config)
      : strMgr(strMgr), config(config) {}

  /// This method gets or creates an index for an operation.
  uint64_t getOpIndex(Operation *op) {
    auto it = opIndexMap.find(op);
    if (it != opIndexMap.end()) {
      return it->second;
    }

    // Check if the operation location has a reserved index and return it.
    auto reserved = getDebugReserved(op->getLoc());
    if (reserved != Bytecode::DebugReserved::SIZE) {
      return static_cast<uint64_t>(reserved);
    }

    // Adjust the index to account for reserved indices.
    uint64_t opIndex = opIndexMap.size() +
                       static_cast<uint64_t>(Bytecode::DebugReserved::SIZE);

    opIndexMap[op] = opIndex;
    return opIndex;
  }

  /// This method adds a debug info attribute to an operation.
  void addDebugInfo(uint64_t opIndex, Attribute attr) {
    // Nothing to do if the operation has a reserved index.
    if (opIndex < static_cast<uint64_t>(Bytecode::DebugReserved::SIZE)) {
      return;
    }

    // Adjust the index to account for reserved indices.
    opIndex -= static_cast<uint64_t>(Bytecode::DebugReserved::SIZE);

    if (opIndex >= debuginfoIndices.size()) {
      debuginfoIndices.resize(opIndex + 1);
    }
    debuginfoIndices[opIndex].push_back(getDebugInfoIndex(attr));
  }

  // debuginfo-section =:
  //   diOpsNum[varint]          // Total number of operations with debug info
  //   padding[bytes]            // Align to 4 bytes
  //   diIndexOffsets[uint32_t]  // Per op offset into the debug info indices
  //   diIndicesNum[varint]      // Total number of debug info indices
  //   padding[bytes]            // Align to 8 bytes
  //   diIndices[uint64_t]       // Array of debug indices to debug info
  //   attributes diAttrNum[varint]         // Total number of debug info
  //   attributes padding[bytes]            // Align to 4 bytes
  //   diOffsets[uint32_t]       // Per debug info attribute offset into the
  //   debug info data diData[bytes]             // Data for each debug info
  //   attribute
  //
  // diData =:
  //   DebugTag[byte]            // Indicates the debug info attribute type
  //   debuginfo-encoding        // Format depends on DebugTag
  LogicalResult writeDebugInfoSection(raw_ostream &stream) {
    // Skip writing the section if there are no debug info attributes.
    if (debuginfoIndices.empty() && debuginfoList.empty()) {
      return success();
    }

    SmallVector<char> diData;
    llvm::raw_svector_ostream diStream(diData);
    EncodingWriter diWriter(diStream);

    // Write the total number of operations with debug info.
    diWriter.writeVarInt(debuginfoIndices.size());

    // Align to 4 bytes for the uint32_t diIndexOffsetsPtr.
    auto alignment = alignof(uint32_t);
    diWriter.alignTo(alignment);

    // Write the per op offset into the debug info indices.
    uint32_t offset = 0;
    for (auto &diIndices : debuginfoIndices) {
      diWriter.writeLE<uint32_t>(offset);
      offset += diIndices.size();
    }

    // Write the total number of debug info indices.
    diWriter.writeVarInt(offset);

    // Align to 8 bytes for the uint64_t diIndicesPtr.
    diWriter.alignTo(alignof(uint64_t));

    // Write the array of debug indices to debug info attributes.
    for (const auto &indices : debuginfoIndices) {
      for (uint64_t diIndex : indices) {
        diWriter.writeLE<uint64_t>(diIndex);
      }
    }

    // Write the total number of debug info attributes.
    diWriter.writeVarInt(debuginfoList.size());

    // Align to 4 bytes for the uint32_t diOffsetsPtr.
    diWriter.alignTo(alignment);

    // Save the current position to fix up offsets later.
    auto offsetsPtr = diWriter.tell();

    // Reserve space for the offset table (filled later).
    for (size_t i = 0; i < debuginfoList.size(); ++i) {
      diWriter.writeLE<uint32_t>(0);
    }

    // Write each debug info attribute and record its starting offset.
    SmallVector<uint32_t> finalOffsets;
    finalOffsets.reserve(debuginfoList.size());

    uint32_t running = 0;
    for (auto attr : debuginfoList) {
      finalOffsets.push_back(running);
      auto before = diWriter.tell();
      if (failed(serializeDebugInfo(attr, diWriter))) {
        return emitError(UnknownLoc::get(attr.getContext()),
                         "failed to serialize debug info attribute");
      }
      running += static_cast<uint32_t>(diWriter.tell() - before);
    }

    // Copy the pre-computed offsets into the reserved slot.
    std::copy_n(finalOffsets.begin(), finalOffsets.size(),
                reinterpret_cast<uint32_t *>(diData.data() + offsetsPtr));

    // Write the debug info section header.
    writeSectionHeader(stream, Bytecode::Section::Debug, diData.size(),
                       diWriter.getRequiredAlignment());

    // Write the debug info section data directly.
    stream.write(diData.data(), diData.size());

    return success();
  }

  LogicalResult validateDebugInfo(Operation *op) {
    return validateDebugInfo(op, op->getLoc());
  }

  LogicalResult validateDebugInfo(Operation *op, Attribute attr) {
    return TypeSwitch<Attribute, LogicalResult>(attr)
        .Case<DILocAttr, FileLineColLoc, UnknownLoc>(
            [&](Attribute attr) { return success(); })
        .Case([&](CallSiteLoc attr) {
          if (failed(validateDebugInfo(op, attr.getCaller())) ||
              failed(validateDebugInfo(op, attr.getCallee()))) {
            return failure();
          }
          return success();
        })
        .Default([&](Attribute attr) { return invalidLocError(op, attr); });
  }

private:
  /// This method gets or creates an index for a debug info attribute.
  uint64_t getDebugInfoIndex(Attribute attr) {
    const void *key = attr.getAsOpaquePointer();
    auto it = diIndexMap.find(key);
    if (it != diIndexMap.end()) {
      return it->second;
    }

    // Check if the debug info attribute has a reserved index and return it.
    auto reserved = getDebugReserved(attr);
    if (reserved != Bytecode::DebugReserved::SIZE) {
      return static_cast<uint64_t>(reserved);
    }

    // Register any dependent debug info attributes.
    registerDebugInfo(attr);

    // Adjust the index to account for reserved indices.
    uint64_t diIndex = debuginfoList.size() +
                       static_cast<uint64_t>(Bytecode::DebugReserved::SIZE);

    diIndexMap[key] = diIndex;
    debuginfoList.push_back(attr);
    return diIndex;
  }

  LogicalResult invalidLocError(Operation *op, Attribute attr) {
    return op->emitError()
           << "unsupported location, got "
           << TypeSwitch<Attribute, StringRef>(attr)
                  .Case([&](OpaqueLoc loc) { return "OpaqueLoc"; })
                  .Case([&](NameLoc loc) { return "NameLoc"; })
                  .Case([&](FusedLoc loc) { return "FusedLoc"; })
                  .Case([&](UnknownLoc loc) { return "UnknownLoc"; })
                  .Case([&](FileLineColLoc loc) { return "FileLineColLoc"; })
                  .Default([&](Attribute loc) { return "Unknown Attribute"; })
           << ", expected DILocAttr or CallSiteLoc";
  }

  Bytecode::DebugReserved getDebugReserved(Attribute attr) {
    return TypeSwitch<Attribute, Bytecode::DebugReserved>(attr)
        .Case<FileLineColLoc, UnknownLoc>(
            [](auto loc) { return Bytecode::DebugReserved::UnknownLoc; })
        .Default([](Attribute) { return Bytecode::DebugReserved::SIZE; });
  }

  void registerDebugInfo(Attribute attr) {
    TypeSwitch<Attribute, void>(attr)
        .Case([&](DICompileUnitAttr diCompileUnit) {
          getDebugInfoIndex(diCompileUnit.getFile());
        })
        .Case([&](DILexicalBlockAttr diLexicalBlock) {
          getDebugInfoIndex(diLexicalBlock.getScope());
          getDebugInfoIndex(diLexicalBlock.getFile());
        })
        .Case([&](DILocAttr diLoc) { getDebugInfoIndex(diLoc.getScope()); })
        .Case([&](DISubprogramAttr diSubprogram) {
          getDebugInfoIndex(diSubprogram.getFile());
          getDebugInfoIndex(diSubprogram.getCompileUnit());
        })
        .Case([&](CallSiteLoc callSiteLoc) {
          getDebugInfoIndex(callSiteLoc.getCallee());
          getDebugInfoIndex(callSiteLoc.getCaller());
        })
        .Case([&](FusedLoc FusedLoc) {
          for (auto subLoc : FusedLoc.getLocations()) {
            getDebugInfoIndex(subLoc);
          }
        })
        .Case(
            [&](NameLoc nameLoc) { getDebugInfoIndex(nameLoc.getChildLoc()); })
        .Case([&](OpaqueLoc opaqueLoc) {
          getDebugInfoIndex(opaqueLoc.getFallbackLocation());
        })
        .Default([](Attribute) {});
  }

  // di-compile-unit =:
  //   DebugTag[DICompileUnit]
  //   diFileIndex[varint] - DIFileAttr
  LogicalResult serialize(DICompileUnitAttr diCompileUnit,
                          EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::DICompileUnit);
    writer.writeVarInt(getDebugInfoIndex(diCompileUnit.getFile()));
    return success();
  }
  // di-file =:
  //   DebugTag[DIFile]
  //   fileNameIndex[varint] - StringAttr
  //   directoryIndex[varint] - StringAttr
  LogicalResult serialize(DIFileAttr diFile, EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::DIFile);
    writer.writeVarInt(strMgr.getStringIndex(diFile.getName()));
    writer.writeVarInt(strMgr.getStringIndex(diFile.getDirectory()));
    return success();
  }
  // di-lexical-block =:
  //   DebugTag[DILexicalBlock]
  //   diScopeIndex[varint] - DILocalScopeAttr
  //   diFileIndex[varint] - DIFileAttr
  //   lineNumber[varint] - unsigned
  //   columnNumber[varint] - unsigned
  LogicalResult serialize(DILexicalBlockAttr diLexicalBlock,
                          EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::DILexicalBlock);
    writer.writeVarInt(getDebugInfoIndex(diLexicalBlock.getScope()));
    writer.writeVarInt(getDebugInfoIndex(diLexicalBlock.getFile()));
    writer.writeVarInt(diLexicalBlock.getLine());
    writer.writeVarInt(diLexicalBlock.getColumn());
    return success();
  }
  // di-loc =:
  //   DebugTag[DILoc]
  //   diScopeIndex[varint] - DILocalScopeAttr
  //   fileNameIndex[varint] - StringAttr
  //   lineNumber[varint] - unsigned
  //   columnNumber[varint] - unsigned
  LogicalResult serialize(DILocAttr diLoc, EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::DILoc);
    writer.writeVarInt(getDebugInfoIndex(diLoc.getScope()));
    writer.writeVarInt(
        strMgr.getStringIndex(diLoc.getSourceLoc().getFilename()));
    writer.writeVarInt(diLoc.getSourceLoc().getLine());
    writer.writeVarInt(diLoc.getSourceLoc().getColumn());
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
  //  flags[varint] - DISubprogramFlags, only present for 13.5 and newer
  LogicalResult serialize(DISubprogramAttr diSubprogram,
                          EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::DISubprogram);
    writer.writeVarInt(getDebugInfoIndex(diSubprogram.getFile()));
    writer.writeVarInt(diSubprogram.getLine());
    writer.writeVarInt(strMgr.getStringIndex(diSubprogram.getName()));
    writer.writeVarInt(strMgr.getStringIndex(diSubprogram.getLinkageName()));
    writer.writeVarInt(getDebugInfoIndex(diSubprogram.getCompileUnit()));
    writer.writeVarInt(diSubprogram.getScopeLine());

    // Older readers have no flags field to fall back on, so a non-default value
    // cannot be represented and must not be silently dropped.
    DISubprogramFlags flags = diSubprogram.getFlags();
    if (config.getBytecodeVersion() >=
        BytecodeVersion::kMinDISubprogramFlagsVersion) {
      writer.writeVarInt(static_cast<uint32_t>(flags));
    } else if (flags != DISubprogramFlags::None) {
      return emitError(UnknownLoc::get(diSubprogram.getContext()))
             << "debug info subprogram '"
             << diSubprogram.getLinkageName().getValue() << "' has flags '"
             << stringifyDISubprogramFlags(flags)
             << "' which require bytecode version "
             << BytecodeVersion::kMinDISubprogramFlagsVersion.toString()
             << " or newer, but the requested version is "
             << config.getBytecodeVersion().toString();
    }
    return success();
  }

  // call-site =:
  //  DebugTag[CallSite]
  //  diCalleeIndex[varint] - LocationAttr
  //  diCallerIndex[varint] - LocationAttr
  LogicalResult serialize(CallSiteLoc callSiteLoc, EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::CallSite);
    writer.writeVarInt(getDebugInfoIndex(callSiteLoc.getCallee()));
    writer.writeVarInt(getDebugInfoIndex(callSiteLoc.getCaller()));
    return success();
  }

  // unknown =:
  //   DebugTag[Unknown]
  LogicalResult serializeUnknown(EncodingWriter &writer) {
    writer.writeVarInt(Bytecode::DebugTag::Unknown);
    return success();
  }

  LogicalResult serializeDebugInfo(Attribute attr, EncodingWriter &writer) {
    return TypeSwitch<Attribute, LogicalResult>(attr)
        // Serialize known debug info attributes.
        .Case<DICompileUnitAttr, DIFileAttr, DILexicalBlockAttr,
              DISubprogramAttr>(
            [&](auto attr) -> LogicalResult { return serialize(attr, writer); })
        // Serialize known locations types.
        .Case<DILocAttr, CallSiteLoc>(
            [&](auto loc) { return serialize(loc, writer); })
        .Case<UnknownLoc, FileLineColLoc>([&](auto loc) { return success(); })
        .Default([&](Attribute) -> LogicalResult { return failure(); });
  }

  StringManager &strMgr;
  const BytecodeWriterConfig &config;
  SmallVector<Attribute> debuginfoList;
  SmallVector<SmallVector<uint64_t>> debuginfoIndices;
  llvm::MapVector<const void *, uint64_t> diIndexMap;
  llvm::MapVector<const void *, uint64_t> opIndexMap;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Function Table Section Management
//===----------------------------------------------------------------------===//
// function-table-section =:
//   numFunctions[varint]
//   function-entry*
//
// function-entry =:
//   nameIndex[varint]         // Index into string table
//   signatureIndex[varint]    // Index into type table
//   entryFlag[byte]           // Bit 0: Visibility(0=Public,1=Private)
//                             // Bit 1: Kind(0=Function,1=Kernel)
//                             // Bits [3:2]: Form for version 13.5+
//   functionLocIndex[varint]  // Index into location table for function
//                             // definition
//   entryPayload              // Kind/form-specific attributes
//   bodyLength[varint]        // Zero selects an extern FuncOp for FormNormal
//   functionBody[bytes]       // Present when bodyLength is non-zero
//   externConstParams         // Present after a zero FormNormal body length
//
// function-body =:
//   instruction*
//
// instruction =:
//   opcode[varint]
//   op-specific-data          // Format depends on the opcode
//  Returns a mapping from operation names to their corresponding bytecode
//  opcodes

// Include generated opcode definitions and map.
#define GEN_OPCODE_ENUM
#include "StaticOpcodes.inc"

#define GEN_OPCODE_MAP
#include "StaticOpcodes.inc"

namespace {
struct FunctionTableWriter {
  FunctionTableWriter(TypeManager &tm, ConstantManager &cm, StringManager &sm,
                      DebugInfoWriter &dm, const BytecodeWriterConfig &cfg)
      : typeMgr(tm), constMgr(cm), strMgr(sm), debuginfo(dm), config(cfg) {}

  LogicalResult writeOperation(Operation *op, EncodingWriter &writer) {
    auto opcode = getOpcodeForOperation(op);
    if (!opcode) {
      return op->emitError("operation not supported in bytecode (missing from "
                           "BytecodeOpcodes.td)");
    }

    // Version checking for public operations.
    uint32_t opcodeValue = static_cast<uint32_t>(*opcode);
    if (!mlir::cuda_tile::detail::isOpcodeAvailableInVersion(
            opcodeValue, config.getBytecodeVersion())) {
      return op->emitError() << "operation '" << op->getName().getStringRef()
                             << "' is not available in bytecode version "
                             << config.getBytecodeVersion().toString();
    }

    writer.writeVarInt(*opcode);

    if (failed(debuginfo.validateDebugInfo(op))) {
      return failure();
    }

    uint64_t functionLocIndex =
        debuginfo.getOpIndex(op->getParentOfType<FunctionOpInterface>());
    debuginfo.addDebugInfo(functionLocIndex, op->getLoc());

    auto numSerializedResults =
        dispatchOpWriter(op, writer, typeMgr, constMgr, strMgr, config);
    if (failed(numSerializedResults)) {
      return failure();
    }
    // Only add serialized results to valueIndexMap. Results that were not
    // serialized (due to version compatibility) should not be indexed.
    for (size_t i = 0; i < *numSerializedResults; ++i) {
      valueIndexMap[op->getResult(i)] = nextValueIndex++;
    }
    return success();
  }

  std::optional<Bytecode::Opcode> getOpcodeForOperation(Operation *op) {
    auto it = Bytecode::getOpcodeMap().find(op->getName().getStringRef());
    if (it != Bytecode::getOpcodeMap().end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // Writes the operands of an operation to the bytecode
  void writeOperands(ValueRange operands, EncodingWriter &writer,
                     bool encodeSize = true) {
    if (encodeSize) {
      writer.writeVarInt(operands.size());
    }
    for (Value operand : operands) {
      uint64_t operandIndex = valueIndexMap.lookup(operand);
      writer.writeVarInt(operandIndex);
    }
  }

  // Writes result types from a TypeRange to the bytecode.
  LogicalResult writeResultTypes(TypeRange resultTypes, EncodingWriter &writer,
                                 TypeManager &typeMgr) {
    for (Type type : resultTypes) {
      if (failed(typeMgr.writeTypeIndex(type, writer))) {
        return failure();
      }
    }
    return success();
  }

  // Writes the result types of an operation to the bytecode.
  LogicalResult writeResultTypes(Operation *op, EncodingWriter &writer,
                                 TypeManager &typeMgr) {
    return writeResultTypes(op->getResultTypes(), writer, typeMgr);
  }

private:
  // Include generated attribute serializers as member functions.
#define GEN_ATTR_WRITERS
#include "AttrBytecode.inc"

  // Include generated attribute serialization dispatch.
#define GEN_ATTR_WRITER_DISPATCH
#include "AttrBytecode.inc"

public:
  // Writes the index or inline representation of an attribute.
  // This function determines whether to serialize inline or use an index based
  // on the attribute type.
  LogicalResult
  writeSingleAttribute(Operation *op, StringRef attrName, Attribute attrValue,
                       EncodingWriter &writer, TypeManager &typeMgr,
                       ConstantManager &constMgr, StringManager &strMgr,
                       bool isSelfContained = false) {
    // Emit an error attached to `op` when available, otherwise fall back to
    // an UnknownLoc-anchored diagnostic.
    auto emitErr = [&](const Twine &msg) -> InFlightDiagnostic {
      if (op) {
        return op->emitError(msg);
      }
      return emitError(UnknownLoc::get(attrValue.getContext()), msg);
    };

    // Dispatch to generated CudaTile attribute serializers.
    // Falls through to MLIR builtins if not a CudaTile attr.
    if (succeeded(writeCudaTileAttr(op, attrValue, writer, isSelfContained))) {
      return success();
    }

    // Builtin MLIR attrs.
    return TypeSwitch<Attribute, LogicalResult>(attrValue)
        .Case<TypeAttr>([&](TypeAttr typeAttr) {
          // Handle TypeAttr: Write index using TypeManager
          if (isSelfContained) {
            writer.writeVarInt(Bytecode::AttributeTag::Type);
          }
          return typeMgr.writeTypeIndex(typeAttr.getValue(), writer);
        })
        .Case<StringAttr>([&](StringAttr strAttr) {
          // Handle StringAttr: Write index using StringManager
          if (isSelfContained) {
            writer.writeVarInt(Bytecode::AttributeTag::String);
          }
          writer.writeVarInt(strMgr.getStringIndex(strAttr.getValue()));
          return success();
        })
        .Case<IntegerAttr>([&](IntegerAttr intAttr) -> LogicalResult {
          if (isSelfContained) {
            writer.writeVarInt(Bytecode::AttributeTag::Integer);
            if (failed(typeMgr.writeTypeIndex(intAttr.getType(), writer))) {
              return emitErr(
                  "failed to write type index for self-contained IntegerAttr");
            }
          }
          writer.writeVarInt(intAttr.getValue().getZExtValue());
          return success();
        })
        .Case<FloatAttr>([&](FloatAttr floatAttr) -> LogicalResult {
          if (isSelfContained) {
            writer.writeVarInt(Bytecode::AttributeTag::Float);
            if (failed(typeMgr.writeTypeIndex(floatAttr.getType(), writer))) {
              return emitErr(
                  "failed to write type index for self-contained FloatAttr");
            }
          }
          writeAPFloatRepresentation(floatAttr.getValue(), writer);
          return success();
        })
        .Case<BoolAttr>([&](BoolAttr boolAttr) -> LogicalResult {
          if (isSelfContained) {
            writer.writeVarInt(Bytecode::AttributeTag::Bool);
          }
          writer.writeByte(boolAttr.getValue() ? 0x01 : 0x00);
          return success();
        })
        .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr)
                                     -> LogicalResult {
          if (isSelfContained) {
            writer.writeVarInt(Bytecode::AttributeTag::DenseElements);
            if (failed(typeMgr.writeTypeIndex(denseAttr.getType(), writer))) {
              return emitErr(
                  "failed to write type index for DenseElementsAttr");
            }
          }

          if (auto intOrFPAttr = dyn_cast<DenseTypedElementsAttr>(denseAttr)) {
            uint64_t constantIndex;
            if (failed(constMgr.addConstant(intOrFPAttr, constantIndex)))
              return emitErr("failed to add constant attribute '")
                     << attrName << "' to pool: " << intOrFPAttr;
            writer.writeVarInt(constantIndex);
            return success();
          }

          return emitErr("unsupported DenseElementsAttr element type "
                         "during serialization");
        })
        .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr arrayAttr) -> LogicalResult {
          if (isSelfContained)
            writer.writeVarInt(Bytecode::AttributeTag::Array);
          writer.writeVarInt(arrayAttr.size());
          for (Attribute elementAttr : arrayAttr.getValue())
            if (failed(writeSelfContainedAttribute(op, "arrayElement",
                                                   elementAttr, writer, typeMgr,
                                                   constMgr, strMgr)))
              return emitErr("failed to write ArrayAttr element: ")
                     << elementAttr;
          return success();
        })
        .Case<mlir::DictionaryAttr>([&](mlir::DictionaryAttr dictAttr)
                                        -> LogicalResult {
          if (isSelfContained)
            writer.writeVarInt(Bytecode::AttributeTag::Dictionary);
          writer.writeVarInt(dictAttr.size());
          for (const NamedAttribute &namedAttr : dictAttr) {
            writer.writeVarInt(strMgr.getStringIndex(namedAttr.getName()));
            if (failed(writeSelfContainedAttribute(op, namedAttr.getName(),
                                                   namedAttr.getValue(), writer,
                                                   typeMgr, constMgr, strMgr)))
              return emitErr("failed to write DictionaryAttr element: ")
                     << namedAttr.getValue();
          }
          return success();
        })
        .Default([&](Attribute) -> LogicalResult {
          // Default case: Error for unsupported types in this context
          // TODO: Need to handle other potential attribute types if they occur
          return emitErr("unsupported attribute type encountered during "
                         "serialization of attribute '")
                 << attrName << "': " << attrValue;
        });
  }

  // Writes a self-contained attribute, including its tag and data.
  LogicalResult writeSelfContainedAttribute(Operation *op, StringRef attrName,
                                            Attribute attrValue,
                                            EncodingWriter &writer,
                                            TypeManager &typeMgr,
                                            ConstantManager &constMgr,
                                            StringManager &strMgr) {
    return writeSingleAttribute(op, attrName, attrValue, writer, typeMgr,
                                constMgr, strMgr, /*isSelfContained=*/true);
  }

  // --- writeOpAttribute Overloads ---
  // This set of functions handles the conversion from native C++ types
  // (as returned by ODS getters) to mlir::Attribute, and then calls
  // the appropriate serialization method (inline or index-based).

  // Template specialization for std::optional<T>
  // The presence of an optional attributes is encoded in the
  // flags field written by TableGen.
  template <typename T>
  LogicalResult writeOpAttribute(Operation *op, StringRef attrName,
                                 const std::optional<T> &nativeValue,
                                 EncodingWriter &writer, TypeManager &typeMgr,
                                 ConstantManager &constMgr,
                                 StringManager &strMgr) {
    if (nativeValue) {
      return writeOpAttribute(op, attrName, *nativeValue, writer, typeMgr,
                              constMgr, strMgr);
    }
    return success();
  }

  /// Helper type trait to check if T is one of the CUDA tile enums.
  /// Auto-generated from CudaTileI32EnumAttr/CudaTileI64EnumAttr definitions.
#define GEN_ENUM_TYPE_TRAIT
#include "AttrBytecode.inc"

  /// Runtime version check for attribute tags.
#define GEN_ATTR_VERSION_CHECK
#include "AttrBytecode.inc"

  // Template for other native C++ types that need conversion
  template <typename T>
  LogicalResult
  writeOpAttribute(Operation *op, StringRef attrName, const T &nativeValue,
                   EncodingWriter &writer, TypeManager &typeMgr,
                   ConstantManager &constMgr, StringManager &strMgr) {
    // Emit an error attached to `op` when available, otherwise fall back to
    // an UnknownLoc-anchored diagnostic. `op` is nullptr when the caller is
    // TypeManager::writeSelfContainedTypeAttribute.
    auto emitErr = [&](const Twine &msg) -> InFlightDiagnostic {
      if (op) {
        return op->emitError(msg);
      }
      MLIRContext *ctx = typeMgr.getContext();
      assert(ctx && "TypeManager has no cached context yet");
      return emitError(UnknownLoc::get(ctx), msg);
    };

    // --- Direct Inline Writes ---
    if constexpr (std::is_same_v<T, bool>) {
      writer.writeByte(nativeValue ? 0x01 : 0x00);
      return success();
    } else if constexpr (is_cuda_tile_enum<T>::value) {
      // Check per-value version requirements.
      if (!isEnumValueAvailableInVersion<T>(nativeValue,
                                            config.getBytecodeVersion())) {
        return emitErr("")
               << "enum value for attribute '" << attrName
               << "' is not available in bytecode version "
               << static_cast<int>(config.getBytecodeVersion().getMajor()) << "."
               << static_cast<int>(config.getBytecodeVersion().getMinor());
      }
      writer.writeVarInt(static_cast<uint64_t>(nativeValue));
      return success();
    } else if constexpr (
        std::is_same_v<std::decay_t<T>, ::llvm::ArrayRef<int32_t>> ||
        std::is_same_v<std::decay_t<T>, ::llvm::ArrayRef<int64_t>> ||
        std::is_same_v<std::decay_t<T>, ::llvm::ArrayRef<int>> ||
        std::is_same_v<std::decay_t<T>, ::llvm::ArrayRef<bool>>) {
      writer.writeLEVarSize(nativeValue);
      return success();
    } else if constexpr (std::is_integral_v<T>) {
      unsigned width = sizeof(T) * CHAR_BIT;
      if (width == 0 || width > 64) {
        return emitErr("") << "unsupported inline integer width for attribute '"
                           << attrName << "': " << width;
      }
      writer.writeVarInt(nativeValue);
      return success();
    } else if constexpr (std::is_base_of_v<mlir::Type, std::decay_t<T>>) {
      return typeMgr.writeTypeIndex(nativeValue, writer);
    } else if constexpr (std::is_base_of_v<Attribute, T>) {
      // If the attribute implements an interface, we need to write it
      // self-contained.
      bool isSelfContained = mlir::detail::IsInterface<T>::value;
      return writeSingleAttribute(op, attrName, nativeValue, writer, typeMgr,
                                  constMgr, strMgr, isSelfContained);
    } else if constexpr (std::is_same_v<std::decay_t<T>, StringRef>) {
      writer.writeVarInt(strMgr.getStringIndex(nativeValue));
      return success();
      // --- Unsupported ---
    } else {
      return emitErr("unsupported native C++ type encountered during attribute "
                     "serialization for attribute '")
             << attrName << "'";
    }
  }
  // Contains generated implementations of the operation-specific
  // bytecode writing functions.
#define GEN_OP_WRITERS
#include "Bytecode.inc"

  // Dispatch to the correct op writer.
  // Returns the number of results that were serialized.
  FailureOr<size_t> dispatchOpWriter(Operation *op, EncodingWriter &writer,
                                     TypeManager &typeMgr,
                                     ConstantManager &constMgr,
                                     StringManager &strMgr,
                                     const BytecodeWriterConfig &config) {
    // Includes the generated TypeSwitch statement for dispatching to the
    // appropriate 'write<OpName>' function. The generated code returns
    // directly.
#define GEN_OP_WRITER_DISPATCH
#include "Bytecode.inc"
    llvm_unreachable("TypeSwitch should handle all cases");
  }

  // Serializes the body of an op with a function interface to bytecode
  LogicalResult writeFunctionBody(FunctionOpInterface func,
                                  SmallVectorImpl<char> &functionBody) {
    llvm::raw_svector_ostream bodyStream(functionBody);
    EncodingWriter writer(bodyStream);
    // Clear state for this function
    valueIndexMap.clear();
    nextValueIndex = 0;
    // Process function arguments using the interface
    for (BlockArgument arg : func.getArguments()) {
      valueIndexMap[arg] = nextValueIndex++;
    }
    // Process operations using the interface
    for (Block &block : func.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (failed(writeOperation(&op, writer))) {
          return failure();
        }
      }
    }
    return success();
  }

public:
  /// Collect all function metadata.
  LogicalResult buildFunctionMap(cuda_tile::ModuleOp module) {
    // Get the body of the module, which contains the function definitions.
    Block *moduleBody = &module.getBody().front();
    if (!moduleBody) {
      return module.emitError("module has no body.");
    }

    // Iterate through all operations in the module's body.
    for (auto func : moduleBody->getOps<FunctionOpInterface>()) {
      // Get the underlying operation pointer.
      Operation *op = func.getOperation();

      uint64_t nameIndex = strMgr.getStringIndex(func.getName());
      uint64_t signatureIndex = typeMgr.getTypeIndex(func.getFunctionType());

      uint64_t functionLocIndex = debuginfo.getOpIndex(op);
      debuginfo.addDebugInfo(functionLocIndex, func.getLoc());

      bool isEntry = isa<cuda_tile::EntryOp>(op);
      Attribute hints;
      if (auto entryOp = dyn_cast<cuda_tile::EntryOp>(op)) {
        hints = entryOp.getOptimizationHintsAttr();
      }

      FunctionMetadata meta{nameIndex, signatureIndex, functionLocIndex,
                            isEntry, hints, func.getArgAttrsAttr()};
#ifdef TILE_IR_INCLUDE_TESTS
      meta.isTestingFunc = isa<cuda_tile::Test_FuncOp>(op);
#endif // TILE_IR_INCLUDE_TESTS
      meta.isExternImpl = isa<cuda_tile::ExternImplOp>(op);
      if (auto funcOp = dyn_cast<cuda_tile::FuncOp>(op)) {
        meta.isExternFunc = funcOp.getIsExtern();
        if (meta.isExternFunc) {
          meta.externFuncConstParams = funcOp.getConstParams();
        }
      }
      functionsMap[op] = meta;
    }
    return success();
  }

  LogicalResult writeFunctionTableSection(raw_ostream &stream) {
    SmallVector<char> buffer;
    llvm::raw_svector_ostream sectionStream(buffer);
    EncodingWriter sectionWriter(sectionStream);

    sectionWriter.writeVarInt(functionsMap.size());

    // Write function metadata and bodies
    for (const auto &pair : functionsMap) {
      Operation *op = pair.first;
      const FunctionMetadata &meta = pair.second;

      auto func = dyn_cast<mlir::FunctionOpInterface>(op);
      assert(
          func &&
          "operation in functionsMap does not implement FunctionOpInterface.");

      sectionWriter.writeVarInt(meta.nameIndex);
      sectionWriter.writeVarInt(meta.signatureIndex);
      // Write entryFlag.
      uint8_t entryFlag = 0;
      // Bit 0 only distinguishes public from private. `nested` has no encoding,
      // so reject it instead of narrowing it to private.
      auto sym = cast<SymbolOpInterface>(op);
      switch (sym.getVisibility()) {
      case SymbolTable::Visibility::Public:
        break;
      case SymbolTable::Visibility::Private:
        entryFlag |=
            static_cast<uint8_t>(Bytecode::FunctionFlags::VisibilityPrivate);
        break;
      case SymbolTable::Visibility::Nested:
        return op->emitError()
               << "symbol `" << sym.getName()
               << "` uses nested symbol visibility, which cannot be encoded in "
                  "bytecode";
      }
      if (meta.isEntry) {
        entryFlag |= static_cast<uint8_t>(Bytecode::FunctionFlags::KindKernel);
        // Pre-13.5: presence bit. At 13.5+ hints are always written for
        // entries.
        if (meta.hints && config.getBytecodeVersion() <
                              BytecodeVersion::kFunctionFlagsFormVersion) {
          entryFlag |= static_cast<uint8_t>(
              Bytecode::FunctionFlags::HasOptimizationHints);
        }
      }
      bool requiresFormLayout = meta.isExternImpl || meta.isExternFunc;
#ifdef TILE_IR_INCLUDE_TESTS
      requiresFormLayout |= meta.isTestingFunc;
#endif // TILE_IR_INCLUDE_TESTS
      if (requiresFormLayout &&
          config.getBytecodeVersion() <
              BytecodeVersion::kFunctionFlagsFormVersion) {
        return op->emitError()
               << "function table entry requires bytecode version "
               << BytecodeVersion::kFunctionFlagsFormVersion.toString();
      }
      if (config.getBytecodeVersion() >=
          BytecodeVersion::kFunctionFlagsFormVersion) {
        // Form layout. Extern FuncOp stays FormNormal; body length 0 selects
        // the extern path.
        if (meta.isExternImpl) {
          entryFlag |=
              static_cast<uint8_t>(Bytecode::FunctionFlags::FormExternImpl);
        }
#ifdef TILE_IR_INCLUDE_TESTS
        else if (meta.isTestingFunc) {
          entryFlag |=
              static_cast<uint8_t>(Bytecode::FunctionFlags::FormTesting);
        }
#endif // TILE_IR_INCLUDE_TESTS
      }

      sectionWriter.writeByte(entryFlag);
      // Continue writing other metadata.
      sectionWriter.writeVarInt(meta.functionLocIndex);

      if (meta.isEntry && (config.getBytecodeVersion() >=
                               BytecodeVersion::kFunctionFlagsFormVersion ||
                           meta.hints)) {
        Attribute hints = meta.hints;
        if (!hints) {
          hints = cuda_tile::OptimizationHintsAttr::get(
              op->getContext(), DictionaryAttr::get(op->getContext()));
        }
        if (failed(writeSelfContainedAttribute(op, kOptimizationHints, hints,
                                               sectionWriter, typeMgr, constMgr,
                                               strMgr))) {
          return failure();
        }
      }

      // Bodiless function-like ops emit their generated op payload followed by
      // a zero body length.
      if (meta.isExternImpl) {
        if (failed(dispatchOpWriter(op, sectionWriter, typeMgr, constMgr,
                                    strMgr, config))) {
          return failure();
        }
        sectionWriter.writeVarInt(0);
        continue;
      }

      // Below 13.5 the function table has no arg_attrs field, and the generated
      // per-op gate never runs for function-like ops. Refuse rather than drop
      // them silently. At 13.5+ the field is positional, so always write it.
      if (config.getBytecodeVersion() < BytecodeVersion::kMinArgAttrsVersion) {
        if (meta.argAttrs) {
          return op->emitError()
                 << "argument attributes require bytecode version "
                 << BytecodeVersion::kMinArgAttrsVersion.toString()
                 << ", targeting " << config.getBytecodeVersion().toString();
        }
      } else {
        // If the attribute is null, replace it with an empty array.
        auto argAttrs = meta.argAttrs;
        if (!argAttrs) {
          argAttrs = ArrayAttr::get(op->getContext(), {});
        }
        if (failed(writeSelfContainedAttribute(op, "arg_attrs", argAttrs,
                                               sectionWriter, typeMgr, constMgr,
                                               strMgr))) {
          return failure();
        }
      }

      if (meta.isExternFunc) {
        // Form layout: body length 0, then always-present const_params.
        sectionWriter.writeVarInt(0);
        ArrayAttr constParams = meta.externFuncConstParams;
        if (!constParams) {
          constParams = ArrayAttr::get(op->getContext(), {});
        }
        if (failed(writeSelfContainedAttribute(op, "const_params", constParams,
                                               sectionWriter, typeMgr, constMgr,
                                               strMgr))) {
          return failure();
        }
        continue;
      }

      SmallVector<char> functionBody;
      if (failed(writeFunctionBody(func, functionBody))) {
        return failure();
      }
      sectionWriter.writeVarInt(functionBody.size());
      sectionWriter.write(functionBody.data(), functionBody.size());
    }
    // Align the function section
    uint64_t alignmentNeeded = alignof(uint64_t);
    sectionWriter.alignTo(alignmentNeeded);
    writeSectionHeader(stream, Bytecode::Section::Func, buffer.size(),
                       sectionWriter.getRequiredAlignment());
    stream.write(buffer.data(), buffer.size());
    return success();
  }

  /// Handles writing regions.
  /// region-bytecode =:
  ///   numBlocks[varint]
  ///   block-bytecode*
  LogicalResult writeRegion(Region &region, EncodingWriter &writer) {
    // Write the number of blocks in the region
    writer.writeVarInt(region.getBlocks().size());

    // Process each block in the region
    for (Block &block : region) {
      if (failed(writeBlock(block, writer))) {
        return failure();
      }
    }

    return success();
  }

  /// Handles writing blocks.
  /// block-bytecode =:
  ///   numArgs[varint]
  ///   argTypeIndex[varint]*  // Type indices for each block argument.
  ///   numOps[varint]
  ///   instruction*           // Bytecode for each operation in the block.
  LogicalResult writeBlock(Block &block, EncodingWriter &writer) {
    // Record the current nextValueIndex. This will be restored after processing
    // the block, effectively rolling back the indices used within this block.
    uint64_t originalNextValueIndex = nextValueIndex;

    // Process block arguments.
    writer.writeVarInt(block.getNumArguments());
    for (BlockArgument arg : block.getArguments()) {
      if (failed(typeMgr.writeTypeIndex(arg.getType(), writer))) {
        return failure();
      }
      // Assign a new index to the block argument.
      // Block arguments are always new values in this scope.
      assert(!valueIndexMap.count(arg) &&
             "block argument encountered that is already in valueIndexMap for "
             "this scope");
      valueIndexMap[arg] = nextValueIndex++;
    }

    // Write number of operations in the block.
    writer.writeVarInt(block.getOperations().size());
    // Process operations in the block.
    for (Operation &op : block) {
      if (failed(writeOperation(&op, writer))) {
        return failure();
      }
    }

    // Remove all of the entries added during parsing of this block.
    for (uint64_t i = 0, e = nextValueIndex - originalNextValueIndex; i < e;
         ++i) {
      valueIndexMap.pop_back();
    }

    // Restore nextValueIndex to what it was before this block.
    nextValueIndex = originalNextValueIndex;
    return success();
  }

private:
  llvm::MapVector<Value, uint64_t> valueIndexMap;
  uint64_t nextValueIndex = 0;

  struct FunctionMetadata {
    uint64_t nameIndex;
    uint64_t signatureIndex;
    uint64_t functionLocIndex;
    bool isEntry;
    Attribute hints;
    ArrayAttr argAttrs;
    bool isExternImpl = false;
    bool isExternFunc = false;
#ifdef TILE_IR_INCLUDE_TESTS
    bool isTestingFunc = false;
#endif // TILE_IR_INCLUDE_TESTS
    ArrayAttr externFuncConstParams = {};
  };
  llvm::MapVector<Operation *, FunctionMetadata> functionsMap;
  TypeManager &typeMgr;
  ConstantManager &constMgr;
  StringManager &strMgr;
  DebugInfoWriter &debuginfo;
  const BytecodeWriterConfig &config;
};
} // end anonymous namespace

LogicalResult
TypeManager::writeSelfContainedTypeAttribute(Attribute attrValue,
                                             EncodingWriter &writer) {
  assert(funcWriter && "type attribute writer requires FunctionTableWriter");
  return funcWriter->writeSelfContainedAttribute(
      /*op=*/nullptr, "type_param", attrValue, writer, *this, *constMgr,
      strMgr);
}

/// Write the global section to the bytecode file.
static LogicalResult
writeGlobalSection(raw_ostream &stream, cuda_tile::ModuleOp module,
                   StringManager &strMgr, TypeManager &typeMgr,
                   ConstantManager &constMgr, DebugInfoWriter &debuginfo,
                   const BytecodeWriterConfig &config) {
  SmallVector<char> buffer;
  llvm::raw_svector_ostream sectionStream(buffer);
  EncodingWriter sectionWriter(sectionStream);

  SmallVector<cuda_tile::GlobalOp> globals;
  for (auto globalOp : module.getOps<cuda_tile::GlobalOp>()) {
    globals.push_back(globalOp);
  }

  if (globals.empty()) {
    return success();
  }

  sectionWriter.writeVarInt(globals.size());

  static const auto kMinGlobalExtendedFields =
      *BytecodeVersion::fromVersion(13, 3, 0);

  for (cuda_tile::GlobalOp globalOp : globals) {
    // Global symbol visibility and the `constant` flag exist only in bytecode
    // 13.3+. Refuse to emit older versions if we would silently lose semantics.
    if (config.getBytecodeVersion() < kMinGlobalExtendedFields) {
      if (globalOp.getSymbolVisibility() != SymbolVisibility::Public) {
        return globalOp.emitError() << "global `" << globalOp.getSymName()
                                    << "` uses non-public symbol visibility, "
                                       "which cannot be encoded "
                                       "in bytecode version "
                                    << config.getBytecodeVersion().toString()
                                    << " (requires bytecode 13.3+)";
      }
      if (globalOp.getConstant()) {
        return globalOp.emitError()
               << "constant global `" << globalOp.getSymName()
               << "` cannot be encoded in bytecode version "
               << config.getBytecodeVersion().toString()
               << " (the `constant` attribute requires bytecode 13.3+)";
      }
    }

    // 1. Write symbol name index.
    sectionWriter.writeVarInt(strMgr.getStringIndex(globalOp.getSymName()));

    // 2. Write type index of the global's value.
    DenseTypedElementsAttr valueAttr = globalOp.getValue();
    sectionWriter.writeVarInt(typeMgr.getTypeIndex(valueAttr.getType()));

    // 3. Write constant index for the global's value.
    uint64_t constIndex;
    if (failed(constMgr.addConstant(valueAttr, constIndex))) {
      return globalOp.emitError("failed to add global constant: '")
             << globalOp.getSymName();
    }
    sectionWriter.writeVarInt(constIndex);

    // 4. Write alignment.
    sectionWriter.writeVarInt(globalOp.getAlignment());

    // 5-6. Symbol visibility and constant flag (bytecode 13.3+ only). Must
    // match parseGlobalSection: older versions read exactly four varints per
    // global.
    if (config.getBytecodeVersion() >= kMinGlobalExtendedFields) {
      sectionWriter.writeVarInt(
          static_cast<uint32_t>(globalOp.getSymbolVisibility()));
      sectionWriter.writeVarInt(static_cast<uint32_t>(globalOp.getConstant()));
    }
  }

  // Write the section header and the buffered content to the main output
  // stream.
  writeSectionHeader(stream, Bytecode::Section::Global, buffer.size(),
                     sectionWriter.getRequiredAlignment());
  stream.write(buffer.data(), buffer.size());
  return success();
}

//===----------------------------------------------------------------------===//
// Producer Section
//===----------------------------------------------------------------------===//
// producer-section =:
//   producerStringIndex[varint]  // Index into the string table
//
/// Write the producer section to the bytecode file.
/// This section is optional and contains producer information (e.g., compiler
/// version, build options) that identifies what tool generated this bytecode.
/// Only available in bytecode version 13.3+.
static LogicalResult writeProducerSection(raw_ostream &stream,
                                          cuda_tile::ModuleOp module,
                                          StringManager &strMgr,
                                          const BytecodeWriterConfig &config) {
  // Producer section is only available in version 13.3+.
  static const auto kMinProducerVersion =
      *BytecodeVersion::fromVersion(13, 3, 0);
  if (config.getBytecodeVersion() < kMinProducerVersion) {
    return success();
  }

  auto producerAttr = module.getProducerAttr();
  if (!producerAttr) {
    return success();
  }

  SmallVector<char> buffer;
  llvm::raw_svector_ostream sectionStream(buffer);
  EncodingWriter sectionWriter(sectionStream);

  // Write the producer string index.
  sectionWriter.writeVarInt(strMgr.getStringIndex(producerAttr.getValue()));

  // Write the section header and the buffered content to the main output
  // stream.
  writeSectionHeader(stream, Bytecode::Section::Producer, buffer.size(),
                     /*alignment=*/1);
  stream.write(buffer.data(), buffer.size());
  return success();
}

//===----------------------------------------------------------------------===//
// BytecodeWriter Implementation
// Manages the overall bytecode writing process by orchestrating different
// layers.
//===----------------------------------------------------------------------===//

LogicalResult cuda_tile::writeBytecode(raw_ostream &os,
                                       cuda_tile::ModuleOp module,
                                       const BytecodeWriterConfig &configIn) {
  // Verify the module (e.g. optimization hints when -Wunsupported-hints
  // -Werr-hints are set).
  if (failed(verify(module.getOperation()))) {
    return failure();
  }
  // Before trying to write the bytecode, verify that the module is
  // self-contained, meaning it does not have any external dependencies that
  // cannot be serialized into bytecode.
  if (failed(verifySelfContainedModuleAndOperationInvariants(module))) {
    return failure();
  }

  BytecodeWriterConfig config = configIn;

  // Public-preview content is intentionally non-portable across versions: when
  // present, it can only be emitted with the explicit opt-in, is version-locked
  // to the current version, and is flagged in the header's version tag so a
  // reader can detect it up front.
  PreviewFeatures requiredPreviewFeatures =
      getRequiredPreviewFeatures(module.getOperation());
  if (!requiredPreviewFeatures.empty()) {
    // Preview content is only valid at the current version. If the caller
    // explicitly asked for a different version, that is a conflict.
    const BytecodeVersion &current = BytecodeVersion::kCurrentVersion;
    if (config.isVersionExplicitlySet() &&
        (config.getBytecodeVersion().getMajor() != current.getMajor() ||
         config.getBytecodeVersion().getMinor() != current.getMinor())) {
      return module->emitError()
             << "module contains public-preview content, which is only "
                "valid at the current bytecode version "
             << current.toString() << ", but version "
             << config.getBytecodeVersion().toString()
             << " was explicitly requested";
    }
    std::optional<BytecodeVersion> previewVersion = BytecodeVersion::fromVersion(
        current.getMajor(), current.getMinor(),
        Bytecode::VersionTag::kContainsPreview);
    assert(previewVersion && "current version must always be valid");
    // Version-lock to the preview-tagged current version. Rebuild the config
    // rather than mutating the version in place so the version/explicitness
    // pair stays consistent.
    config = BytecodeWriterConfig(*previewVersion,
                                  config.isVersionExplicitlySet());
  }

  // Write the header of the bytecode file.
  if (failed(writeHeader(os, module, config))) {
    return failure();
  }

  // Initialize Managers
  StringManager stringMgr;
  TypeManager typeMgr(config, stringMgr);
  ConstantManager constantMgr(config);
  typeMgr.setConstantManager(constantMgr);
  DebugInfoWriter debuginfo(stringMgr, config);

  // Collect all function information to populate the type, string, and constant
  // tables
  FunctionTableWriter funcWriter(typeMgr, constantMgr, stringMgr, debuginfo,
                                 config);
  typeMgr.setFunctionWriter(&funcWriter);
  if (failed(funcWriter.buildFunctionMap(module))) {
    return failure();
  }
  if (failed(writeGlobalSection(os, module, stringMgr, typeMgr, constantMgr,
                                debuginfo, config))) {
    return failure();
  }
  if (failed(funcWriter.writeFunctionTableSection(os))) {
    return failure();
  }
  if (failed(constantMgr.writeConstantSection(os))) {
    return failure();
  }
  if (failed(debuginfo.writeDebugInfoSection(os))) {
    return failure();
  }
  if (failed(typeMgr.writeTypeSection(os))) {
    return failure();
  }
  if (failed(writeProducerSection(os, module, stringMgr, config))) {
    return failure();
  }
  if (failed(stringMgr.writeStringSection(os))) {
    return failure();
  }

  // Write the end section to indicate the end of the bytecode.
  os.write(Bytecode::Section::EndOfBytecode);
  return success();
}
