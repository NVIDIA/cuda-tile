//===- TileIRType.h - CUDA Tile IR Type wrapper for TableGen ---------*- C++
//-*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_TOOLS_CUDATILETBLGEN_TILEIRTYPE_H_
#define CUDA_TILE_TOOLS_CUDATILETBLGEN_TILEIRTYPE_H_

#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using mlir::tblgen::Operator;

namespace cudatile {
namespace tblgen {

enum CudaTileElementType {
  kI1,
  kI8,
  kI16,
  kI32,
  kI64,
  kF8E4M3FN,
  kF8E5M2,
  kF16,
  kF32,
  kF64,
  kBF16,
  kTF32,
  kUnknown,
};

CudaTileElementType elementTypeFromString(StringRef name);

std::vector<CudaTileElementType> allElementTypes();

std::ostream &operator<<(std::ostream &os, CudaTileElementType elementType);

std::string elementTypeToString(CudaTileElementType elementType);

enum TileIRTypeKind {
  kElementType,
  kTile,
  kTensorView,
  kToken,
  kName,
  kAttributeType,
  kPointer,
  kVariadic,
  kAnyType,
  kUninitialized,
};

// The base class for all CUDA Tile IR types.
struct TileIRTypeBase {
  TileIRTypeKind type;

  TileIRTypeBase(TileIRTypeKind type) : type(type) {}
  TileIRTypeBase() : type(kUninitialized) {}
};

// A wrapper around a shared pointer to a TileIRTypeBase
// to make it easier to pass around and manage.
struct TileIRType {
  std::shared_ptr<TileIRTypeBase> ty_ptr;

  TileIRType(std::shared_ptr<TileIRTypeBase> ty_ptr)
      : ty_ptr(std::move(ty_ptr)) {}

  TileIRType(const TileIRType &other) = default;

  TileIRType(TileIRType &&other) : ty_ptr(std::move(other.ty_ptr)) {}

  TileIRType &operator=(const TileIRType &other) = default;
  TileIRType &operator=(TileIRType &&other) = default;

  TileIRType() : ty_ptr(std::make_shared<TileIRTypeBase>()) {}

  template <typename T>
  T *as() const {
    return std::static_pointer_cast<T>(ty_ptr).get();
  }

  TileIRTypeKind kind() const { return ty_ptr->type; }

  // A type that represents a element with a set of allowed element types.
  static TileIRType tile(const std::vector<TileIRType> &allowedTypes);

  // A type that represents any valid CUDA Tile IR
  static TileIRType any_type();

  // A type that represents a token.
  static TileIRType token();

  // A type that represents a CUDA Tile IR tensor view.
  static TileIRType tensor_view();

  // A type that represents a CUDA Tile IR integer tensor (i1/i8/i16/i32/i64).
  static TileIRType int_tile();

  // A type that represents a CUDA Tile IR base float tensor (f16/bf16/f32/f64).
  static TileIRType base_float_tile();

  // A type that represents a CUDA Tile IR float tensor
  // (f8e4m3fn/f8e5m2/f16/bf16/f32/tf32/f64).
  static TileIRType float_tile();

  // A type that represents a CUDA Tile IR numeric tensor.
  static TileIRType numeric_tile();

  // A type that represents a CUDA Tile IR tile with any element type.
  static TileIRType any_tile();

  // A type that represents a pointer to a CUDA Tile IR type with the given
  // element types.
  static TileIRType pointer(const std::vector<TileIRType> &elementTypes);

  // A type that represents an builtin type.
  static TileIRType builtin(std::string name);

  // A type that represents an attribute.
  static TileIRType attribute(std::string operationName,
                              std::string attributeName);

  // A type that represents a variadic argument taking N or more arguments
  // of the provided type.
  static TileIRType variadic(TileIRType type);

  // The set of "meta types" used in the dialect definition.
  //
  // A type that represents a symbol.
  static TileIRType symbol();
  // A type that represents a flag.
  static TileIRType flag();

  std::string toString() const;

  friend raw_ostream &operator<<(raw_ostream &os, const TileIRType &descriptor);

  friend std::ostream &operator<<(std::ostream &os,
                                  const TileIRType &descriptor);
};

// A type that represents any valid CUDA Tile IR type.
struct AnyType : TileIRTypeBase {
  AnyType() : TileIRTypeBase(kAnyType) {}
};

// A type that represents a memory ordering token.
struct TokenType : TileIRTypeBase {
  TokenType() : TileIRTypeBase(kToken) {}
};

// A type that represents a tile.
struct TileType : TileIRTypeBase {
  std::vector<int> allowedRanks;
  std::vector<TileIRType> allowedTypes;

  TileType(std::vector<TileIRType> allowedTypes)
      : TileIRTypeBase(kTile), allowedTypes(std::move(allowedTypes)) {}
};

// A type that represents a tensor view.
struct TensorViewType : TileIRTypeBase {
  TensorViewType() : TileIRTypeBase(kTensorView) {}
};

// A type that represents an element type.
struct ElementType : TileIRTypeBase {
  CudaTileElementType elementType;
  ElementType(CudaTileElementType elementType)
      : TileIRTypeBase(kElementType), elementType(elementType) {}
};

// A type that represents a built-in type with
// a description defined in the specification inside of
// operations.rst.

struct BuiltinType : TileIRTypeBase {
  std::string name;
  std::optional<std::string> anchor;

  BuiltinType(std::string name)
      : TileIRTypeBase(kName), name(std::move(name)), anchor() {}

  BuiltinType(std::string name, std::string anchor)
      : TileIRTypeBase(kName), name(std::move(name)),
        anchor(std::move(anchor)) {}
};

// A type that represents an opaque named type.
struct AttributeType : TileIRTypeBase {
  std::string operationName;
  std::string attributeName;

  AttributeType(std::string opName, std::string attributeName)
      : TileIRTypeBase(kAttributeType), operationName(std::move(opName)),
        attributeName(std::move(attributeName)) {}
};

// A type that represents a pointer type.
struct PointerType : TileIRTypeBase {
  // The set of possible element types for the pointer.
  //
  // Note: empty means that there are no element type constraints.
  std::vector<TileIRType> elementTypes;

  PointerType(std::vector<TileIRType> elementTypes)
      : TileIRTypeBase(kPointer), elementTypes(std::move(elementTypes)) {}
};

// A type that represents a variadic number of arguments of a given type.
struct VariadicType : TileIRTypeBase {
  TileIRType type;
  VariadicType(TileIRType type)
      : TileIRTypeBase(kVariadic), type(std::move(type)) {}
};

// Convert a type from llvm::Record to CudaTileType.
TileIRType getType(const Record &tcDef);

// Convert an attribute from an llvm::Record to CudaTileType.
TileIRType convertAttribute(const std::string &opName, const Attribute &attr);

} // namespace tblgen
} // namespace cudatile

#endif //  CUDA_TILE_TOOLS_CUDATILETBLGEN_TILEIRTYPE_H_
