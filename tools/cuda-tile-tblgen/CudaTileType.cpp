//===- CudaTileType.cpp - CUDA Tile IR Type wrapper for TableGen -------*- C++
//-*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the CUDA Tile dialect type parsing and printing
// utilities.
//
//===----------------------------------------------------------------------===//

#include "CudaTileType.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"

#include <iostream>
#include <sstream>

using namespace llvm;
using namespace mlir;

namespace cudatile {
namespace tblgen {

CudaTileElementType elementTypeFromString(StringRef name) {
  if (name == "CudaTile_Int1")
    return kI1;
  if (name == "CudaTile_Int8")
    return kI8;
  if (name == "CudaTile_Int16")
    return kI16;
  if (name == "CudaTile_Int32")
    return kI32;
  if (name == "CudaTile_Int64")
    return kI64;
  if (name == "CudaTile_Float8E4M3FN")
    return kF8E4M3FN;
  if (name == "CudaTile_Float8E5M2")
    return kF8E5M2;
  if (name == "CudaTile_Float16")
    return kF16;
  if (name == "CudaTile_BFloat16")
    return kBF16;
  if (name == "CudaTile_Float32")
    return kF32;
  if (name == "CudaTile_TF32")
    return kTF32;
  if (name == "CudaTile_Float64")
    return kF64;
  return kUnknown;
}

std::vector<CudaTileElementType> allElementTypes() {
  return {kI1,  kI8,  kI16,  kI32,  kI64,      kF16,
          kF32, kF64, kBF16, kTF32, kF8E4M3FN, kF8E5M2};
}

std::ostream &operator<<(std::ostream &os, CudaTileElementType elementType) {
  switch (elementType) {
  case kI1:
    os << "i1";
    break;
  case kI8:
    os << "i8";
    break;
  case kI16:
    os << "i16";
    break;
  case kI32:
    os << "i32";
    break;
  case kI64:
    os << "i64";
    break;
  case kF8E4M3FN:
    os << "fp8e4m3fn";
    break;
  case kF8E5M2:
    os << "fp8e5m2";
    break;
  case kF16:
    os << "f16";
    break;
  case kF32:
    os << "f32";
    break;
  case kF64:
    os << "f64";
    break;
  case kBF16:
    os << "bf16";
    break;
  case kTF32:
    os << "tf32";
    break;
  case kUnknown:
    os << "unknown";
    break;
  default:
    std::cout << static_cast<int>(elementType) << std::endl;
    llvm_unreachable("Invalid element type");
    break;
  }

  return os;
}

std::string elementTypeToString(CudaTileElementType elementType) {
  std::stringstream ss;
  ss << elementType;
  return ss.str();
}

TileIRType TileIRType::tile(const std::vector<TileIRType> &allowedTypes) {
  return TileIRType(std::make_shared<TileType>(allowedTypes));
}

TileIRType TileIRType::any_type() {
  return TileIRType(std::make_shared<AnyType>());
}

TileIRType TileIRType::token() {
  return TileIRType(std::make_shared<TokenType>());
}

TileIRType TileIRType::tensor_view() {
  return TileIRType(std::make_shared<TensorViewType>());
}

TileIRType TileIRType::float_tile() {
  // Get base float types (f16, bf16, f32, f64)
  auto baseTypes = TileIRType::base_float_tile();
  auto types = baseTypes.as<TileType>()->allowedTypes;

  // Add fp8 and tf32 types
  types.emplace_back(std::make_shared<ElementType>(kF8E4M3FN));
  types.emplace_back(std::make_shared<ElementType>(kF8E5M2));
  types.emplace_back(std::make_shared<ElementType>(kTF32));

  return TileIRType(std::make_shared<TileType>(types));
}

TileIRType TileIRType::int_tile() {
  auto types = {
      TileIRType(std::make_shared<ElementType>(kI1)),
      TileIRType(std::make_shared<ElementType>(kI8)),
      TileIRType(std::make_shared<ElementType>(kI16)),
      TileIRType(std::make_shared<ElementType>(kI32)),
      TileIRType(std::make_shared<ElementType>(kI64)),
  };
  return TileIRType(std::make_shared<TileType>(types));
}

TileIRType TileIRType::base_float_tile() {
  auto types = {
      TileIRType(std::make_shared<ElementType>(kF16)),
      TileIRType(std::make_shared<ElementType>(kBF16)),
      TileIRType(std::make_shared<ElementType>(kF32)),
      TileIRType(std::make_shared<ElementType>(kF64)),
  };
  return TileIRType(std::make_shared<TileType>(types));
}

TileIRType TileIRType::numeric_tile() {
  auto floatTypes = TileIRType::float_tile();
  auto intTypes = TileIRType::int_tile();
  auto types = intTypes.as<TileType>()->allowedTypes;
  types.insert(types.end(), floatTypes.as<TileType>()->allowedTypes.begin(),
               floatTypes.as<TileType>()->allowedTypes.end());
  return TileIRType(std::make_shared<TileType>(types));
}

TileIRType TileIRType::any_tile() {
  std::vector<TileIRType> types;
  return TileIRType(std::make_shared<TileType>(types));
}

TileIRType TileIRType::pointer(const std::vector<TileIRType> &elementTypes) {
  return TileIRType(std::make_shared<PointerType>(elementTypes));
}

TileIRType TileIRType::builtin(std::string name) {
  return TileIRType(std::make_shared<BuiltinType>(std::move(name)));
}

TileIRType TileIRType::attribute(std::string operationName,
                                 std::string attributeName) {
  return TileIRType(std::make_shared<AttributeType>(std::move(operationName),
                                                    std::move(attributeName)));
}

TileIRType TileIRType::variadic(TileIRType type) {
  return TileIRType(std::make_shared<VariadicType>(std::move(type)));
}

TileIRType TileIRType::symbol() {
  return TileIRType(std::make_shared<BuiltinType>(std::string("Symbol"),
                                                  std::string("type-Symbol")));
}

TileIRType TileIRType::flag() {
  return TileIRType(std::make_shared<BuiltinType>(std::string("Flag"),
                                                  std::string("type-Flag")));
}

std::string kindToString(TileIRTypeKind type) {
  switch (type) {
  case kElementType:
    return "ElementType";
  case kTile:
    return "Tile";
  case kTensorView:
    return "TensorView";
  case kName:
    return "Name";
  case kPointer:
    return "Pointer";
  case kVariadic:
    return "Variadic";
  case kUninitialized:
    return "Uninitialized";
  default:
    llvm_unreachable("Invalid type descriptor type");
  }
};

void printAppliedType(std::ostream &os, const std::string &ty_ctor,
                      const std::vector<TileIRType> &args,
                      const std::string &sep) {
  os << ty_ctor;
  size_t i = 0;
  if (!args.empty()) {
    os << "<";
    for (auto &arg : args) {
      os << arg.toString();
      if (i != args.size() - 1) {
        os << sep << " ";
      }
      i++;
    }
    os << ">";
  }
}

std::string TileIRType::toString() const {
  std::stringstream ss;
  switch (kind()) {
  case kTile: {
    auto tensor = this->as<TileType>();
    // if ranks + dtype is empty we print the polymorphic version.
    // i.e tile<_, _> which we shorthand to `tile`
    // if ranks is empty but we have types we print tile<_, a | b | c>
    // if both are popualted we print something like tile<(), a | b | c> for
    // zero or for scalars we can print tile<(), a | b | c> as a | b | c
    printAppliedType(ss, "tile", tensor->allowedTypes, " | ");
    break;
  }
  case kAnyType: {
    ss << "any";
    break;
  }
  case kToken: {
    ss << "token";
    break;
  }
  case kTensorView: {
    ss << "tensor_view";
    break;
  }
  case kName: {
    auto nominal = this->as<BuiltinType>();
    ss << nominal->name;
    break;
  }
  case kAttributeType: {
    auto attribute = this->as<AttributeType>();
    ss << "Attribute<" << attribute->operationName << ", "
       << attribute->attributeName << ">";
    break;
  }
  case kPointer: {
    auto pointer = this->as<PointerType>();
    if (pointer->elementTypes.empty()) {
      // We want to print the polymorphic version.
      printAppliedType(ss, "ptr", pointer->elementTypes, " | ");
    } else {
      printAppliedType(ss, "ptr", {}, " | ");
    }
    break;
  }
  case kVariadic: {
    auto variadic = this->as<VariadicType>();
    ss << "Variadic<" << variadic->type.toString() << ">";
    break;
  }
  case kElementType: {
    auto elementType = this->as<ElementType>();
    ss << elementTypeToString(elementType->elementType);
    break;
  }
  default: {
    auto msg = "Invalid type descriptor type: " + kindToString(kind());
    llvm_unreachable(msg.c_str());
  }
  }
  return ss.str();
}

std::ostream &operator<<(std::ostream &os, const TileIRType &ty) {
  os << ":tileirty:`";
  os << ty.toString();
  os << "`";
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const TileIRType &ty) {
  os << ":tileirty:`";
  os << ty.toString();
  os << "`";
  return os;
}

TileIRType convertAttributeDef(const std::string &opName,
                               const Record &attrDef) {
  auto attrName = attrDef.getName().str();
  // std::cout << "attrName: " << attrName.str() << std::endl;
  if (attrName == "UnitAttr") {
    return TileIRType::flag();
  } else if (attrName == "DenseI32ArrayAttr") {
    return TileIRType::builtin("Array<i32>");
  } else if (attrName == "IntegerAttr") {
    return TileIRType::builtin("int");
  } else if (attrName == "I32Attr") {
    return TileIRType::builtin("i32");
  } else if (attrName == "I64Attr") {
    return TileIRType::builtin("i64");
  } else if (attrName == "StrAttr") {
    return TileIRType::builtin("String");
  } else if (attrName == "SymbolNameAttr" || attrName == "SymbolRefAttr" ||
             attrName == "FlatSymbolRefAttr") {
    return TileIRType::symbol();

  } else if (attrDef.isSubClassOf("TypeAttrOf")) {
    // Consider refining this to be more specific in the future
    // right now all `TypeAttrOf` will be rendered as `Type`.
    return TileIRType::builtin("Type");
  } else if (attrDef.isSubClassOf("OptionalAttr") ||
             attrDef.isSubClassOf("ConfinedAttr")) {
    auto baseAttr = attrDef.getValueAsDef("baseAttr");
    return convertAttributeDef(opName, *baseAttr);
  } else if (attrDef.isSubClassOf("DefaultValuedAttr")) {
    // TODO: Add a new case for defaulted valued attributes.
    auto baseAttr = attrDef.getValueAsDef("baseAttr");
    return convertAttributeDef(opName, *baseAttr);
  } else if (attrName == "DictArrayAttr") {
    // TODO(@jroesch): what do we render these as?
    return TileIRType::builtin("Attributes");
  } else if (attrName == "BoolAttr") {
    return TileIRType::builtin("bool");
  } else if (attrName == "ArrayAttr") {
    return TileIRType::builtin("Array");

  } else if (attrName == "StringElementsAttr") {
    return TileIRType::builtin("String");
    // Attributes
  } else if (attrName == "CudaTile_AssumePredicateAttrInterface") {
    return TileIRType::attribute(opName, "AssumePredicate");
  } else if (attrName == "CudaTile_ComparisonPredicateAttr") {
    return TileIRType::attribute(opName, "ComparisonPredicate");
  } else if (attrName == "CudaTile_ComparisonOrderingAttr") {
    return TileIRType::attribute(opName, "ComparisonOrdering");
  } else if (attrName == "CudaTile_SignednessAttr") {
    return TileIRType::attribute(opName, "Signedness");
  } else if (attrName == "CudaTile_RoundingModeAttr") {
    return TileIRType::attribute(opName, "RoundingMode");
  } else if (attrName == "CudaTile_AtomicRMWModeAttr") {
    return TileIRType::attribute(opName, "AtomicRMWMode");
  } else if (attrName == "CudaTile_MemoryOrderingSemanticsAttr") {
    return TileIRType::attribute(opName, "MemoryOrderingSemantics");
  } else if (attrName == "CudaTile_MemoryScopeAttr") {
    return TileIRType::attribute(opName, "MemoryScope");
  } else if (attrName == "CudaTile_OptimizationHintsAttr") {
    return TileIRType::attribute(opName, "OptimizationHints");
  } else if (attrName == "CudaTile_IntegerOverflowAttr") {
    return TileIRType::attribute(opName, "IntegerOverflow");
  } else if (attrName == "CudaTile_PaddingValueAttr") {
    return TileIRType::attribute(opName, "PaddingValue");
  } else if (attrName == "CudaTile_DivRoundingModeAttr") {
    return TileIRType::attribute(opName, "DivRoundingMode");
  } else if (attrName == "Builtin_DenseIntOrFPElementsAttr") {
    return TileIRType::builtin("DenseConstant");
  } else {
    PrintFatalError("convertAttributeDef: unhandled attribute type: `" +
                    attrName + "`");
  }
}

TileIRType convertAttribute(const std::string &opName, const Attribute &attr) {
  auto &attrDef = attr.getDef();
  return convertAttributeDef(opName, attrDef);
}

// Forward declaration.
TileIRType getType(const Record &tcDef);

static std::vector<TileIRType>
getAllowedElementTypes(const llvm::Record &tcDef) {
  auto allowedTypes = tcDef.getValueAsListOfDefs("allowedElementTypes");
  std::vector<TileIRType> types;
  for (auto type : allowedTypes) {
    // std::cout << "record: " << type->getName().str() << std::endl;
    auto t = getType(*type);
    // std::cout << "type: " << t << std::endl;
    types.push_back(t);
  }

  return types;
}

// static std::vector<CudaTileType> getAllowedTypes(const llvm::Record &tcDef) {
//   auto allowedTypes = tcDef.getValueAsListOfDefs("allowedTypes");
//   std::vector<CudaTileType> types;
//   for (auto type : allowedTypes) {
//     // std::cout << "record: " << type->getName().str() << std::endl;
//     auto t = getType(*type);
//     // std::cout << "type: " << t << std::endl;
//     types.push_back(t);
//   }

//   return types;
// }

TileIRType getType(const Record &tcDef) {
  // std::cout << "-----" << tcDef.getName().str() << std::endl;
  // for (auto superclass : tcDef.getSuperClasses()) {
  // std::cout << "superclass: " << superclass.first->getName().str() <<
  // std::endl;
  // }
  // std::cout << "-----" << std::endl;

  auto el_type = elementTypeFromString(tcDef.getName().str());

  if (el_type != kUnknown) {
    return TileIRType(std::make_shared<ElementType>(el_type));
  }
  // If the type is a number tensor type, return the numeric tensor type.
  //
  // We put this one first because it is more specific than the other tensor
  // types.
  if (tcDef.getName() == "CudaTile_NumberTileType") {
    return TileIRType::numeric_tile();
  }

  // Base Types
  if (tcDef.getName() == "CudaTile_AnyType" || tcDef.getName() == "AnyType") {
    return TileIRType::builtin("Any");
  } else if (tcDef.getName() == "CudaTile_Float8E4M3FN") {
    return TileIRType::tile(
        {TileIRType(std::make_shared<ElementType>(kF8E4M3FN))});
  } else if (tcDef.getName() == "CudaTile_Float8E5M2") {
    return TileIRType::tile(
        {TileIRType(std::make_shared<ElementType>(kF8E5M2))});
  } else if (tcDef.getName() == "CudaTile_TFloat32") {
    return TileIRType::tile({TileIRType(std::make_shared<ElementType>(kTF32))});
  } else if (tcDef.getName() == "CudaTile_Tf32FloatTileType") {
    return TileIRType::tile({TileIRType(std::make_shared<ElementType>(kTF32))});
  } else if (tcDef.getName() == "CudaTile_FloatTileType") {
    return TileIRType::float_tile();
  } else if (tcDef.getName() == "CudaTile_PointerTileType") {
    return TileIRType::pointer({});
  } else if (tcDef.getName() == "CudaTile_PointerType") {
    return TileIRType::pointer({});
  } else if (tcDef.getName() == "CudaTile_TokenType") {
    return TileIRType::token();
  } else if (tcDef.getName() == "CudaTile_TensorViewType") {
    return TileIRType::tensor_view();
    // This should be a builtin type.
  } else if (tcDef.getName() == "CudaTile_PartitionViewType") {
    return TileIRType::builtin("partition_view");
  } else if (tcDef.getName() == "CudaTile_StridedViewType") {
    return TileIRType::builtin("strided_view");
  } else if (tcDef.getName() == "CudaTile_TileView") {
    // Today we represent the view type interface as a builtin type.
    return TileIRType::builtin("view_type");
  } else if (tcDef.getName() == "CudaTile_IntTileType") {
    return TileIRType::int_tile();
  } else if (tcDef.getName() == "CudaTile_BaseFloatTileType") {
    return TileIRType::base_float_tile();
    // TensorOf
  } else if (tcDef.isSubClassOf("CudaTile_ScalarTileOf")) {
    auto allowedElementTypes = getAllowedElementTypes(tcDef);
    // for (auto type : allowedElementTypes) {
    // std::cout << "type22: " << type << std::endl;
    // }
    auto t = TileIRType::tile(allowedElementTypes);
    // std::cout << "t: " << std::endl;
    //  std::cout << t << std::endl;
    return t;
  } else if (tcDef.isSubClassOf("CudaTile_TileOf")) {
    auto allowedElementTypes = getAllowedElementTypes(tcDef);
    return TileIRType::tile(allowedElementTypes);
  } else if (tcDef.isSubClassOf("AnyTypeOf")) {
    // auto allowedTypes = getAllowedTypes(tcDef);
    // return allowedTypes[0];
    return TileIRType::builtin("any");

  } else if (tcDef.isSubClassOf("FlagType")) {
    return TileIRType::builtin("Flag");
  } else if (tcDef.getName() == "CudaTile_TileType") {
    return TileIRType::any_tile();
  } else if (tcDef.isSubClassOf("Variadic")) {
    auto baseType = tcDef.getValueAsDef("baseType");
    auto baseTypeDesc = getType(*baseType);
    return TileIRType::variadic(baseTypeDesc);
  } else if (tcDef.isSubClassOf("Optional")) {
    auto baseType = tcDef.getValueAsDef("baseType");
    auto baseTypeDesc = getType(*baseType);
    // TODO(@jroesch): add optional
    return baseTypeDesc;
  } else if (tcDef.getName() == "I32") {
    return TileIRType::builtin("i32");
  } else {
    PrintFatalError("getType: unsupported type `" + tcDef.getName().str() +
                    "`");
  }
}

} // namespace tblgen
} // namespace cudatile
