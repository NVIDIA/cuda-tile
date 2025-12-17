//===- CudaTileOp.h - CUDA Tile operation definitions -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the CUDA Tile dialect operation constraints.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_CUDATILETBLGEN_CUDATILEOP_H_
#define CUDA_TILE_TOOLS_CUDATILETBLGEN_CUDATILEOP_H_

#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

#include "CudaTileAttr.h"
#include "CudaTileType.h"
#include "Emitter.h"
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using mlir::tblgen::Operator;

namespace cudatile {
namespace tblgen {

enum ParameterType {
  kArgument,
  kAttribute,
  kProperty,
};

struct OperationParameter {
  std::string opName;
  std::string name;
  std::string sinceVersion;
  std::string specDesc;
  std::optional<TypeConstraint> typeConstraint;
  std::optional<Attribute> attribute;
  std::optional<Property> property;
  std::optional<std::vector<std::string>> selectedVariants;
  ParameterType type;

  OperationParameter(const OperationParameter &other) = default;
  OperationParameter(OperationParameter &&other) = default;
  OperationParameter &operator=(const OperationParameter &other) = default;
  OperationParameter &operator=(OperationParameter &&other) = default;

  std::string getDescription() const;
  TileIRType getTypeDescription() const;
};

struct AllTypesMatch {
  std::vector<std::string> names;
  std::string type;
  AllTypesMatch() : names(), type() {}
  AllTypesMatch(std::vector<std::string> names, std::string type)
      : names(std::move(names)), type(std::move(type)) {}
};

struct AllElementTypeMatch {
  std::vector<std::string> names;
  std::string type;
  AllElementTypeMatch() : names(), type() {}
  AllElementTypeMatch(std::vector<std::string> names, std::string type)
      : names(std::move(names)), type(std::move(type)) {}
};

struct TypesMatchWith {
  std::string message;
  TypesMatchWith() : message() {}
  TypesMatchWith(std::string message) : message(std::move(message)) {}
};

struct AnyTypeOf {
  std::vector<std::string> names;
  AnyTypeOf() : names() {}
  AnyTypeOf(std::vector<std::string> names) : names(std::move(names)) {}
};

struct AllRanksMatch {
  std::vector<std::string> names;
  AllRanksMatch() : names() {}
  AllRanksMatch(std::vector<std::string> names) : names(std::move(names)) {}
};

struct SameTypeOperands {
  std::vector<std::string> names;
  std::string type;
  SameTypeOperands() : names(), type() {}
  SameTypeOperands(std::vector<std::string> names, std::string type)
      : names(std::move(names)), type(std::move(type)) {}
};

struct SameOperandsAndResultShape {
  std::vector<std::string> names;
  SameOperandsAndResultShape() : names() {}
  SameOperandsAndResultShape(std::vector<std::string> names)
      : names(std::move(names)) {}
};

struct SameOperandsAndResultElementType {
  std::vector<std::string> names;
  std::string type;
  SameOperandsAndResultElementType() : names(), type() {}
  SameOperandsAndResultElementType(std::vector<std::string> names,
                                   std::string type)
      : names(std::move(names)), type(std::move(type)) {}
};

struct OperationTrait {
  std::string description;
  OperationTrait() : description() {}
  OperationTrait(std::string description)
      : description(std::move(description)) {}
};

using OperationConstraint =
    std::variant<AllTypesMatch, AllElementTypeMatch, SameOperandsAndResultShape,
                 TypesMatchWith, AnyTypeOf, SameOperandsAndResultElementType,
                 AllRanksMatch, SameTypeOperands, OperationTrait>;

struct OperationSignature {
  std::string name;
  std::vector<OperationParameter> parameters;
  std::vector<OperationParameter> results;
  std::vector<OperationConstraint> constraints;
  // This copies for now but we could optimize if it matters.
  std::unordered_map<std::string, OperationParameter> parameterMap;

  std::optional<OperationParameter>
  getParamOrResult(const std::string &name) const {
    auto it = this->parameterMap.find(name);
    if (it != this->parameterMap.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  OperationSignature()
      : name(""), parameters(), results(), constraints(), parameterMap() {}
  OperationSignature(const Operator &op);
  OperationSignature(const OperationSignature &other) = default;
  OperationSignature(OperationSignature &&other) = default;
  OperationSignature &operator=(const OperationSignature &other) = default;
};

class CudaTileOp {
public:
  CudaTileOp(const mlir::tblgen::Operator &op);
  CudaTileOp(const Record *op) : CudaTileOp(mlir::tblgen::Operator(op)) {}
  CudaTileOp(const CudaTileOp &other);
  CudaTileOp(CudaTileOp &&other) = default;

  std::string getOperationName() const { return this->op.getOperationName(); }

  llvm::StringRef getDescription() const;
  std::vector<TileIRAttr> getAttributes();
  std::string getCudaTileSpecGroup();
  std::string getCudaTileSpecSubGroup();
  std::vector<std::string> getMLIRExamples();

  std::vector<Table> getDescriptionTables();

  OperationSignature signature;
  // protected:
  mlir::tblgen::Operator op;
};

} // namespace tblgen
} // namespace cudatile

#endif //  CUDA_TILE_TOOLS_CUDATILETBLGEN_CUDATILEOP_H_
