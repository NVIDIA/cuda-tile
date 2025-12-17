//===- CudaTileAttr.cpp - CUDA Tile IR Attribute wrapper for TableGen --*- C++
//-*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CudaTileAttr.h"

#include "mlir/TableGen/Attribute.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"

#include <iostream>
#include <sstream>

using namespace llvm;
using namespace mlir;

namespace cudatile {
namespace tblgen {

static std::vector<std::string> getMLIRExamples(const llvm::Record &record) {
  auto mdef = record.getValueAsListOfStrings("mlirExamples");
  return std::vector<std::string>(mdef.begin(), mdef.end());
}

static StringRef cleanName(StringRef name) {
  // Remove the "CudaTile_" prefix from the attribute name if present.
  static const std::string prefix = "CudaTile_";
  if (name.starts_with(prefix)) {
    name = name.substr(prefix.size());
  }

  if (name.ends_with("Attr")) {
    name = name.substr(0, name.size() - 4);
  }
  return name;
}

std::string TileIREnumAttr::getAnchor() const {
  return "op-attribute-" + this->opName + "-" + this->name + "-attr";
}

std::string TileIRAttrDef::getAnchor() const {
  return "op-attribute-" + this->opName + "-" + this->name + "-attr";
}

std::string TileIRAttrInterface::getAnchor() const {
  return "op-attribute-" + this->opName + "-" + this->name + "-attr";
}

TileIREnumAttr TileIREnumAttr::fromTableGen(
    const std::string &opName, const llvm::Record &enumInfoRecord,
    const std::optional<std::vector<std::string>> &selectedVariants) {
  TileIREnumAttr attr;
  attr.opName = opName;
  attr.name = cleanName(enumInfoRecord.getName());
  attr.prefixDescription =
      enumInfoRecord.getValueAsString("specPrefixDescription");
  attr.suffixDescription =
      enumInfoRecord.getValueAsString("specSuffixDescription");

  auto enumerants = enumInfoRecord.getValueAsListOfDefs("enumerants");
  for (const auto &enumerant : enumerants) {
    auto sym = enumerant->getValueAsString("symbol");
    std::cout << "sym: " << sym.str() << std::endl;
    // If selectedVariants is not set, all variants are selected.
    //
    // Otherwise only the variants in the selectedVariants are selected.
    bool isSelected = !selectedVariants;
    // If variant does not appear in the selected variants, skip it.
    if (selectedVariants &&
        std::find(selectedVariants->begin(), selectedVariants->end(),
                  sym.str()) != selectedVariants->end()) {
      isSelected = true;
    }
    auto description = enumerant->getValueAsString("description");
    // Get the human readable representation of the enum case.
    auto str = enumerant->getValueAsString("str").trim();
    attr.enumcases.push_back(
        TileIREnumCase{str.str(), description.str(), isSelected});
  }
  return attr;
}

TileIRAttrDef TileIRAttrDef::fromTableGen(const std::string &opName,
                                          const llvm::Record &record) {
  TileIRAttrDef attr;
  attr.opName = opName;

  attr.name = cleanName(record.getName());
  attr.prefixDescription = record.getValueAsString("description");
  attr.examples = getMLIRExamples(record);
  return attr;
}

TileIRAttrInterface
TileIRAttrInterface::fromTableGen(const std::string &opName,
                                  const llvm::Record &record) {
  TileIRAttrInterface attr;

  auto name = record.getName();
  attr.name = cleanName(name);
  attr.opName = opName;
  attr.prefixDescription =
      record.getValueAsString("cudaTileAttrPrefixDescription");
  attr.suffixDescription =
      record.getValueAsString("cudaTileAttrSuffixDescription");
  return attr;
}

std::vector<TileIRAttrDef>
findInterfaceImplementors(const TileIRAttrInterface &attrInterface,
                          const std::vector<const Record *> &attrDefs) {
  std::cout << "findInterfaceImplementors: " << attrInterface.name << std::endl;
  std::cout << "findInterfaceImplementors: numCandidateAttributeDefs: "
            << attrDefs.size() << std::endl;

  // Move to find implementators.
  std::vector<StringRef> implementators;
  for (const auto &attrDef : attrDefs) {
    auto traits = attrDef->getValueAsListOfDefs("traits");
    for (const auto &trait : traits) {
      if (trait->isSubClassOf("DeclareAttrInterfaceMethods")) {
        auto cppInterfaceName = trait->getValueAsString("cppInterfaceName");
        // This is a bit of hack to check that it implements the interface but
        // works for now.
        std::cout << "findInterfaceImplementors: cppInterfaceName: "
                  << cppInterfaceName.str() << std::endl;
        std::cout
            << "findInterfaceImplementors: attrInterface.name + AttrInterface: "
            << attrInterface.name + "AttrInterface" << std::endl;
        if (cppInterfaceName.str() == attrInterface.name + "AttrInterface") {
          implementators.push_back(attrDef->getName());
        }
      }
    }
  }

  std::cout << "findInterfaceImplementors: numImplementators: "
            << implementators.size() << std::endl;

  std::vector<TileIRAttrDef> implementatorsDefs;
  for (auto &attrDef : attrDefs) {
    if (std::find(implementators.begin(), implementators.end(),
                  attrDef->getName()) != implementators.end()) {
      // Probably should allow this to be other types too.
      implementatorsDefs.emplace_back(
          TileIRAttrDef::fromTableGen(attrInterface.opName, *attrDef));
    }
  }

  return implementatorsDefs;
}

} // namespace tblgen
} // namespace cudatile
