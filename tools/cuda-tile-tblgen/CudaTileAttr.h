//===- CudaTileAttr.h - CUDA Tile IR Attribute wrapper for TableGen ----*- C++
//-*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_TOOLS_CUDATILETBLGEN_TILEIRATTR_H_
#define CUDA_TILE_TOOLS_CUDATILETBLGEN_TILEIRATTR_H_

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

struct TileIREnumCase {
  std::string str;
  std::string description;
  bool isSelected;
};

struct TileIREnumAttr {
  std::string opName;
  std::string name;
  std::string prefixDescription;
  std::string suffixDescription;
  std::vector<TileIREnumCase> enumcases;

  std::string getAnchor() const;
  static TileIREnumAttr
  fromTableGen(const std::string &opName, const llvm::Record &record,
               const std::optional<std::vector<std::string>> &selectedVariants);
};

struct TileIRAttrDef {
  std::string opName;
  std::string name;
  std::string prefixDescription;
  std::string suffixDescription;
  std::vector<std::string> examples;

  std::string getAnchor() const;
  static TileIRAttrDef fromTableGen(const std::string &opName,
                                    const llvm::Record &record);
};

struct TileIRAttrInterface {
  std::string opName;
  std::string name;
  std::string prefixDescription;
  std::string suffixDescription;

  std::string getAnchor() const;

  static TileIRAttrInterface fromTableGen(const std::string &opName,
                                          const llvm::Record &record);
};

using TileIRAttr =
    std::variant<TileIREnumAttr, TileIRAttrDef, TileIRAttrInterface>;

std::vector<TileIRAttrDef>
findInterfaceImplementors(const TileIRAttrInterface &attrInterface,
                          const std::vector<const Record *> &attrDefs);

} // namespace tblgen
} // namespace cudatile

#endif // CUDA_TILE_TOOLS_CUDATILETBLGEN_TILEIRATTR_H_
