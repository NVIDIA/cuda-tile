//===- SpecGen.h - CUDA Tile dialect spec generator helpers -----*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_TOOLS_CUDATILETBLGEN_SPECGEN_H_
#define CUDA_TILE_TOOLS_CUDATILETBLGEN_SPECGEN_H_

#include "mlir/Support/LLVM.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

namespace cudatile {
namespace tblgen {

void generateSpec(mlir::raw_ostream &os, const llvm::RecordKeeper &records,
                  const std::optional<std::string> &examplesDirectory);

} // namespace tblgen
} // namespace cudatile

#endif // CUDA_TILE_TOOLS_CUDATILETBLGEN_SPECGEN_H_
