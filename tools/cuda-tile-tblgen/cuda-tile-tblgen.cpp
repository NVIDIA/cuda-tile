//===- cuda-tile-tblgen.cpp - CUDA Tile dialect tblgen ----------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains the main function for generating the CudaTile spec from
// MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

#include "Emitter.h"
#include "SpecGen.h"

using namespace llvm;
using namespace mlir;
using namespace cudatile::tblgen;

int main(int argc, char **argv) {

static mlir::GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

static cl::OptionCategory opDefSpecCat("Options for specification generation.");

static cl::opt<std::string> opExamplesDirectory(
    "examples-directory",
    llvm::cl::desc("The path of the directory to write out the examples."),
    llvm::cl::cat(opDefSpecCat));

static mlir::GenRegistration genSpecificationRegister(
    "gen-op-spec", "Generate the CUDA Tile IR specification from the dialect.",
    [](const RecordKeeper &records, raw_ostream &os) {
      std::optional<std::string> examplesDirectory;
      if (!opExamplesDirectory.empty()) {
        examplesDirectory = opExamplesDirectory;
      }
      cudatile::tblgen::generateSpec(os, records, examplesDirectory);
      return false;
    });

return MlirTblgenMain(argc, argv); 

}
