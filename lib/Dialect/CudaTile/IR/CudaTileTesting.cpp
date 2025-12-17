//===- CudaTileTesting.cpp - CUDA Tile Testing Op Parsing -------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Interfaces.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedFuncParserAndPrinter.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// Test_FuncOp
//===----------------------------------------------------------------------===//

ParseResult Test_FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFuncOp<Test_FuncOp>(parser, result);
}

void Test_FuncOp::print(OpAsmPrinter &printer) { printFuncOp(*this, printer); }
