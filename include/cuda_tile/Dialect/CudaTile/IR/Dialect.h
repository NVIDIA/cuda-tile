//===- Dialect.h - CUDA Tile Dialect Utilities ------------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_DIALECT_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h.inc"
#include <functional>
#include <optional>
#include <variant>

namespace mlir::cuda_tile {

/// Compute the maximum signed value for an integer with the given bitwidth.
int64_t getMaxSignedValueForBitwidth(int64_t n);

/// Compute the minimum signed value for an integer with the given bitwidth.
int64_t getMinSignedValueForBitwidth(int64_t n);

/// Compute the maximum unsigned value for an integer with the given bitwidth.
uint64_t getMaxUnsignedValueForBitwidth(int64_t n);

/// Main function signature parser with cuda_tile dialect support.
/// This function extends MLIR's standard function signature parsing
/// to support cuda_tile dialect-specific argument and result attributes.
mlir::ParseResult parseFunctionSignatureWithArguments(
    mlir::OpAsmParser &parser, bool allowVariadic,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &arguments,
    bool &isVariadic, llvm::SmallVectorImpl<mlir::Type> &resultTypes,
    llvm::SmallVectorImpl<mlir::DictionaryAttr> &resultAttrs);

/// Print function signature with cuda_tile dialect type support.
/// This function prints function signatures while omitting the !cuda_tile.
/// prefix from tile types and using custom type printing for CudaTile types.
void printFunctionSignatureWithCudaTileTypes(mlir::OpAsmPrinter &printer,
                                             mlir::Operation *op,
                                             mlir::TypeRange inputs,
                                             mlir::TypeRange results);

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_DIALECT_H
