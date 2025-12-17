//===- Types.h - CUDA Tile Type Utilities -----------------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_TYPES_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_TYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Interfaces.h"

#define GET_TYPEDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/Types.h.inc"

namespace mlir {
namespace cuda_tile {

// The rationale for this trait is to prevent users from creating programs
// that would have catastrophic register pressure and cause the compiler to
// hang.
// Since H100 has 256KB registers, we should allow users to create tiles
// of size up to 256K elements.
//
// We can relax the constraint a little bit since we will apply the slice
// optimization whenever we can in the latest implementation. We still need
// the constraint because a very large tile size may lead to very long
// compilation time even with the slicing (also very likely to have bad
// performance sine it doesn't fit to the hardware).
// A very rough estimation for the limit may be something like:
// factor(4) x max-num-of-ctas-per-cga(16) x maxOnChipRegisterPerCta(256k)
// factor > 1  means the tile size can be larger than the hardware capacity
// but not too much larger.
int64_t constexpr maxTileNumElements = 16777216;

// Generate C++ functions for certain type constraints.
#include "cuda_tile/Dialect/CudaTile/IR/TypeConstraints.h.inc"

/// Return "true" if the given type is an pointer or a tensor of pointer.
bool isPointerLike(Type t);

/// Return a type with same shape as the argument, with i1 element type.
Type getI1SameShape(Type type);

/// Parse a type, if type is unprefixed, assume it is from the cuda_tile dialect
ParseResult parseCudaTileType(AsmParser &p, Type &type);
ParseResult parseCudaTileType(AsmParser &p, SmallVectorImpl<Type> &types);

/// Parses a single cuda tile type and splats 'types' to contain as many
/// instances of that type as 'values'.
ParseResult parseCudaTileTypeSplat(AsmParser &p, SmallVectorImpl<Type> &types,
                                   ArrayRef<OpAsmParser::UnresolvedOperand> values);

/// Print a type, stripping prefix if belonging to cuda_tile dialect
void printCudaTileType(AsmPrinter &p, Type type);
void printCudaTileType(AsmPrinter &p, Operation *op, Type type);
void printCudaTileType(AsmPrinter &p, TypeRange types);
void printCudaTileType(AsmPrinter &p, Operation *op, TypeRange types);

/// Print a splatted cuda tile type. Asserts that all of types are equal and
/// prints only one instance of that type using 'printCudaTileType'.
/// This allows using the function in a custom assembly format using:
///   custom<CudaTileTypeSplat>(type($values), $values)
void printCudaTileTypeSplat(AsmPrinter &p, Operation *op, TypeRange types,
                            ValueRange values);

/// This class represents any cuda tile type.
struct CudaTileType : public Type {
  using Type::Type;

  /// Classof support for casting functionality.
  static bool classof(Type type);
};

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_TYPES_H
