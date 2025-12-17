//===- Ops.h - CUDA Tile Operation Utilities --------------------*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_OPS_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Interfaces.h"
#include "cuda_tile/Dialect/CudaTile/IR/Traits.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

namespace mlir::cuda_tile::impl {
struct IfOpImplicitTerminatorType;
struct LoopOpImplicitTerminatorType;

/// Verify the given memory model components.
LogicalResult verifyMemoryModelLoad(Operation *op,
                                    MemoryOrderingSemantics memoryOrdering,
                                    std::optional<MemoryScope> scope);
LogicalResult verifyMemoryModelStore(Operation *op,
                                     MemoryOrderingSemantics memoryOrdering,
                                     std::optional<MemoryScope> scope);

/// Verify the debug information within the given function operation.
LogicalResult verifyFuncDebugInfo(FunctionOpInterface funcOp);
LogicalResult verifyFuncBodyDebugInfo(FunctionOpInterface funcOp);
} // namespace mlir::cuda_tile::impl

//===----------------------------------------------------------------------===//
// Tablegen Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h.inc"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace mlir::cuda_tile {
namespace impl {
//===----------------------------------------------------------------------===//
// ControlFlowImplicitTerminatorOperation
//===----------------------------------------------------------------------===//

/// This class provides an interface compatible with
/// SingleBlockImplicitTerminator, but allows multiple types of potential
/// terminators aside from just one. If a terminator isn't present, this will
/// generate a `ImplicitOpT` operation.
template <typename ImplicitOpT, typename... OtherTerminatorOpTs>
struct ControlFlowImplicitTerminatorOpType {
  /// Implementation of `classof` that supports all of the potential terminator
  /// operations.
  static bool classof(Operation *op) {
    return isa<ImplicitOpT, OtherTerminatorOpTs...>(op);
  }

  //===--------------------------------------------------------------------===//
  // Implicit Terminator Methods

  /// The following methods are all used when interacting with the "implicit"
  /// terminator.

  template <typename... Args>
  static void build(Args &&...args) {
    ImplicitOpT::build(std::forward<Args>(args)...);
  }
  static constexpr StringLiteral getOperationName() {
    return ImplicitOpT::getOperationName();
  }
};
/// An implicit terminator type for `if` operations, which can contain:
/// break, continue, yield.
struct IfOpImplicitTerminatorType
    : public ControlFlowImplicitTerminatorOpType<YieldOp, BreakOp, ContinueOp,
                                                 ReturnOp> {};
struct LoopOpImplicitTerminatorType
    : public ControlFlowImplicitTerminatorOpType<ContinueOp, BreakOp> {};
} // namespace impl
} // namespace mlir::cuda_tile

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_OPS_H
