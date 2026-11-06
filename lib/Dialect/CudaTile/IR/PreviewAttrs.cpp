//===- PreviewAttrs.cpp - CUDA Tile Public-Preview Attributes ---*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/CompactLayoutParser.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// ScratchRequestAttr
//===----------------------------------------------------------------------===//

LogicalResult
ScratchRequestAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           SymbolicExprAttr size, SymbolicExprAttr alignment) {
  if (!size) {
    return emitError() << "scratch size must be an integer symbolic expression";
  }
  if (failed(verifySymbolicExprKind(size, SymbolicExprKind::Int, emitError))) {
    return failure();
  }
  if (!alignment) {
    return emitError()
           << "scratch alignment must be an integer symbolic expression";
  }
  if (failed(verifySymbolicExprKind(alignment, SymbolicExprKind::Int,
                                    emitError))) {
    return failure();
  }
  // The range checks run on the simplified expressions so that a request stays
  // rejected however it was spelled: `-(8)` and `4 - 12` mean the same thing as
  // `-8`, and the compact printer spells all three the same way once folded.
  if (auto sizeInt = llvm::dyn_cast<SymbolicExprIntAttr>(size.simplify());
      sizeInt && sizeInt.getValue() < 0) {
    return emitError() << "scratch size must be non-negative";
  }
  if (auto alignmentInt =
          llvm::dyn_cast<SymbolicExprIntAttr>(alignment.simplify())) {
    int64_t value = alignmentInt.getValue();
    if (value <= 0) {
      return emitError() << "scratch alignment must be positive";
    }
    if (!llvm::isPowerOf2_64(value)) {
      return emitError() << "scratch alignment must be a power of two";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConstParamAttr
//===----------------------------------------------------------------------===//

LogicalResult
ConstParamAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       StringRef name, TypeAttr type) {
  if (name.empty()) {
    return emitError() << "constant parameter name cannot be empty";
  }
  if (!type) {
    return emitError() << "constant parameter type cannot be null";
  }
  Type parameterType = type.getValue();
  if (name == "?" && llvm::isa<NoneType>(parameterType)) {
    return success();
  }
  if (!llvm::isa<IndexType, ShapeParamType, DTypeParamType, StringParamType>(
          parameterType)) {
    return emitError()
           << "constant parameter type must be index, "
              "!cuda_tile.preview$shape, !cuda_tile.preview$Datatype, or "
              "!cuda_tile.preview$string, got "
           << parameterType;
  }
  return success();
}

/// Print symbolic expressions in diagnostics using the compact spelling.
///
/// Concrete datatype expressions are rendered as the underlying MLIR type
/// (`f32`) rather than the storage attribute
/// (`#cuda_tile.preview$concrete_type<f32>`), matching the textual syntax users
/// write for constant arguments.
static void printSymbolicExprForDiagnostic(SymbolicExprAttr expr,
                                           InFlightDiagnostic &diag) {
  if (auto concreteType = llvm::dyn_cast<ConcreteTypeAttr>(expr)) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    os << concreteType.getType().getValue();
    diag << os.str();
  } else if (auto string = llvm::dyn_cast<SymbolicExprConstStrAttr>(expr)) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    os << '"';
    llvm::printEscapedString(string.getValue(), os);
    os << '"';
    diag << os.str();
  } else {
    diag << expr;
  }
}

/// Recursively verify that a symbolic expression has the requested kind.
///
/// Integer expressions may contain arithmetic over integer-valued
/// subexpressions. Tuple expressions model shape parameters, and their elements
/// must be integer-valued dimensions unless the expression is a single shape
/// variable. Datatype expressions are either concrete types or datatype
/// variables. String expressions are concrete string constants or symbolic
/// variables forwarding an in-scope string parameter.
LogicalResult mlir::cuda_tile::verifySymbolicExprKind(
    SymbolicExprAttr expr, SymbolicExprKind expectedKind,
    function_ref<InFlightDiagnostic()> emitError) {
  switch (expectedKind) {
  case SymbolicExprKind::Int: {
    if (llvm::isa<SymbolicExprIntAttr, SymbolicExprVarAttr,
                  SymbolicExprBuiltinAttr>(expr)) {
      // Variables forward in-scope constant parameters; their declared kind is
      // checked when the binding is matched to the declaration parameter.
      // Builtins (`context.*`) are integer-valued and resolved from the
      // execution context during concretization.
      return success();
    }
    if (llvm::isa<ConcreteTypeAttr>(expr)) {
      InFlightDiagnostic diag = emitError();
      diag << "expected integer expression, got ";
      printSymbolicExprForDiagnostic(expr, diag);
      return failure();
    }
    if (llvm::isa<SymbolicTupleAttr>(expr)) {
      return emitError() << "expected integer expression, got tuple";
    }
    if (auto mul = llvm::dyn_cast<SymbolicExprMulAttr>(expr)) {
      if (failed(verifySymbolicExprKind(mul.getLhs(), SymbolicExprKind::Int,
                                        emitError))) {
        return failure();
      }
      return verifySymbolicExprKind(mul.getRhs(), SymbolicExprKind::Int,
                                    emitError);
    }
    if (auto add = llvm::dyn_cast<SymbolicExprAddAttr>(expr)) {
      if (failed(verifySymbolicExprKind(add.getLhs(), SymbolicExprKind::Int,
                                        emitError))) {
        return failure();
      }
      return verifySymbolicExprKind(add.getRhs(), SymbolicExprKind::Int,
                                    emitError);
    }
    if (auto neg = llvm::dyn_cast<SymbolicExprNegAttr>(expr)) {
      return verifySymbolicExprKind(neg.getOperand(), SymbolicExprKind::Int,
                                    emitError);
    }
    if (auto sub = llvm::dyn_cast<SymbolicExprSubAttr>(expr)) {
      if (failed(verifySymbolicExprKind(sub.getLhs(), SymbolicExprKind::Int,
                                        emitError))) {
        return failure();
      }
      return verifySymbolicExprKind(sub.getRhs(), SymbolicExprKind::Int,
                                    emitError);
    }
    if (auto div = llvm::dyn_cast<SymbolicExprFloorDivAttr>(expr)) {
      if (failed(verifySymbolicExprKind(div.getLhs(), SymbolicExprKind::Int,
                                        emitError))) {
        return failure();
      }
      return verifySymbolicExprKind(div.getRhs(), SymbolicExprKind::Int,
                                    emitError);
    }
    if (auto div = llvm::dyn_cast<SymbolicExprCeilDivAttr>(expr)) {
      if (failed(verifySymbolicExprKind(div.getLhs(), SymbolicExprKind::Int,
                                        emitError))) {
        return failure();
      }
      return verifySymbolicExprKind(div.getRhs(), SymbolicExprKind::Int,
                                    emitError);
    }
    if (llvm::isa<SymbolicExprDivAttr>(expr)) {
      return emitError() << "unsupported expression in integer context";
    }
    return emitError() << "unknown expression in integer context";
  }
  case SymbolicExprKind::Tuple: {
    if (llvm::isa<SymbolicExprVarAttr>(expr)) {
      // A shape parameter variable represents the whole rank-polymorphic shape,
      // not a single dimension inside a tuple.
      return success();
    }
    auto tuple = llvm::dyn_cast<SymbolicTupleAttr>(expr);
    if (!tuple) {
      InFlightDiagnostic diag = emitError();
      diag << "expected tuple expression, got ";
      printSymbolicExprForDiagnostic(expr, diag);
      return failure();
    }
    for (SymbolicExprAttr elem : tuple.getValues()) {
      SymbolicExprKind elemKind = llvm::isa<SymbolicTupleAttr>(elem)
                                      ? SymbolicExprKind::Tuple
                                      : SymbolicExprKind::Int;
      if (failed(verifySymbolicExprKind(elem, elemKind, emitError))) {
        return failure();
      }
    }
    return success();
  }
  case SymbolicExprKind::Datatype: {
    if (llvm::isa<ConcreteTypeAttr, SymbolicExprVarAttr>(expr)) {
      // Datatype variables forward an in-scope datatype parameter; concrete
      // datatypes are represented as ConcreteTypeAttr.
      return success();
    }
    InFlightDiagnostic diag = emitError();
    diag << "expected datatype expression, got ";
    printSymbolicExprForDiagnostic(expr, diag);
    return failure();
  }
  case SymbolicExprKind::String: {
    if (llvm::isa<SymbolicExprConstStrAttr, SymbolicExprVarAttr>(expr)) {
      return success();
    }
    InFlightDiagnostic stringDiag = emitError();
    stringDiag << "expected string expression, got ";
    printSymbolicExprForDiagnostic(expr, stringDiag);
    return failure();
  }
  }
  llvm::report_fatal_error("unhandled SymbolicExprKind");
}

//===----------------------------------------------------------------------===//
// Compact expression/tuple/layout printers
//===----------------------------------------------------------------------===//

static void printTupleCompact(llvm::raw_ostream &os, SymbolicTupleAttr tuple);

static CompactExprOperator getCompactExprOperator(Attribute attr) {

  if (llvm::isa<SymbolicExprAddAttr>(attr)) {
    return CompactExprOperator::Add;
  }
  if (llvm::isa<SymbolicExprNegAttr>(attr)) {
    return CompactExprOperator::Neg;
  }
  if (llvm::isa<SymbolicExprSubAttr>(attr)) {
    return CompactExprOperator::Sub;
  }
  if (llvm::isa<SymbolicExprMulAttr>(attr)) {
    return CompactExprOperator::Mul;
  }
  if (llvm::isa<SymbolicExprFloorDivAttr>(attr)) {
    return CompactExprOperator::FloorDiv;
  }
  if (llvm::isa<SymbolicExprCeilDivAttr>(attr)) {
    return CompactExprOperator::CeilDiv;
  }
  return CompactExprOperator::Primary;
}

/// Print an expression attribute in compact form.
static void printExprCompact(
    llvm::raw_ostream &os, Attribute attr,
    CompactExprPrecedence parentPrec = CompactExprPrecedence::Additive,
    bool isRight = false) {
  printCompactExprWithPrecedence(
      os, getCompactExprOperator(attr), parentPrec, isRight,
      [&](CompactExprPrecedence myPrec) {
        if (auto v = llvm::dyn_cast<SymbolicExprIntAttr>(attr)) {
          os << v.getValue();
        } else if (auto v = llvm::dyn_cast<SymbolicExprVarAttr>(attr)) {
          os << v.getName();
        } else if (auto v = llvm::dyn_cast<SymbolicTupleAttr>(attr)) {
          printTupleCompact(os, v);
        } else if (auto v = llvm::dyn_cast<SymbolicExprMulAttr>(attr)) {
          printExprCompact(os, v.getLhs(), myPrec, /*isRight=*/false);
          os << " * ";
          printExprCompact(os, v.getRhs(), myPrec, /*isRight=*/true);
        } else if (auto v = llvm::dyn_cast<SymbolicExprFloorDivAttr>(attr)) {
          printExprCompact(os, v.getLhs(), myPrec, /*isRight=*/false);
          os << " / ";
          printExprCompact(os, v.getRhs(), myPrec, /*isRight=*/true);
        } else if (auto v = llvm::dyn_cast<SymbolicExprCeilDivAttr>(attr)) {
          printExprCompact(os, v.getLhs(), myPrec, /*isRight=*/false);
          os << " /+ ";
          printExprCompact(os, v.getRhs(), myPrec, /*isRight=*/true);
        } else if (auto v = llvm::dyn_cast<SymbolicExprAddAttr>(attr)) {
          printExprCompact(os, v.getLhs(), myPrec, /*isRight=*/false);
          os << " + ";
          printExprCompact(os, v.getRhs(), myPrec, /*isRight=*/true);
        } else if (auto v = llvm::dyn_cast<SymbolicExprNegAttr>(attr)) {
          os << "-";
          printExprCompact(os, v.getOperand(), myPrec);
        } else if (auto v = llvm::dyn_cast<SymbolicExprSubAttr>(attr)) {
          printExprCompact(os, v.getLhs(), myPrec, /*isRight=*/false);
          os << " - ";
          printExprCompact(os, v.getRhs(), myPrec, /*isRight=*/true);
        }
      });
}

/// Print a SymbolicTupleAttr as a compact parenthesized tuple: (a, b, c)
static void printTupleCompact(llvm::raw_ostream &os, SymbolicTupleAttr tuple) {
  os << "(";
  llvm::interleaveComma(tuple.getValues(), os,
                        [&](Attribute v) { printExprCompact(os, v); });
  os << ")";
}
