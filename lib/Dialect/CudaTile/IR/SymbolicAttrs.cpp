//===- SymbolicAttrs.cpp - CUDA Tile Symbolic Attributes --------*- C++ -*-===//
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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CheckedArithmetic.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/CompactLayoutParser.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {

static void printTupleCompact(llvm::raw_ostream &os, SymbolicTupleAttr tuple,
                              bool isNested = false);

static constexpr StringLiteral kBuiltinEvalContextPrefix = "builtin.";

/// Check if an expression can be printed in compact unquoted form.
static bool canPrintCompact(Attribute attr) {
  if (llvm::isa<SymbolicExprIntAttr, SymbolicExprVarAttr,
                SymbolicExprBuiltinAttr>(attr)) {
    return true;
  }
  if (auto tuple = llvm::dyn_cast<SymbolicTupleAttr>(attr)) {
    return llvm::all_of(tuple.getValues(), canPrintCompact);
  }
  if (auto v = llvm::dyn_cast<SymbolicExprMulAttr>(attr)) {
    return canPrintCompact(v.getLhs()) && canPrintCompact(v.getRhs());
  }
  if (auto v = llvm::dyn_cast<SymbolicExprFloorDivAttr>(attr)) {
    return canPrintCompact(v.getLhs()) && canPrintCompact(v.getRhs());
  }
  if (auto v = llvm::dyn_cast<SymbolicExprCeilDivAttr>(attr)) {
    return canPrintCompact(v.getLhs()) && canPrintCompact(v.getRhs());
  }
  if (auto v = llvm::dyn_cast<SymbolicExprAddAttr>(attr)) {
    return canPrintCompact(v.getLhs()) && canPrintCompact(v.getRhs());
  }
  if (auto v = llvm::dyn_cast<SymbolicExprNegAttr>(attr)) {
    // A negated literal has no compact spelling of its own: the parser folds a
    // leading '-' into the literal, so `-4` reads back as the literal -4 and
    // not as this node. Fall back to the verbose form to keep the round trip
    // exact.
    if (llvm::isa<SymbolicExprIntAttr>(v.getOperand())) {
      return false;
    }
    return canPrintCompact(v.getOperand());
  }
  if (auto v = llvm::dyn_cast<SymbolicExprSubAttr>(attr)) {
    return canPrintCompact(v.getLhs()) && canPrintCompact(v.getRhs());
  }
  return false;
}

static CompactExprOperator getCompactExprOperator(Attribute attr) {

  if (llvm::isa<SymbolicExprAddAttr>(attr))
    return CompactExprOperator::Add;
  if (llvm::isa<SymbolicExprNegAttr>(attr))
    return CompactExprOperator::Neg;
  if (llvm::isa<SymbolicExprSubAttr>(attr))
    return CompactExprOperator::Sub;
  if (llvm::isa<SymbolicExprMulAttr>(attr))
    return CompactExprOperator::Mul;
  if (llvm::isa<SymbolicExprFloorDivAttr>(attr))
    return CompactExprOperator::FloorDiv;
  if (llvm::isa<SymbolicExprCeilDivAttr>(attr))
    return CompactExprOperator::CeilDiv;
  return CompactExprOperator::Primary;
}

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
        } else if (auto v = llvm::dyn_cast<SymbolicExprBuiltinAttr>(attr)) {
          os << v.getName();
        } else if (auto v = llvm::dyn_cast<SymbolicTupleAttr>(attr)) {
          printTupleCompact(os, v, /*isNested=*/true);
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

static void printTupleCompact(llvm::raw_ostream &os, SymbolicTupleAttr tuple,
                              bool isNested) {
  os << "(";
  llvm::interleaveComma(tuple.getValues(), os,
                        [&](Attribute v) { printExprCompact(os, v); });
  // A nested one-element tuple needs a trailing comma to survive re-parsing:
  // `(16)` reads back as a grouped scalar, `(16,)` as the tuple. This mirrors
  // printCompactTupleConstArg in PreviewOps.cpp. The outermost shapes/strides
  // tuples never need it -- their parens *are* the tuple, and
  // parseParenTupleBody does not accept a trailing comma there.
  if (isNested && tuple.getValues().size() == 1) {
    os << ",";
  }
  os << ")";
}

enum class SignedDivRounding { TowardZero, Floor, Ceil };

/// Negate an integer literal, or return a null attribute when `operand` is not
/// a literal or its negation is not representable. As with llvm::checkedAdd and
/// friends in the binary folds, an unrepresentable result leaves the expression
/// node in place, so the caller that needs a concrete value is the one to
/// report the failure.
static SymbolicExprAttr foldNeg(SymbolicExprAttr operand) {
  auto intAttr = llvm::dyn_cast<SymbolicExprIntAttr>(operand);
  if (!intAttr) {
    return {};
  }
  if (intAttr.getValue() == std::numeric_limits<int64_t>::min()) {
    return {};
  }
  return SymbolicExprIntAttr::get(operand.getContext(), -intAttr.getValue());
}

static std::optional<int64_t> foldSignedDiv(int64_t lhs, int64_t rhs,
                                            SignedDivRounding rounding) {
  if (rhs == 0) {
    return std::nullopt;
  }
  // The only signed division that overflows.
  if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
    return std::nullopt;
  }

  int64_t quotient = lhs / rhs;
  if (rounding == SignedDivRounding::TowardZero) {
    return quotient;
  }

  int64_t remainder = lhs % rhs;
  if (remainder == 0) {
    return quotient;
  }

  // A non-zero remainder implies |rhs| >= 2, so the adjustment below cannot
  // push the quotient out of range.
  bool signsDiffer = (lhs < 0) != (rhs < 0);
  if (rounding == SignedDivRounding::Floor) {
    return signsDiffer ? quotient - 1 : quotient;
  }
  return signsDiffer ? quotient : quotient + 1;
}

/// Rebuild `original` as `lhs <op> rhs`, or return `original` unchanged when
/// the operands did not change or when `rhs` is the literal zero.
///
/// The division attributes reject a literal zero divisor in their verifiers, so
/// `DivAttr::get` would assert on one. Substitution and simplification can
/// nevertheless produce a zero divisor from an expression that was legal as
/// written, as in `1 / N` specialized with `N = 0`. Leaving `original` in place
/// keeps the expression symbolic, so the caller that needed a concrete value
/// reports the failure rather than the compiler aborting here.
template <typename DivAttr>
static SymbolicExprAttr rebuildDivision(DivAttr original, SymbolicExprAttr lhs,
                                        SymbolicExprAttr rhs) {
  if (lhs == original.getLhs() && rhs == original.getRhs()) {
    return original;
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhs)) {
    if (rhsInt.getValue() == 0) {
      return original;
    }
  }
  return DivAttr::get(original.getContext(), lhs, rhs);
}

} // namespace

//===----------------------------------------------------------------------===//
// SymbolicExprIntAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprIntAttr::instantiate(DictionaryAttr evalContext) const {
  return *this;
}

SymbolicExprAttr SymbolicExprIntAttr::simplify() const { return *this; }

//===----------------------------------------------------------------------===//
// SymbolicExprConstStrAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprConstStrAttr::instantiate(DictionaryAttr evalContext) const {
  return *this;
}

SymbolicExprAttr SymbolicExprConstStrAttr::simplify() const { return *this; }

//===----------------------------------------------------------------------===//
// SymbolicExprVarAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprVarAttr::instantiate(DictionaryAttr evalContext) const {
  if (auto value = evalContext.get(getName())) {
    return cast<SymbolicExprAttr>(value);
  }
  return *this;
}

SymbolicExprAttr SymbolicExprVarAttr::simplify() const { return *this; }

LogicalResult
SymbolicExprVarAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            StringRef name) {
  if (name.empty()) {
    return emitError() << "expression variable name cannot be empty";
  }
  if (name.contains('.')) {
    return emitError() << "expression variable name cannot contain '.'; "
                          "dotted names are reserved for builtins";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprBuiltinAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprBuiltinAttr::instantiate(DictionaryAttr evalContext) const {
  auto key = (kBuiltinEvalContextPrefix + getName()).str();
  if (auto value = evalContext.get(key)) {
    return cast<SymbolicExprAttr>(value);
  }
  return *this;
}

SymbolicExprAttr SymbolicExprBuiltinAttr::simplify() const { return *this; }

LogicalResult
SymbolicExprBuiltinAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                StringRef name, TypeAttr type) {
  if (!name.starts_with("context.") || name == "context.") {
    return emitError() << "builtin expression name must be in the 'context.' "
                          "namespace";
  }
  if (!type.getValue()) {
    return emitError() << "builtin expression result type cannot be null";
  }
  if (!llvm::isa<IndexType>(type.getValue())) {
    return emitError() << "builtin expression result type must be index, got "
                       << type.getValue();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConcreteTypeAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
ConcreteTypeAttr::instantiate(DictionaryAttr evalContext) const {
  return *this;
}

SymbolicExprAttr ConcreteTypeAttr::simplify() const { return *this; }

//===----------------------------------------------------------------------===//
// SymbolicExprMulAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprMulAttr::instantiate(DictionaryAttr evalContext) const {
  auto lhsInst = getLhs().instantiate(evalContext);
  auto rhsInst = getRhs().instantiate(evalContext);

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsInst)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsInst)) {
      if (auto result =
              llvm::checkedMul(lhsInt.getValue(), rhsInt.getValue())) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }

  if (lhsInst == getLhs() && rhsInst == getRhs()) {
    return *this;
  }
  return SymbolicExprMulAttr::get(getContext(), lhsInst, rhsInst);
}

SymbolicExprAttr SymbolicExprMulAttr::simplify() const {
  auto lhsSimp = getLhs().simplify();
  auto rhsSimp = getRhs().simplify();

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsSimp)) {
    if (lhsInt.getValue() == 0) {
      return lhsInt;
    }
    if (lhsInt.getValue() == 1) {
      return rhsSimp;
    }
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
      if (auto result =
              llvm::checkedMul(lhsInt.getValue(), rhsInt.getValue())) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
    if (rhsInt.getValue() == 0) {
      return rhsInt;
    }
    if (rhsInt.getValue() == 1) {
      return lhsSimp;
    }
  }

  if (lhsSimp == getLhs() && rhsSimp == getRhs()) {
    return *this;
  }
  return SymbolicExprMulAttr::get(getContext(), lhsSimp, rhsSimp);
}

LogicalResult
SymbolicExprMulAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            SymbolicExprAttr lhs, SymbolicExprAttr rhs) {
  (void)emitError;
  (void)lhs;
  (void)rhs;
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprDivAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprDivAttr::instantiate(DictionaryAttr evalContext) const {
  auto lhsInst = getLhs().instantiate(evalContext);
  auto rhsInst = getRhs().instantiate(evalContext);

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsInst)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsInst)) {
      if (auto result = foldSignedDiv(lhsInt.getValue(), rhsInt.getValue(),
                                      SignedDivRounding::TowardZero)) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }

  return rebuildDivision(*this, lhsInst, rhsInst);
}

SymbolicExprAttr SymbolicExprDivAttr::simplify() const {
  auto lhsSimp = getLhs().simplify();
  auto rhsSimp = getRhs().simplify();

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsSimp)) {
    if (lhsInt.getValue() == 0) {
      if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
        if (rhsInt.getValue() != 0) {
          return lhsInt;
        }
      }
    }
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
      if (auto result = foldSignedDiv(lhsInt.getValue(), rhsInt.getValue(),
                                      SignedDivRounding::TowardZero)) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
    if (rhsInt.getValue() == 1) {
      return lhsSimp;
    }
  }

  return rebuildDivision(*this, lhsSimp, rhsSimp);
}

LogicalResult
SymbolicExprDivAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            SymbolicExprAttr lhs, SymbolicExprAttr rhs) {
  (void)lhs;
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhs)) {
    if (rhsInt.getValue() == 0) {
      return emitError() << "division by zero is not allowed";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprAddAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprAddAttr::instantiate(DictionaryAttr evalContext) const {
  auto lhsInst = getLhs().instantiate(evalContext);
  auto rhsInst = getRhs().instantiate(evalContext);

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsInst)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsInst)) {
      if (auto result =
              llvm::checkedAdd(lhsInt.getValue(), rhsInt.getValue())) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }

  if (lhsInst == getLhs() && rhsInst == getRhs()) {
    return *this;
  }
  return SymbolicExprAddAttr::get(getContext(), lhsInst, rhsInst);
}

SymbolicExprAttr SymbolicExprAddAttr::simplify() const {
  auto lhsSimp = getLhs().simplify();
  auto rhsSimp = getRhs().simplify();

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsSimp)) {
    if (lhsInt.getValue() == 0) {
      return rhsSimp;
    }
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
      if (auto result =
              llvm::checkedAdd(lhsInt.getValue(), rhsInt.getValue())) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
    if (rhsInt.getValue() == 0) {
      return lhsSimp;
    }
  }

  if (lhsSimp == getLhs() && rhsSimp == getRhs()) {
    return *this;
  }
  return SymbolicExprAddAttr::get(getContext(), lhsSimp, rhsSimp);
}

LogicalResult
SymbolicExprAddAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            SymbolicExprAttr lhs, SymbolicExprAttr rhs) {
  (void)emitError;
  (void)lhs;
  (void)rhs;
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprNegAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprNegAttr::instantiate(DictionaryAttr evalContext) const {
  auto operandInst = getOperand().instantiate(evalContext);

  if (SymbolicExprAttr folded = foldNeg(operandInst)) {
    return folded;
  }

  if (operandInst == getOperand()) {
    return *this;
  }
  return SymbolicExprNegAttr::get(getContext(), operandInst);
}

SymbolicExprAttr SymbolicExprNegAttr::simplify() const {
  auto operandSimp = getOperand().simplify();

  if (SymbolicExprAttr folded = foldNeg(operandSimp)) {
    return folded;
  }
  if (auto negAttr = llvm::dyn_cast<SymbolicExprNegAttr>(operandSimp)) {
    return negAttr.getOperand();
  }

  if (operandSimp == getOperand()) {
    return *this;
  }
  return SymbolicExprNegAttr::get(getContext(), operandSimp);
}

LogicalResult
SymbolicExprNegAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            SymbolicExprAttr operand) {
  (void)emitError;
  (void)operand;
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprSubAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprSubAttr::instantiate(DictionaryAttr evalContext) const {
  auto lhsInst = getLhs().instantiate(evalContext);
  auto rhsInst = getRhs().instantiate(evalContext);

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsInst)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsInst)) {
      if (auto result =
              llvm::checkedSub(lhsInt.getValue(), rhsInt.getValue())) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }

  if (lhsInst == getLhs() && rhsInst == getRhs()) {
    return *this;
  }
  return SymbolicExprSubAttr::get(getContext(), lhsInst, rhsInst);
}

SymbolicExprAttr SymbolicExprSubAttr::simplify() const {
  auto lhsSimp = getLhs().simplify();
  auto rhsSimp = getRhs().simplify();

  if (lhsSimp == rhsSimp) {
    return SymbolicExprIntAttr::get(getContext(), 0);
  }

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsSimp)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
      if (auto result =
              llvm::checkedSub(lhsInt.getValue(), rhsInt.getValue())) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
    if (rhsInt.getValue() == 0) {
      return lhsSimp;
    }
  }

  if (lhsSimp == getLhs() && rhsSimp == getRhs()) {
    return *this;
  }
  return SymbolicExprSubAttr::get(getContext(), lhsSimp, rhsSimp);
}

LogicalResult
SymbolicExprSubAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            SymbolicExprAttr lhs, SymbolicExprAttr rhs) {
  (void)emitError;
  (void)lhs;
  (void)rhs;
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprFloorDivAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprFloorDivAttr::instantiate(DictionaryAttr evalContext) const {
  auto lhsInst = getLhs().instantiate(evalContext);
  auto rhsInst = getRhs().instantiate(evalContext);

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsInst)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsInst)) {
      if (auto result = foldSignedDiv(lhsInt.getValue(), rhsInt.getValue(),
                                      SignedDivRounding::Floor)) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }

  return rebuildDivision(*this, lhsInst, rhsInst);
}

SymbolicExprAttr SymbolicExprFloorDivAttr::simplify() const {
  auto lhsSimp = getLhs().simplify();
  auto rhsSimp = getRhs().simplify();

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsSimp)) {
    if (lhsInt.getValue() == 0) {
      if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
        if (rhsInt.getValue() != 0) {
          return lhsInt;
        }
      }
    }
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
      if (auto result = foldSignedDiv(lhsInt.getValue(), rhsInt.getValue(),
                                      SignedDivRounding::Floor)) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
    if (rhsInt.getValue() == 1) {
      return lhsSimp;
    }
  }

  return rebuildDivision(*this, lhsSimp, rhsSimp);
}

LogicalResult
SymbolicExprFloorDivAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 SymbolicExprAttr lhs, SymbolicExprAttr rhs) {
  (void)lhs;
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhs)) {
    if (rhsInt.getValue() == 0) {
      return emitError() << "floor division by zero is not allowed";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicExprCeilDivAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicExprCeilDivAttr::instantiate(DictionaryAttr evalContext) const {
  auto lhsInst = getLhs().instantiate(evalContext);
  auto rhsInst = getRhs().instantiate(evalContext);

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsInst)) {
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsInst)) {
      if (auto result = foldSignedDiv(lhsInt.getValue(), rhsInt.getValue(),
                                      SignedDivRounding::Ceil)) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }

  return rebuildDivision(*this, lhsInst, rhsInst);
}

SymbolicExprAttr SymbolicExprCeilDivAttr::simplify() const {
  auto lhsSimp = getLhs().simplify();
  auto rhsSimp = getRhs().simplify();

  if (auto lhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(lhsSimp)) {
    if (lhsInt.getValue() == 0) {
      if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
        if (rhsInt.getValue() != 0) {
          return lhsInt;
        }
      }
    }
    if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
      if (auto result = foldSignedDiv(lhsInt.getValue(), rhsInt.getValue(),
                                      SignedDivRounding::Ceil)) {
        return SymbolicExprIntAttr::get(getContext(), *result);
      }
    }
  }
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhsSimp)) {
    if (rhsInt.getValue() == 1) {
      return lhsSimp;
    }
  }

  return rebuildDivision(*this, lhsSimp, rhsSimp);
}

LogicalResult
SymbolicExprCeilDivAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                SymbolicExprAttr lhs, SymbolicExprAttr rhs) {
  (void)lhs;
  if (auto rhsInt = llvm::dyn_cast<SymbolicExprIntAttr>(rhs)) {
    if (rhsInt.getValue() == 0) {
      return emitError() << "ceil division by zero is not allowed";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolicTupleAttr
//===----------------------------------------------------------------------===//

SymbolicExprAttr
SymbolicTupleAttr::instantiate(DictionaryAttr evalContext) const {
  SmallVector<SymbolicExprAttr> instantiatedValues;
  instantiatedValues.reserve(getValues().size());
  bool changed = false;
  for (SymbolicExprAttr value : getValues()) {
    auto instantiated = value.instantiate(evalContext);
    changed |= instantiated != value;
    instantiatedValues.push_back(instantiated);
  }
  if (!changed) {
    return *this;
  }
  return SymbolicTupleAttr::get(getContext(), instantiatedValues);
}

SymbolicExprAttr SymbolicTupleAttr::simplify() const {
  SmallVector<SymbolicExprAttr> simplifiedValues;
  simplifiedValues.reserve(getValues().size());
  bool changed = false;
  for (SymbolicExprAttr value : getValues()) {
    auto simplified = value.simplify();
    changed |= simplified != value;
    simplifiedValues.push_back(simplified);
  }
  if (!changed) {
    return *this;
  }
  return SymbolicTupleAttr::get(getContext(), simplifiedValues);
}

//===----------------------------------------------------------------------===//
// LayoutAttr
//===----------------------------------------------------------------------===//

Attribute LayoutAttr::parse(AsmParser &parser, Type type) {
  MLIRContext *ctx = parser.getContext();

  if (parser.parseLess())
    return {};

  auto startLoc = parser.getCurrentLocation();
  if (failed(parser.parseOptionalKeyword("shapes"))) {
    bool emittedDiagnostic = false;
    CudaTileCompactLayoutParser compactParser(parser, ctx, startLoc,
                                              emittedDiagnostic);
    FailureOr<LayoutAttr> result =
        compactParser.parse(CompactLayoutTerminator::Greater);
    if (failed(result) && !emittedDiagnostic) {
      parser.emitError(startLoc, "failed to parse compact layout syntax");
      return {};
    }
    return succeeded(result) ? *result : LayoutAttr();
  }

  SymbolicTupleAttr shapes, strides;
  if (parser.parseEqual() || parser.parseAttribute(shapes))
    return {};
  if (parser.parseComma() || parser.parseKeyword("strides") ||
      parser.parseEqual() || parser.parseAttribute(strides))
    return {};
  if (parser.parseGreater())
    return {};

  return LayoutAttr::getChecked([&]() { return parser.emitError(startLoc); },
                                ctx, shapes, strides);
}

void LayoutAttr::print(AsmPrinter &printer) const {
  if (canPrintCompact(getShapes()) && canPrintCompact(getStrides())) {
    auto &os = printer.getStream();
    os << "<";
    printTupleCompact(os, getShapes());
    os << ":";
    printTupleCompact(os, getStrides());
    os << ">";
    return;
  }

  printer << "<shapes = " << getShapes() << ", strides = " << getStrides()
          << ">";
}

/// Walk the shape and stride trees of one mode in lockstep, reporting the first
/// position at which their structure disagrees. `mode` spells the position in
/// CuTe mode notation, e.g. `1` or `1.0`.
static LogicalResult
verifyModeStructure(function_ref<InFlightDiagnostic()> emitError,
                    SymbolicExprAttr shape, SymbolicExprAttr stride,
                    StringRef mode) {
  auto shapeTuple = llvm::dyn_cast<SymbolicTupleAttr>(shape);
  auto strideTuple = llvm::dyn_cast<SymbolicTupleAttr>(stride);
  if (!shapeTuple && !strideTuple) {
    return success();
  }
  if (!shapeTuple || !strideTuple) {
    // Substituting a shape parameter can turn one mode into a tuple of modes,
    // so a parametric mode may legitimately face a tuple until the parameters
    // are concretized. An integer never expands, so pairing one with a tuple
    // is a structure the layout algebra can never walk.
    SymbolicExprAttr scalar = shapeTuple ? stride : shape;
    if (!llvm::isa<SymbolicExprIntAttr>(scalar.simplify())) {
      return success();
    }
    return emitError() << "shapes and strides must have the same nesting "
                          "structure, but mode "
                       << mode << " is a tuple in "
                       << (shapeTuple ? "shapes" : "strides")
                       << " and an integer in "
                       << (shapeTuple ? "strides" : "shapes");
  }
  ArrayRef<SymbolicExprAttr> shapeModes = shapeTuple.getValues();
  ArrayRef<SymbolicExprAttr> strideModes = strideTuple.getValues();
  if (shapeModes.size() != strideModes.size()) {
    return emitError() << "shapes and strides must have the same nesting "
                          "structure, but mode "
                       << mode << " has rank " << shapeModes.size()
                       << " in shapes and rank " << strideModes.size()
                       << " in strides";
  }
  for (size_t index = 0, e = shapeModes.size(); index < e; ++index) {
    if (failed(verifyModeStructure(emitError, shapeModes[index],
                                   strideModes[index],
                                   (Twine(mode) + "." + Twine(index)).str()))) {
      return failure();
    }
  }
  return success();
}

/// Walk one side of the layout, reporting the first integer leaf outside its
/// permitted range. Shape extents count elements and must be positive; strides
/// step forward through storage and must be non-negative. A parametric leaf is
/// left to concretization. `mode` spells the position in CuTe mode notation,
/// e.g. `1` or `1.0`.
static LogicalResult
verifyLeafRange(function_ref<InFlightDiagnostic()> emitError,
                SymbolicExprAttr side, StringRef mode, bool isShape) {
  if (auto tuple = llvm::dyn_cast<SymbolicTupleAttr>(side)) {
    ArrayRef<SymbolicExprAttr> modes = tuple.getValues();
    for (size_t index = 0, e = modes.size(); index < e; ++index) {
      if (failed(verifyLeafRange(emitError, modes[index],
                                 (Twine(mode) + "." + Twine(index)).str(),
                                 isShape))) {
        return failure();
      }
    }
    return success();
  }
  auto leafInt = llvm::dyn_cast<SymbolicExprIntAttr>(side.simplify());
  if (!leafInt) {
    return success();
  }
  if (isShape ? leafInt.getValue() > 0 : leafInt.getValue() >= 0) {
    return success();
  }
  return emitError() << (isShape ? "shape extents must be positive"
                                 : "strides must be non-negative")
                     << ", but mode " << mode << " is " << leafInt.getValue();
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 SymbolicTupleAttr shapes,
                                 SymbolicTupleAttr strides) {
  if (shapes.getValues().size() != strides.getValues().size()) {
    return emitError() << "shapes and strides must have the same rank, got "
                       << shapes.getValues().size() << " vs "
                       << strides.getValues().size();
  }
  // The layout algebra walks the two trees in lockstep, so the nesting has to
  // agree below the top level as well, up to the modes a shape parameter has
  // yet to expand into. Concrete leaves are additionally range-checked.
  for (size_t index = 0, e = shapes.getValues().size(); index < e; ++index) {
    if (failed(verifyModeStructure(emitError, shapes.getValues()[index],
                                   strides.getValues()[index],
                                   Twine(index).str()))) {
      return failure();
    }
    if (failed(verifyLeafRange(emitError, shapes.getValues()[index],
                               Twine(index).str(), /*isShape=*/true))) {
      return failure();
    }
    if (failed(verifyLeafRange(emitError, strides.getValues()[index],
                               Twine(index).str(), /*isShape=*/false))) {
      return failure();
    }
  }
  return success();
}
