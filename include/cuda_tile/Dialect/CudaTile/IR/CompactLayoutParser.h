//===- CompactLayoutParser.h - CUDA Tile compact layout parser --*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_COMPACTLAYOUTPARSER_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_COMPACTLAYOUTPARSER_H

#include "mlir/IR/OpImplementation.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include <cstdint>
#include <utility>

namespace mlir::cuda_tile {

// Terminator mode for the compact layout parser.
// All variants except None consume the corresponding token. None leaves the
// terminator unconsumed so the caller (e.g., parseCommaSeparatedList) can
// handle its own separators.
enum class CompactLayoutTerminator { Comma, Greater, RBrace, None };

/// Compact expression operators shared by the parser and printers.
enum class CompactExprOperator {
  /// Binary addition, printed and parsed as `lhs + rhs`.
  Add,
  /// Binary subtraction, printed and parsed as `lhs - rhs`.
  Sub,
  /// Binary multiplication, printed and parsed as `lhs * rhs`.
  Mul,
  /// Integer floor division, printed and parsed as `lhs / rhs`.
  FloorDiv,
  /// Integer ceiling division, printed and parsed as `lhs /+ rhs`.
  CeilDiv,
  /// Unary negation, printed and parsed as `-expr`.
  Neg,
  /// Primary or leaf expression, such as an integer, variable, type, or tuple.
  Primary,
};

// Shared compact expression precedence for both parsing and printing.
// The parser below applies these levels in parseAdditive, parseMultiplicative,
// parseUnary, and parsePrimary; printers use the same table for parentheses.
enum class CompactExprPrecedence : unsigned {
  Additive = 0,
  Multiplicative = 1,
  Unary = 2,
  Primary = 3,
};

inline CompactExprPrecedence getCompactExprPrecedence(CompactExprOperator op) {
  switch (op) {
  case CompactExprOperator::Add:
  case CompactExprOperator::Sub:
    return CompactExprPrecedence::Additive;
  case CompactExprOperator::Mul:
  case CompactExprOperator::FloorDiv:
  case CompactExprOperator::CeilDiv:
    return CompactExprPrecedence::Multiplicative;
  case CompactExprOperator::Neg:
    return CompactExprPrecedence::Unary;
  case CompactExprOperator::Primary:
    return CompactExprPrecedence::Primary;
  }
  llvm_unreachable("unexpected compact expression operator");
}

constexpr bool shouldPrintCompactExprParens(CompactExprPrecedence precedence,
                                            CompactExprPrecedence parent,
                                            bool isRightChild) {
  // Neg is the only unary operator, so a unary child of a unary parent is
  // always a negated negation. It needs parentheses even though the two
  // precedences are equal: `-(-M)` round-trips, while `--M` does not parse.
  // parseUnary() consumes the first '-' and then hands the rest to
  // AsmParser::parseOptionalInteger(), which consumes the second '-' and
  // commits to reading an integer literal, so it fails on the following name
  // instead of recursing into parseUnary() for the inner negation.
  if (precedence == CompactExprPrecedence::Unary &&
      parent == CompactExprPrecedence::Unary) {
    return true;
  }

  unsigned precedenceRank = static_cast<unsigned>(precedence);
  unsigned parentRank = static_cast<unsigned>(parent);
  return precedenceRank < parentRank ||
         (precedenceRank == parentRank && isRightChild);
}

/// Public API helper that prints one compact expression with precedence-aware
/// parentheses.
///
/// \tparam PrinterT Type of printer/stream that supports `operator<<`.
/// \tparam BodyFn Callable used to print the expression body.
/// \param printer Destination printer or stream.
/// \param op Operator represented by the expression being printed.
/// \param parent Precedence of the parent expression.
/// \param isRightChild True when this expression is the right child of its
/// parent, which requires parentheses for equal-precedence non-associative
/// cases.
/// \param body Callable invoked as `body(precedence)`. It must use the passed
/// CompactExprPrecedence to print child expressions with the correct precedence
/// rules.
/// \returns void; output is written to `printer`.
///
/// The helper computes the child precedence with getCompactExprPrecedence(op)
/// and decides whether to emit parentheses with
/// shouldPrintCompactExprParens(precedence, parent, isRightChild).
template <typename PrinterT, typename BodyFn>
void printCompactExprWithPrecedence(PrinterT &printer, CompactExprOperator op,
                                    CompactExprPrecedence parent,
                                    bool isRightChild, BodyFn &&body) {
  CompactExprPrecedence precedence = getCompactExprPrecedence(op);
  bool needParens =
      shouldPrintCompactExprParens(precedence, parent, isRightChild);
  if (needParens) {
    printer << "(";
  }

  std::forward<BodyFn>(body)(precedence);

  if (needParens) {
    printer << ")";
  }
}

/// Parser for the compact `shapes:strides` layout syntax and the symbolic
/// expressions it is built from. Parsed values are CUDA Tile symbolic
/// expression and layout attributes.
class CudaTileCompactLayoutParser {
public:
  CudaTileCompactLayoutParser(AsmParser &parser, MLIRContext *ctx,
                              SMLoc startLoc, bool &emittedDiagnostic);

  FailureOr<LayoutAttr> parse(CompactLayoutTerminator terminator);

  /// Parse a standalone compact expression.
  ///
  /// This is for callers that need a single SymbolicExprAttr outside the full
  /// shape:stride layout grammar handled by parse(). The parser must be
  /// positioned at the first token of a valid compact expression before
  /// calling. The returned FailureOr<SymbolicExprAttr> contains the parsed
  /// expression on success, or failure on syntax/semantic errors reported
  /// through the underlying AsmParser.
  FailureOr<SymbolicExprAttr> parseExpr();

  /// Parse a bracketed compact constant argument list.
  ///
  /// This owns the shared `[arg, ...]` list grammar used by call and impl
  /// constant arguments.
  ParseResult parseConstantArgumentList(ArrayAttr &result);

  /// Print \p expr using compact expression syntax.
  static void printCompactExpr(
      llvm::raw_ostream &os, SymbolicExprAttr expr,
      CompactExprPrecedence parentPrec = CompactExprPrecedence::Additive,
      bool isRight = false);

private:
  /// Parse one compact constant argument: a string, a concrete datatype
  /// constant, or a symbolic expression.
  ///
  /// Compact constant arguments use the following parenthesized forms:
  ///
  ///   scalar-expression ::= ... | '(' scalar-expression ')'
  ///   tuple             ::= '(' ')' | '(' expr ',' ... ')'
  ///
  /// Thus `(M)` groups a scalar expression, while `(M,)` and `(M, N)` create
  /// singleton and multi-element tuples.
  FailureOr<SymbolicExprAttr> parseConstantArgument();

  ParseResult parseTerminator(CompactLayoutTerminator terminator);
  FailureOr<SymbolicExprAttr> parseTuple();
  FailureOr<SymbolicExprAttr> parseParenTupleBody();
  FailureOr<SymbolicExprAttr> parseAdditive();
  FailureOr<SymbolicExprAttr> parseMultiplicative();

  /// Parse an integer literal if the parser is positioned on one, negating the
  /// parsed value when `negate` is set.
  OptionalParseResult parseOptionalIntegerLiteral(bool negate, int64_t &result);

  FailureOr<SymbolicExprAttr> parseUnary();
  FailureOr<SymbolicExprAttr> parsePrimary();

  /// Build an integer literal node. Compact expressions only admit literals
  /// that fit in a signed 64-bit integer; wider ones are rejected by
  /// parseOptionalIntegerLiteral before reaching here.
  FailureOr<SymbolicExprAttr> makeInt(int64_t value);
  FailureOr<SymbolicExprAttr> makeVariable(StringRef name);
  FailureOr<SymbolicExprAttr> makeNeg(SymbolicExprAttr operand);
  FailureOr<SymbolicExprAttr> makeAdd(SymbolicExprAttr lhs,
                                      SymbolicExprAttr rhs);
  FailureOr<SymbolicExprAttr> makeSub(SymbolicExprAttr lhs,
                                      SymbolicExprAttr rhs);
  FailureOr<SymbolicExprAttr> makeMul(SymbolicExprAttr lhs,
                                      SymbolicExprAttr rhs);
  FailureOr<SymbolicExprAttr> makeFloorDiv(SymbolicExprAttr lhs,
                                           SymbolicExprAttr rhs);
  FailureOr<SymbolicExprAttr> makeCeilDiv(SymbolicExprAttr lhs,
                                          SymbolicExprAttr rhs);
  FailureOr<SymbolicExprAttr> makeTuple(ArrayRef<SymbolicExprAttr> values);
  FailureOr<LayoutAttr> makeLayout(SymbolicExprAttr shapes,
                                   SymbolicExprAttr strides);

  MLIRContext *ctx;
  AsmParser &parser;
  SMLoc startLoc;
  bool &emittedDiagnostic;
};

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_COMPACTLAYOUTPARSER_H
