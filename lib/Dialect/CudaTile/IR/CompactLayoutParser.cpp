//===- CompactLayoutParser.cpp ----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CUDA Tile compact layout parser
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/CompactLayoutParser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {

template <typename AttrT, typename... Args>
FailureOr<SymbolicExprAttr>
getCheckedExprAttr(MLIRContext *ctx, AsmParser &parser, SMLoc startLoc,
                   bool &emittedDiagnostic, Args &&...args) {
  auto attr = AttrT::getChecked(
      [&]() {
        emittedDiagnostic = true;
        return parser.emitError(startLoc);
      },
      ctx, std::forward<Args>(args)...);
  if (!attr) {
    return failure();
  }
  return SymbolicExprAttr(attr);
}

} // namespace

CudaTileCompactLayoutParser::CudaTileCompactLayoutParser(
    AsmParser &parser, MLIRContext *ctx, SMLoc startLoc,
    bool &emittedDiagnostic)
    : ctx(ctx), parser(parser), startLoc(startLoc),
      emittedDiagnostic(emittedDiagnostic) {}

FailureOr<LayoutAttr>
CudaTileCompactLayoutParser::parse(CompactLayoutTerminator terminator) {
  auto shapes = parseTuple();
  if (failed(shapes) || parser.parseColon()) {
    return failure();
  }

  auto strides = parseTuple();
  if (failed(strides) || parseTerminator(terminator)) {
    return failure();
  }

  return makeLayout(*shapes, *strides);
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parseExpr() {
  return parseAdditive();
}

ParseResult
CudaTileCompactLayoutParser::parseConstantArgumentList(ArrayAttr &result) {
  if (parser.parseLSquare()) {
    return failure();
  }

  SmallVector<Attribute> args;
  if (succeeded(parser.parseOptionalRSquare())) {
    result = ArrayAttr::get(parser.getContext(), args);
    return success();
  }

  do {
    auto arg = parseConstantArgument();
    if (failed(arg)) {
      return failure();
    }
    args.push_back(*arg);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare()) {
    return failure();
  }
  result = ArrayAttr::get(parser.getContext(), args);
  return success();
}

ParseResult CudaTileCompactLayoutParser::parseTerminator(
    CompactLayoutTerminator terminator) {
  switch (terminator) {
  case CompactLayoutTerminator::Comma:
    return parser.parseComma();
  case CompactLayoutTerminator::Greater:
    return parser.parseGreater();
  case CompactLayoutTerminator::RBrace:
    return parser.parseRBrace();
  case CompactLayoutTerminator::None:
    return success();
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "unknown compact layout terminator");
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parseTuple() {
  if (succeeded(parser.parseOptionalLParen())) {
    return parseParenTupleBody();
  }

  auto expr = parseExpr();
  if (failed(expr)) {
    return failure();
  }

  SmallVector<SymbolicExprAttr, 1> values{*expr};
  return makeTuple(values);
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parseParenTupleBody() {
  SmallVector<SymbolicExprAttr, 4> values;
  if (succeeded(parser.parseOptionalRParen())) {
    return makeTuple(values);
  }

  while (true) {
    // A '(' here is ambiguous: it opens either a nested tuple, `(M, 16)`, or
    // a grouped scalar sub-expression, `(M + N) * 2`. Do not commit to the
    // tuple reading -- parsePrimary() resolves the ambiguity by parsing the
    // parenthesized body and then looking at what follows the ')', and going
    // through parseExpr() lets a binary operator continue with the
    // group as its left operand. Consuming the '(' here instead would parse
    // `(M + N) * 2` as the tuple `(M + N)` and then fail on the '*'.
    auto expr = parseExpr();
    if (failed(expr)) {
      return failure();
    }
    values.push_back(*expr);

    if (succeeded(parser.parseOptionalRParen())) {
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
  }

  return makeTuple(values);
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parseAdditive() {
  auto left = parseMultiplicative();
  if (failed(left)) {
    return failure();
  }

  while (true) {
    if (succeeded(parser.parseOptionalPlus())) {
      auto right = parseMultiplicative();
      if (failed(right)) {
        return failure();
      }
      left = makeAdd(*left, *right);
      if (failed(left)) {
        return failure();
      }
    } else if (succeeded(parser.parseOptionalMinus())) {
      auto right = parseMultiplicative();
      if (failed(right)) {
        return failure();
      }
      left = makeSub(*left, *right);
      if (failed(left)) {
        return failure();
      }
    } else {
      break;
    }
  }

  return left;
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parseMultiplicative() {
  auto left = parseUnary();
  if (failed(left)) {
    return failure();
  }

  while (true) {
    if (succeeded(parser.parseOptionalStar())) {
      auto right = parseUnary();
      if (failed(right)) {
        return failure();
      }
      left = makeMul(*left, *right);
      if (failed(left)) {
        return failure();
      }
      continue;
    }

    if (succeeded(parser.parseOptionalSlash())) {
      bool isCeilDiv = succeeded(parser.parseOptionalPlus());
      auto right = parseUnary();
      if (failed(right)) {
        return failure();
      }
      left =
          isCeilDiv ? makeCeilDiv(*left, *right) : makeFloorDiv(*left, *right);
      if (failed(left)) {
        return failure();
      }
      continue;
    }

    break;
  }

  return left;
}

OptionalParseResult
CudaTileCompactLayoutParser::parseOptionalIntegerLiteral(bool negate,
                                                         int64_t &result) {
  SMLoc loc = parser.getCurrentLocation();
  llvm::APInt value;
  OptionalParseResult parsed = parser.parseOptionalInteger(value);
  if (!parsed.has_value() || failed(*parsed)) {
    return parsed;
  }

  // AsmParser::parseOptionalInteger widens the literal so that its sign bit
  // stays clear, which makes `value` a signed quantity however the literal
  // was spelled and leaves room for the negation below.
  if (negate) {
    value.negate();
  }
  if (!value.isSignedIntN(64)) {
    return parser.emitError(
        loc, "integer literal does not fit in a signed 64-bit integer");
  }
  result = value.getSExtValue();
  return success();
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parseUnary() {
  bool negated = succeeded(parser.parseOptionalMinus());

  int64_t value = 0;
  OptionalParseResult intResult = parseOptionalIntegerLiteral(negated, value);
  if (intResult.has_value()) {
    if (failed(*intResult)) {
      return failure();
    }
    return makeInt(value);
  }

  if (!negated) {
    return parsePrimary();
  }
  auto operand = parseUnary();
  if (failed(operand)) {
    return failure();
  }
  return makeNeg(*operand);
}

FailureOr<SymbolicExprAttr> CudaTileCompactLayoutParser::parsePrimary() {
  if (succeeded(parser.parseOptionalLParen())) {
    if (succeeded(parser.parseOptionalRParen())) {
      return makeTuple({});
    }

    auto first = parseExpr();
    if (failed(first)) {
      return failure();
    }

    if (succeeded(parser.parseOptionalRParen())) {
      return first;
    }
    if (parser.parseComma()) {
      return failure();
    }

    SmallVector<SymbolicExprAttr, 4> values{*first};
    while (true) {
      if (succeeded(parser.parseOptionalRParen())) {
        return makeTuple(values);
      }

      auto value = parseExpr();
      if (failed(value)) {
        return failure();
      }
      values.push_back(*value);

      if (succeeded(parser.parseOptionalRParen())) {
        return makeTuple(values);
      }
      if (parser.parseComma()) {
        return failure();
      }
    }
  }

  StringRef identifier;
  if (failed(parser.parseOptionalKeyword(&identifier))) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected integer, identifier, or '('");
    return failure();
  }
  return makeVariable(identifier);
}

static CompactExprOperator getCompactExprOperator(SymbolicExprAttr expr) {
  if (isa<SymbolicExprAddAttr>(expr)) {
    return CompactExprOperator::Add;
  }
  if (isa<SymbolicExprSubAttr>(expr)) {
    return CompactExprOperator::Sub;
  }
  if (isa<SymbolicExprMulAttr>(expr)) {
    return CompactExprOperator::Mul;
  }
  if (isa<SymbolicExprFloorDivAttr>(expr)) {
    return CompactExprOperator::FloorDiv;
  }
  if (isa<SymbolicExprCeilDivAttr>(expr)) {
    return CompactExprOperator::CeilDiv;
  }
  if (isa<SymbolicExprNegAttr>(expr)) {
    return CompactExprOperator::Neg;
  }
  return CompactExprOperator::Primary;
}

void CudaTileCompactLayoutParser::printCompactExpr(
    llvm::raw_ostream &os, SymbolicExprAttr expr,
    CompactExprPrecedence parentPrec, bool isRight) {
  printCompactExprWithPrecedence(
      os, getCompactExprOperator(expr), parentPrec, isRight,
      [&](CompactExprPrecedence prec) {
        if (auto intAttr = dyn_cast<SymbolicExprIntAttr>(expr)) {
          os << intAttr.getValue();
        } else if (auto varAttr = dyn_cast<SymbolicExprVarAttr>(expr)) {
          os << varAttr.getName();
        } else if (auto builtinAttr = dyn_cast<SymbolicExprBuiltinAttr>(expr)) {
          os << builtinAttr.getName();
        } else if (auto tupleAttr = dyn_cast<SymbolicTupleAttr>(expr)) {
          os << "(";
          llvm::interleaveComma(
              tupleAttr.getValues(), os,
              [&](SymbolicExprAttr value) { printCompactExpr(os, value); });
          // Keep a one-element tuple distinguishable from a grouped scalar.
          if (tupleAttr.getValues().size() == 1) {
            os << ",";
          }
          os << ")";
        } else if (auto concreteAttr = dyn_cast<ConcreteTypeAttr>(expr)) {
          os << concreteAttr.getType().getValue();
        } else if (auto mulAttr = dyn_cast<SymbolicExprMulAttr>(expr)) {
          printCompactExpr(os, mulAttr.getLhs(), prec, /*isRight=*/false);
          os << "*";
          printCompactExpr(os, mulAttr.getRhs(), prec, /*isRight=*/true);
        } else if (auto divAttr = dyn_cast<SymbolicExprFloorDivAttr>(expr)) {
          printCompactExpr(os, divAttr.getLhs(), prec, /*isRight=*/false);
          os << "/";
          printCompactExpr(os, divAttr.getRhs(), prec, /*isRight=*/true);
        } else if (auto divAttr = dyn_cast<SymbolicExprCeilDivAttr>(expr)) {
          printCompactExpr(os, divAttr.getLhs(), prec, /*isRight=*/false);
          os << "/+";
          printCompactExpr(os, divAttr.getRhs(), prec, /*isRight=*/true);
        } else if (auto addAttr = dyn_cast<SymbolicExprAddAttr>(expr)) {
          printCompactExpr(os, addAttr.getLhs(), prec, /*isRight=*/false);
          os << "+";
          printCompactExpr(os, addAttr.getRhs(), prec, /*isRight=*/true);
        } else if (auto negAttr = dyn_cast<SymbolicExprNegAttr>(expr)) {
          os << "-";
          printCompactExpr(os, negAttr.getOperand(), prec);
        } else if (auto subAttr = dyn_cast<SymbolicExprSubAttr>(expr)) {
          printCompactExpr(os, subAttr.getLhs(), prec, /*isRight=*/false);
          os << "-";
          printCompactExpr(os, subAttr.getRhs(), prec, /*isRight=*/true);
        } else {
          llvm::report_fatal_error(
              "unsupported symbolic expression in compact printer");
        }
      });
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeInt(int64_t value) {
  return SymbolicExprAttr(SymbolicExprIntAttr::get(ctx, value));
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeVariable(StringRef name) {
  if (name.starts_with("context.")) {
    return getCheckedExprAttr<SymbolicExprBuiltinAttr>(
        ctx, parser, startLoc, emittedDiagnostic, name,
        TypeAttr::get(IndexType::get(ctx)));
  }
  return getCheckedExprAttr<SymbolicExprVarAttr>(ctx, parser, startLoc,
                                                 emittedDiagnostic, name);
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::parseConstantArgument() {
  std::string stringValue;
  if (succeeded(parser.parseOptionalString(&stringValue))) {
    return SymbolicExprAttr(SymbolicExprConstStrAttr::get(ctx, stringValue));
  }

  // Parenthesized constant arguments use the symbolic expression grammar,
  // including its grouped-expression and tuple forms.
  const char *cursor = parser.getCurrentLocation().getPointer();
  if (cursor && *cursor == '(')
    return parseExpr();

  Type type;
  auto typeResult = parser.parseOptionalType(type);
  if (typeResult.has_value()) {
    if (failed(*typeResult)) {
      emittedDiagnostic = true;
      return failure();
    }
    return SymbolicExprAttr(ConcreteTypeAttr::get(ctx, TypeAttr::get(type)));
  }
  return parseExpr();
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeNeg(SymbolicExprAttr operand) {
  // The negation of an integer literal is a negative literal, whatever
  // parentheses it was written with. The compact printer emits `-4` for both
  // `-4` and `-(4)`, so keeping a SymbolicExprNegAttr around the literal would
  // print an attribute that reads back as a different one.
  if (auto intAttr = dyn_cast<SymbolicExprIntAttr>(operand)) {
    int64_t value = intAttr.getValue();
    if (value == std::numeric_limits<int64_t>::min()) {
      emittedDiagnostic = true;
      parser.emitError(startLoc, "negated integer literal does not fit in a "
                                 "signed 64-bit integer");
      return failure();
    }
    return SymbolicExprAttr(SymbolicExprIntAttr::get(ctx, -value));
  }
  return SymbolicExprAttr(SymbolicExprNegAttr::get(ctx, operand));
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeAdd(SymbolicExprAttr lhs,
                                     SymbolicExprAttr rhs) {
  return getCheckedExprAttr<SymbolicExprAddAttr>(ctx, parser, startLoc,
                                                 emittedDiagnostic, lhs, rhs);
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeSub(SymbolicExprAttr lhs,
                                     SymbolicExprAttr rhs) {
  return getCheckedExprAttr<SymbolicExprSubAttr>(ctx, parser, startLoc,
                                                 emittedDiagnostic, lhs, rhs);
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeMul(SymbolicExprAttr lhs,
                                     SymbolicExprAttr rhs) {
  return getCheckedExprAttr<SymbolicExprMulAttr>(ctx, parser, startLoc,
                                                 emittedDiagnostic, lhs, rhs);
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeFloorDiv(SymbolicExprAttr lhs,
                                          SymbolicExprAttr rhs) {
  return getCheckedExprAttr<SymbolicExprFloorDivAttr>(
      ctx, parser, startLoc, emittedDiagnostic, lhs, rhs);
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeCeilDiv(SymbolicExprAttr lhs,
                                         SymbolicExprAttr rhs) {
  return getCheckedExprAttr<SymbolicExprCeilDivAttr>(
      ctx, parser, startLoc, emittedDiagnostic, lhs, rhs);
}

FailureOr<SymbolicExprAttr>
CudaTileCompactLayoutParser::makeTuple(ArrayRef<SymbolicExprAttr> values) {
  return SymbolicExprAttr(SymbolicTupleAttr::get(ctx, values));
}

FailureOr<LayoutAttr>
CudaTileCompactLayoutParser::makeLayout(SymbolicExprAttr shapes,
                                        SymbolicExprAttr strides) {
  auto layout = LayoutAttr::getChecked(
      [&]() {
        emittedDiagnostic = true;
        return parser.emitError(startLoc);
      },
      ctx, cast<SymbolicTupleAttr>(shapes), cast<SymbolicTupleAttr>(strides));
  if (!layout) {
    return failure();
  }
  return layout;
}
