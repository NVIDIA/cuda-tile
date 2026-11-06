//===- Remark.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/Remark.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/TileIRRemarks.cpp.inc"

#define DEBUG_TYPE "tileir-remarks"

using namespace llvm;
using namespace mlir;
using namespace mlir::cuda_tile::remark;

using CategoryID = cuda_tile::remark::CategoryID;

//------------------------------------------------------------------------------
// emitRemarkImpl + helpers
//------------------------------------------------------------------------------

union Packed {
  std::uint32_t raw;
  struct {
    std::uint32_t kind : 3;
    std::uint32_t cat : 4;
    std::uint32_t num : 25;
  } bits;
  /// Construct from any of RemarkID / ErrorID / WarningID (all I32 enums
  /// that share the same bit layout).
  template <typename IdT, typename = std::enable_if_t<std::is_enum_v<IdT>>>
  constexpr explicit Packed(IdT id) : raw(static_cast<std::uint32_t>(id)) {}
};

template <typename IdT>
static inline constexpr cuda_tile::remark::DiagnosticKind getKind(IdT id) {
  return static_cast<cuda_tile::remark::DiagnosticKind>(Packed(id).bits.kind);
}

template <typename IdT>
static inline constexpr CategoryID getCategory(IdT id) {
  return static_cast<CategoryID>(Packed(id).bits.cat);
}

/// Get the category string for MLIR remark filtering.
/// Must match the filter strings in tileirAs-cli.cpp.
template <typename IdT>
static StringRef getCategoryString(IdT id) {
  return stringifyCategoryID(getCategory(id));
}

/// Get the remark message from the TableGen-generated doc string.
/// Works for RemarkID / ErrorID / WarningID via the overloaded
/// stringifyEnum() TableGen emits for each I32EnumAttr.
template <typename IdT>
static StringRef getRemarkMessage(IdT id) {
  return stringifyEnum(id).split('@').second;
}

template <typename IdT>
static StringRef getRemarkName(IdT id) {
  return stringifyEnum(id).split('@').first;
}

/// Function pointer type for remark emitters (passed, missed, failed,
/// analysis).
using RemarkEmitFn =
    mlir::remark::detail::InFlightRemark (*)(Location, remark::RemarkOpts);

static StringRef findParentFunctionName(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return funcOp.getName();
  }
  if (auto funcOp = op->getParentOfType<FunctionOpInterface>()) {
    return funcOp.getName();
  }
  llvm_unreachable("no function op found");
}

/// User-facing label prefixes for always-on diagnostics. Consumed by
/// formatRemarkId to produce `RemarkId` strings like "E0000" / "W0000".
static constexpr char kErrorPrefix = 'E';
static constexpr char kWarningPrefix = 'W';

/// Build the user-facing identifier ("E0000" / "W0000").
/// The TableGen `Num` field for Error/Warning entries is the display number
/// directly — `Num=0` renders as `E0000`, `Num=1` as `E0001`, etc.
template <typename IdT>
static std::string formatRemarkId(IdT id, char prefix) {
  unsigned num = Packed(id).bits.num;
  std::string out;
  llvm::raw_string_ostream os(out);
  os << prefix << llvm::format("%04u", num);
  return out;
}

//===----------------------------------------------------------------------===//
// Always-on emission (Error / Warning)
//===----------------------------------------------------------------------===//
//
// Constructs a Remark manually and reports it via RemarkEngine::report()
// directly, bypassing the per-category filter. The only gate is the
// presence of an attached RemarkEngine; once an engine is attached, these
// diagnostics always surface regardless of the `-remarks-filter` options.
// Errors / Warnings don't flow through the InFlightRemark / finalizer path
// — they're a flat emission point with no notes / reason / suggestion
// streaming, just an extra message.

template <typename IdT>
static void emitAlwaysOn(Operation *op, IdT id, StringRef extraMsg,
                         StringRef reason, StringRef suggestion) {
  StringRef name = getRemarkName(id);
  StringRef category = getCategoryString(id);
  StringRef function = findParentFunctionName(op);
  StringRef tmpl = getRemarkMessage(id);
  std::string message = extraMsg.empty() ? std::string(tmpl)
                                         : llvm::formatv(tmpl.data(), extraMsg);
  cuda_tile::remark::DiagnosticKind kind = getKind(id);

  auto *engine = op->getContext()->getRemarkEngine();
  if (!engine) {
    // Fallback to MLIR native diagnostics when no remark engine is attached.
    if (kind == cuda_tile::remark::DiagnosticKind::Error)
      op->emitError() << message;
    else
      op->emitWarning() << message;
    return;
  }

  auto opts = mlir::remark::RemarkOpts::name(name).category(category).function(
      function);

  bool isError = kind == cuda_tile::remark::DiagnosticKind::Error;
  auto rkind = isError ? mlir::remark::RemarkKind::RemarkFailure
                       : mlir::remark::RemarkKind::RemarkMissed;
  auto severity = isError ? mlir::DiagnosticSeverity::Error
                          : mlir::DiagnosticSeverity::Warning;
  char prefix = isError ? kErrorPrefix : kWarningPrefix;

  mlir::remark::detail::Remark remark(rkind, severity, op->getLoc(), opts);
  remark.insert(message);
  remark.insert(mlir::remark::detail::Remark::Arg("RemarkId",
                                                  formatRemarkId(id, prefix)));
  // Explicit classification arg consumed by the TileIR streamer so it can
  // route/format always-on diagnostics without sniffing the RemarkId prefix.
  remark.insert(mlir::remark::detail::Remark::Arg(
      "DiagClass", isError ? StringRef("Error") : StringRef("Warning")));
  if (!reason.empty()) {
    remark.insert(mlir::remark::detail::Remark::Arg("Reason", reason.str()));
  }
  if (!suggestion.empty()) {
    remark.insert(
        mlir::remark::detail::Remark::Arg("Suggestion", suggestion.str()));
  }
  engine->report(std::move(remark));
}

//===----------------------------------------------------------------------===//
// emitRemarkImpl
//===----------------------------------------------------------------------===//
//
// Filter-gated path for RemarkID-based (Pass / Miss / Fail / Analysis)
// remarks. Called from the InFlightRemark defaultFinalizer with already-
// materialized notes / reasons / suggestions. A TableGen-level assert
// forbids TileIRRemark<> from being constructed with DK_Error /
// DK_Warning, so this path should only ever see the four filter-gated
// kinds.
static void emitRemarkImpl(Operation *op, RemarkID id, StringRef extraMsg,
                           ArrayRef<std::pair<std::string, std::string>> notes,
                           ArrayRef<std::string> reasons,
                           ArrayRef<std::string> suggestions) {
  StringRef category = getCategoryString(id);
  StringRef messageTemplate = getRemarkMessage(id);
  StringRef name = getRemarkName(id);
  auto function = findParentFunctionName(op);
  auto opts = mlir::remark::RemarkOpts::name(name).category(category).function(
      function);
  cuda_tile::remark::DiagnosticKind diagnosticKind = getKind(id);

  std::string message = extraMsg.empty()
                            ? std::string(messageTemplate)
                            : llvm::formatv(messageTemplate.data(), extraMsg);

  RemarkEmitFn emitFn;
  switch (diagnosticKind) {
  case cuda_tile::remark::DiagnosticKind::Pass:
    emitFn = mlir::remark::passed;
    break;
  case cuda_tile::remark::DiagnosticKind::Miss:
    emitFn = mlir::remark::missed;
    break;
  case cuda_tile::remark::DiagnosticKind::Fail:
    emitFn = mlir::remark::failed;
    break;
  case cuda_tile::remark::DiagnosticKind::Analysis:
    emitFn = mlir::remark::analysis;
    break;
  default:
    llvm_unreachable("Invalid diagnostic kind for filter-gated remark path; "
                     "Error/Warning must use reportError / reportWarning");
  }
  auto diag = emitFn(op->getLoc(), opts);

  diag << message;

  for (const auto &[key, value] : notes) {
    diag << mlir::remark::metric(key, value);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Category: " << category << "\n";
    llvm::dbgs() << "Name: " << name << "\n";
    llvm::dbgs() << "Message: " << message << "\n";
    for (const auto &[key, value] : notes) {
      llvm::dbgs() << "Metric[" << key << "]: " << value << "\n";
    }
    llvm::dbgs() << "Reasons: [";
    llvm::interleaveComma(reasons, llvm::dbgs());
    llvm::dbgs() << "]\n";
    llvm::dbgs() << "Suggestions: [";
    llvm::interleaveComma(suggestions, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  for (auto r : reasons) {
    diag << mlir::remark::reason("{0}", r);
  }
  for (auto s : suggestions) {
    diag << mlir::remark::suggest("{0}", s);
  }
}

//------------------------------------------------------------------------------
// defaultFinalizer + helpers
//------------------------------------------------------------------------------

/// Pretty-print a location to the provided stream.
static llvm::raw_ostream &prettyPrintLocation(Location loc,
                                              llvm::raw_ostream &os) {
  // If this is a FileLineColLoc, we can pretty-print it.
  if (auto fileLoc = llvm::dyn_cast<FileLineColLoc>(loc)) {
    os << fileLoc.getFilename().getValue() << ":" << fileLoc.getLine() << ":"
       << fileLoc.getColumn();
  } else {
    os << loc;
  }

  return os;
}

/// Pretty-print an attribute to ensure we get the thing we expect across the
/// various tests/etc. This helps us ensure uniformity whenever possible.
static void prettyPrintAttr(Attribute attr, llvm::raw_ostream &os) {
  auto printArray = [&](auto arr) {
    os << "[";
    llvm::interleaveComma(arr.asArrayRef(), os);
    os << "]";
  };
  llvm::TypeSwitch<Attribute>(attr)
      .Case([&](StringAttr str) { os << str.getValue(); })
      .Case<DenseI8ArrayAttr>(printArray)
      .Case<DenseI16ArrayAttr>(printArray)
      .Case<DenseI32ArrayAttr>(printArray)
      .Case<DenseI64ArrayAttr>(printArray)
      .Case<ArrayAttr>([&](ArrayAttr arr) {
        os << "[";
        // Recurse for ArrayAttr - we don't want to print the types for the
        // stuff inside the array either (for example).
        llvm::interleaveComma(arr, os,
                              [&](Attribute a) { prettyPrintAttr(a, os); });
        os << "]";
      })
      .Case([&](IntegerAttr i) { os << i.getValue(); })
      .Case([&](FloatAttr f) { os << f.getValue(); })
      .Default([&](Attribute attr) { os << attr; });
}

/// Provides a default finalizer for InFlightRemark that just outputs to the
/// mlir::remark infrastructure directly via emitRemarkImpl. Useful as a
/// fallback if debuginfo isn't enabled, for example.
static LogicalResult defaultFinalizer(Operation *op, InFlightRemark &rmk) {
  // Otherwise, use the default remark emission.
  std::vector<std::pair<std::string, std::string>> notes;
  for (auto note : rmk.getNotes()) {
    auto &back = notes.emplace_back();
    llvm::raw_string_ostream keyStream(back.first);
    prettyPrintAttr(note.key, keyStream);

    llvm::raw_string_ostream valueStream(back.second);

    if (note.loc) {
      prettyPrintLocation(*note.loc, valueStream) << ": ";
    }

    prettyPrintAttr(note.value, valueStream);
  }

  SmallVector<std::string> fmtSuggestions, fmtReasons;
  for (const auto &suggestion : rmk.getSuggestions()) {
    auto &back = fmtSuggestions.emplace_back();
    llvm::raw_string_ostream stream(back);
    if (suggestion.loc) {
      prettyPrintLocation(*suggestion.loc, stream) << ": ";
    }

    stream << suggestion.message;
  }

  for (const auto &reason : rmk.getReasons()) {
    auto &back = fmtReasons.emplace_back();
    llvm::raw_string_ostream stream(back);
    // Only print the reason location if the reason is on a different op.
    if (reason.op != op) {
      prettyPrintLocation(reason.op->getLoc(), stream) << ": ";
    }
    stream << reason.message;
  }

  emitRemarkImpl(op, rmk.getName(), rmk.getMessage(), notes, fmtReasons,
                 fmtSuggestions);
  return success();
}

//------------------------------------------------------------------------------
// InFlightRemark
//------------------------------------------------------------------------------

mlir::cuda_tile::remark::InFlightRemark::~InFlightRemark() {
  // If we don't have a finalizer, or the finalizer fails, run the default one.
  if (!finalizer || failed(finalizer(op, *this))) {
    (void)defaultFinalizer(op, *this);
  }
}

InFlightString InFlightRemark::addSuggestion(std::optional<Location> loc) {
  // Push back a new Suggestion. The message field will be filled in by the
  // InFlightString as the user fills it in.
  suggestions.push_back({loc, {}});
  auto &suggestion = suggestions.back();
  // The callback to add a note is provided here so the user can stream in a
  // symbol and we can handle it the same way as we do here.
  return InFlightString{
      [this](Location loc, Attribute k, Attribute v) { addNote(loc, k, v); },
      suggestion.message};
}

InFlightString InFlightRemark::addReason(Operation *reasonOp) {
  // Default to this operation.
  if (!reasonOp) {
    reasonOp = op;
  }
  // Push back a new Reason. The message field will be filled in by the
  // InFlightString as the user fills it in.
  reasons.push_back({reasonOp, {}});
  auto &reason = reasons.back();
  // The callback to add a note is provided here so the user can stream in a
  // symbol and we can handle it the same way as we do here.
  return InFlightString{
      [this](Location loc, Attribute k, Attribute v) { addNote(loc, k, v); },
      reason.message};
}

//------------------------------------------------------------------------------
// reportRemark
//------------------------------------------------------------------------------

InFlightRemark
mlir::cuda_tile::remark::reportRemark(Operation *op, RemarkID id,
                                      InFlightRemark::Finalizer finalizer) {
  // Check if the dialect has a finalizer defined on it. No finalizer provided
  // by the user means we use the one on the dialect.
  if (!finalizer) {
    auto *ctx = op->getContext();
    if (auto *ctDialect = ctx->getLoadedDialect<cuda_tile::CudaTileDialect>()) {
      finalizer = ctDialect->getRemarkFinalizer();
    }
  }
  return InFlightRemark(finalizer, op, id);
}

InFlightRemark mlir::cuda_tile::remark::reportRemark(
    Operation *op, RemarkID id, StringRef message, StringRef reason,
    StringRef suggestion, InFlightRemark::Finalizer finalizer) {
  auto rmk = reportRemark(op, id, finalizer);
  rmk << message;

  // Add the reason if we have one.
  if (!reason.empty()) {
    rmk.addReason(op) << reason;
  }

  // Add the suggestion if we have one.
  if (!suggestion.empty()) {
    rmk.addSuggestion() << suggestion;
  }
  return rmk;
}

void cuda_tile::remark::reportError(Operation *op, ErrorID id,
                                    StringRef extraMsg, StringRef reason,
                                    StringRef suggestion) {
  // Emits through the engine bypassing the per-category filter; does NOT
  // end compilation — caller must signalPassFailure() / return failure()
  // to propagate.
  emitAlwaysOn(op, id, extraMsg, reason, suggestion);
}

void cuda_tile::remark::reportWarning(Operation *op, WarningID id,
                                      StringRef extraMsg, StringRef reason,
                                      StringRef suggestion) {
  emitAlwaysOn(op, id, extraMsg, reason, suggestion);
}
